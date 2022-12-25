import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import cos_sim


class AttentionLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout, alpha, decive):
        super(AttentionLayer, self).__init__()
        self.dropout = dropout

        self.W = nn.Parameter(
            nn.init.xavier_normal_(torch.FloatTensor(hidden_dim, hidden_dim).to(decive), gain=1.414),
            requires_grad=True)
        self.register_parameter('W', self.W)

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(in_dim, hidden_dim)

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x_1, x_2):
        h_1 = self.leakyrelu(self.fc1(x_1))
        h_2 = self.leakyrelu(self.fc2(x_2))
        out = torch.matmul(h_1, torch.matmul(self.W, h_2.transpose(-1, -2)))
        out = F.dropout(out, self.dropout, training=self.training)
        return out


class AttributeCompletion(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, dropout, alpha, decive):
        super(AttributeCompletion, self).__init__()
        self.attentions = [AttentionLayer(in_dim, hidden_dim, dropout, alpha, decive) for _ in range(num_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x_1, x_2):
        out = [att(x_1, x_2).unsqueeze(0) for att in self.attentions]
        out = torch.cat(out, dim=0)
        out = torch.mean(out, dim=0, keepdim=False)
        return out


class TransformerLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout, alpha, att):
        super(TransformerLayer, self).__init__()
        self.dropout = dropout
        self.fc = nn.Linear(in_dim, hidden_dim)
        self.Q = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.K = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.attention = att

    def forward(self, x, y=None):
        if y is None:
            return self.leakyrelu(self.fc(x))
        x = self.leakyrelu(self.fc(x))
        y = self.leakyrelu(self.fc(y))

        query = self.Q(x)
        key = self.K(y)
        vec = self.V(y)
        attention = self.attention(key, query)
        e = F.softmax(attention, dim=1)
        h = torch.sum(self.leakyrelu(torch.mul(vec, e)), dim=1, keepdim=True)
        h = F.dropout(h, self.dropout, training=self.training)
        out = (x + h) / 2
        return out


class TransformerAgg(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, dropout, alpha, att):
        super(TransformerAgg, self).__init__()
        self.transformers = [TransformerLayer(in_dim, hidden_dim, dropout, alpha, att[k]) for k in range(num_heads)]
        for i, transformer in enumerate(self.transformers):
            self.add_module('transformer_{}'.format(i), transformer)

    def forward(self, x, y=None):
        out = [transformer(x, y) for transformer in self.transformers]
        out = torch.cat(out, dim=-1)

        return out


class SemanticAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1, keepdim=True)


class HGNN(nn.Module):
    def __init__(self, expected_metapaths, in_dim, hidden_dim, num_heads, dropout, alpha, decive):
        super(HGNN, self).__init__()
        self.expected_metapaths = expected_metapaths

        self.attentions = [
            [AttentionLayer(hidden_dim, hidden_dim, dropout, alpha, decive) for _ in range(num_heads)]
            for
            _ in expected_metapaths]
        for i in range(len(self.attentions)):
            for j, att in enumerate(self.attentions[i]):
                self.add_module('attention_path{}_head{}'.format(i, j), att)
        # ====================================================================
        self.transformer_agg = [
            [TransformerAgg(in_dim, hidden_dim, num_heads, dropout, alpha, self.attentions[k]) for _ in range(len(expected_metapaths[k]) - 1)]
            for
            k in range(len(expected_metapaths))]
        for i in range(len(self.transformer_agg)):
            for j, transformer in enumerate(self.transformer_agg[i]):
                self.add_module('transformer_{}{}'.format(i, j), transformer)
        # ====================================================================
        self.inner_metapath_semantic_attention = [SemanticAttention(hidden_dim * num_heads, hidden_dim) for _ in
                                                  range(len(expected_metapaths))]
        for i, inner in enumerate(self.inner_metapath_semantic_attention):
            self.add_module('inner_metapath_semantic_attention{}'.format(i), inner)
        self.between_metapath_semantic_attention = SemanticAttention(hidden_dim * num_heads, hidden_dim)

    def forward(self, x, target_nodes_mask=None, neighbors=None):
        if target_nodes_mask is None and neighbors is None:
            x = x.unsqueeze(1)
            metapath_based_emb = []
            for k, metapath in enumerate(self.expected_metapaths):
                emb = []
                for i in range(len(metapath) - 1):
                    cur_emb = self.transformer_agg[k][i](x)
                    emb.append(cur_emb)
                emb = torch.cat(emb, dim=1)
                metapath_based_emb.append(self.inner_metapath_semantic_attention[k](emb))
            metapath_based_emb = torch.cat(metapath_based_emb, dim=1)
            return self.between_metapath_semantic_attention(metapath_based_emb).squeeze(1)

        target_nodes_x = F.embedding(target_nodes_mask, x).unsqueeze(1)
        metapath_based_emb = []
        for k, metapath in enumerate(self.expected_metapaths):
            emb = []
            for i in range(len(metapath) - 1):
                neighbors_x = F.embedding(neighbors[k][i], x)
                # print(target_nodes_x.shape, neighbors_x.shape)
                cur_emb = self.transformer_agg[k][i](target_nodes_x, neighbors_x)
                emb.append(cur_emb)
            emb = torch.cat(emb, dim=1)
            metapath_based_emb.append(self.inner_metapath_semantic_attention[k](emb))
        metapath_based_emb = torch.cat(metapath_based_emb, dim=1)
        return self.between_metapath_semantic_attention(metapath_based_emb).squeeze(1)


class MyModel(nn.Module):
    def __init__(self, emb_dim, hidden_dim, out_dim, num_feats, num_heads, expected_metapaths, dropout,
                 alpha=0.2, loss_weight=None, tau=0.5, device=torch.device('cpu')):
        super(MyModel, self).__init__()
        self.loss_weight = loss_weight
        self.tau = tau
        self.device = device

        self.fc_nodeEmb2hidden = nn.Linear(emb_dim, hidden_dim, bias=True)
        self.fc_featEmb2hidden = nn.Linear(emb_dim, hidden_dim, bias=True)

        self.attributeCompletion = AttributeCompletion(in_dim=hidden_dim, hidden_dim=out_dim, num_heads=num_heads,
                                                       dropout=dropout, alpha=alpha, decive=device)
        self.fc_cmpFeat2hidden = nn.Linear(num_feats, hidden_dim)
        self.fc_realFeat2hidden = nn.Linear(num_feats, hidden_dim)

        self.hgnn = HGNN(expected_metapaths=expected_metapaths, in_dim=hidden_dim, hidden_dim=out_dim,
                         num_heads=num_heads, dropout=dropout, alpha=alpha, decive=device)

        self.W = nn.Sequential(nn.Linear(out_dim * num_heads, out_dim),
                               nn.LeakyReLU(alpha),
                               nn.Linear(out_dim, out_dim))
        self.W1 = nn.Sequential(nn.Linear(out_dim * num_heads, out_dim),
                                nn.LeakyReLU(alpha),
                                nn.Linear(out_dim, out_dim))
        self.W2 = nn.Sequential(nn.Linear(num_feats, out_dim),
                                nn.LeakyReLU(alpha),
                                nn.Linear(out_dim, out_dim))

        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU(alpha)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def contrastive_loss(self, x, y, bias=None):
        f = lambda x: torch.exp(x / self.tau)
        cos_inner = f(cos_sim(x, x))
        cos_between = f(cos_sim(x, y))
        if bias is None:
            cost = - torch.log(cos_between.diag() / (cos_inner.sum(1) + cos_between.sum(1) - cos_inner.diag()))
        else:
            pos = cos_between.diag() + torch.mul(cos_between, bias).sum(1) + torch.mul(cos_inner, bias).sum(1)
            reverse_bias = torch.ones_like(bias) - bias
            neg = torch.mul(cos_between, reverse_bias).sum(1) + torch.mul(cos_inner, reverse_bias).sum(1) - cos_inner.diag()
            cost = - torch.log(pos / neg)
        return cost.mean()

    def forward(self, node_emb, feat_emb, lbl_feat, feat_mask, target_nodes_mask, neighbor_mask, loss_bias):
        # Attribute Completion
        node_emb = self.leakyrelu(self.fc_nodeEmb2hidden(node_emb))
        feat_emb = self.leakyrelu(self.fc_featEmb2hidden(feat_emb))
        cmp_feat = self.attributeCompletion(node_emb, feat_emb)
        # mapping
        x_lbl = self.leakyrelu(self.fc_realFeat2hidden(lbl_feat))
        x_cmp = self.leakyrelu(self.fc_cmpFeat2hidden(self.leakyrelu(cmp_feat)))
        # Data Augmentation
        x_view1 = torch.zeros_like(x_cmp)
        x_view1[feat_mask['exist']] = x_lbl
        x_view1[feat_mask['miss']] = x_cmp[feat_mask['miss']]
        x_view2 = x_cmp
        # HGNN
        final_emb1 = self.hgnn(x_view1, target_nodes_mask, neighbor_mask)
        final_emb2 = self.hgnn(x_view2, target_nodes_mask, neighbor_mask)
        # loss
        loss_1 = 0.5 * (self.contrastive_loss(self.W(final_emb1), self.W(final_emb2), loss_bias) +
                        self.contrastive_loss(self.W(final_emb2), self.W(final_emb1), loss_bias))
        loss_2 = 0.5 * (self.contrastive_loss(self.W1(final_emb1), self.W2(lbl_feat[target_nodes_mask])) +
                        self.contrastive_loss(self.W1(final_emb2), self.W2(lbl_feat[target_nodes_mask])))

        return final_emb1.detach(), final_emb2.detach(), self.loss_weight[0] * loss_1, self.loss_weight[1] * loss_2

    def evaluation(self, node_emb, feat_emb, lbl_feat, feat_mask, target_nodes_mask, neighbor_mask):
        # Attribute Completion
        node_emb = self.leakyrelu(self.fc_nodeEmb2hidden(node_emb))
        feat_emb = self.leakyrelu(self.fc_featEmb2hidden(feat_emb))
        cmp_feat = self.attributeCompletion(node_emb, feat_emb)
        # mapping
        x_lbl = self.leakyrelu(self.fc_realFeat2hidden(lbl_feat))
        x_cmp = self.leakyrelu(self.fc_cmpFeat2hidden(self.leakyrelu(cmp_feat)))

        x = torch.zeros_like(x_cmp)
        x[feat_mask['exist']] = x_lbl
        x[feat_mask['miss']] = x_cmp[feat_mask['miss']]

        emb1 = self.hgnn(x, target_nodes_mask, neighbor_mask)
        emb2 = self.hgnn(x_cmp, target_nodes_mask, neighbor_mask)
        return emb1.detach(), emb2.detach()
