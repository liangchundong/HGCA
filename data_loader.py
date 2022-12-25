import numpy as np
import scipy.sparse as sp
import torch
import math
import random


class DataLoader:
    def __init__(self, dataset, feat_node_type, batch_size, neighbor_size, expected_metapaths, metapath_weight,
                 device=torch.device('cpu')):
        self.type_mask = np.load('data/{}/node_types.npy'.format(dataset))
        self.labels = np.load('data/{}/labels.npy'.format(dataset))
        train_val_test_idx = np.load('data/{}/train_val_test_idx.npz'.format(dataset))
        self.train_idx = train_val_test_idx['train_idx']
        self.val_idx = train_val_test_idx['val_idx']
        self.test_idx = train_val_test_idx['test_idx']

        feats = []
        for i in range(feat_node_type + 1):
            feats.append(sp.load_npz('data/{}/features_{}.npz'.format(dataset, i)).toarray().astype(np.float))
        self.lbl_feat = np.concatenate(feats, axis=0)
        self.lbl_feat_mask = np.where(self.type_mask <= feat_node_type)[0]
        self.node_emb = np.load('data/{}/metapath2vec_emb_node.npy'.format(dataset))
        self.feat_emb = np.load('data/{}/metapath2vec_emb_word.npy'.format(dataset))
        self.feat_emb = torch.FloatTensor(self.feat_emb).to(device)

        self.neighbor_nodes = [[] for _ in range(len(expected_metapaths))]
        for k, metapath in enumerate(expected_metapaths):
            for i in range(1, len(metapath)):
                dic = np.load(
                    'data/{}/indices/('.format(dataset) + '-'.join(map(str, metapath)) + '){}-{}.npy'.format(
                        i, metapath[i]),
                    allow_pickle=True).item()
                self.neighbor_nodes[k].append(dic)

        self.metapath_neighbors = []
        for i, metapath in enumerate(expected_metapaths):
            adj = sp.load_npz('data/{}/adj_{}.npz'.format(dataset, metapath)).toarray().astype(np.float)
            adj = adj * metapath_weight[i]
            np.fill_diagonal(adj, 0.0)
            self.metapath_neighbors.append(adj)

        self.indices_all = np.arange(np.where(self.type_mask == 0)[0].shape[0])
        self.indices_test = np.copy(self.test_idx)
        np.random.shuffle(self.indices_all)
        self.iter_counter_all = 0
        self.iter_counter_test = 0
        self.num_data_all = len(self.indices_all)
        self.num_data_test = len(self.indices_test)

        self.expected_metapaths = expected_metapaths
        self.batch_size = batch_size
        self.neighbor_size = neighbor_size
        self.device = device

    def data_info(self):
        emb_dim = self.node_emb.shape[1]
        num_feats = self.lbl_feat.shape[1]
        num_classes = np.max(self.labels) + 1
        return emb_dim, num_feats, num_classes

    def data_partition(self):
        return self.train_idx, self.val_idx, self.test_idx

    def num_iterations_all(self):
        return int(np.ceil(self.num_data_all / self.batch_size))

    def num_iterations_test(self):
        return int(np.ceil(self.num_data_test / self.batch_size))

    def reset_all(self):
        np.random.shuffle(self.indices_all)
        self.iter_counter_all = 0

    def reset_test(self):
        self.iter_counter_test = 0

    def next(self):
        if self.num_iterations_all() - self.iter_counter_all <= 0:
            self.reset_all()
        self.iter_counter_all += 1
        cur_indices = np.copy(self.indices_all[(self.iter_counter_all - 1) * self.batch_size:self.iter_counter_all * self.batch_size])
        cur_indices = np.sort(cur_indices)
        cur_label = self.labels[cur_indices]
        cur_all_nodes = set(cur_indices)
        # print(cur_all_nodes)

        cur_neighbors = [
            [np.zeros((len(cur_indices), self.neighbor_size)) for _ in range(len(self.expected_metapaths[k]) - 1)] for k
            in range(len(self.expected_metapaths))]
        for cur_id, node_id in enumerate(cur_indices):
            for k, metapath in enumerate(self.expected_metapaths):
                for i in range(len(self.expected_metapaths[k]) - 1):
                    tmp = self.neighbor_nodes[k][i][node_id]
                    tmp = tmp * math.ceil(self.neighbor_size / len(tmp))
                    tmp = random.sample(tmp, self.neighbor_size)
                    cur_neighbors[k][i][cur_id] = tmp
                    cur_all_nodes = cur_all_nodes.union(tmp)

        cur_all_nodes = sorted(cur_all_nodes)
        mapping = {map_from: map_to for map_to, map_from in enumerate(cur_all_nodes)}
        lbl_feat_indices = np.array([k for k in cur_all_nodes if k in self.lbl_feat_mask])
        lbl_feat_mask = np.array([mapping[k] for k in lbl_feat_indices])
        target_nodes_mask = np.array([mapping[k] for k in cur_indices])

        loss_bias = []
        for i, adj in enumerate(self.metapath_neighbors):
            loss_bias.append(torch.FloatTensor(adj[target_nodes_mask][:, target_nodes_mask]).to(self.device).unsqueeze(0))
        loss_bias = torch.sum(torch.cat(loss_bias, dim=0), dim=0, keepdim=False)
        loss_bias = torch.where(loss_bias > 1.0, torch.ones_like(loss_bias), loss_bias)

        target_nodes_mask = torch.LongTensor(target_nodes_mask).to(self.device)
        not_lbl_feat_indices = np.array([k for k in cur_all_nodes if k not in self.lbl_feat_mask])
        not_lbl_feat_mask = np.array([mapping[k] for k in not_lbl_feat_indices])

        for metapath_id, metapath_neighbors in enumerate(cur_neighbors):
            for type_id in range(len(metapath_neighbors)):
                neighbor_indice = cur_neighbors[metapath_id][type_id]
                for i in range(neighbor_indice.shape[0]):
                    for j in range(neighbor_indice.shape[1]):
                        neighbor_indice[i, j] = mapping[neighbor_indice[i, j]]
                neighbor_mask = torch.LongTensor(neighbor_indice).to(self.device)
                cur_neighbors[metapath_id][type_id] = neighbor_mask

        cur_all_nodes = np.array(list(cur_all_nodes))
        cur_node_emb = torch.FloatTensor(self.node_emb[cur_all_nodes]).to(self.device)
        lbl_feat = torch.FloatTensor(self.lbl_feat[lbl_feat_indices]).to(self.device)
        cur_label = torch.LongTensor(cur_label).to(self.device)
        feat_mask = {'exist': lbl_feat_mask, 'miss': not_lbl_feat_mask}

        return cur_node_emb, self.feat_emb, lbl_feat, feat_mask, target_nodes_mask, cur_label, cur_neighbors, loss_bias

    def next4test(self):
        if self.num_iterations_test() - self.iter_counter_test <= 0:
            self.reset_test()
        self.iter_counter_test += 1
        cur_indices = np.copy(
            self.indices_test[(self.iter_counter_test - 1) * self.batch_size:self.iter_counter_test * self.batch_size])
        cur_indices = np.sort(cur_indices)
        cur_label = self.labels[cur_indices]
        cur_all_nodes = set(cur_indices)
        # print(cur_all_nodes)

        cur_neighbors = [
            [np.zeros((len(cur_indices), self.neighbor_size)) for _ in range(len(self.expected_metapaths[k]) - 1)] for k
            in range(len(self.expected_metapaths))]
        for cur_id, node_id in enumerate(cur_indices):
            for k, metapath in enumerate(self.expected_metapaths):
                for i in range(len(self.expected_metapaths[k]) - 1):
                    tmp = self.neighbor_nodes[k][i][node_id]
                    tmp = tmp * math.ceil(self.neighbor_size / len(tmp))
                    tmp = random.sample(tmp, self.neighbor_size)
                    cur_neighbors[k][i][cur_id] = tmp
                    cur_all_nodes = cur_all_nodes.union(tmp)

        cur_all_nodes = sorted(cur_all_nodes)
        mapping = {map_from: map_to for map_to, map_from in enumerate(cur_all_nodes)}
        lbl_feat_indices = np.array([k for k in cur_all_nodes if k in self.lbl_feat_mask])
        lbl_feat_mask = np.array([mapping[k] for k in lbl_feat_indices])
        target_nodes_mask = np.array([mapping[k] for k in cur_indices])
        target_nodes_mask = torch.LongTensor(target_nodes_mask).to(self.device)
        not_lbl_feat_indices = np.array([k for k in cur_all_nodes if k not in self.lbl_feat_mask])
        not_lbl_feat_mask = np.array([mapping[k] for k in not_lbl_feat_indices])

        # print(mapping)
        for metapath_id, metapath_neighbors in enumerate(cur_neighbors):
            for type_id in range(len(metapath_neighbors)):
                neighbor_indice = cur_neighbors[metapath_id][type_id]
                for i in range(neighbor_indice.shape[0]):
                    for j in range(neighbor_indice.shape[1]):
                        neighbor_indice[i, j] = mapping[neighbor_indice[i, j]]
                neighbor_mask = torch.LongTensor(neighbor_indice).to(self.device)
                cur_neighbors[metapath_id][type_id] = neighbor_mask

        cur_all_nodes = np.array(list(cur_all_nodes))
        cur_node_emb = torch.FloatTensor(self.node_emb[cur_all_nodes]).to(self.device)
        lbl_feat = torch.FloatTensor(self.lbl_feat[lbl_feat_indices]).to(self.device)
        cur_label = torch.LongTensor(cur_label).to(self.device)
        feat_mask = {'exist': lbl_feat_mask, 'miss': not_lbl_feat_mask}

        return cur_node_emb, self.feat_emb, lbl_feat, feat_mask, target_nodes_mask, cur_label, cur_neighbors
