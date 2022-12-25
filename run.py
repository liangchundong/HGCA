import argparse
import torch
import time
import re
import os
from utils import evaluate_results_nc, setup_seed, EarlyStopping
from model import MyModel as Model
from data_loader import DataLoader


'''
python -u -W ignore run.py --cuda --dataset Yelp --metapath-weight 0.2#0.6#0.2 --loss-weight 0.7#0.3 --lr-decay-start 10
python -u -W ignore run.py --cuda --dataset ACM --metapath-weight 0.4#0.6 --loss-weight 0.5#0.5 --lr-decay-start 10
python -u -W ignore run.py --cuda --dataset DBLP --metapath-weight 0.1#0.1#0.8 --loss-weight 0.5#0.5 --lr-decay-start 20

nohup python3.7 -u -W ignore run.py --cuda --dataset ACM --metapath-weight 0.4#0.6 > log-ACM.out 2>&1 &
nohup python3.7 -u -W ignore run.py --cuda --dataset Yelp --metapath-weight 0.2#0.6#0.2 > log-Yelp.out 2>&1 &
nohup python3.7 -u -W ignore run.py --cuda --dataset DBLP --metapath-weight 0.1#0.1#0.8 > log-DBLP.out 2>&1 &
'''

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
ap = argparse.ArgumentParser(description='args')
ap.add_argument('--dataset', default='ACM', help='Dataset.')
ap.add_argument('--batch-size', type=int, default=512, help='Batch Size.')
ap.add_argument('--neighbor-size', type=int, default=64, help='Neighbor Size.')
ap.add_argument('--hidden-dim', type=int, default=128, help='Hidden dim.')
ap.add_argument('--out-dim', type=int, default=64, help='Output dim.')
ap.add_argument('--alpha', type=float, default=0.2, help='Alpha of LeakyReLu.')
ap.add_argument('--dropout', type=float, default=0.3, help='Dropout. Default is 0.3.')
ap.add_argument('--num_heads', type=int, default=4, help='Multi Head. Default is 4.')
ap.add_argument('--epoch', type=int, default=40, help='Number of epochs. Default is 20.')
ap.add_argument('--patience', type=int, default=5, help='Patience. Default is 5.')
ap.add_argument('--repeat', type=int, default=1, help='Repeat. Default is 1.')
ap.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
ap.add_argument('--weight-decay', type=float, default=0.001, help='Weight decay.')
ap.add_argument('--seed', type=int, default=65535, help='Random seed.')
ap.add_argument('--cuda', action='store_true', default=False, help='Using GPU or not.')
ap.add_argument('--tau', type=float, default=0.5, help='Temperature.')
ap.add_argument('--loss-weight', type=str, default='0.5#0.5', help='Loss weight.')
ap.add_argument('--metapath-weight', type=str, default='0.4#0.6', help='Meta-path weight.')
args = ap.parse_args()
setup_seed(args.seed, args.cuda)
print(args)
print('')

# split
loss_weight = [float(x) for x in re.split('#', args.loss_weight)]
metapath_weight = [float(x) for x in re.split('#', args.metapath_weight)]

# dataset info
feat_node_type = {'DBLP': 1, 'ACM': 0, 'Yelp': 0}
feat_node_type = feat_node_type[args.dataset]
expected_metapaths = {'ACM': [(0, 1, 0), (0, 2, 0)],
                      'Yelp': [(0, 1, 0), (0, 2, 0), (0, 3, 0)],
                      'DBLP': [(0, 1, 0), (0, 1, 2, 1, 0), (0, 1, 3, 1, 0)]}
expected_metapaths = expected_metapaths[args.dataset]
assert len(metapath_weight) == len(expected_metapaths), 'meta-path weight error!'

# device
if args.cuda and torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('Using CUDA')
else:
    device = torch.device('cpu')
    print('Using CPU')

# train
print('start train with repeat = {}\n'.format(args.repeat))
for cur_repeat in range(args.repeat):
    print('cur_repeat = {}   ==============================================================='.format(cur_repeat))
    data_loader = DataLoader(dataset=args.dataset, feat_node_type=feat_node_type, batch_size=args.batch_size,
                             neighbor_size=args.neighbor_size, expected_metapaths=expected_metapaths, device=device,
                             metapath_weight=metapath_weight)
    emb_dim, num_feats, num_classes = data_loader.data_info()
    net = Model(emb_dim=emb_dim, hidden_dim=args.hidden_dim, out_dim=args.out_dim, num_feats=num_feats,
                num_heads=args.num_heads, expected_metapaths=expected_metapaths, dropout=args.dropout,
                alpha=args.alpha, loss_weight=loss_weight, tau=args.tau, device=device)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print('model init finish')

    # training loop
    print('training...')
    net.train()
    early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                   save_path='checkpoint/checkpoint_{}'.format(args.dataset))
    for epoch in range(args.epoch):
        # training
        train_time = time.time()
        net.train()
        train_loss_avg = 0
        for iteration in range(data_loader.num_iterations_all()):
            node_emb, feat_emb, lbl_feat, feat_mask, target_nodes_mask, label, neighbor_mask, loss_bias = data_loader.next()
            _, _, loss_1, loss_2 = net(node_emb, feat_emb, lbl_feat, feat_mask, target_nodes_mask, neighbor_mask, loss_bias)

            train_loss = loss_1 + loss_2
            train_loss_avg += train_loss.item()
            # auto grad
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            # print('\titeration {:02d} | loss_train: {:.6f} | loss_detail: {:.6f} # {:.6f} # {:.6f}'.format(
            #     iteration, train_loss.item(), loss_recon.item(), loss_innerOutput.item(), loss_betweenOutAndIn.item()))

        train_loss_avg /= data_loader.num_iterations_all()
        train_time = time.time() - train_time
        print('Epoch {:04d} | Loss {:.4f} | Time(s) {:.4f}'.format(epoch, train_loss_avg, train_time))

        # testing
        embeddings, labels = [], []
        net.eval()
        with torch.no_grad():
            for iteration in range(data_loader.num_iterations_test()):
                node_emb, feat_emb, lbl_feat, feat_mask, target_nodes_mask, label, neighbor_mask = data_loader.next4test()
                emb, _ = net.evaluation(node_emb, feat_emb, lbl_feat, feat_mask, target_nodes_mask, neighbor_mask)
                embeddings.append(emb)
                labels.append(label)
            embeddings = torch.cat(embeddings, 0)
            labels = torch.cat(labels, 0)
            embeddings = embeddings.cpu().numpy()
            labels = labels.detach().cpu().numpy()
        evaluate_results_nc(embeddings, labels)

        # early stopping
        early_stopping(train_loss_avg, net, epoch)
        if early_stopping.early_stop:
            print('  Early stopping!')
            break

print('all finished\n\n')
