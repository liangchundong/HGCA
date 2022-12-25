import numpy as np
import os
import networkx as nx
import scipy.sparse as sp


def get_metapath_neighbor_pairs(M, type_mask, expected_metapaths):
    """
    This code is copied from https://github.com/cynricfu/MAGNN
    :param M: the raw adjacency matrix
    :param type_mask: an array of types of all node
    :param expected_metapaths: a list of expected metapaths
    :return: a list of python dictionaries, consisting of metapath-based neighbor pairs and intermediate paths
    """
    outs = []
    for metapath in expected_metapaths:
        # consider only the edges relevant to the expected metapath
        mask = np.zeros(M.shape, dtype=bool)
        for i in range((len(metapath) - 1) // 2):
            temp = np.zeros(M.shape, dtype=bool)
            temp[np.ix_(type_mask == metapath[i], type_mask == metapath[i + 1])] = True
            temp[np.ix_(type_mask == metapath[i + 1], type_mask == metapath[i])] = True
            mask = np.logical_or(mask, temp)
        partial_g_nx = nx.from_numpy_matrix((M * mask).astype(int))

        # only need to consider the former half of the metapath
        # e.g., we only need to consider 0-1-2 for the metapath 0-1-2-1-0
        metapath_to_target = {}
        for source in (type_mask == metapath[0]).nonzero()[0]:
            # print((len(metapath) + 1) // 2 - 1)
            single_source_paths = nx.single_source_shortest_path(
                partial_g_nx, source, cutoff=(len(metapath) + 1) // 2 - 1)
            for target in (type_mask == metapath[(len(metapath) - 1) // 2]).nonzero()[0]:
                # check if there is a possible valid path from source to target node
                has_path = False
                if target in single_source_paths:
                    has_path = True
                if source == target:
                    has_path = True

                if has_path:
                    shortests = [p for p in nx.all_shortest_paths(partial_g_nx, source, target) if
                                 len(p) == (len(metapath) + 1) // 2]
                    if target == source:
                        if len(metapath) == 3:
                            shortests.append([target, target])
                        elif len(metapath) == 5:
                            neighbors = nx.all_neighbors(partial_g_nx, target)
                            for neighbor in neighbors:
                                shortests.append([target, neighbor, target])
                    if len(shortests) > 0:
                        metapath_to_target[target] = metapath_to_target.get(target, []) + shortests
        metapath_neighbor_paris = {}
        for key, value in metapath_to_target.items():
            for p1 in value:
                for p2 in value:
                    metapath_neighbor_paris[(p1[0], p2[0])] = metapath_neighbor_paris.get((p1[0], p2[0]), []) + [
                        p1 + p2[-2::-1]]
        outs.append(metapath_neighbor_paris)
    return outs


def get_edge_metapath_idx_array(neighbor_pairs):
    all_edge_metapath_idx_array = []
    for metapath_neighbor_pairs in neighbor_pairs:
        sorted_metapath_neighbor_pairs = sorted(metapath_neighbor_pairs.items())
        edge_metapath_idx_array = []
        for _, paths in sorted_metapath_neighbor_pairs:
            edge_metapath_idx_array .extend(paths)
        edge_metapath_idx_array = np.array(edge_metapath_idx_array, dtype=int)
        all_edge_metapath_idx_array.append(edge_metapath_idx_array)
        print(edge_metapath_idx_array.shape)
    return all_edge_metapath_idx_array


def save_edge_metapath_idx_array(expected_metapaths, adjM, type_mask, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # get metapath based neighbor pairs
    neighbor_pairs = get_metapath_neighbor_pairs(adjM, type_mask, expected_metapaths)

    # save data - node indices of edge metapaths
    all_edge_metapath_idx_array = get_edge_metapath_idx_array(neighbor_pairs)
    for metapath, edge_metapath_idx_array in zip(expected_metapaths, all_edge_metapath_idx_array):
        np.save(save_path + '/' + '-'.join(map(str, metapath)) + '_idx.npy', edge_metapath_idx_array)


def sampling_ACM():
    save_path = '../data/ACM/indices'
    target_type = 0
    expected_metapaths = [(0, 1, 0), (0, 2, 0)]
    adjM = sp.load_npz('../data/ACM/adjM.npz').toarray()
    type_mask = np.load('../data/ACM/node_types.npy')
    num_target = np.where(type_mask == target_type)[0].shape[0]
    if not os.path.exists(save_path):
        save_edge_metapath_idx_array(expected_metapaths, adjM, type_mask, save_path)

    for metapath in expected_metapaths:
        neighbors_dict = [{i: [] for i in range(num_target)} for _ in range(len(metapath) - 1)]
        edge_metapath_idx_array = np.load(save_path + '/' + '-'.join(map(str, metapath)) + '_idx.npy')
        print(edge_metapath_idx_array.shape)
        for idx in edge_metapath_idx_array:
            for i, d in enumerate(neighbors_dict):
                d[idx[0]].append(idx[i+1])
        for i, d in enumerate(neighbors_dict):
            np.save(save_path + '/(' + '-'.join(map(str, metapath)) + '){}-{}.npy'.format(i+1, metapath[i+1]), d)


def sampling_DBLP():
    save_path = '../data/DBLP/indices'
    target_type = 1
    expected_metapaths = [(0, 1, 0), (0, 1, 2, 1, 0), (0, 1, 3, 1, 0)]
    adjM = sp.load_npz('../data/DBLP/adjM.npz').toarray()
    type_mask = np.load('../data/DBLP/node_types.npy')
    num_target = np.where(type_mask == target_type)[0].shape[0]
    if not os.path.exists(save_path):
        save_edge_metapath_idx_array(expected_metapaths, adjM, type_mask, save_path)

    for metapath in expected_metapaths:
        neighbors_dict = [{i: [] for i in range(num_target)} for _ in range(len(metapath) - 1)]
        edge_metapath_idx_array = np.load(save_path + '/' + '-'.join(map(str, metapath)) + '_idx.npy')
        print(edge_metapath_idx_array.shape)
        # print(edge_metapath_idx_array)
        for idx in edge_metapath_idx_array:
            for i, d in enumerate(neighbors_dict):
                d[idx[0]].append(idx[i+1])
        for i, d in enumerate(neighbors_dict):
            np.save(save_path + '/(' + '-'.join(map(str, metapath)) + '){}-{}.npy'.format(i+1, metapath[i+1]), d)


def sampling_Yelp():
    save_path = '../data/Yelp/indices'
    target_type = 0
    expected_metapaths = [(0, 1, 0), (0, 2, 0), (0, 3, 0)]
    adjM = sp.load_npz('../data/Yelp/adjM.npz').toarray()
    type_mask = np.load('../data/Yelp/node_types.npy')
    num_target = np.where(type_mask == target_type)[0].shape[0]
    if not os.path.exists(save_path):
        save_edge_metapath_idx_array(expected_metapaths, adjM, type_mask, save_path)

    for metapath in expected_metapaths:
        neighbors_dict = [{i: [] for i in range(num_target)} for _ in range(len(metapath) - 1)]
        edge_metapath_idx_array = np.load(save_path + '/' + '-'.join(map(str, metapath)) + '_idx.npy')
        print(edge_metapath_idx_array.shape)
        # print(edge_metapath_idx_array)
        for idx in edge_metapath_idx_array:
            for i, d in enumerate(neighbors_dict):
                d[idx[0]].append(idx[i+1])
        for i, d in enumerate(neighbors_dict):
            np.save(save_path + '/(' + '-'.join(map(str, metapath)) + '){}-{}.npy'.format(i+1, metapath[i+1]), d)


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description='args')
    ap.add_argument('--dataset', default='ACM', help='Dataset.')
    args = ap.parse_args()

    if args.dataset == 'Yelp':
        sampling_Yelp()
        print('Yelp Done')
    if args.dataset == 'ACM':
        sampling_ACM()
        print('ACM Done')
    if args.dataset == 'DBLP':
        sampling_DBLP()
        print('All Done')
