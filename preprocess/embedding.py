from itertools import islice
import numpy as np
import scipy.sparse as sp
import re
import argparse

ap = argparse.ArgumentParser(description='args')
ap.add_argument('--dataset', default='DBLP', help='Dataset.')
ap.add_argument('--dim', type=int, default=128, help='Dimension')
args = ap.parse_args()

if args.dataset == 'DBLP':
    A_n = 4057
    P_n = 14328
    V_n = 20
    T_n = 7723
    W_n = sp.load_npz('../data/DBLP/features_1.npz').toarray().shape[1]
    dim = args.dim

    author_feature = np.zeros((A_n, dim), dtype=np.float)
    paper_feature = np.zeros((P_n, dim), dtype=np.float)
    venue_feature = np.zeros((V_n, dim), dtype=np.float)
    term_feature = np.zeros((T_n, dim), dtype=np.float)
    word_feature = np.zeros((W_n, dim), dtype=np.float)

    path = r'metapath2vec_DBLP_embeddings.txt'

    A_counter = 0
    P_counter = 0
    V_counter = 0
    T_counter = 0
    W_counter = 0

    f = open(path, "r")
    for line in islice(f, 2, None):
        line = line.strip()
        line = re.split(' ', line)
        node_info = line[0].split('#')
        node_type = node_info[0]
        node_id = int(node_info[1])
        if node_type == 'vVenue':
            venue_feature[node_id] = line[1:]
            V_counter += 1
        if node_type == 'aAuthor':
            author_feature[node_id] = line[1:]
            A_counter += 1
        if node_type == 'pPaper':
            paper_feature[node_id] = line[1:]
            P_counter += 1
        if node_type == 'tTerm':
            term_feature[node_id] = line[1:]
            T_counter += 1
        if node_type == 'wWord':
            word_feature[node_id] = line[1:]
            W_counter += 1

    if V_counter != V_n:
        print('{} venue node miss'.format(V_n - V_counter))
    if A_counter != A_n:
        print('{} author node miss'.format(A_n - A_counter))
    if P_counter != P_n:
        print('{} paper node miss'.format(P_n - P_counter))
    if T_counter != T_n:
        print('{} term node miss'.format(T_n - T_counter))
    if W_counter != W_n:
        print('{} word node miss'.format(W_n - W_counter))

    f = np.concatenate((author_feature, paper_feature, term_feature, venue_feature), axis=0)
    print(f.shape)
    np.save('../data/DBLP/metapath2vec_emb_node.npy', f)
    np.save('../data/DBLP/metapath2vec_emb_word.npy', word_feature)
    print('done!')

elif args.dataset == 'ACM':
    P_n = 4019
    A_n = 7167
    S_n = 60
    W_n = sp.load_npz('../data/ACM/features_0.npz').toarray().shape[1]
    dim = args.dim

    p_feature = np.zeros((P_n, dim), dtype=np.float)
    a_feature = np.zeros((A_n, dim), dtype=np.float)
    s_feature = np.zeros((S_n, dim), dtype=np.float)
    w_feature = np.zeros((W_n, dim), dtype=np.float)

    path = r'metapath2vec_ACM_embeddings.txt'

    P_counter = 0
    A_counter = 0
    S_counter = 0
    W_counter = 0

    f = open(path, "r")
    for line in islice(f, 2, None):
        line = line.strip()
        line = re.split(' ', line)
        node_info = line[0].split('#')
        node_type = node_info[0]
        node_id = int(node_info[1])
        if node_type == 'pPaper':
            p_feature[node_id] = line[1:]
            P_counter += 1
        if node_type == 'aAuthor':
            a_feature[node_id] = line[1:]
            A_counter += 1
        if node_type == 'sSubject':
            s_feature[node_id] = line[1:]
            S_counter += 1
        if node_type == 'wWord':
            w_feature[node_id] = line[1:]
            W_counter += 1

    if P_counter != P_n:
        print('{} p node miss'.format(P_n - P_counter))
    if A_counter != A_n:
        print('{} a node miss'.format(A_n - A_counter))
    if S_counter != S_n:
        print('{} s node miss'.format(S_n - S_counter))
    if W_counter != W_n:
        print('{} w node miss'.format(W_n - W_counter))

    f = np.concatenate((p_feature, a_feature, s_feature), axis=0)
    print(f.shape)
    np.save('../data/ACM/metapath2vec_emb_node.npy', f)
    np.save('../data/ACM/metapath2vec_emb_word.npy', w_feature)
    print('done!')

elif args.dataset == 'Yelp':
    B_n = 2614
    U_n = 1286
    S_n = 4
    L_n = 9
    W_n = sp.load_npz('../data/Yelp/features_0.npz').toarray().shape[1]
    dim = args.dim

    b_feature = np.zeros((B_n, dim), dtype=np.float)
    u_feature = np.zeros((U_n, dim), dtype=np.float)
    s_feature = np.zeros((S_n, dim), dtype=np.float)
    l_feature = np.zeros((L_n, dim), dtype=np.float)
    w_feature = np.zeros((W_n, dim), dtype=np.float)

    path = r'metapath2vec_Yelp_embeddings.txt'

    B_counter = 0
    U_counter = 0
    S_counter = 0
    L_counter = 0
    W_counter = 0

    f = open(path, "r")
    for line in islice(f, 2, None):
        line = line.strip()
        line = re.split(' ', line)
        node_info = line[0].split('#')
        node_type = node_info[0]
        node_id = int(node_info[1])
        if node_type == 'bBusinesses':
            b_feature[node_id] = line[1:]
            B_counter += 1
        if node_type == 'uUser':
            u_feature[node_id] = line[1:]
            U_counter += 1
        if node_type == 'sService':
            s_feature[node_id] = line[1:]
            S_counter += 1
        if node_type == 'lLevel':
            l_feature[node_id] = line[1:]
            L_counter += 1
        if node_type == 'wWord':
            w_feature[node_id] = line[1:]
            W_counter += 1

    if B_counter != B_n:
        print('{} b node miss'.format(B_n - B_counter))
    if U_counter != U_n:
        print('{} u node miss'.format(U_n - U_counter))
    if S_counter != S_n:
        print('{} s node miss'.format(S_n - S_counter))
    if L_counter != L_n:
        print('{} l node miss'.format(L_n - L_counter))
    if W_counter != W_n:
        print('{} w node miss'.format(W_n - W_counter))

    f = np.concatenate((b_feature, u_feature, s_feature, l_feature), axis=0)
    print(f.shape)
    np.save('../data/Yelp/metapath2vec_emb_node.npy', f)
    np.save('../data/Yelp/metapath2vec_emb_word.npy', w_feature)
    print('done!')

else:
    raise Exception('Dataset error!')

'''
./metapath2vec -train ../preprocess/walks_ACM.txt -output ../preprocess/metapath2vec_ACM_embeddings -pp 0 -size 128 -window 4 -negative 10 -threads 32
./metapath2vec -train ../preprocess/walks_DBLP.txt -output ../preprocess/metapath2vec_DBLP_embeddings -pp 0 -size 128 -window 4 -negative 10 -threads 32
./metapath2vec -train ../preprocess/walks_Yelp.txt -output ../preprocess/metapath2vec_Yelp_embeddings -pp 0 -size 128 -window 4 -negative 10 -threads 32
'''