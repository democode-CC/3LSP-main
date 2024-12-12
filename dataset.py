import numpy as np
import torch as th
import scipy.sparse as sp
from scipy.linalg import fractional_matrix_power, inv
import os
import pickle
import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, RedditDataset
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler



def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    if isinstance(features, np.ndarray):
        return features
    else:
        return features.todense(), sparse_to_tuple(features)


def sparse_to_tuple(sparse_mx):
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape
    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def compute_ppr(graph: nx.Graph, alpha=0.2, self_loop=True):
    a = nx.convert_matrix.to_numpy_array(graph)
    if self_loop:
        a = a + np.eye(a.shape[0])  # A^ = A + I_n
    d = np.diag(np.sum(a, 1))  # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)  # D^(-1/2)
    at = np.matmul(np.matmul(dinv, a), dinv)  # A~ = D^(-1/2) x A^ x D^(-1/2)
    return alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))  # a(I_n-(1-a)A~)^-1



def data_augment(name, epsilon, subgraph_sampling=None, ratio=None, adj_init='cos_sim', adj_init_random_p=0.5, alpha1=0.2, alpha2=0.4, sim_k=5):
    if name == 'cora':
        dataset = CoraGraphDataset()
        partition_num = int(1/ratio)
    elif name == 'citeseer':
        dataset = CiteseerGraphDataset()
        partition_num = int(1/ratio)
    elif name == 'pubmed':
        dataset = PubmedGraphDataset()
        partition_num = int(4/ratio)
    elif name == 'reddit':
        dataset = RedditDataset()
        partition_num = int(100/ratio)

    graph = dataset[0]
    graph = dgl.add_self_loop(graph)
    n_nodes = graph.number_of_nodes()

    if subgraph_sampling == 'metis':
        if name == 'pubmed':
            part_no = 1
        elif name == 'reddit':
            part_no = 4
        else:
            part_no = 0
        if os.path.exists(f'tmp_partition/{name}/{name}.json'):
            print('partition result exists!')
            graph, node_feature, edge_feature, _, _, _, _ = dgl.distributed.load_partition(f'tmp_partition/{name}/{name}.json', part_no)

            graph.ndata['feat'] = node_feature['_N/feat']
            nx_g = dgl.to_networkx(graph, node_attrs=['feat'])  # convert to networkx and save for ASC and SSC
            pickle.dump(nx_g, open(f'nx_{name}.pkl', 'wb'))
        else:
            if not os.path.isdir(f'tmp_partition/{name}'):
                os.makedirs(f'tmp_partition/{name}')
            else:
                pass

            if 'train_mask' in graph.ndata.keys():
                bn = graph.ndata['train_mask']
            else:
                bn = None


            dgl.distributed.partition_graph(graph, name, partition_num, num_hops=0, part_method='metis',
                                            out_path=f'tmp_partition/{name}', reshuffle=False,
                                            balance_ntypes=bn,
                                            balance_edges=True)
            graph, node_feature, edge_feature, _, _, _, _ = dgl.distributed.load_partition(f'tmp_partition/{name}/{name}.json', part_no)

        feat = node_feature['_N/feat']
        label = node_feature['_N/label']
        n_nodes = graph.number_of_nodes()

        train_mask = node_feature['_N/train_mask']
        val_mask = node_feature['_N/val_mask']
        test_mask = node_feature['_N/test_mask']
    else:
        if name == 'fraudyelp':
            feat = graph.ndata.pop('feature')
        else:
            feat = graph.ndata.pop('feat')
        label = graph.ndata.pop('label')
        train_mask = graph.ndata.pop('train_mask')
        val_mask = graph.ndata.pop('val_mask')
        test_mask = graph.ndata.pop('test_mask')

    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()

    if adj_init == 'ori':
        nx_g = dgl.to_networkx(graph)

    elif adj_init == 'zero':
        init_graph_adj = np.zeros((n_nodes, n_nodes))
        nx_g = nx.from_numpy_matrix(init_graph_adj)
    elif adj_init == 'full':
        init_graph_adj = np.ones((n_nodes, n_nodes))
        nx_g = nx.from_numpy_matrix(init_graph_adj)

    elif adj_init == 'random':
        nx_g = nx.generators.random_graphs.gnp_random_graph(n_nodes, adj_init_random_p)

    elif adj_init == 'cos_sim':
        KNN = sim_k
        init_graph_adj = np.zeros((n_nodes, n_nodes))
        feat_np = feat.cpu().numpy()
        sim = cosine_similarity(feat_np)
        for i in range(sim.shape[0]):
            ind = np.argpartition(sim[i], -KNN)[-KNN:]
            init_graph_adj[i][ind] = 1
            for j in range(sim.shape[0]):
                if init_graph_adj[j][i] == 1:
                    init_graph_adj[i][j] = 1
        nx_g = nx.from_numpy_matrix(init_graph_adj)




    print('computing ppr')
    diff1_adj = compute_ppr(nx_g, alpha1)
    print('computing end')
    diff2_adj = compute_ppr(nx_g, alpha2)


    # Obtain target_matrix and relation matrix
    target_mat = graph.adj().to_dense().numpy()
    target_u, target_sv, _ = np.linalg.svd(target_mat, full_matrices=True)
    np.save(f'target_sv_{name}.npy', target_sv)
    np.save(f'target_u_{name}.npy', target_u)
    r_mat = np.transpose(np.linalg.solve(np.transpose(diff1_adj), feat))
    r_u, r_sv, _ = np.linalg.svd(r_mat, full_matrices=True)
    np.save(f'r_sv_{name}.npy', r_sv)
    np.save(f'r_u_{name}.npy', r_u)

    if name == 'citeseer' or name == 'pubmed' or name == 'reddit':
        print('additional processing')
        feat = th.tensor(preprocess_features(feat.numpy())).float()
        diff1_adj[diff1_adj < epsilon] = 0
        scaler = MinMaxScaler()
        scaler.fit(diff1_adj)
        diff1_adj = scaler.transform(diff1_adj)


        diff2_adj[diff2_adj < epsilon] = 0
        scaler = MinMaxScaler()
        scaler.fit(diff2_adj)
        diff2_adj = scaler.transform(diff2_adj)


    diff1_edges = np.nonzero(diff1_adj)
    diff1_weight = diff1_adj[diff1_edges]
    diff1_graph = dgl.graph(diff1_edges)

    diff2_edges = np.nonzero(diff2_adj)
    diff2_weight = diff2_adj[diff2_edges]
    diff2_graph = dgl.graph(diff2_edges)

    graph = graph.add_self_loop()

    return graph, diff1_graph, diff2_graph, feat, label, train_idx, val_idx, test_idx, diff1_weight, diff2_weight, dataset.num_classes






