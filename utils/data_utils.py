"""Data utils functions for pre-processing and data loading."""
import os
import pickle as pkl
import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch


def load_data(args, datapath):
   
    data = load_data_lp(args.dataset, args.use_feats, datapath,args.avg)
    adj = data['adj_train']
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges(
            adj,data['edges'],data['obj'], args.val_prop, args.test_prop, args.split_seed
    )
    split_adj_train = generate_train_split_graph(train_edges,data['graph_split']["father_subnodes"],len(data['graph_split']["child_nodes"]))
    data['adj_train'] = adj_train
    data['split_adj_train'] = split_adj_train
    data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false
    data['val_edges'], data['val_edges_false'] = val_edges, val_edges_false
    data['test_edges'], data['test_edges_false'] = test_edges, test_edges_false
    #data['features'] = mask_features(data['test_edges'],data['features'],data['a'], data['idx'],data['edges'])
    data['adj_train_norm'], data['features'] = process(
    data['adj_train'], data['features'], args.normalize_adj, args.normalize_feats
)
    return data

# ############### GRAPH SPLIT
def split_graph(obj):
    father_subnodes ={}
    child_nodes = []
    for i,_ in obj.items():
        child = _.split(" ")
        for c in child:
            if i not in father_subnodes:
                father_subnodes[i] = [len(child_nodes)]
            else:
                father_subnodes[i].append(len(child_nodes))
            child_nodes.append(c)
    return {"father_subnodes":father_subnodes,"child_nodes":child_nodes}

def generate_train_split_graph(train_edges,father_subnodes,num):
    adj = np.zeros((num,num))
    for (i,j) in train_edges:
             chi = father_subnodes[i.item()]
             par = father_subnodes[j.item()]
             for c in chi:
                    for p in par:
                         adj[c,p] = 1.
    return torch.from_numpy(adj)
    
    
# ############### FEATURES PROCESSING ####################################
        
                  

def process(adj, features, normalize_adj, normalize_feats):
    if type(features) is not np.ndarray:
        if sp.isspmatrix(features):
            features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def augment(adj, features, normalize_feats=True):
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.size(0), 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    return features


# ############### DATA SPLITS #####################################################


def mask_edges(adj,edges,obj,val_prop, test_prop, seed):
    np.random.seed(seed)  # get tp edges
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    # get tn edges
    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
   
    return adj_train, torch.LongTensor(train_edges), torch.LongTensor(train_edges_false), torch.LongTensor(val_edges), \
           torch.LongTensor(val_edges_false), torch.LongTensor(test_edges), torch.LongTensor(
            test_edges_false)  


def split_data(labels, val_prop, test_prop, seed):
    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)
    pos_idx = labels.nonzero()[0]
    neg_idx = (1. - labels).nonzero()[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    pos_idx = pos_idx.tolist()
    neg_idx = neg_idx.tolist()
    nb_pos_neg = min(len(pos_idx), len(neg_idx))
    nb_val = round(val_prop * nb_pos_neg)
    nb_test = round(test_prop * nb_pos_neg)
    idx_val_pos, idx_test_pos, idx_train_pos = pos_idx[:nb_val], pos_idx[nb_val:nb_val + nb_test], pos_idx[
                                                                                                   nb_val + nb_test:]
    idx_val_neg, idx_test_neg, idx_train_neg = neg_idx[:nb_val], neg_idx[nb_val:nb_val + nb_test], neg_idx[
                                                                                                   nb_val + nb_test:]
    return idx_val_pos + idx_val_neg, idx_test_pos + idx_test_neg, idx_train_pos + idx_train_neg


def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()


# ############### LINK PREDICTION DATA LOADERS ####################################


def load_data_lp(dataset, use_feats, data_path,avg):
    adj, features,edges,obj,graph_split = load_my_data(dataset, data_path, data_path,avg)

    data = {'adj_train':adj, 'features': features,'edges':edges,'obj':obj,'graph_split':graph_split}
    return data




# ############### DATASETS ####################################


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_synthetic_data(dataset_str, use_feats, data_path):
    object_to_idx = {}
    idx_counter = 0
    edges = []
    with open(os.path.join(data_path, "{}.edges.csv".format(dataset_str)), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.
    if use_feats:
        features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
    else:
        features = sp.eye(adj.shape[0])
    #labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)))
    return sp.csr_matrix(adj), features

def incremation_edge(edges):
    ancestors = {}
    for _ in range(30):
        for (chi, par) in edges:
            if chi in ancestors:
                if par in ancestors:
                    for an in ancestors[par]:
                        if an not in ancestors[chi]:
                            ancestors[chi] += [an]
                if par not in ancestors[chi]:
                    ancestors[chi] += [par]
            else:
                ancestors[chi] = [par]
   
    return ancestors

def load_my_data(dataset_str, use_feats, data_path,avg):
   
    object_to_idx = {}
    idx_counter = 0
    edges = []
    with open(os.path.join(data_path, "{}.edges.csv".format(dataset_str)), 'r') as f:
        all_edges = f.readlines()
    
        for line in all_edges[1:]:
            k = line.strip('\n').split(',')
            n1 = k[0]
            n2 = k[1]
            if n1 in object_to_idx:
                i = object_to_idx[n1]
            else:
                i = idx_counter
                object_to_idx[n1] = i
                idx_counter += 1
            if n2 in object_to_idx:
                j = object_to_idx[n2]
            else:
                j = idx_counter
                object_to_idx[n2] = j
                idx_counter += 1
            edges.append((i, j))
    #print(object_to_idx)
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    k = np.array([len(key.split(" ")) for key,val in object_to_idx.items()])
   
    length = np.expand_dims(np.array([len(key.split(" ")) for key,val in object_to_idx.items()]),0).transpose()
    for i, j in edges:
        adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.
   
    if use_feats:

        features = np.load(os.path.join(data_path,'feature.npy'))
        if avg == 1:
            dim = 300
            num = features.shape[0]//dim
            avg_feature = np.zeros((features.shape[0],dim))
            for i in range(num):
                 avg_feature += features[:,i*dim:(i+1)*dim]
            avg_feature/= (length+ 1e-6)
            features = avg_feature
        
    else:
        features = sp.eye(adj.shape[0])
    #features = sp.eye(adj.shape[0])
    # labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)))
    idx = {key:val for val,key in object_to_idx.items()}
    graph_split = split_graph(idx)
    return  sp.csr_matrix(adj), features, edges,object_to_idx,graph_split



