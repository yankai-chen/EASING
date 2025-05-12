import dgl
import numpy as np
import pickle
import random
import torch
from sklearn.metrics import mean_absolute_error
from .metric import ndcg, spearman_sci
import pdb
import torch.nn as nn
from tqdm import tqdm


def convert_to_gpu(*data, device):
    res = []
    for item in data:
        item = item.to(device)
        res.append(item)
    return tuple(res)


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_model(model, model_path):
    print(f"load model {model_path}")
    model.load_state_dict(torch.load(model_path))


def count_parameters_in_KB(model):
    param_num = np.sum(np.prod(v.size()) for v in model.parameters()) / 1e3
    return param_num


def get_rank_metrics(predicts, labels, NDCG_k, spearman=False):
    if spearman:
        return ndcg(labels, predicts, NDCG_k), spearman_sci(labels, predicts)
    return ndcg(labels, predicts, NDCG_k)


def rank_evaluate(predicts, labels, NDCG_k, loss_func, spearman=False):
    with torch.no_grad():
        loss = loss_func(predicts, labels.reshape(-1)) 
        mae= mean_absolute_error(predicts.cpu().numpy(), labels.cpu().numpy())
    if spearman:
        ndcg_score, spear_score = get_rank_metrics(predicts, labels, NDCG_k, spearman)
        return loss, ndcg_score, spear_score, mae
    else:
        ndcg_score = get_rank_metrics(predicts, labels, NDCG_k, spearman)
        return loss, ndcg_score,mae

def load_split_data(split_data_path, num_split_idx):
    dataset_spilt=[]
    labels_idx=[]
    for i in range(num_split_idx):
        with open(f"{split_data_path}/split_dataset_idx{i}.pkl","rb") as input:
            data=pickle.load(input)
            dataset_spilt.append(data['idx'])
            labels_idx.append(data['labels'])
    return dataset_spilt, labels_idx

def split_train_val_test(dataset_spilt, labels_idx, train_num, num_split_idx):
    dataset_index=[i for i in range(num_split_idx)]
    train_labels=np.array([],dtype=np.float32)
    train_idx=np.array([],dtype=np.int64)
    val_labels=np.array([],dtype=np.float32)
    val_idx=np.array([],dtype=np.int64)
    test_labels=np.array([],dtype=np.float32)
    test_idx=np.array([],dtype=np.int64)
    unlabeled_labels=np.array([],dtype=np.float32)
    unlabeled_idx=np.array([],dtype=np.int64)
    split_num = num_split_idx//10
    for _ in range(split_num):
        subdataset=random.choice(dataset_index)
        test_idx=np.concatenate((test_idx,dataset_spilt[subdataset]))
        test_labels=np.concatenate((test_labels,labels_idx[subdataset]))
        test_labels=torch.tensor(test_labels)
        dataset_index.remove(subdataset)
    for _ in range(split_num):
        subdataset=random.choice(dataset_index)
        val_idx=np.concatenate((val_idx,dataset_spilt[subdataset]))
        val_labels=np.concatenate((val_labels,labels_idx[subdataset]))
        val_labels=torch.tensor(val_labels)
        dataset_index.remove(subdataset)
        
    num=int(train_num*split_num)
    for _ in range(num):
        subdataset=random.choice(dataset_index)
        train_idx=np.concatenate((train_idx,dataset_spilt[subdataset]))
        train_labels=np.concatenate((train_labels,labels_idx[subdataset]))
        train_labels=torch.tensor(train_labels)
        dataset_index.remove(subdataset)
    
    for i in range(len(dataset_index)):
        unlabeled_idx=np.concatenate((unlabeled_idx,dataset_spilt[dataset_index[i]]))
        unlabeled_labels=np.concatenate((unlabeled_labels,labels_idx[dataset_index[i]]))
        unlabeled_labels=torch.tensor(unlabeled_labels)
    return train_idx, val_idx, test_idx,train_labels,val_labels,test_labels,unlabeled_idx,unlabeled_labels

def load_fb15k_rel_data(
    graph_data_path, 
    semantic_data_path, 
    train_num=8.0, 
    split_data_path='',
    num_split_idx=1000):
    
    # load data
    with open(graph_data_path, 'rb') as f:
        data = pickle.load(f)
    
    # edge list
    edges = data['edges']
    
    # structure features
    node_feat1 = data['features']
    
    # semantic features
    node_feat2 = pickle.load(open(semantic_data_path, 'rb'))
    node_feat2 = torch.from_numpy(node_feat2).float()
    
    # edge types
    edge_types = data['edge_types']
    rel_num = (max(edge_types) + 1).item()
    
    # construct a heterogeneous graph
    hg = dgl.graph(edges)
    g = hg.local_var()
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    node_norm = torch.from_numpy(norm).view(-1, 1)
    g.ndata['norm'] = node_norm
    g.apply_edges(lambda edges: {'norm': edges.dst['norm']})
    edge_norm = g.edata['norm']
    
    # split dataset
    dataset_spilt, labels_idx = load_split_data(split_data_path, num_split_idx)
    #  train_idx, val_idx, test_idx
    train_idx, val_idx, test_idx,train_labels,val_labels,test_labels,unlabeled_idx,unlabeled_labels = split_train_val_test(dataset_spilt, labels_idx, train_num, num_split_idx)
    print(len(test_idx), len(val_idx), len(train_idx))

    return hg, edge_types, edge_norm, rel_num, node_feat1, node_feat2, train_idx, val_idx, test_idx,train_labels,val_labels,test_labels,unlabeled_idx,unlabeled_labels
    




def load_imdb_s_rel_subgraph_data(
    graph_data_path, 
    structure_data_path,
    semantic_data_path, 
    train_num=8.0, 
    split_data_path='', 
    num_split_idx=1000):
    
    with open(graph_data_path, 'rb') as f:
        data = pickle.load(f)

    node_feat1 = torch.from_numpy(pickle.load(open(structure_data_path, 'rb')))
    node_feat2 = pickle.load(open(semantic_data_path, 'rb'))
    node_feat2 = torch.from_numpy(node_feat2).float()

    # edge list
    edges = data['edges']
    edge_types = data['edge_types']
    rel_num = (max(edge_types) + 1).item()

    # construct a heterogeneous graph
    hg = dgl.graph(edges)
    
    # generate edge norm
    g = hg.local_var()
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    node_norm = torch.from_numpy(norm).view(-1, 1)
    g.ndata['norm'] = node_norm
    g.apply_edges(lambda edges: {'norm': edges.dst['norm']})
    edge_norm = g.edata['norm']
    
    # split dataset
    dataset_spilt, labels_idx = load_split_data(split_data_path, num_split_idx)
    train_idx, val_idx, test_idx,train_labels,val_labels,test_labels,unlabeled_idx,unlabeled_labels = split_train_val_test(dataset_spilt, labels_idx, train_num, num_split_idx)
    print(len(test_idx), len(val_idx), len(train_idx))
    
    return hg, edge_types, edge_norm, rel_num, node_feat1, node_feat2, train_idx, val_idx, test_idx,train_labels,val_labels,test_labels,unlabeled_idx,unlabeled_labels



def load_tmdb_rel_data(
    graph_data_path, 
    semantic_data_path, 
    train_num=8.0, 
    split_data_path='', 
    num_split_idx=1000
    ):
    
    with open(graph_data_path, 'rb') as f:
        data = pickle.load(f)

    # edge list
    edges = data['edges']
    edge_types = data['edge_types']
    # rel_num = (max(edge_types) + 1).item()
    rel_num = 34

    node_feat1 = data['features']
    node_feat2 = pickle.load(open(semantic_data_path, 'rb'))
    node_feat2 = torch.from_numpy(node_feat2).float()
    
    # construct a heterogeneous graph
    hg = dgl.graph(edges)

    
    # split dataset
    dataset_spilt, labels_idx = load_split_data(split_data_path, num_split_idx)
    train_idx, val_idx, test_idx,train_labels,val_labels,test_labels,unlabeled_idx,unlabeled_labels = split_train_val_test(dataset_spilt, labels_idx, train_num, num_split_idx)

    print(len(test_idx), len(val_idx), len(train_idx))

    # generate edge norm
    g = hg.local_var()
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    node_norm = torch.from_numpy(norm).view(-1, 1)
    g.ndata['norm'] = node_norm
    g.apply_edges(lambda edges: {'norm': edges.dst['norm']})
    edge_norm = g.edata['norm']


    return hg, edge_types, edge_norm, rel_num, node_feat1, node_feat2, train_idx, val_idx, test_idx,train_labels,val_labels,test_labels,unlabeled_idx,unlabeled_labels



def load_data(
    graph_data_path, 
    structure_data_path, 
    semantic_data_path, 
    dataset_name, 
    train_num=8.0,
    split_data_path='',
    num_split_idx=1000
    ):

    if dataset_name.startswith('FB15K'):
        return load_fb15k_rel_data(
            graph_data_path=graph_data_path, 
            semantic_data_path=semantic_data_path, 
            train_num=train_num,
            split_data_path=split_data_path, 
            num_split_idx=num_split_idx
            )
        
    elif dataset_name.startswith('TMDB'):
        return load_tmdb_rel_data(
            graph_data_path=graph_data_path, 
            semantic_data_path=semantic_data_path, 
            train_num=train_num,
            split_data_path=split_data_path, 
            num_split_idx=num_split_idx
            )
        
    elif dataset_name.startswith('IMDB'):
        return load_imdb_s_rel_subgraph_data(
            graph_data_path=graph_data_path, 
            structure_data_path=structure_data_path, 
            semantic_data_path=semantic_data_path, 
            train_num=train_num,
            split_data_path=split_data_path, 
            num_split_idx=num_split_idx
            )
    else:
        return NotImplementedError('Unsupported dataset {}'.format(dataset_name))


def get_centrality(graph):
    g = graph.local_var()
    in_deg = g.in_degrees(range(g.number_of_nodes())).float()
    theta = 1e-4
    centrality = torch.log(in_deg + theta)
    print(centrality)
    return centrality

def get_relative_entropy(graph, cont_feat):
    theta = 1e-4
    cont_feat_copy=cont_feat.clone()
    g = graph.local_var()
    REnt_check=[]
    
    # softmax
    cont_feat_copy=nn.functional.softmax(cont_feat_copy,dim=1)
    print("--------------------calculating KL-div--------------------")
    # calculate REnt
    for node in tqdm(range(g.number_of_nodes())):
        # get target node's neighbors
        neighbours=g.in_edges(node)
        neighbours_list=neighbours[0].cpu().numpy().tolist()
        if node in neighbours_list:
            neighbours_list.remove(node)
        kl_current_node=[]
        kl_current_node_check=[]
        for neighbour in neighbours_list:
            # check kl_div
            c=(torch.log(cont_feat_copy[node])-torch.log(cont_feat_copy[neighbour])).cpu().numpy()
            nan_indices=np.where(np.isnan(c))
            inf_indices=np.where(np.isinf(c))
            c[nan_indices]=0
            c[inf_indices]=0
            kl_div_check=cont_feat_copy[node].cpu().numpy()*c
            kl_div_check=kl_div_check.sum()
            kl_current_node_check.append(kl_div_check)
        kl_current_node_check=np.array(kl_current_node_check)
        REnt_check.append(kl_current_node_check.sum())
    
    REnt_check=torch.tensor(np.array(REnt_check,dtype=np.float32))
    REnt_check=torch.log(REnt_check+theta+1)
    print(REnt_check)
    return REnt_check
