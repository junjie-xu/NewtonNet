import os
import torch
import random
import argparse
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from load_data import *
from config import parser_add_main_args
from utils import eval_rocauc, eval_acc
from torch_geometric.utils import get_laplacian, to_dense_adj


def edge_homo_ratio(data):
    sum = 0
    for i in range(len(data.edge_index[0])):
        if data.y[data.edge_index[0][i]] == data.y[data.edge_index[1][i]]:
            sum += 1
    return sum / len(data.edge_index[0])


def remove_homo_edges(data, remove_homo_ratio):
    edge_index_T = data.edge_index.T.cpu().numpy()
    homo_edge, hetero_edge = [], []

    for i in edge_index_T:
        if data.y[i[0]] == data.y[i[1]]:
            homo_edge.append(i)
        else:
            hetero_edge.append(i)

    hetero_edge = np.array(hetero_edge)
    homo_edge = np.array(homo_edge)
    np.random.shuffle(homo_edge)

    new_edge_index = np.concatenate((hetero_edge, homo_edge[:int(len(homo_edge) * (1 - remove_homo_ratio))]), 0)
    data.edge_index = torch.from_numpy(new_edge_index).T
    return data


def remove_hetero_edges(data, remove_hetero_ratio):
    edge_index_T = data.edge_index.T.cpu().numpy()
    homo_edge, hetero_edge = [], []

    for i in edge_index_T:
        if data.y[i[0]] == data.y[i[1]]:
            homo_edge.append(i)
        else:
            hetero_edge.append(i)

    hetero_edge = np.array(hetero_edge)
    homo_edge = np.array(homo_edge)
    np.random.shuffle(hetero_edge)

    new_edge_index = np.concatenate((homo_edge, hetero_edge[:int(len(hetero_edge) * (1 - remove_hetero_ratio))]), 0)
    data.edge_index = torch.from_numpy(new_edge_index).T
    return data


def synthetic_dataset(data, target_ratio):
    num_classes = data.y.max().item() + 1
    edge_index_T = data.edge_index.T.cpu().numpy()
    homo_edges, hetero_edges = [], []

    for i in edge_index_T:
        if data.y[i[0]] == data.y[i[1]]:
            homo_edges.append(i)
        else:
            hetero_edges.append(i)

    homo_edges, hetero_edges = np.array(homo_edges), np.array(hetero_edges)
    num_edges, num_homo_edges, num_hetero_edges = edge_index_T.shape[0], homo_edges.shape[0], hetero_edges.shape[0]
    ratio = num_homo_edges / num_edges

    if target_ratio > ratio:
        change = int(target_ratio * num_edges - num_homo_edges)

        np.random.shuffle(hetero_edges)
        new_hetero_edges = hetero_edges[:num_hetero_edges-change]

        nodes_each_class = [[] for _ in range(num_classes)]
        for i in range(len(data.y)):
            nodes_each_class[data.y[i]].append(i)

        new_homo_edges = []
        for i in nodes_each_class:
            for j in range(int(change / num_classes)):
                new_homo_edges.append(random.sample(i, 2))

        new_homo_edges = np.array(new_homo_edges)
        new_homo_edges = np.concatenate((new_homo_edges, homo_edges), 0)
        new_edge_index = np.concatenate((new_homo_edges, new_hetero_edges), 0)

    else:
        change = int((num_homo_edges - target_ratio * num_edges) / (1 - target_ratio))
        new_homo_edges = homo_edges[:num_homo_edges - change]
        new_edge_index = np.concatenate((new_homo_edges, hetero_edges), 0)

    new_edge_index = np.unique(new_edge_index, axis=0)
    data.edge_index = torch.from_numpy(new_edge_index).T
    return data


def CSBM(n, d, ratio, p, mu, train_prop=.6, valid_prop=.2, num_masks=5):
    Lambda = np.sqrt(d) * (2 * ratio - 1)
    c_in = d + np.sqrt(d) * Lambda
    c_out = d - np.sqrt(d)*Lambda
    print('c_in: ', c_in, 'c_out: ', c_out)
    y = np.ones(n)
    y[int(n/2)+1:] = -1
    y = np.asarray(y, dtype=int)

    # creating edge_index
    edge_index = [[], []]
    for i in range(n-1):
        for j in range(i+1, n):
            if y[i]*y[j] > 0:
                Flip = np.random.binomial(1, c_in/n)
            else:
                Flip = np.random.binomial(1, c_out/n)
            if Flip > 0.5:
                edge_index[0].append(i)
                edge_index[1].append(j)
                edge_index[0].append(j)
                edge_index[1].append(i)

    # creating node features
    x = np.zeros([n, p])
    u = np.random.normal(0, 1/np.sqrt(p), [1, p])
    for i in range(n):
        Z = np.random.normal(0, 1, [1, p])
        x[i] = np.sqrt(mu/n)*y[i]*u + Z/np.sqrt(p)

    data = Data(x=torch.tensor(x, dtype=torch.float32),
                edge_index=torch.tensor(edge_index),
                y=torch.tensor((y + 1) // 2, dtype=torch.int64))

    data.coalesce()

    splits_lst = [rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop)
                  for _ in range(num_masks)]
    data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)

    return data


