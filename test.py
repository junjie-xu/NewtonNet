import os
import torch
import random
import argparse
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from models import *
from load_data import *
from config import *
from utils import *
from torch_geometric.utils import get_laplacian, to_dense_adj
import matplotlib.pyplot as plt
import time
from synthetic import *


# parser = argparse.ArgumentParser(description='General Training Pipeline')
# parser_add_main_args(parser)
# args = parser.parse_args()
# device = args.device
# torch.manual_seed(args.seed)
# np.random.seed(args.seed)
# random.seed(args.seed)
#
# # Load Data
# data = CSBM(n=args.num_nodes, d=5, ratio=0.99, p=args.num_features, mu=1,
#                 train_prop=args.train_prop, valid_prop=args.valid_prop, num_masks=args.num_masks)
# print("Homophily Ratio: ", edge_homo_ratio(data))
#
# print('Data:', data)
# num_nodes = data.num_nodes
# num_classes = data.y.max().item() + 1
# num_features = data.x.shape[1]
# num_edges = data.edge_index.shape[1]
#
# node_set = range(num_nodes)
#
# edge_index1, norm1 = get_laplacian(data.edge_index, None, normalization='sym', dtype=data.x.dtype,
#                                                num_nodes=data.num_nodes)
#
# L = to_dense_adj(edge_index1, edge_attr=norm1, max_num_nodes=data.x.shape[0]).squeeze()
#
# e, U = torch.linalg.eigh(L)
#
# torch.set_printoptions(threshold=np.inf)
# print(e)
# plt.hist(e.cpu().numpy(), bins=50, range=[0, 2])
# plt.show()

# for i in range(data.x.shape[1]):
#     sig = data.x[:, i]
#     normal_sig = F.normalize(sig, dim=0)
#     C = U.mH @ normal_sig
#     print(C.max(), C.min())

a = torch.tensor([1,2,3,4,5,6])

print(a[5: 10])

