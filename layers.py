from torch import Tensor
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros, ones, constant
from typing import Optional
from torch_geometric.typing import OptTensor
from math import factorial
import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.utils import get_laplacian, to_dense_adj, dense_to_sparse
from scipy.special import comb
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np
import scipy
import pandas as pd
from torch_sparse import SparseTensor, matmul



class DenseGCNConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        bias: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved

        self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)


    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        # x = x.unsqueeze(0) if x.dim() == 2 else x
        # adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        # B, N, _ = adj.size()
        out = self.lin(x)
        out = torch.matmul(adj, out)

        if self.bias is not None:
            out = out + self.bias
        return out



class BernConv(MessagePassing):
    def __init__(self, in_channels, out_channels, K, shared=True, bias=True, **kwargs):
        super(BernConv, self).__init__(aggr='add', **kwargs)

        self.K = K

        self.Lap_edge_index = None
        self.Lap_norm = None

        self.lin = Linear(in_channels, out_channels)
        self.shared = shared

        if shared:
            self.register_parameter('temp', None)
        else:
            self.temp = Parameter(torch.Tensor(self.K + 1))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.fill_(0)
        self.temp.data.fill_(1)

    def forward(self, x, edge_index, coef, edge_weight=None):
        if self.shared:
            TEMP = F.relu(coef)
        else:
            TEMP = F.relu(self.temp)

        if self.Lap_norm is None:
            # L=I-D^(-0.5)AD^(-0.5)
            self.Lap_edge_index, self.Lap_norm = get_laplacian(edge_index, edge_weight, normalization='sym', dtype=x.dtype,
                                               num_nodes=x.size(self.node_dim))
            # 2I-L
            self.edge_index2, self.norm2 = add_self_loops(self.Lap_edge_index, -self.Lap_norm, fill_value=2., num_nodes=x.size(self.node_dim))

        tmp = []
        tmp.append(x)
        for i in range(self.K):
            x = self.propagate(self.edge_index2, x=x, norm=self.norm2, size=None)
            tmp.append(x)

        out = (comb(self.K, 0) / (2 ** self.K)) * TEMP[0] * tmp[self.K]

        for i in range(self.K):
            x = tmp[self.K - i - 1]
            x = self.propagate(self.Lap_edge_index, x=x, norm=self.Lap_norm, size=None)
            for j in range(i):
                x = self.propagate(self.Lap_edge_index, x=x, norm=self.Lap_norm, size=None)

            out = out + (comb(self.K, i + 1) / (2 ** self.K)) * TEMP[i + 1] * x

        out = self.lin(out)
        out += self.bias
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j



class ChebIIConv(MessagePassing):
    def __init__(self, in_channels, out_channels, K, shared=True, bias=True, **kwargs):
        super(ChebIIConv, self).__init__(aggr='add', **kwargs)

        self.K = K
        self.weights = None
        self.Lap_edge_index = None
        self.Lap_norm = None
        self.lin = Linear(in_channels, out_channels)
        self.shared = shared
        self.cheb_values = self.cheb_value(self.K)

        if shared:
            self.register_parameter('temp', None)
        else:
            self.temp = Parameter(torch.Tensor(self.K + 1))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.fill_(0)
        self.temp.data.fill_(1)

    def cheb_value(self, K):
        cheb_nodes = np.polynomial.chebyshev.chebpts1(K + 1)
        values = torch.zeros((K+1, len(cheb_nodes)))

        for i in range(K + 1):
            for j in range(len(cheb_nodes)):
                values[i][j] = scipy.special.eval_chebyt(i, cheb_nodes[j])
        return values

    def forward(self, x, edge_index, coef, edge_weight=None):
        if self.shared:
            TEMP = F.relu(coef)
        else:
            TEMP = F.relu(self.temp)

        if self.Lap_norm is None:
            self.Lap_edge_index, self.Lap_norm = get_laplacian(edge_index, edge_weight, normalization='sym',
                                                               dtype=x.dtype, num_nodes=x.size(self.node_dim))

        Tx_0 = x
        Tx_1 = self.propagate(self.Lap_edge_index, x=x, norm=self.Lap_norm, size=None)
        self.cheb_values = self.cheb_values.to(TEMP.get_device())
        w = self.cheb_values @ TEMP
        w = w * 2 / (self.K + 1)
        out = w[0] * Tx_0 + w[1] * Tx_1

        for k in range(2, self.K + 1):
            Tx_2 = self.propagate(self.Lap_edge_index, x=Tx_1, norm=self.Lap_norm, size=None)
            Tx_2 = 2. * Tx_2 - Tx_0
            out += w[k] * Tx_2
            Tx_0, Tx_1 = Tx_1, Tx_2

        self.weights = w.detach().cpu().numpy()
        self.weights[0] /= 2

        out = self.lin(out)
        out += self.bias
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j



class NewtonConv(MessagePassing):
    def __init__(self, in_channels, out_channels, K, points, shared=True, bias=True, **kwargs):
        super(NewtonConv, self).__init__(aggr='add', **kwargs)

        self.K = K
        self.weights = None
        self.Lap_edge_index = None
        self.Lap_norm = None
        self.lin = Linear(in_channels, out_channels)
        self.points = points
        self.shared = shared

        if shared:
            self.register_parameter('temp', None)
        else:
            self.temp = Parameter(torch.Tensor(self.K + 1))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.bias is not None:
            self.bias.data.fill_(0)
        if self.temp is not None:
            self.temp.data.fill_(1)

    def k_quotient(self, k, points, values):
        table = torch.zeros((k + 1, k + 1))
        table[0] = values
        for i in range(1, k + 1):
            for j in range(i, k + 1):
                table[i][j] = (table[i - 1][j] - table[i - 1][j - 1]) / (points[j] - points[j - i])
        return table.diag()

    def forward(self, x, edge_index, coef, edge_weight=None):
        if self.shared:
            TEMP = F.relu(coef)
        else:
            TEMP = F.relu(self.temp)

        if self.Lap_norm is None:
            self.Lap_edge_index, self.Lap_norm = get_laplacian(edge_index, edge_weight, normalization='sym',
                                                               dtype=x.dtype, num_nodes=x.size(self.node_dim))

        quotients = self.k_quotient(self.K, self.points, TEMP)
        self.weights = quotients
        out = quotients[0] * x

        for k in range(1, self.K+1):
            tmp_lap_edge, tmp_lap_norm = add_self_loops(self.Lap_edge_index, self.Lap_norm, fill_value=self.points[k-1], num_nodes=x.size(self.node_dim))
            x = self.propagate(tmp_lap_edge, x=x, norm=tmp_lap_norm, size=None)
            out += quotients[k] * x

        out = self.lin(out)
        out += self.bias
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j



class MixHopLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hops=2):
        super(MixHopLayer, self).__init__()
        self.hops = hops
        self.lins = torch.nn.ModuleList()
        for hop in range(self.hops + 1):
            lin = torch.nn.Linear(in_channels, out_channels)
            self.lins.append(lin)

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, adj_t):
        xs = [self.lins[0](x)]
        for j in range(1, self.hops + 1):
            x_j = self.lins[j](x)
            for hop in range(j):
                x_j = matmul(adj_t, x_j)
            xs += [x_j]
        return torch.cat(xs, dim=1)



class Bern_prop(MessagePassing):
    def __init__(self, args, bias=True, **kwargs):
        super(Bern_prop, self).__init__(aggr='add', **kwargs)

        self.K = args.K
        self.device = args.device
        self.Lap_edge_index = None
        self.Lap_norm = None
        self.temp = Parameter(torch.Tensor(self.K + 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1)

    def forward(self, x, edge_index, edge_weight=None):
        TEMP=F.relu(self.temp)

        if self.Lap_norm is None:
            self.Lap_edge_index, self.Lap_norm = get_laplacian(edge_index, edge_weight, normalization='sym',
                                                               dtype=x.dtype, num_nodes=x.size(self.node_dim))
            self.edge_index2, self.norm2 = add_self_loops(self.Lap_edge_index, -self.Lap_norm, fill_value=2.,
                                                          num_nodes=x.size(self.node_dim))

        tmp = []
        tmp.append(x)
        for i in range(self.K):
            x = self.propagate(self.edge_index2, x=x, norm=self.norm2, size=None)
            tmp.append(x)

        out = (comb(self.K, 0) / (2 ** self.K)) * TEMP[0] * tmp[self.K]

        for i in range(self.K):
            x = tmp[self.K - i - 1]
            x = self.propagate(self.Lap_edge_index, x=x, norm=self.Lap_norm, size=None)
            for j in range(i):
                x = self.propagate(self.Lap_edge_index, x=x, norm=self.Lap_norm, size=None)

            out = out + (comb(self.K, i + 1) / (2 ** self.K)) * TEMP[i + 1] * x

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j



class ChebII_prop(MessagePassing):
    def __init__(self, args, **kwargs):
        super(ChebII_prop, self).__init__(aggr='add', **kwargs)

        self.K = args.K
        self.device = args.device
        self.weights = [0] * (self.K + 1)
        self.Lap_edge_index = None
        self.Lap_norm = None
        self.cheb_values = self.cheb_value(self.K).to(args.device)
        self.temp = Parameter(torch.Tensor(self.K + 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1)

    def cheb_value(self, K):
        cheb_nodes = np.polynomial.chebyshev.chebpts1(K + 1)
        values = torch.zeros((K+1, len(cheb_nodes)))

        for i in range(K + 1):
            for j in range(len(cheb_nodes)):
                values[i][j] = scipy.special.eval_chebyt(i, cheb_nodes[j])
        return values

    def forward(self, x, edge_index, edge_weight=None):
        TEMP = F.relu(self.temp)

        if self.Lap_norm is None:
            self.Lap_edge_index, self.Lap_norm = get_laplacian(edge_index, edge_weight, normalization='sym',
                                                               dtype=x.dtype, num_nodes=x.size(self.node_dim))

        Tx_0 = x
        Tx_1 = self.propagate(self.Lap_edge_index, x=x, norm=self.Lap_norm, size=None)
        w = self.cheb_values @ TEMP
        # w = w * 2 / (self.K + 1)
        out = w[0] * Tx_0 + w[1] * Tx_1

        for k in range(2, self.K+1):
            Tx_2 = self.propagate(self.Lap_edge_index, x=Tx_1, norm=self.Lap_norm, size=None)
            Tx_2 = 2. * Tx_2 - Tx_0

            out += w[k] * Tx_2
            Tx_0, Tx_1 = Tx_1, Tx_2

        self.weights = w.detach().cpu().numpy() * 2 / (self.K+1)
        self.weights[0] /= 2

        return out * 2 / (self.K+1)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j



class GPR_prop(MessagePassing):
    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            TEMP = 0.0 * np.ones(K + 1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            TEMP = alpha * (1 - alpha) ** np.arange(K + 1)
            TEMP[-1] = (1 - alpha) ** K
        elif Init == 'NPPR':
            TEMP = (alpha) ** np.arange(K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == 'Random':
            bound = np.sqrt(3 / (K + 1))
            TEMP = np.random.uniform(-bound, bound, K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == 'WS':
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K + 1):
            self.temp.data[k] = self.alpha * (1 - self.alpha) ** k
        self.temp.data[-1] = (1 - self.alpha) ** self.K

    def forward(self, x, edge_index, edge_weight=None):
        if isinstance(edge_index, torch.Tensor):
            edge_index, norm = gcn_norm(
                edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
            norm = None

        hidden = x * (self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k + 1]
            hidden = hidden + gamma * x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j



class Newton_prop(MessagePassing):
    def __init__(self, args, points, **kwargs):
        super(Newton_prop, self).__init__(aggr='add', **kwargs)

        self.K = args.K
        self.device = args.device
        self.weights = None
        self.Lap_edge_index = None
        self.Lap_norm = None
        self.temp = None
        self.points = points
        print('Interpolated points: ', self.points)

        self.temp = Parameter(torch.Tensor(self.K + 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1)

    def k_quotient(self, k, points, values):
        table = torch.zeros((k + 1, k + 1))
        table[0] = values
        for i in range(1, k + 1):
            for j in range(i, k + 1):
                table[i][j] = (table[i - 1][j] - table[i - 1][j - 1]) / (points[j] - points[j - i])
        return table.diag()

    def forward(self, x, edge_index, edge_weight=None):
        TEMP = F.relu(self.temp)

        if self.Lap_norm is None:
            self.Lap_edge_index, self.Lap_norm = get_laplacian(edge_index, edge_weight, normalization='sym',
                                                               dtype=x.dtype, num_nodes=x.size(self.node_dim))

        quotients = self.k_quotient(self.K, self.points, TEMP)
        self.weights = quotients
        out = quotients[0] * x

        for k in range(1, self.K+1):
            tmp_lap_edge, tmp_lap_norm = add_self_loops(self.Lap_edge_index, self.Lap_norm, fill_value=-self.points[k-1], num_nodes=x.size(self.node_dim))
            x = self.propagate(tmp_lap_edge, x=x, norm=tmp_lap_norm, size=None)
            out += quotients[k] * x

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j



class Filter_prop(MessagePassing):
    def __init__(self, args, **kwargs):
        super(Filter_prop, self).__init__(aggr='add', **kwargs)

        self.adj = None
        self.low = args.low
        self.middle = args.middle
        self.high = args.high
        self.mode = args.mode

    def filter(self, e):
        if self.mode == '3':
            for i in range(len(e)):
                if e[i] < 2/3:
                    e[i] = self.low
                elif 2/3 <= e[i] < 4/3:
                    e[i] = self.middle
                else:
                    e[i] = self.high

        elif self.mode == '2':
            for i in range(len(e)):
                if e[i] <= 1:
                    e[i] = self.low
                else:
                    e[i] = self.high
        else:
            print("Wrong mode", '!' * 100)
        return e

    def forward(self, x, edge_index, edge_weight=None):
        if self.adj == None:
            edge_index1, norm1 = get_laplacian(edge_index, edge_weight, normalization='sym', dtype=x.dtype,
                                               num_nodes=x.size(self.node_dim))

            L = to_dense_adj(edge_index1, edge_attr=norm1, max_num_nodes=x.shape[0]).squeeze()
            e, U = torch.linalg.eigh(L)

            e = self.filter(e)
            self.adj = U @ torch.diag_embed(e) @ U.mH

        out = self.adj @ x
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


