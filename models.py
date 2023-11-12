import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv, APPNP, GATConv
from torch_geometric.utils import get_laplacian
import numpy as np
from layers import *



class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, args):
        super(MLP, self).__init__()

        self.L = args.L
        self.dropout = args.dropout
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()

        if self.L == 1:
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(self.L - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index=None):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x



class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, args, use_bn=True):
        super(GCN, self).__init__()

        self.L = args.L
        self.dropout = args.dropout
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(args.L - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x



class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, args):
        super(GAT, self).__init__()

        self.L = args.L
        self.dropout = args.dropout
        heads = args.gat_heads
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
        for _ in range(self.L - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True))
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=heads, concat=False))

        self.activation = F.elu

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x



class DenseGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, args):
        super(DenseGCN, self).__init__()
        self.L = args.L
        self.dropout = args.dropout
        self.convs = nn.ModuleList()

        self.convs.append(DenseGCNConv(in_channels, hidden_channels))
        for _ in range(self.L - 2):
            self.convs.append(DenseGCNConv(hidden_channels, hidden_channels))
        self.convs.append(DenseGCNConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj)
        return x


class APPNPNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, args):
        super(APPNPNet, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.prop = APPNP(args.K, args.gpr_alpha)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop(x, edge_index)
        return x



class MixHop(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, args):
        super(MixHop, self).__init__()

        self.dropout = args.dropout
        hops = args.hops
        self.L = args.L
        self.convs = nn.ModuleList()
        self.convs.append(MixHopLayer(in_channels, hidden_channels, hops=hops))
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels * (hops + 1)))
        for _ in range(self.L - 2):
            self.convs.append(MixHopLayer(hidden_channels * (hops + 1), hidden_channels, hops=hops))
            self.bns.append(nn.BatchNorm1d(hidden_channels * (hops + 1)))

        self.convs.append(MixHopLayer(hidden_channels * (hops + 1), out_channels, hops=hops))
        self.final_project = nn.Linear(out_channels * (hops + 1), out_channels)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.final_project.reset_parameters()

    def forward(self, x, edge_index):
        num_nodes = x.shape[0]
        edge_weight = None
        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm(
                edge_index, edge_weight, num_nodes, False,
                dtype=x.dtype)
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=edge_weight, sparse_sizes=(num_nodes, num_nodes))
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(edge_index, edge_weight, num_nodes, False, dtype=x.dtype)
            edge_weight = None
            adj_t = edge_index

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        x = self.final_project(x)
        return x



class ChebNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, args):
        super(ChebNet, self).__init__()
        self.K = args.K
        self.L = args.L
        self.dropout = args.dropout
        self.convs = nn.ModuleList()

        self.convs.append(ChebConv(in_channels, hidden_channels, self.K))
        for _ in range(self.L - 2):
            self.convs.append(ChebConv(hidden_channels, hidden_channels, self.K))
        self.convs.append(ChebConv(hidden_channels, out_channels, self.K))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GPRGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, args, Init='PPR', Gamma=None, ppnp='GPR_prop'):
        super(GPRGNN, self).__init__()

        K = args.K
        alpha = args.gpr_alpha
        self.Init = Init
        self.dprate = args.dprate
        self.dropout = args.dropout

        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)

        if ppnp == 'PPNP':
            self.prop = APPNP(K, alpha)
        elif ppnp == 'GPR_prop':
            self.prop = GPR_prop(K, alpha, Init, Gamma)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop(x, edge_index)
            return x
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop(x, edge_index)
            return x



class Bern(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, args):
        super(Bern, self).__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.prop = Bern_prop(args)
        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.prop.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop(x, edge_index)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop(x, edge_index)
        return x



class ChebII(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, args):
        super(ChebII, self).__init__()

        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.prop = ChebII_prop(args)

        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.prop.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop(x, edge_index)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop(x, edge_index)
        return x



class Newton(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, args, points):
        super(Newton, self).__init__()

        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.prop = Newton_prop(args, points)

        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.prop.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop(x, edge_index)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop(x, edge_index)
        return x



class Filter(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, args):
        super(Filter, self).__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

        self.dprate = args.dprate
        self.dropout = args.dropout
        self.prop = Filter_prop(args)

    def reset_parameters(self):
        self.prop.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop(x, edge_index)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop(x, edge_index)
        return x


