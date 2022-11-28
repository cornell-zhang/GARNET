import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor
import numpy as np


class ada_prop(MessagePassing):
    def __init__(self, P, coe, bias=True, **kwargs):
        super(ada_prop, self).__init__(aggr='add', **kwargs)
        self.P = P
        self.coe = coe

        coes = coe*(1-coe)**np.arange(P+1)
        coes[-1] = (1-coe)**P

        self.coes = nn.Parameter(torch.tensor(coes))

    def reset_parameters(self):
        nn.init.zeros_(self.coes)
        for p in range(self.P+1):
            self.coes.data[p] = self.coe*(1-self.coe)**p
        self.coes.data[-1] = (1-self.coe)**self.P

    def forward(self, x, edge_index, edge_weight=None):
        if isinstance(edge_index, torch.Tensor):
            edge_index, norm = gcn_norm(
                edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
            norm = None

        hidden = x*(self.coes[0])
        for p in range(self.P):
            if norm is None:
                x = edge_index @ x
            else:
                x = self.propagate(edge_index, x=x, norm=norm)
            c = self.coes[p+1]
            hidden = hidden + c*x
        return hidden

    def message(self, x_j, norm):
        if norm is not None:
            return norm.view(-1, 1) * x_j
        else:
            return x_j

class ada_filter(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=.5, coe=.5, P=10):
        super(ada_filter, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.prop = ada_prop(P, coe)

        self.dropout = dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)


        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.prop(x, edge_index)
        return F.log_softmax(x, dim=1)


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3,
                 dropout=0.5, use_bn=False, norm=True):
        super(GCN, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=True, normalize=norm))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True, normalize=norm))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(
            GCNConv(hidden_channels, out_channels, cached=True, normalize=norm))

        self.dropout = dropout
        self.activation = F.relu
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
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)
