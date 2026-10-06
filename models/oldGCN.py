# Install libraries
import torch

import torch_geometric
import torch_geometric.nn as pyg_nn

from torch.nn import Linear, LayerNorm
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class oldGCN(torch.nn.Module):
    def __init__(self, config,input_dim, output_dim):

        ''' config should have
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            hidden_dim: dimensionality of hidden units at ALL layers
            input_dim: num of node features in input
            output_dim: number of classes for prediction
            edge_dim : if edge weight exist=dim of edge weight, otherwise None
            activ_funct: activation function to use in ALL layers
            dropout : dropout ratio in all layers except last
            final_dropout: dropout ratio on the final linear layer
            graph_pooling_type: how to aggregate entire nodes in a graph (mean, average)
        '''

        super(oldGCN, self).__init__()
        torch.manual_seed(12345)

        self.num_layers = config['num_layers']
        self.hidden_dim = config['hidden_dim']
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_dim = config['edge_attr']
        self.activ_funct = getattr(torch.nn, config['activ_funct'])()
        self.dropout = config['dropout']
        self.final_dropout = config['final_dropout']
        self.graph_pooling_type = getattr(pyg_nn, config['graph_pooling_type'])

        # Separate conv + LayerNorm per layer
        self.convs = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()

        self.convs.append(GCNConv(input_dim, self.hidden_dim))
        self.layer_norms.append(LayerNorm(input_dim))

        for _ in range(max(0, self.num_layers - 1)):
            self.convs.append(GCNConv(self.hidden_dim, self.hidden_dim))
            self.layer_norms.append(LayerNorm(self.hidden_dim))

        self.lin = Linear(self.hidden_dim, output_dim)


    def forward(self, x, edge_index, batch, edge_weight=None):

        ew = edge_weight.abs() if edge_weight is not None else None

        for conv, ln in zip(self.convs, self.layer_norms):
            x = ln(x)
            x = conv(x, edge_index) if ew is None else conv(x, edge_index, ew)
            if self.dropout is not None:
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.activ_funct(x)

        # Readout layer
        x = self.graph_pooling_type(x, batch)  # [batch_size, hidden_channels]

        # Final classifier
        if self.final_dropout is not None:
            x = F.dropout(x, p=self.final_dropout, training=self.training)

        x = self.lin(x)

        return x
