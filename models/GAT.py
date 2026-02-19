# Install libraries
import os
import torch
from torch import nn

import torch_geometric
import torch_geometric.nn as pyg_nn

from torch.nn import Linear, Softmax, LayerNorm
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class xGAT(torch.nn.Module):
    def __init__(self, config, input_dim, output_dim):

        '''
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            hidden_dim: dimensionality of hidden units at ALL layers
            input_dim: num of node features in input
            output_dim: number of classes for prediction
            activ_funct: activation function
            final_dropout: dropout ratio on the final linear layer
            neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
            graph_pooling_type: how to aggregate entire nodes in a graph (mean, average)
            device: which device to use
        '''
        super(xGAT, self).__init__()
        torch.manual_seed(12345)

        self.num_layers = config['num_layers']
        self.hidden_dim = config['hidden_dim']
        self.final_hidden_dim = config['final_hidden_dim']
        self.num_heads = config['num_heads']
        self.num_heads_final = config['num_heads_final']
        self.edge_dim = config['edge_attr']
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activ_funct = config['activ_funct']
        self.final_dropout = config['final_dropout']
        self.graph_pooling_type = config['graph_pooling_type']

        # Independent conv + LayerNorm per layer
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        # First layer: input_dim → hidden_dim * num_heads
        self.convs.append(GATv2Conv(self.input_dim, self.hidden_dim,
                                    heads=self.num_heads, edge_dim=self.edge_dim))
        self.layer_norms.append(LayerNorm(self.input_dim))

        # Hidden layers: (hidden_dim * num_heads) → hidden_dim * num_heads
        for _ in range(max(0, self.num_layers - 2)):
            self.convs.append(GATv2Conv(self.num_heads * self.hidden_dim, self.hidden_dim,
                                        heads=self.num_heads, edge_dim=self.edge_dim))
            self.layer_norms.append(LayerNorm(self.num_heads * self.hidden_dim))

        # Final GAT layer: returns attention weights, concat=False → final_hidden_dim
        self.convs.append(GATv2Conv(self.num_heads * self.hidden_dim, self.final_hidden_dim,
                                    heads=self.num_heads_final, edge_dim=self.edge_dim, concat=False))
        self.layer_norms.append(LayerNorm(self.num_heads * self.hidden_dim))

        self.attention_weights = None
        self.lin = Linear(self.final_hidden_dim, self.output_dim)


    def forward(self, x, edge_index, batch, edge_attr=None):

        # All layers except the last (no attention weight return)
        for conv, ln in zip(self.convs[:-1], self.layer_norms[:-1]):
            x = ln(x)
            x = conv(x, edge_index, edge_attr=edge_attr) if self.edge_dim is not None else conv(x, edge_index)
            x = self.activ_funct(x)

        # Final layer: return attention weights
        x = self.layer_norms[-1](x)
        if self.edge_dim is not None:
            x, (edge_index, alpha) = self.convs[-1](x, edge_index, edge_attr, return_attention_weights=True)
        else:
            x, (edge_index, alpha) = self.convs[-1](x, edge_index, return_attention_weights=True)
        x = self.activ_funct(x)

        # Readout layer
        self._pooling_func = getattr(pyg_nn, self.graph_pooling_type)
        x = self._pooling_func(x, batch)

        # Final classifier
        if self.final_dropout > 0:
            x = F.dropout(x, p=self.final_dropout, training=self.training)

        x = self.lin(x)

        return x