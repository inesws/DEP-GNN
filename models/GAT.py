# Install libraries
import os
import torch
from torch import nn

import torch_geometric
import torch_geometric.nn as pyg_nn

import torch_geometric.datasets as Datasets
import torch_geometric.data as Data
import torch_geometric.transforms as transforms
from torch_geometric.utils import remove_self_loops

import numpy as np
import scipy as sp, scipy.io
import pandas as pd
import matplotlib.pyplot as plt

from torch.nn import Linear, Softmax
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class xGAT(torch.nn.Module):
    def __init__(self, config, input_dim, output_dim):
        #num_layers, num_heads, num_heads_final, hidden_dim, input_dim, output_dim, activ_funct, final_dropout, graph_pooling_type, device, edge_dim=None

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


        self.gat0 = GATv2Conv(self.input_dim, self.hidden_dim, heads=self.num_heads, edge_dim=self.edge_dim)
        self.gat = GATv2Conv(self.num_heads*self.hidden_dim, self.hidden_dim, heads=self.num_heads,  edge_dim=self.edge_dim)
        self.gatF = GATv2Conv(self.num_heads*self.hidden_dim, self.final_hidden_dim, heads=self.num_heads_final, edge_dim=self.edge_dim, concat=False)
        self.attention_weights = None
        self.lin = Linear(self.final_hidden_dim, self.output_dim)
        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax()


    #def get_e

    def block(self, x, edge_index):
        
        x = self.gat(x, edge_index)
        af = self.activ_funct
        x = af(x)
        
        return x
    
    def block_w_edge(self, x, edge_index, edge_attr):
        
        x = self.gat(x, edge_index,  edge_attr)
        af = self.activ_funct
        x = af(x)
        
        return x


    def forward(self, x, edge_index, batch, edge_attr=None):

        if self.edge_dim is not None:

            x = self.gat0(x, edge_index,  edge_attr)
            af = self.activ_funct
            x = af(x)

            hidden_layers = max(0, self.num_layers -2)

            for _ in range(hidden_layers):

                x = self.block_w_edge(x, edge_index, edge_attr)

            # Final Layer
            x, (edge_index, alpha) = self.gatF(x, edge_index, edge_attr, return_attention_weights =True)
            af = self.activ_funct
            x = af(x)


        else:

            x = self.gat0(x, edge_index)
            af = self.activ_funct
            x = af(x)

            hidden_layers = max(0, self.num_layers -2)

            for _ in range(hidden_layers):

                x = self.block(x, edge_index)

            # Final Layer
            x, (edge_index, alpha) = self.gatF(x, edge_index, return_attention_weights =True)
            af = self.activ_funct
            x = af(x)


        # 2. Readout layer
        self._pooling_func = getattr(pyg_nn, self.graph_pooling_type)

        x = self._pooling_func(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        if self.final_dropout > 0:

            x = F.dropout(x, p=self.final_dropout, training=self.training)
        
        x = self.lin(x)

        #x= self.sig(x) # OPTIONAL
        #x = self.softmax(x)

        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        return x#, alpha