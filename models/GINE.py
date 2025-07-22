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
from torch_geometric.nn import GINEConv
import sys
sys.path.append("models/")
from MLP import MLP


class xGINE(torch.nn.Module):
    def __init__(self, config, input_dim, output_dim):

        '''
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            num_mlp_layers :
            hidden_dim: dimensionality of hidden units at ALL layers
            input_dim: num of node features in input
            output_dim: number of classes for prediction
            activ_funct: activation function
            final_dropout: dropout ratio on the final linear layer
            neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
            graph_pooling_type: how to aggregate entire nodes in a graph (mean, average)
            train_eps:
            eps:
            device: which device to use
        '''
        super(xGINE, self).__init__()
        torch.manual_seed(12345)

        self.num_layers = config['num_layers']
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activ_funct = config['activ_funct']
        self.final_dropout = config['final_dropout']
        self.graph_pooling_type = config['graph_pooling_type']
        self.train_eps = config['train_eps']
        self.edge_dim = config['edge_attr']
        
        self.eps_ = nn.Parameter(torch.zeros(config['num_layers']))
        self.mlps_ = torch.nn.ModuleList()
        self.batch_norms_ = torch.nn.ModuleList()
        #self.linears_predictions_ = torch.nn.ModuleList()

        for layer in range(self.num_layers):
            if layer == 0:
                self.mlps_.append(MLP(config, input_dim, config['hidden_dim']))
            else:
                self.mlps_.append(MLP(config, config['hidden_dim'], config['hidden_dim']))

            self.batch_norms_.append(nn.BatchNorm1d(config['hidden_dim']))
            #self.linears_prediction_.append(nn.Linear( config['hidden_dim'], output_dim))


        self.gin0 = GINEConv(self.mlps_[0], train_eps=self.train_eps, edge_dim=self.edge_dim)
        self.gin = GINEConv(self.mlps_[1], train_eps=self.train_eps, edge_dim=self.edge_dim)
        self.lin = Linear(config['hidden_dim'], output_dim)


    def block(self, x, edge_index, layer_idx, edge_attr):
        
        x = self.gin(x, edge_index, edge_attr)

        x = self.batch_norms_[layer_idx](x)

        af = self.activ_funct
        x = af(x)
        
        return x
    


    def forward(self, x, edge_index,  batch, edge_attr):
            

        self.hidden_rep_ = []
        
        edge_attr = edge_attr.unsqueeze(-1)

        x = self.gin0(x, edge_index, edge_attr)
        x = self.batch_norms_[0](x)
        af = self.activ_funct
        x = af(x)

        self.hidden_rep_.append(x)

        hidden_layers = max(0, self.num_layers -1)

        for layer_idx in range(hidden_layers):

            x = self.block(x, edge_index, layer_idx+1, edge_attr)
            self.hidden_rep_.append(x)


        # 2. Readout layer
        self._pooling_func = getattr(pyg_nn, self.graph_pooling_type)

        x = self._pooling_func(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        if self.final_dropout > 0:

            x = F.dropout(x, p=self.final_dropout, training=self.training)
        
        x = self.lin(x)


        return x
    
def compute_saliency(self, x, edge_index, batch, target_class=None):
    """
    Compute saliency map for the input with respect to a specific target class.

    Args:
        x (torch.Tensor): Node features.
        edge_index (torch.Tensor): Edge list.
        batch (torch.Tensor): Batch tensor for graph pooling.
        target_class (int, optional): The specific class to compute saliency for.
                                      If None, computes saliency for the predicted class.
    
    Returns:
        torch.Tensor: Saliency map with the same shape as `x`.
    """
    self.eval()  # Ensure the model is in evaluation mode
    self.zero_grad()  # Clear any existing gradients

    # Enable gradient tracking for input features
    x.requires_grad_()

    # Forward pass
    logits = self.forward(x, edge_index, batch)  # [num_graphs, num_classes]

    if target_class is None:
        # Use the predicted class if no specific target class is provided
        target_class = logits.argmax(dim=1)

    # One-hot encode the target class for gradient computation
    # Ensure proper batching if multiple graphs are present
    one_hot = torch.zeros_like(logits)
    for i, cls in enumerate(target_class):
        one_hot[i, cls] = 1.0

    # Backward pass to compute gradients
    logits.backward(gradient=one_hot)

    # Saliency is the gradient of input w.r.t. the target class
    saliency = x.grad

    return saliency


