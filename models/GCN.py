# Install libraries
import torch

import torch_geometric
import torch_geometric.nn as pyg_nn

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
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

        super(GCN, self).__init__()
        torch.manual_seed(12345)

        self.num_layers = config['num_layers']
        self.hidden_dim = config['hidden_dim']
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_dim = config['edge_attr']
        self.activ_funct = config['activ_funct']
        self.dropout = config['dropout']
        self.final_dropout = config['final_dropout']
        self.graph_pooling_type = config['graph_pooling_type']

        self.conv0 = GCNConv(input_dim, self.hidden_dim)
        self.conv = GCNConv(self.hidden_dim, self.hidden_dim)
        self.lin = Linear(self.hidden_dim, output_dim)


    def block(self, x, edge_index, edge_weight=None):

        if edge_weight==None:
            
            x = self.conv(x, edge_index)

        else:
            x = self.conv(x, edge_index,  edge_weight.abs() )
            
        af = self.activ_funct
        if self.dropout != None:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = af(x)
        
        return x


    def forward(self, x, edge_index, batch, edge_weight=None):


        if edge_weight==None:
            x = self.conv0(x, edge_index)

        else:

            x = self.conv0(x, edge_index, edge_weight.abs() ) # we apply sigmoid() to edge weight to ensure they are all positve and the plus of being normalized(?)
        
        if self.dropout != None:
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        af = self.activ_funct
        x = af(x)
        
        hidden_layers = max(0, self.num_layers -1)

        for _ in range(hidden_layers):

            if edge_weight==None:
                 
                 x = self.block(x, edge_index)

            else:

                x = self.block(x, edge_index, edge_weight.abs())

        # 2. Readout layer
        self._pooling_func = getattr(pyg_nn, self.graph_pooling_type)

        x = self._pooling_func(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        if self.final_dropout > 0:
            x = F.dropout(x, p=self.final_dropout, training=self.training)
        
        x = self.lin(x)

        return x

