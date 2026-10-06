# Install libraries
import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as pyg_nn

from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_geometric.utils import degree


class GCNConv(MessagePassing):
    """Custom GCN Convolution with mask support for topology sparsification."""
    
    def __init__(self, input_dim, emb_dim):
        super(GCNConv, self).__init__(aggr='add')
        self.linear = nn.Linear(input_dim, emb_dim)

    def forward(self, x, edge_index, data_mask=None):
        x = self.linear(x)
        row, col = edge_index
        
        # Degree normalization
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        # Apply mask if provided (for topology sparsification)
        if data_mask is not None:
            norm = deg_inv_sqrt[row] * data_mask * deg_inv_sqrt[col]
        else:
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * F.relu(x_j)

    def update(self, aggr_out):
        return aggr_out

    def reset_parameters(self):
        self.linear.reset_parameters()


class GCN_MoG(torch.nn.Module):
    """
    GCN architecture compatible with MoG topology sparsification.
    Organized to match DEP-GNN's GCN.py structure.
    
    Config should have:
        num_layers: number of layers (including input layer)
        hidden_dim: dimensionality of hidden units at ALL layers
        input_dim: num of node features in input
        output_dim: number of classes for prediction
        dropout: dropout ratio in all layers except last
        final_dropout: dropout ratio on the final linear layer
        graph_pooling_type: how to aggregate nodes (mean, sum, max, attention, set2set)
        residual: whether to use residual connections
        jk_mode: JK aggregation mode (last, sum)
    """

    def __init__(self, config, input_dim, output_dim):
        super(GCN_MoG, self).__init__()
        torch.manual_seed(12345)

        self.num_layers = config['num_layers']
        self.hidden_dim = config['hidden_dim']
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = config['dropout']
        self.final_dropout = config['final_dropout']
        self.residual = config.get('residual', False)
        self.jk_mode = config.get('jk_mode', 'last')

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # GCN convolution layers
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        # First layer: input_dim -> hidden_dim
        self.convs.append(GCNConv(input_dim, self.hidden_dim))
        self.batch_norms.append(BatchNorm1d(self.hidden_dim))

        # Hidden layers: hidden_dim -> hidden_dim
        for _ in range(max(0, self.num_layers - 1)):
            self.convs.append(GCNConv(self.hidden_dim, self.hidden_dim))
            self.batch_norms.append(BatchNorm1d(self.hidden_dim))

        # Pooling function to generate whole-graph embeddings
        pooling_type = config.get('graph_pooling_type', 'mean')
        if pooling_type == 'sum':
            self.pool = global_add_pool
        elif pooling_type == 'mean':
            self.pool = global_mean_pool
        elif pooling_type == 'max':
            self.pool = global_max_pool
        elif pooling_type == 'attention':
            self.pool = GlobalAttention(
                gate_nn=nn.Sequential(
                    nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
                    nn.BatchNorm1d(2 * self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(2 * self.hidden_dim, 1)
                )
            )
        elif pooling_type == 'set2set':
            self.pool = Set2Set(self.hidden_dim, processing_steps=2)
        else:
            raise ValueError(f"Invalid graph pooling type: {pooling_type}")

        # Graph-level projection layer
        self.lin1 = Linear(self.hidden_dim, self.hidden_dim)

        # Final classifier
        if pooling_type == 'set2set':
            self.lin_final = Linear(2 * self.hidden_dim, output_dim)
        else:
            self.lin_final = Linear(self.hidden_dim, output_dim)

    def forward(self, x, edge_index, batch, edge_weight=None):
        """
        Forward pass for GCN.
        
        Args:
            x: Node feature tensor
            edge_index: Edge connectivity
            batch: Batch assignment for nodes
            edge_weight: Optional edge weights (not used - sparsification handled externally)
        
        Returns:
            Predictions
        """
        # Node representation from GCN layers
        h_list = [x]
        for layer in range(self.num_layers):
            h = self.convs[layer](h_list[layer], edge_index)
            h = self.batch_norms[layer](h)

            # No ReLU on final GCN layer
            if layer == self.num_layers - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)
            else:
                h = F.dropout(F.relu(h), p=self.dropout, training=self.training)

            # Residual connection
            if self.residual and h.size(-1) == h_list[layer].size(-1):
                h = h + h_list[layer]

            h_list.append(h)

        # JK aggregation
        if self.jk_mode == 'last':
            node_representation = h_list[-1]
        elif self.jk_mode == 'sum':
            node_representation = torch.zeros_like(h_list[-1])
            for layer_repr in h_list:
                node_representation = node_representation + layer_repr
        else:
            raise ValueError(f"Invalid JK mode: {self.jk_mode}")

        # Graph-level pooling
        h_graph = self.pool(node_representation, batch)

        # Graph-level MLP
        h_graph = F.relu(self.lin1(h_graph))
        h_graph = F.dropout(h_graph, p=self.final_dropout, training=self.training)
        h_graph = self.lin_final(h_graph)

        return h_graph

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.batch_norms:
            bn.reset_parameters()
        self.lin1.reset_parameters()
        self.lin_final.reset_parameters()
