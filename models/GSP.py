import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GPSConv, GCNConv, GINEConv
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU


class GSP(torch.nn.Module):
    def __init__(self, config, input_dim, output_dim):
        """GPS (General, Powerful, Scalable) Graph Transformer backbone.

        Each GPSConv layer combines a local MPNN (GCN) with global
        multi-head self-attention, following the config-based pattern
        used by the other backbones in this project.

        Positional encoding is derived at forward time via torch.eye(N)
        where N = number of nodes per graph (= input_dim for brain graphs
        with fixed ROI parcellation). The PE is BatchNorm-ed, projected to
        pe_dim, and concatenated with the projected node features, following
        the original GPS recipe.

        Config keys (from yaml):
            num_layers:         number of GPS layers
            hidden_dim:         hidden dimensionality
            pe_dim:             positional encoding projection dim (default 16)
            heads:              number of attention heads (default 4)
            edge_attr:          edge feature dim (int) or null for no edge features
            dropout:            dropout in GPS layers (null for None)
            final_dropout:      dropout before the final linear classifier
            activ_funct:        activation function name (ReLU, ELU, …)
            graph_pooling_type: global pooling (global_mean_pool, global_add_pool)
            local_mpnn:         local message-passing type: 'GCN' or 'GIN' (default 'GCN')
        """
        super(GSP, self).__init__()
        torch.manual_seed(12345)

        self.num_layers = config['num_layers']
        self.hidden_dim = config['hidden_dim']
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pe_dim = config.get('pe_dim', 16)
        self.edge_dim = config.get('edge_attr', None)
        self.heads = config.get('heads', 4)
        self.dropout = config.get('dropout', None) or 0.0
        self.final_dropout = config.get('final_dropout', None)
        self.activ_funct = getattr(torch.nn, config.get('activ_funct', 'ReLU'))()
        self.graph_pooling_type = getattr(pyg_nn, config.get('graph_pooling_type', 'global_mean_pool'))
        self.local_mpnn = config.get('local_mpnn', 'GCN')

        # Positional encoding branch 
        # input_dim = N (number of ROIs) since PE is torch.eye(N)
        self.pe_norm = BatchNorm1d(input_dim)
        self.pe_lin = Linear(input_dim, self.pe_dim)

        # Node feature projection (to hidden_dim - pe_dim so concat = hidden_dim)
        self.node_proj = Linear(input_dim, self.hidden_dim - self.pe_dim)

        # Build GPS layers
        self.convs = nn.ModuleList()

        for _ in range(self.num_layers):
            # Local MPNN inside each GPS layer
            if self.local_mpnn == 'GIN':
                local_nn = Sequential(
                    Linear(self.hidden_dim, self.hidden_dim),
                    ReLU(),
                    Linear(self.hidden_dim, self.hidden_dim),
                )
                if self.edge_dim is not None:
                    local_conv = GINEConv(local_nn, edge_dim=self.edge_dim)
                else:
                    local_conv = GINEConv(local_nn)
            else:  # default: GCN
                local_conv = GCNConv(self.hidden_dim, self.hidden_dim)

            conv = GPSConv(
                channels=self.hidden_dim,
                conv=local_conv,
                heads=self.heads,
                dropout=self.dropout,
                attn_type='multihead',
            )
            self.convs.append(conv)

        # MLP classifier (like original GPS: channels → channels//2 → channels//4 → output)
        self.mlp = Sequential(
            Linear(self.hidden_dim, self.hidden_dim // 2),
            ReLU(),
            Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            ReLU(),
            Linear(self.hidden_dim // 4, output_dim),
        )

    def forward(self, x, edge_index, batch, edge_weight=None):
        # --- Positional encoding via torch.eye(N) ---
        # All brain graphs share the same ROI parcellation, so N is constant.
        num_graphs = batch.max().item() + 1
        num_nodes_per_graph = x.size(0) // num_graphs
        pe = torch.eye(num_nodes_per_graph, device=x.device).repeat(num_graphs, 1)
        pe = self.pe_norm(pe)
        pe = self.pe_lin(pe)

        # --- Node feature projection ---
        h = self.node_proj(x)

        # --- Concatenate node features + PE → hidden_dim ---
        x = torch.cat([h, pe], dim=-1)

        # GPS layers (no extra norm/activation — GPSConv has internal LayerNorm)
        for conv in self.convs:
            if edge_weight is not None and self.local_mpnn == 'GCN':
                x = conv(x, edge_index, batch, edge_attr=edge_weight.abs())
            elif edge_weight is not None:
                x = conv(x, edge_index, batch, edge_attr=edge_weight.unsqueeze(-1) if edge_weight.dim() == 1 else edge_weight)
            else:
                x = conv(x, edge_index, batch)

        # Global pooling
        x = self.graph_pooling_type(x, batch)

        # MLP classifier
        return self.mlp(x)
