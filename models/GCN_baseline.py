import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool


class GCN_baseline(nn.Module):
    """Small dense GCN classifier with the repository model interface."""

    def __init__(self, config, input_dim, output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = int(config["hidden_dim"])
        self.num_layers = int(config.get("num_layers", config.get("layers", 3)))
        self.dropout = float(config.get("dropout", 0.0))
        self.use_edge_weight = bool(
            config.get("use_edge_weight", config.get("edge_attr", False))
        )

        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.convs = nn.ModuleList(
            [GCNConv(input_dim, self.hidden_dim)]
            + [
                GCNConv(self.hidden_dim, self.hidden_dim)
                for _ in range(self.num_layers - 1)
            ]
        )
        self.classifier = nn.Linear(self.hidden_dim, output_dim)

    def forward(
        self,
        data: Data,
        edge_index=None,
        batch=None,
        edge_attr=None,
    ) -> torch.Tensor:
        # The Data-object path is the canonical baseline interface. The
        # tensor-argument path keeps compatibility with the existing DEP
        # trainer, which calls models as model(x, edge_index, batch, edge_attr).
        if isinstance(data, Data):
            x = data.x
            edge_index = data.edge_index
            batch = data.batch
            edge_attr = getattr(data, "edge_attr", None)
        else:
            x = data

        edge_weight = None
        if self.use_edge_weight and edge_attr is not None:
            # GCN normalization can produce invalid values for signed weights.
            edge_weight = edge_attr.abs()
            if edge_weight.ndim > 1:
                edge_weight = edge_weight.squeeze(-1)

        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight))
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.classifier(global_mean_pool(x, batch))
