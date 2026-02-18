import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.utils import softmax , dropout_edge

class IndiDEP(nn.Module):
    def __init__(self, params, input_dim, output_dim):
        super(IndiDEP, self).__init__()
        
        self.alpha = params.alpha 
        self.beta = params.beta
        self.lr = params.lr
        self.dropout = params.dropout
        self.min_sp = params.curr_sp
        self.input_dim = input_dim # num_node_features 
        self.output_dim = output_dim

        self.weight_mask = nn.Parameter(torch.zeros(input_dim, input_dim)) 
        self.sig = nn.Sigmoid()
        self.edge_mask = None  # binary edge-selection mask from last prune()

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.xavier_normal_(self.weight_mask)
      
    def forward(self, data): #, ptr
        """
        Forward pass 
        Args:
            data: always the original data non-prunned -> a BatchData from DataLoader

        Returns:
            sample_data: According to current weight_mask -> returns a sampled version of data
        """
        
        self.sym_weight_mask = self.sig(self.weight_mask.T + self.weight_mask)

        if self.dropout is not None:
            self.sym_weight_mask = F.dropout( self.sym_weight_mask, p=self.dropout, training=self.training)

        sample_data, mask = self.prune(data)
        
        return sample_data, mask 

    def update_prune(self, sp_increase ):

        self.min_sp = round(self.min_sp + sp_increase,5)


    def l1_norm(self):

        """
        Lasso regularization loss (L1 loss) on the weights.
        Args:
            None
        Returns:
            lasso_loss: The L1 regularization loss.
        """
        l1 = self.alpha * torch.sum(torch.abs(self.sym_weight_mask))  

        return l1  # L1 regularization
    
    def l2_norm(self):

        l2 = self.beta * torch.norm(self.sym_weight_mask, p=2) **2

        return l2
    

    def prune(self, data):
        """
        Subject-based pruning:
        1. Multiply the learned weight_mask with each subject's actual edge_attr
        2. For each subject independently, prune the smallest-valued edges
        
        This means an edge pruned for subject A may survive for subject B
        if its edge_attr weight differs.

        data: the original non-pruned data -> ensures edges can recover
              if weight_mask regrows them
        """
        # Reshape data
        batch_size = data.ptr.shape[0] - 1
        num_edges_per_batch = data.edge_index.shape[1] // batch_size

        x_reshape = data.edge_attr.view(batch_size, -1)  # (B, E)
        edge_ind_reshape = data.edge_index.T.reshape(batch_size, num_edges_per_batch, 2).permute(0, 2, 1)  # (B, 2, E)

        # Get weight_mask values for the edge positions (same topology for all subjects)
        edge_ind_0 = edge_ind_reshape[0]  # (2, E) — use first subject's edge indices
        weight_edge_attr = self.sym_weight_mask[edge_ind_0[0], edge_ind_0[1]]  # (E,)

        # Multiply weight_mask scores with each subject's edge_attr → (B, E)
        # Each subject gets a personalised importance score per edge
        subject_scores = weight_edge_attr.unsqueeze(0) * x_reshape  # (B, E)

        # Per-subject pruning: for each subject, prune the smallest scored edges
        num_to_prune = int(subject_scores.size(1) * self.min_sp)

        # Sort each subject's scores and build per-subject masks (B, E)
        _, sorted_indices = torch.sort(subject_scores, dim=1)  # ascending
        prune_indices = sorted_indices[:, :num_to_prune]  # (B, num_to_prune)

        mask = torch.ones(batch_size, subject_scores.size(1), dtype=torch.bool,
                          device=subject_scores.device)  # (B, E)
        mask.scatter_(1, prune_indices, False)  # set pruned positions to False

        # Store per-subject mask (B, E)
        self.edge_mask = mask

        # Filter edge_index and edge_attr per subject
        # Since each subject may keep different edges, we collect per-subject
        filtered_edge_indices = []
        filtered_edge_attrs = []
        new_ptr = [0]

        for b in range(batch_size):
            subj_mask = mask[b]  # (E,)
            subj_edge_index = edge_ind_reshape[b, :, subj_mask]  # (2, kept_edges)
            subj_edge_attr = x_reshape[b, subj_mask]  # (kept_edges,)

            filtered_edge_indices.append(subj_edge_index)
            filtered_edge_attrs.append(subj_edge_attr)
            new_ptr.append(new_ptr[-1] + subj_mask.sum().item())

        sample_data = data.clone()
        sample_data.edge_index = torch.cat(filtered_edge_indices, dim=1)  # (2, total_kept)
        sample_data.edge_attr = torch.cat(filtered_edge_attrs, dim=0)  # (total_kept,)

        # Rebuild batch vector for edges if needed (ptr stays valid for nodes)
        return sample_data, mask


    def get_params(self):
        """
        Method to return the current weight mask
        """
        return self.sym_weight_mask

    def get_edge_mask(self):
        """
        Return the binary edge-selection mask from the last forward/prune call.
        True = edge kept, False = edge pruned.
        """
        return self.edge_mask
