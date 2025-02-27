import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.utils import softmax , dropout_edge

class DEP(nn.Module):
    def __init__(self, input_dim, output_dim, config):
        super(DEP, self).__init__()
        
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.lr = config['lr']
        self.dropout = config['dropout']
        self.min_sp = config['min_sp']
        self.input_dim = input_dim # num_node_features 
        self.output_dim = output_dim

        self.weight_mask = nn.Parameter(torch.zeros(input_dim, input_dim)) 
        self.sig = nn.Sigmoid()

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
        Return the sample data 
        
        data: the original non-pruned data -> this ensures that although some edges were removed from iter 1
        they can still recover if their weight is regrown in weight_mask

        """
        # Reshape data 
        batch_size = data.ptr.shape[0] - 1 # batch size = max int + 1 eg [0 0 0 0 .... 15 15 15]

        num_edges_per_batch = data.edge_index.shape[1] // batch_size
        
        x_reshape = data.edge_attr.view(batch_size, -1)
        edge_ind_reshape = data.edge_index.T.reshape(batch_size, num_edges_per_batch, 2).permute(0, 2, 1)

        # Reshape weight mask
        edge_ind_reshape_0 = edge_ind_reshape[0] # it's enought to use the example of one subject in batch 1
        # all edge_attr are the same for all subjs

        weight_edge_attr = self.sym_weight_mask[edge_ind_reshape_0[0], edge_ind_reshape_0[1]] # get the entries from weight_mask that have an edge in our data

        num_elements = int(weight_edge_attr.numel() * self.min_sp)
        sorted_indices = torch.argsort(weight_edge_attr)[:num_elements]

        mask = torch.ones(x_reshape.size(1), dtype=torch.bool) 

        mask[sorted_indices] = False

        # Step 2: Filter edge_index and edge_attr using the mask
        filtered_edge_index = edge_ind_reshape[:, :, mask]
        filtered_edge_attr = x_reshape[:, mask]

        sample_data = data

        sample_data.edge_index = filtered_edge_index.permute(1,0,2).view( 2, -1)
        sample_data.edge_attr = filtered_edge_attr.view(-1)

        return sample_data, mask


    def get_params(self):
        """
        Method to return the current weight mask
        """
        return self.sym_weight_mask
