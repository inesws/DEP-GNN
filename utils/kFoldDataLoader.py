import torch
import numpy as np
import torch_geometric


from torch_geometric.loader import DataLoader

from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold


class KFold_DataLoader(object):

    def __init__(self, num_repeats, num_k, batch_size, stratify= False, random_seed=None):
        
        self.num_repeats = num_repeats
        self.num_k = num_k
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.stratify = stratify
        self.rskf = RepeatedStratifiedKFold(n_repeats=num_repeats, n_splits=num_k, random_state=random_seed)
        self.rkf = RepeatedKFold(n_repeats=num_repeats, n_splits=num_k, random_state=random_seed)
        


    def get_nk_loaders(self, dataset):

        train_loaders = []
        val_loaders = []

        if hasattr(dataset, 'y'): 
            y = dataset.y
        else:
            y = [np.array(dataset[i].y) for i in range(len(dataset))] 

        if self.stratify==True:
            for i, (train_idx, val_idx) in enumerate(self.rskf.split(dataset, y)):

                # Split dataset into train and validation subsets
                train_subset = torch.utils.data.Subset(dataset, train_idx)
                val_subset = torch.utils.data.Subset(dataset, val_idx)

                # Create DataLoaders for each subset
                train_loaders.append(DataLoader(train_subset, batch_size=self.batch_size, shuffle=True))
                val_loaders.append(DataLoader(val_subset, batch_size=self.batch_size, shuffle=False))
        else:
            
            for i, (train_idx, val_idx) in enumerate(self.rkf.split(dataset)):

                # Split dataset into train and validation subsets
                train_subset = torch.utils.data.Subset(dataset, train_idx)
                val_subset = torch.utils.data.Subset(dataset, val_idx)

                # Create DataLoaders for each subset
                train_loaders.append(DataLoader(train_subset, batch_size=self.batch_size, shuffle=True))
                val_loaders.append(DataLoader(val_subset, batch_size=self.batch_size, shuffle=False))


        return train_loaders, val_loaders
            