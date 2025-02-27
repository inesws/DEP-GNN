# Install libraries
import os
import torch

import torch_geometric
import torch_sparse

import h5py
from torch_geometric.data import Data,  Dataset

class BrainGraphDataset(Dataset):
    def __init__(self, root, hdf5_path=None, transform=None, pre_transform=None):
        self.hdf5_path = hdf5_path  # Path to the HDF5 file
        self.graph_keys = []  # Placeholder for graph keys
        super().__init__(root, transform, pre_transform)

        # Load the graph keys during initialization
        with h5py.File(self.hdf5_path, 'r') as h5file:
            self.graph_keys = list(h5file.keys())

    def len(self):
        # Return the number of graphs in the dataset
        return len(self.graph_keys)

    def get(self, idx):
        # Load a single graph from the HDF5 file
        with h5py.File(self.hdf5_path, 'r') as h5file:
            group = h5file[self.graph_keys[idx]]

            # Extract data for the graph
            edge_index = torch.tensor(group['edge_index'][:], dtype=torch.long)
            node_features = torch.tensor(group['node_features'][:], dtype=torch.float)
            edge_att = torch.tensor(group['edge_att'][:], dtype=torch.float)
            label = torch.tensor(group['labels'][()], dtype=torch.long)

            # Create a Data object
            data = Data(x=node_features, edge_index=edge_index, y=label, edge_attr=edge_att)

        # Apply transformations if any
        if self.transform:
            data = self.transform(data)

        return data

    @property
    def raw_file_names(self):
        # Specify the raw HDF5 file
        return [os.path.basename(self.hdf5_path)]

    @property
    def processed_file_names(self):
        # This is unused in this implementation but required by the Dataset class
        return ['data.pt']

    def process(self):
        # No preprocessing needed as we load directly from the HDF5 file
        pass
