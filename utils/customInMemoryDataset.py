# Install libraries
import torch

import torch_geometric

from torch_geometric.data import Data, InMemoryDataset

import h5py


class BrainGraphDataset(InMemoryDataset):
    def __init__(self, root, hdf5_path=None, transform=None, pre_transform=None):
        self.hdf5_path = hdf5_path
        super().__init__(root, transform, pre_transform)


        # Load the processed data if it already exists, otherwise process it
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']  # Specify a name for the processed data file

    def process(self):
        data_list = []  # List to store all Data objects

        # Load data from the HDF5 file and create Data objects
        with h5py.File(self.hdf5_path, 'r') as h5file:
            for graph_id in h5file.keys():
                group = h5file[graph_id]

                # Extract data for each graph
                edge_index = torch.tensor(group['edge_index'][:], dtype=torch.long)
                node_features = torch.tensor(group['node_features'][:], dtype=torch.float)
                edge_att = torch.tensor(group['edge_att'][:], dtype=torch.float)
                label = torch.tensor(group['labels'][()], dtype=torch.long)

                # Create a Data object and add to data_list
                data = Data(x=node_features, edge_index=edge_index, y=label, edge_attr=edge_att)
                data_list.append(data)

        # If there are any transformations, apply them
        if self.pre_transform:
            data_list = [self.pre_transform(data) for data in data_list]

        # Collate the list of Data objects into a single InMemoryDataset-compatible format
        data, slices = self.collate(data_list)

        # Save the processed data
        torch.save((data, slices), self.processed_paths[0])

    def len(self):
        # len is automatically handled by InMemoryDataset
        return super().len()

    def get(self, idx):
        # get is automatically handled by InMemoryDataset
        return super().get(idx)
