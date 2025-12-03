# Install libraries
import os
import torch

import torch_geometric

import torch_geometric.datasets as Datasets
import torch_geometric.data as Data
import torch_geometric.transforms as transforms
from torch_geometric.utils import remove_self_loops

import torch_sparse
from torch_sparse import coalesce

import networkx as nx
from networkx.convert_matrix import from_numpy_array


import numpy as np
import scipy as sp, scipy.io
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

import h5py
from pathlib import Path
import glob

def find_output_dim(dataset):
  """
  Docstring for find_output_dim. For binary classes, catergorical 
  classes, and one-hot encoded labels.
  
  :dataset: PyG dataset object
  :returns: output_dim (int) - number of classes for classification
  """
  sample_y = dataset[0].y
  if sample_y.dim() > 0 and sample_y.shape[0] > 1 and len(sample_y.shape) == 1:
      # One-hot encoded
      output_dim = sample_y.shape[0]
  else:
      # Categorical - find max class
      all_labels = torch.cat([data.y.view(-1) for data in dataset])
      output_dim = len(torch.unique(all_labels))

  return output_dim
   

def adjust_labels(data):
    """Pre-transform function to adjust labels"""
    # Handle different label scenarios
    if (data.y.item() == 1)  :
        data.y = torch.tensor(0)
    elif data.y.item() == 2:
        data.y = torch.tensor(1)
    # Add more cases if needed:
    # elif data.y.item() == 3:
    #     data.y = torch.tensor(2)
    
    return data

# Fisher's r-to-z transformation

def fishers_r_z_transform(cm):
    """
    cm : np.array with dimension N_ROI X N_ROI
    ____________________________________________
    outputs:

    fisher r-to-z transform cm

    """

    cm_transform = np.arctanh(cm - np.eye(cm.shape[0]))
    
    return cm_transform


# Threshold a connectivity matrix (adjacency)

def threshold_matrix(cm, k_percent):


    num_elements = np.triu_indices_from(cm, k=1)[0].shape[0]
    k = max(1, int(num_elements * k_percent / 100))  # Ensure at least 1 element is kept


    unique_values = torch.from_numpy(np.unique(cm.flatten()))
    top_values, top_indices = torch.topk(np.abs(unique_values), k)

    condition_met = (np.abs(cm[..., np.newaxis]) >= top_values.numpy()).any(axis=-1)

    thresholded_matrix = np.where(condition_met, cm, 0)

    return thresholded_matrix


# Create a sparse graph ( adjacency )
from scipy.sparse import csr_matrix

def create_sparse_graph( cm, sparsity_level, decrease_step=None):

  """
  cm: 

  sparsity_level: top N% connections

  decrease_step : Defined to assure the final graph is not disconnected due to sparsity_level
  ( default= None, otherwise define a N% step to decrease sparsity level)
  """

  thresholded_matrix = threshold_matrix(cm, sparsity_level)

  if decrease_step is not None:

     sparse_matrix = csr_matrix(thresholded_matrix)
     G = nx.from_scipy_sparse_array(sparse_matrix)

     # Check if the graph is connected
     while not nx.is_connected(G):  # While the graph is not connected
        

        print("Graph is disconnected, decreasing the sparsity level by {} %.".format(decrease_step))
    
        thresholded_matrix= threshold_matrix(cm, sparsity_level + decrease_step)

        # Update the graph with the new thresholded matrix
        sparse_matrix = csr_matrix(thresholded_matrix)
        G = nx.from_scipy_sparse_array(sparse_matrix)



  return thresholded_matrix

# Return graph properties needed to create a Data object in PyG: edge_index, edge_att, node_attr,labels, pos

def extract_graph_properties(cm, node_att, label, sparsity_level, decrease_step=None, absolute=False, node_id= False, node_pos=False):

  """
  cm : the connectivity matrix that will define the edges and edges weights

  node_att : the matrix that will define the node attributes
  e.g will be the same as cm when the node features are the connectivity profiles

  label : the graph/node labels

  sparsity_level: top K % values - the level to threshold the cm
  keep top 10 % positives

  absolute: False = keep negative connectivity values

  returns

  data.Data(x=node_att_processed, edge_index=edge_index.long(), y=y_torch, edge_attr=edge_att)
  graph

  """

  num_nodes = cm.shape[0]

  # Absolute CM
  if absolute==True:
    cm = np.abs(cm)

  # Threshold CM
  if sparsity_level is not None:

    cm = create_sparse_graph( cm, sparsity_level, decrease_step)

  G = nx.from_numpy_array(cm)
  A = nx.to_scipy_sparse_array(G) # is == cm
  adj = A.tocoo()

  edge_att = np.zeros(len(adj.row))
  for i in range(len(adj.row)):
      edge_att[i] = cm[adj.row[i], adj.col[i]]

  edge_index = np.stack([adj.row, adj.col])
  edge_index, edge_att = remove_self_loops(torch.from_numpy(edge_index), torch.from_numpy(edge_att))
  edge_index = edge_index.long()


  edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes, num_nodes) # edge index are duplicated 
  # and edge_att are coherent with it: they are all the non-zero and without self-connections 

  if node_id == True:
     att_torch = torch.eye(node_att.shape[1])
     
  else:
     att_torch = torch.from_numpy(node_att).float()

  y_torch = torch.from_numpy(np.array(label)).long()  # classification

  # Include node position
  if node_pos==True:
    pos= np.where(cm==1, node_att, 0)
  else:
    pos=0

  return edge_index, edge_att, att_torch, y_torch, pos



####################################
# Convert mat graph files to a graph property hdf5 file

def save_dataset_pkl(cov_matrix, hdf5_path, FC_folder, config):
    # Open HDF5 file for writing
    with h5py.File(hdf5_path, 'w') as h5file:
        count = 0
 
        node_feat_path = [ os.path.join(FC_folder, path) for path in  config['node_feat']]
        edge_feat_path = [ os.path.join(FC_folder, path) for path in  config['edge_feat']]

        subj_ids_list= [ int(l[0:6]) for l in os.listdir(node_feat_path[0])]
        print('Will load {} node features'.format(len(node_feat_path)))
        print('Will load {} edge features'.format(len(edge_feat_path)))


        # Iterate over subject IDs
        for i in np.sort(np.array(subj_ids_list)):

            print('Loading subject {}'.format(i))

            isnan = np.isnan(cov_matrix.loc[count, config['label']])

            if isnan:
               print('Subject {} has NaN value for label. Passing to next subject'.format(i))
               count += 1  # Increment your count
               continue

            # Concatenate all node features
            node_feat = []
            for p in node_feat_path:
               
               file = sp.io.loadmat(glob.glob(os.path.join(p, f"{i}_*.mat"))[0])
               k = list(file.keys())[-1:]
               node_feat.append(file[k[0]])
            
            #print('Loaded {} node features'.format(len(node_feat)))

            node_feat = np.hstack(node_feat)  

            # Concanate all edge_features
            edge_feat = []
            for p in edge_feat_path:
               
               file = sp.io.loadmat(glob.glob(os.path.join(p, f"{i}_*.mat"))[0]) 
               k = list(file.keys())[-1:]
               edge_feat.append(file[k[0]])
            #print('Loaded {} edge features'.format(len(edge_feat)))

            edge_feat = np.hstack(edge_feat)  

            # Extract the label from the covariance matrix
            label = cov_matrix.loc[count, config['label']]
               
              
            # Extract graph properties
            edge_index, edge_att, att_torch, y_torch, pos = extract_graph_properties(
                edge_feat, node_feat, label,
                sparsity_level=config['top_k'],decrease_step=config['decrease_step'], absolute=config['absolute'], 
                node_id=config['node_id'], node_pos=config['node_pos']
            )

          
            # Create a group in the HDF5 file for each graph
            group = h5file.create_group(f'graph_{i}')

            # Save edge_index and node features into HDF5
            group.create_dataset('edge_index', data=edge_index)  # Ensure edge_index is in a compatible shape
            group.create_dataset('edge_att', data=edge_att)
            group.create_dataset('node_features', data=att_torch)  # Ensure this is also in a compatible shape
            group.create_dataset('labels', data=y_torch)  # Labels stored as an array

            # if (pos==0) TO IMPLEMENT

            count += 1  # Increment your count

    print("All graph properties saved to HDF5 file.")