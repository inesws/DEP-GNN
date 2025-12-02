# Install libraries
import os
import torch

import torch_geometric

import torch_geometric.datasets as Datasets
from torch_geometric.data import Data, InMemoryDataset
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


import sys
sys.path.insert(1, 'C:/Users/inesw/OneDrive/Área de Trabalho/Polimi/PhD/pytorch_learning')
from dataset_utils import extract_graph_properties, fishers_r_z_transform, save_dataset_pkl

sys.path.insert(1, 'C:/Users/inesw/OneDrive/Área de Trabalho/Polimi/PhD/pytorch_learning/my_classes')
from customDataset import BrainGraphDataset
#from customInMemoryDataset import BrainGraphDataset

from pathlib import Path
cov_matrix = pd.read_csv(Path(r"C:\Users\inesw\OneDrive\Área de Trabalho\Polimi\PhD\MIP_LAB\HCP_MIP_FC\covariates_new_noERROR.csv"))

fluid_intel_scores = pd.read_csv(Path(r"C:\Users\inesw\OneDrive\Área de Trabalho\Polimi\PhD\MIP_LAB\HCP_MIP_FC\cogn_composite_score_new_noERROR.csv"))

cogn_domain_scores = pd.read_csv(Path(r"C:\Users\inesw\OneDrive\Área de Trabalho\Polimi\PhD\MIP_LAB\HCP_MIP_FC\cognitive_domain_scores_noERROR.csv"))

cogn_domain_scores_unadj = pd.read_csv(Path(r"C:\Users\inesw\OneDrive\Área de Trabalho\Polimi\PhD\MIP_LAB\HCP_MIP_FC\all_cov_for_classes_corrected_noERROR.csv"))

#cogn_domain_scores_unadj = pd.read_csv(Path(r"C:\Users\inesw\OneDrive\Área de Trabalho\Polimi\PhD\MIP_LAB\HCP_MIP_FC\cognitive_domain_scores_unadj_plus_cognfluid_noERROR.csv"))

#nan_ind = np.where(np.isnan(fluid_intel_scores['CogFluidComp_Unadj'].values))[0] # 'CogFluidComp_AgeAdj'
#fluid_intel_scores = fluid_intel_scores.drop(nan_ind)

## CHOOSE OPTIONS 

topk = '_fully_connected' #'_fully_connected', '50', '1', '56.25'
sp = None # OR None(for fully connected option), int(topk), float(topk)

edge_type = 'EdgeW' # EdgeB , EdgeAttr, 'EdgeW'

edge_feat = 'PearC' # MI, PearC, ALL 

node_feat =  'NodeID'# 'PearC', MI, ROIvol, Vcount, AllStruct, AllFC, All , NodeID
node_id = True #False, True

transform = 'raw' # fisherz

label = 'PicVocab_Unadj_class' # Handedness_classCogFluidComp_Unadj_class' ReadEng_Unadj_class ReadEng_AgeAdj_class, Sex, 'fluidint_Ageadj', 'handedness' 'fluidint_Unadj', 'Age', 'Education', 'Flanker'

## ANY TRANSFORMATIONS? 
# In order to ignore certain subjects we need to put their label value to NaN

#cogn_domain_scores.loc[cogn_domain_scores['ReadEng_AgeAdj_class']==3, 'ReadEng_AgeAdj_class'] = np.nan
cogn_domain_scores_unadj.loc[cogn_domain_scores_unadj['PicVocab_Unadj_class']==3, 'PicVocab_Unadj_class'] = np.nan


## Dataset Name encoding <Node Feature Type>_<Edge Feature>_<Sparsity Level>_<Raw or Transformed values?>

#dataset_name = f"{node_feat}_{edge_type}_{edge_feat}_Sp{top_k}_{transform}" # connectivity profile ; edge_weight ; 10% sparsity level; Raw pearson corr values

dataset_name = f"{node_feat}_{edge_type}_{edge_feat}_Sp{topk}_{transform}_{label}" # connectivity profile ; edge_weight ; 10% sparsity level; Raw pearson corr values; label= target name in data.y

folder_path = "C:/Users/inesw/OneDrive/Área de Trabalho/Polimi/PhD/MIP_LAB/HCP_MIP_FC/Custom_Datasets_PyG/PyG_Data_hdf5/"

hdf5_path = folder_path + dataset_name + '.h5'

##

config = {'node_feat':['CM360Glasser'] , # ['CM360Glasser']  name of the folder where connectivity matrix type to use for nodes are saved
          'edge_feat': ['CM360Glasser'],  # name of the folder where connectivity matrix type to use for edges are saved
          'label': 'PicVocab_Unadj_class',  # ReadEng_AgeAdj_class'Flanker_AgeAdj_class', 'Age_in_Yrs', 'Male', 'CogFluidComp_Unadj', CogFluidComp_AgeAdj,'ReadEng_AgeAdj_class', 'Handedness' : column name of the label in the covariates matrix
          'decrease_step': None, # None : will allow for a graph to be disconnected but will keep sparsity level equal to all
          'absolute': None, # None or 
          'node_pos': None, # None or 
          'node_id' : node_id, # Id True will Include only node ID and not actual node features
          'top_k': sp, # int(topk) or None
          'transform': transform 
}


# Select parent folder for brain connectivity data ( where adjancency matrices are) 

FC_folder = Path(r"C:\Users\inesw\OneDrive\Área de Trabalho\Polimi\PhD\MIP_LAB\HCP_MIP_FC")

## Create list of graph Data in pkl

#save_dataset_pkl(cov_matrix, hdf5_path, FC_folder, config)
save_dataset_pkl(cogn_domain_scores_unadj, hdf5_path, FC_folder, config) #cov_matrix, fluid_intel_scores, cogn_domain_scores

## Create Custom Dataset PyG
dataset_dir = Path(rf"C:\Users\inesw\OneDrive\Área de Trabalho\Polimi\PhD\MIP_LAB\HCP_MIP_FC\Custom_Datasets_PyG\{dataset_name}_Dataset")  # 
dataset = BrainGraphDataset(root=dataset_dir, hdf5_path=hdf5_path)

print(dataset[0])

