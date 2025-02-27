# Install libraries

import pandas as pd

import sys
sys.path.insert(1, '/path/utils')
from dataset_utils import save_dataset_pkl

sys.path.insert(1, '/path/utisl')
from customDataset import BrainGraphDataset
#from customInMemoryDataset import BrainGraphDataset

from pathlib import Path

# The dataframes loaded bellow should contain a column with the LABEL to add to the graph dataset
cov_matrix = pd.read_csv(Path(r"C:\path\covariates.csv")) 


## CHOOSE OPTIONS 
"""
 topk : the percentage % to threshold the fully-connected brain adjancancy matrix: keep top k% connections
        to keep fully connected write '_fully_connected' and set 'sp'= None
 sp: float(topk) or None for fully connected

 edge_type: adding a edge weight, a edge feature set or binary=No edge?

 edge_feat: which type of edge feature: 
 'PearC': pearson correlation
 'MI': mutual information
 ect

 node_feat: name of nome feature to use
 'PearC', 'ROIvol', 'NodeID' etc

 node_id: for using NodeID, put node_id= True: 
 you still load the original adjacency matrix but it will create a node id matrix from it

 transform: raw data or applying any transformation to data?

 label: name of task/label
"""

topk = '56.25' #'_fully_connected', '50', '1'
sp = float(topk) # OR None, int(topk)  (for fully connected option)

edge_type = 'EdgeW' # EdgeB , EdgeAttr, 'EdgeW'

edge_feat = 'PearC' # MI, PearC, ALL 

node_feat =  'PearC'# 'PearC', MI, ROIvol, Vcount, AllStruct, AllFC, All , NodeID
node_id = False

transform = 'raw' # fisherz

label = 'Sex' # Sex, 'fluidint_Ageadj', 'handedness' 'fluidint_Unadj', 'Age', 'Education', 'Flanker'


# Dataset Name encoding <Node Feature Type>_<Edge Feature>_<Sparsity Level>_<Raw or Transformed values?>


dataset_name = f"{node_feat}_{edge_type}_{edge_feat}_Sp{topk}_{transform}_{label}" # connectivity profile ; edge_weight ; 10% sparsity level; Raw pearson corr values; label= target name in data.y

folder_path = "C:/folder_save/hdf5_files/"

hdf5_path = folder_path + dataset_name + '.h5'

##

config = {'node_feat':['CM360Glasser'] , # ['CM360Glasser']  name of the folder where connectivity matrix type to use for nodes are saved
          'edge_feat': ['CM360Glasser'],  # name of the folder where connectivity matrix type to use for edges are saved
          'label': 'Male',  # 'Flanker_AgeAdj_class', 'Age_in_Yrs', 'Male', 'CogFluidComp_Unadj', CogFluidComp_AgeAdj, 'Handedness' : column name of the label in the covariates matrix
          'decrease_step': None, # None : will allow for a graph to be disconnected but will keep sparsity level equal to all
          'absolute': None, # None or 
          'node_pos': None, # None or 
          'node_id' : node_id, # Include only node ID and not actual node features
          'top_k': sp, # int(topk) or None
          'transform': transform 
}


# Select parent folder for brain connectivity data ( where adjancency matrices are) 

FC_folder = Path(r"C:\data\folder")

## Create list of graph Data in pkl

save_dataset_pkl(cov_matrix, hdf5_path, FC_folder, config) 

## Create Custom Dataset PyG
dataset_dir = Path(rf"C:\path\Custom_Datasets_PyG\{dataset_name}_Dataset")  # 
dataset = BrainGraphDataset(root=dataset_dir, hdf5_path=hdf5_path)

print(dataset[0])

