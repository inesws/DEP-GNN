
import argparse
import random
import os

import numpy as np
import torch
import sys
from pathlib import Path
from torch_geometric.loader import DataLoader


# Add main directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from Trainer import Trainer

# Dataset
sys.path.insert(1, os.path.join('/home/isampaio/Desktop/Ines/DEPGNN/DEP-GNN/', 'utils'))

import utils
from utils.customDataset import BrainGraphDataset
from utils.kFoldDataLoader import KFold_DataLoader
from utils.EarlyStopping2 import EarlyStopping2
from utils.model_utils import read_yaml
from utils.dataset_utils import adjust_labels

# Model
sys.path.insert(1, os.path.join('/home/isampaio/Desktop/Ines/DEPGNN/DEP-GNN/', 'models'))
import models


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():

    """############ GNN Model Training ############"""
    parser = argparse.ArgumentParser(description='DEP-GNN Training')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 3407)')
    parser.add_argument('--cuda', type=int, default=0, help='cuda number (default: 0)')
    parser.add_argument('--loss', type=str, default='CrossEntropy', help='CrossEntropyLoss, BCEWithLogitsLoss')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs (default: 50)')
    parser.add_argument('--model_name', type=str, default='GCN', help='model name from model/<GNN_name>.py file')
    parser.add_argument('--model_config', type=str, default='/home/isampaio/Desktop/Ines/DEPGNN/DEP-GNN/config_GCN', help='model config files')
    parser.add_argument('--patience', type=int, default=10, help='num of epochs patience for early_stopping (default: 10)')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for training (default: 128)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=None, help='weight decay (default: 5e-2)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer AdamW,(Adam)')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout, 0=None')

    # regression: use regression head for regression tasks;
    # bin_regression: uses a classification in bines + regression for regression tasks;

    """############ DEP Sampler  ############"""
    parser.add_argument('--alpha', type=float, default=1e-5, help='alpha sparsity hyperparameter (default: 1e-5)')
    parser.add_argument('--beta', type=float, default=0.001, help='beta sparsity hyperparameter (default: 1e-5)')
    parser.add_argument('--curr_sp', type=float, default=0.05, help='current sparsity level (default: 0.05)')
    parser.add_argument('--iter_step', type=int, default=9, help='num epoch to increase sp level (default: 10)')
    parser.add_argument('--prune_sp', type=float, default=0.05, help='incremental prunning sparsity (default: 0.05)')
    parser.add_argument('--dropout', type=float, default=None, help='dropout')

    """############ Graph Dataset  ############"""
    parser.add_argument('--dataset_name', type=str,
                        default='NodeID_EdgeW_PearC_Sp_fully_connected_raw_Handedness_class',
                        help='name of the dataset folder to use?')
    
    parser.add_argument('--data_dir', type=str,
                        default='/home/isampaio/Desktop/Ines/DEPGNN/Data/',
                        help='datasets_dir') # full_ccd_r1_to_11_nor5_challenge_1.pkl , train_ccd_r1_to_r11_nor5_challenge_2.pkl
    parser.add_argument('--split', type=float, # TH
                        nargs= '+',
                        default= '0.8 0.1 0.1',
                        help='Train/val/test proportions, e.g.,--split 0.9 0.1 0 --split 0.8 0.1 0.1')
    parser.add_argument('--num_repeats', type=int, default=1, help='num of repeats for k-fold (default: 1)')
    parser.add_argument('--k_folds',type=int, default= 4, help='num of k folds (default: 4)')
    parser.add_argument('--num_workers', type=int, default=8, help='num_workers in dataloader') # to be used in data_loaders

    """############ CRITICAL SETTINGS: Downstream dataset settings ############"""

    ## freeze cbramod backbone:
    parser.add_argument('--freeze', type=bool,
                        default=False, help='freeze mode for DEP') 
    parser.add_argument('--DEP_weights', type=str,
                        default='/home/isampaio/Desktop/Ines/DEPGNN/DEP_trainedweights/name.pth', help='DEP_weights') 
    

    ## choose regression head:
    parser.add_argument('--regression_head', type=str, default='ClassificationRegressionHead',
                        help='[EEGRegressor, ClassificationRegressionHead, ContrastiveRegressionHead, ContrastiveWithMemoryRegressionHead]')
    ## the regression head lr is different from cbramod lr by lr*multi_lr_scaler:
    parser.add_argument('--multi_lr_scaler', type=float, default=5,
                        help='scales the regression head lr based on --lr * scale')  # set different learning rates for different modules

    ########## Parse parameters ####

    params = parser.parse_args()
    print(params)
    setup_seed(params.seed)

    #### Dataset #####
    #  Split Data: 0.8 kfold( Train, Val) | 0.2 Test


    hdf5_path =os.path.join(params.data_dir, f"{params.dataset_name}.h5") 
    dataset_dir = os.path.join(params.data_dir, f"{params.dataset_name}_Dataset")
    dataset = BrainGraphDataset(root=Path(dataset_dir), hdf5_path=hdf5_path,pre_transform=adjust_labels )

    train_size = int(len(dataset)*params.split[0])
    dataset = dataset.shuffle()
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]

    sfk = KFold_DataLoader(params, stratify= True)
    train_loaders, val_loaders = sfk.get_nk_loaders(train_dataset)

    test_loader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False)

    data_loaders = { 'train': train_loaders,
                     'val': val_loaders,
                     'test': test_loader
                    }

    #### DEP Sampler#####
    node_feat_dim = dataset.num_node_features
    output_dim = len(dataset[0].y.shape) +2 

    sampler = models.DEP( params, node_feat_dim, output_dim) # the input_dim is always the same as the number of ROIs or number of row/column of original adjancecy

    #### GNN Model #####
    import importlib
    module_name = params.model_name
    module = importlib.import_module(f"model_folder.{module_name}")
    ModelClass = getattr(module, module_name)
    model_config = read_yaml(params.model_config)[0]
    model = ModelClass(config=model_config, input_dim=node_feat_dim, output_dim=output_dim)
    
    #### Train!!! #####

    trainer = Trainer(params, data_loaders, model, sampler)
    trainer.train_for_classification()
    #trainer.train_for_regression()


    print('Done!!!!!')

if __name__ == '__main__':
    main()

