
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

from TrainerGNN import TrainerGNN

# Dataset
sys.path.insert(1, os.path.join('/home/isampaio/Desktop/Ines/DEPGNN/DEP-GNN/', 'utils'))

import utils
from utils.customDataset import BrainGraphDataset
from utils.kFoldDataLoader import KFold_DataLoader
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


def train_one_fold(params, data_loaders, model):
    """Train a single fold or single run"""
    trainer = TrainerGNN(params, data_loaders, model)
    results = trainer.train_for_classification()
    return results


def main():

    """############ GNN Model Training ############"""
    parser = argparse.ArgumentParser(description='DEP-GNN Training')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 3407)')
    parser.add_argument('--cuda', type=int, default=0, help='cuda number (default: 0)')
    parser.add_argument('--loss', type=str, default='CrossEntropyLoss', help='CrossEntropyLoss, BCEWithLogitsLoss')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs (default: 50)')
    parser.add_argument('--model_name', type=str, default='GCN', help='model name from model/<GNN_name>.py file')
    parser.add_argument('--model_config', type=str, default='/home/isampaio/Desktop/Ines/DEPGNN/DEP-GNN/config_GCN', help='model config files')
    parser.add_argument('--patience', type=int, default=10, help='num of epochs patience for early_stopping (default: 10)')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for training (default: 128)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=None, help='weight decay (default: 5e-2)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer AdamW,(Adam)')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout, 0=None')
    parser.add_argument('--model_dir', type=str, 
                        default='/home/isampaio/Desktop/Ines/DEPGNN/noSampler_results/',
                        help='directory to save trained models')

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



    ########## Parse parameters ####

    params = parser.parse_args()
    print(params)
    setup_seed(params.seed)

    #### Dataset #####
    #  Split Data: 0.8 kfold( Train, Val) | 0.2 Test

    hdf5_path =os.path.join(params.data_dir, f"{params.dataset_name}.h5") 
    dataset_dir = os.path.join(params.data_dir, f"{params.dataset_name}_Dataset")
    #dataset = BrainGraphDataset(root=Path(dataset_dir), hdf5_path=hdf5_path,pre_transform=adjust_labels )
    dataset = BrainGraphDataset(root=Path(dataset_dir), hdf5_path=hdf5_path)

    train_size = int(len(dataset)*params.split[0])

    dataset = dataset.shuffle()
    train_dataset = dataset[:train_size] # if train is 0.8 then test is 0.2 
    test_dataset = dataset[train_size:]

    # Create k-fold loaders or single split
    if params.num_repeats == 1 and params.k_folds == 1:
        # Single train/val split (no k-fold)
        val_size = int(len(train_dataset) * params.split[1])
        val_dataset = train_dataset[val_size:]
        train_dataset = train_dataset[:val_size]
        
        train_loaders = [DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)]
        val_loaders = [DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False)]
    else:
        # K-fold cross-validation
        sfk = KFold_DataLoader(params, stratify=True)
        train_loaders, val_loaders = sfk.get_nk_loaders(train_dataset)

    test_loader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False)

    #### Model Setup #####
    node_feat_dim = dataset.num_node_features
    output_dim = len(dataset[0].y.shape) + 2
    
    # Import model class
    import importlib
    module_name = params.model_name
    module = importlib.import_module(f"models.{module_name}")
    ModelClass = getattr(module, module_name)
    model_config = read_yaml(params.model_config)[0]

    #### Train!!! #####
    original_model_dir = params.model_dir
    
    if params.num_repeats == 1 and params.k_folds == 1:
        # Single train/val split - no k-fold
        print("=" * 60)
        print("Training with single train/val split (no k-fold)")
        print("=" * 60)
        
        data_loaders = {
            'train': train_loaders[0],
            'val': val_loaders[0],
            'test': test_loader
        }
        
        # Initialize model and sampler
        model = ModelClass(config=model_config, input_dim=node_feat_dim, output_dim=output_dim)
        
        # Train
        results = train_one_fold(params, data_loaders, model)
        
    else:
        # K-fold cross-validation
        # Note: num_repeats * k_folds = total number of training runs
        # e.g., num_repeats=2, k_folds=4 → 8 total folds
        print("=" * 60)
        print(f"Training with {params.k_folds}-fold CV, {params.num_repeats} repeat(s)")
        print(f"Total training runs: {len(train_loaders)} (num_repeats × k_folds)")
        print("=" * 60)
        
        all_results = []
        
        for fold_idx, (train_loader, val_loader) in enumerate(zip(train_loaders, val_loaders)):
            # Determine repeat and fold numbers for clarity
            repeat_num = (fold_idx // params.k_folds) + 1
            fold_num = (fold_idx % params.k_folds) + 1
            
            print(f"\n{'=' * 60}")
            print(f"Repeat {repeat_num}/{params.num_repeats}, Fold {fold_num}/{params.k_folds} (Run {fold_idx + 1}/{len(train_loaders)})")
            print("=" * 60)
            
            # Create data loaders for this fold
            data_loaders = {
                'train': train_loader,
                'val': val_loader,
                'test': test_loader
            }
            
            # Reinitialize model and sampler for each fold (fresh start)
            fold_model = ModelClass(config=model_config, input_dim=node_feat_dim, output_dim=output_dim)

            # Update model_dir to save fold-specific models
            # Include repeat number if multiple repeats
            if params.num_repeats > 1:
                folder_name = f"repeat{repeat_num}_fold{fold_num}"
            else:
                folder_name = f"fold_{fold_num}"
            params.model_dir = os.path.join(original_model_dir, folder_name)
            
            # Train this fold
            results = train_one_fold(params, data_loaders, fold_model)
            all_results.append(results)
            
            # Restore original model_dir
            params.model_dir = original_model_dir
        
        # Print summary of all folds
        print("\n" + "=" * 60)
        print("K-Fold Cross-Validation Summary")
        print("=" * 60)
        print(f"Completed {len(all_results)} folds")

    print('\n' + "=" * 60)
    print('Training Complete!')
    print("=" * 60)

if __name__ == '__main__':
    main()