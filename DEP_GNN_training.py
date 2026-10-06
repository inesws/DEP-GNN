
import argparse
import json
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

from TrainerDEPGNN import Trainer

# Dataset
sys.path.insert(1, os.path.join('/home/isampaio/Desktop/Ines/DEPGNN/DEP-GNN/', 'utils'))

import utils
from utils.customDataset import BrainGraphDataset
from utils.kFoldDataLoader import KFold_DataLoader
from utils.model_utils import read_yaml
from utils.dataset_utils import adjust_labels, find_output_dim

# Model
sys.path.insert(1, os.path.join('/home/isampaio/Desktop/Ines/DEPGNN/DEP-GNN/', 'models'))
import models
from models.DEP import DEP

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_pretrained_sampler(sampler, params):
    """Helper function to load pretrained sampler weights"""
    if params.use_pretrained_sampler and os.path.exists(params.sampler_pretrain_dir):
        map_location = torch.device(f'cuda:{params.cuda}')
        checkpoint = torch.load(params.sampler_pretrain_dir, map_location=map_location)
        
        # Handle both old format (state_dict only) and new format (checkpoint dict)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            # New format: checkpoint dictionary with state_dict and min_sp
            sampler.load_state_dict(checkpoint['state_dict'])
            sampler.min_sp = checkpoint['min_sp']
            print(f"  - Loaded pretrained DEP sampler with min_sp={sampler.min_sp:.4f}")
            if 'epoch' in checkpoint:
                print(f"  - From epoch: {checkpoint['epoch']}")
            if 'f1' in checkpoint:
                print(f"  - F1 score: {checkpoint['f1']:.4f}")
        else:
            # Old format: state_dict only
            sampler.load_state_dict(checkpoint)
            print(f"  - Loaded pretrained DEP sampler (old format, default min_sp={sampler.min_sp:.4f})")
    return sampler


def train_one_fold(params, data_loaders, model, sampler):
    """Train a single fold or single run"""
    trainer = Trainer(params, data_loaders, model, sampler)
    results = trainer.train_for_classification()
    return results


def main():

    """############ GNN Model Training ############"""
    parser = argparse.ArgumentParser(description='DEP-GNN Training')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 3407)')
    parser.add_argument('--cuda', type=int, default=0, help='cuda number (default: 0)')
    parser.add_argument('--loss', type=str, default='CrossEntropyLoss', help='CrossEntropyLoss, BCEWithLogitsLoss')
    parser.add_argument('--y_dim', type=int, default=2, help='Num of classes,either categorical or one-hot: 2 for binary or 2+ for multiclass with CrossEntropyLoss, 1 for binary with BCEWithLogitsLoss')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs (default: 50)')
    parser.add_argument('--model_name', type=str, default='GCN', help='model name from model/<GNN_name>.py file')
    parser.add_argument('--model_config', type=str, default='/home/isampaio/Desktop/Ines/DEPGNN/DEP-GNN/model_configs/gcn.yaml', help='model config files')
    parser.add_argument('--patience', type=int, default=50, help='num of epochs patience for early_stopping (default: 10)')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for training (default: 128)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay (default: 5e-2)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer AdamW,(Adam)')

    """############ DEP Sampler  ############"""
    parser.add_argument('--alpha', type=float, default=0.00001, help='alpha sparsity hyperparameter (default: 1e-5)')
    parser.add_argument('--beta', type=float, default=0.001, help='beta sparsity hyperparameter (default: 1e-5)')
    parser.add_argument('--DEP_lr', type=float, default=None, help='use different lr for DEP training (default: None)')
    parser.add_argument('--curr_sp', type=float, default=0.0, help='current sparsity level (default: 0.05)')
    parser.add_argument('--iter_step', type=int, default=5, help='num epoch to increase sp level (default: 10)')
    parser.add_argument('--prune_sp', type=float, default=0.05, help='incremental prunning sparsity (default: 0.05)')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')

    """############ Graph Dataset  ############"""
    parser.add_argument('--dataset_name', type=str,
                        default='NodeID_EdgeW_PearC_Sp_fully_connected_raw_Sex',
                        help='name of the dataset folder to use?')
    parser.add_argument('--data_dir', type=str,
                        default='/mnt/datafast/ines/DEPGNN/Data/',
                        help='datasets_dir') 
    parser.add_argument('--split', type=float, 
                        nargs= '+',
                        default=[0.8, 0.1, 0.1],
                        help='Train/val/test proportions, e.g.,--split 0.9 0.1 0 --split 0.8 0.1 0.1')
    parser.add_argument('--num_repeats', type=int, default=2, help='num of repeats for k-fold (default: 1)')
    parser.add_argument('--k_folds',type=int, default= 5, help='num of k folds (default: 4)')
    parser.add_argument('--num_workers', type=int, default=8, help='num_workers in dataloader') # to be implemented 

    """############ CRITICAL DEP SETTINGS############"""

    ## freeze backbone:
    parser.add_argument('--freeze', type=bool,
                        default=False, help='freeze mode for DEP') 
    parser.add_argument('--use_pretrained_sampler', type=bool, default=False, 
                    help='whether to load pretrained DEP sampler weights')
    parser.add_argument('--sampler_pretrain_dir', type=str,
                    default='/home/isampaio/Desktop/Ines/DEPGNN/DEP_weights/sampler.pth', 
                    help='path to pretrained DEP sampler checkpoint (includes weights and prune percentage)')
    parser.add_argument('--model_dir', type=str, 
                    default='/home/isampaio/Desktop/Ines/DEPGNN/results/myoldGCN_combo31_v5/',
                    help='directory to save trained models')


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

    output_dim = params.y_dim #find_output_dim(dataset)
    
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
        
        # Create directory if it doesn't exist
        if not os.path.isdir(params.model_dir):
            os.makedirs(params.model_dir)
        
        # Initialize model and sampler
        model = ModelClass(config=model_config, input_dim=node_feat_dim, output_dim=output_dim)
        sampler = DEP(params, node_feat_dim, output_dim)
        
        # Load pretrained weights if specified
        if params.use_pretrained_sampler:
            sampler = load_pretrained_sampler(sampler, params)
        
        # Train
        results = train_one_fold(params, data_loaders, model, sampler)
        
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
            fold_sampler =DEP(params, node_feat_dim, output_dim)
            
            # Load pretrained weights if specified
            if params.use_pretrained_sampler:
                fold_sampler = load_pretrained_sampler(fold_sampler, params)
            
            # Update model_dir to save fold-specific models
            # Include repeat number if multiple repeats
            if params.num_repeats > 1:
                folder_name = f"repeat{repeat_num}_fold{fold_num}"
            else:
                folder_name = f"fold_{fold_num}"
            params.model_dir = os.path.join(original_model_dir, folder_name)
            
            # Create directory if it doesn't exist
            if not os.path.isdir(params.model_dir):
                os.makedirs(params.model_dir)
            
            # Train this fold
            results, sampler, model = train_one_fold(params, data_loaders, fold_model, fold_sampler)
            all_results.append(results)
            
            # Restore original model_dir
            params.model_dir = original_model_dir
        
        # Print summary of all folds
        print("\n" + "=" * 60)
        print("K-Fold Cross-Validation Summary")
        print("=" * 60)
        print(f"Completed {len(all_results)} folds")
        
        # Calculate mean and std across all folds
        if all_results:
            import pandas as pd
            
            # Convert results to DataFrame for easy aggregation
            results_df = pd.DataFrame(all_results)
            
            # Calculate statistics
            mean_results = results_df.mean()
            std_results = results_df.std()
            
            print("\n" + "=" * 60)
            print("Aggregated K-Fold Results (Mean ± Std)")
            print("=" * 60)
            print(f"Best Val Accuracy:  {mean_results['best_val_acc']:.4f} ± {std_results['best_val_acc']:.4f}")
            print(f"Best Val F1:        {mean_results['best_val_f1']:.4f} ± {std_results['best_val_f1']:.4f}")
            print(f"Best Val Precision: {mean_results['best_val_precision']:.4f} ± {std_results['best_val_precision']:.4f}")
            print(f"Best Val Recall:    {mean_results['best_val_recall']:.4f} ± {std_results['best_val_recall']:.4f}")
            print(f"Best Val AUC:       {mean_results['best_val_auc']:.4f} ± {std_results['best_val_auc']:.4f}")
            print(f"Best Val Loss:      {mean_results['best_val_loss']:.4f} ± {std_results['best_val_loss']:.4f}")
            print(f"\nTest Accuracy:      {mean_results['test_acc']:.4f} ± {std_results['test_acc']:.4f}")
            print(f"Test F1:            {mean_results['test_f1']:.4f} ± {std_results['test_f1']:.4f}")
            print(f"Test Precision:     {mean_results['test_precision']:.4f} ± {std_results['test_precision']:.4f}")
            print(f"Test Recall:        {mean_results['test_recall']:.4f} ± {std_results['test_recall']:.4f}")
            print(f"Test AUC:           {mean_results['test_auc']:.4f} ± {std_results['test_auc']:.4f}")
            print(f"Test Loss:          {mean_results['test_loss']:.4f} ± {std_results['test_loss']:.4f}")
            if 'final_sparsity' in results_df.columns:
                print(f"Final Sparsity:     {mean_results['final_sparsity']:.4f} ± {std_results['final_sparsity']:.4f}")
            
            # Create directory if it doesn't exist
            if not os.path.isdir(original_model_dir):
                os.makedirs(original_model_dir)
            
            # Save results to CSV
            results_save_path = os.path.join(original_model_dir, "kfold_results.csv")
            results_df.to_csv(results_save_path, index=False)
            print(f"\nDetailed results saved to: {results_save_path}")
            
            # Save summary statistics
            summary_save_path = os.path.join(original_model_dir, "kfold_summary.csv")
            summary_df = pd.DataFrame({
                'metric': results_df.columns,
                'mean': mean_results.values,
                'std': std_results.values
            })
            summary_df.to_csv(summary_save_path, index=False)
            print(f"Summary statistics saved to: {summary_save_path}")

    # ── Save hyperpar to JSON ─────────────────────────────────────────────
    results = {
        "dataset_name": params.dataset_name,
        "split": params.split,
        "k_folds": params.k_folds,
        "num_repeats": params.num_repeats,
        "epochs": params.epochs,
        "model_name": params.model_name,
        "batch_size": params.batch_size,
        "optimizer": params.optimizer,
        "weight_decay": params.weight_decay,
        "alpha": params.alpha,
        "beta": params.beta,
        "DEP_lr": params.DEP_lr,
        "curr_sp": params.curr_sp,
        "iter_step": params.iter_step,
        "prune_sp": params.prune_sp,
        "dropout": params.dropout,
        "lr": params.lr,
        "freeze": params.freeze,
        "seed": params.seed,
        "loss": params.loss,
        "y_dim": params.y_dim,
        "patience": params.patience,
        "cuda": params.cuda

    }

    results_path = os.path.join(
        original_model_dir,
        f"DEP_{params.dataset_name}_seed{params.seed}.json",
    )
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")        

    print('\n' + "=" * 60)
    print('Training Complete!')
    print("=" * 60)

if __name__ == '__main__':
    main()

