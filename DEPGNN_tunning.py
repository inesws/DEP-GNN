
import argparse
import copy
import csv
import itertools
import random
import os
import shutil
import time

import numpy as np
import pandas as pd
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


def train_one_fold(params, data_loaders, model, sampler):
    """Train a single fold or single run"""
    trainer = Trainer(params, data_loaders, model, sampler)
    results = trainer.train_for_classification()
    return results


def build_grid(params_namespace):
    """
    Inspect every attribute of the parsed params.  If the default value is a
    list, treat it as a set of candidate values for grid-search.  Scalar
    defaults are kept fixed.  Returns:
        grid_keys  – list of arg names that vary
        grid_combos – list[dict] with every combination
        fixed_params – dict of arg names with fixed (scalar) values
    """
    grid_axes = {}
    fixed_params = {}

    for key, value in vars(params_namespace).items():
        if isinstance(value, list) and key != 'split':   # split is a real list arg
            grid_axes[key] = value
        else:
            fixed_params[key] = value

    grid_keys = sorted(grid_axes.keys())
    grid_values = [grid_axes[k] for k in grid_keys]
    grid_combos = [dict(zip(grid_keys, combo))
                   for combo in itertools.product(*grid_values)]

    return grid_keys, grid_combos, fixed_params


def extract_binary_mask(sampler, data_loader):
    """Extract the binary edge-selection mask from a trained DEP sampler.

    Runs one forward pass through the sampler (in eval mode) with a single
    batch so that ``prune()`` populates ``sampler.edge_mask`` with the
    correct mask for the restored best-epoch weights and current ``min_sp``.

    Returns:
        binary_mask: 1D BoolTensor (num_edges_per_graph,).
                     True = edge kept, False = edge pruned.
    """
    sampler.eval()
    with torch.no_grad():
        batch = next(iter(data_loader)).cuda()
        sampler(batch)  # populates sampler.edge_mask via prune()
    return sampler.get_edge_mask().cpu()


def compute_mean_jaccard(masks):
    """Compute the mean pairwise Jaccard index over a list of binary masks.

    Jaccard(A, B) = |A & B| / |A | B|  (over True entries).
    Returns mean Jaccard and std across all pairs.
    """
    n = len(masks)
    if n < 2:
        return float('nan'), float('nan')

    jaccards = []
    for i in range(n):
        for j in range(i + 1, n):
            intersection = (masks[i] & masks[j]).sum().float()
            union = (masks[i] | masks[j]).sum().float()
            if union == 0:
                jac = 1.0  # both empty → identical
            else:
                jac = (intersection / union).item()
            jaccards.append(jac)

    return float(np.mean(jaccards)), float(np.std(jaccards))


def save_combo_csv(path, combo_id, combo_dict):
    """Append one row to the hyperparameter-combinations CSV."""
    file_exists = os.path.isfile(path)
    fieldnames = ['combo_id'] + sorted(combo_dict.keys())
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        row = {'combo_id': combo_id}
        row.update(combo_dict)
        writer.writerow(row)


def append_results_csv(path, combo_id, mean_results, std_results, total_time_sec):
    """Append one row (mean +/- std from k-fold) to the results CSV.
    Contains only combo_id, metrics, and total training time.
    Hyperparameter values live in the separate combos CSV.
    """
    file_exists = os.path.isfile(path)

    metric_keys = [
        'final_sparsity_mean', 'final_sparsity_std',
        'best_val_loss_mean', 'best_val_loss_std',
        'best_val_acc_mean', 'best_val_acc_std',
        'best_val_f1_mean', 'best_val_f1_std',
        'best_val_precision_mean', 'best_val_precision_std',
        'best_val_recall_mean', 'best_val_recall_std',
        'best_val_auc_mean', 'best_val_auc_std',
        'test_loss_mean', 'test_loss_std',
        'test_acc_mean', 'test_acc_std',
        'test_f1_mean', 'test_f1_std',
        'test_precision_mean', 'test_precision_std',
        'test_recall_mean', 'test_recall_std',
        'test_auc_mean', 'test_auc_std',
        'jaccard_mean', 'jaccard_std',
    ]
    fieldnames = ['combo_id'] + metric_keys + ['total_time_sec']

    row = {'combo_id': combo_id, 'total_time_sec': round(total_time_sec, 1)}

    for m in mean_results.index:
        row[f'{m}_mean'] = mean_results[m]
        row[f'{m}_std'] = std_results[m]

    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():

    """############ GNN Model Training ############"""
    parser = argparse.ArgumentParser(description='DEP-GNN Hyperparameter Tuning (Grid Search)')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--cuda', type=int, default=0, help='cuda number (default: 0)')
    parser.add_argument('--loss', type=str, default='CrossEntropyLoss', help='CrossEntropyLoss, BCEWithLogitsLoss')
    parser.add_argument('--y_dim', type=int, default=2, help='Num of classes,either categorical or one-hot: 2 for binary or 2+ for multiclass with CrossEntropyLoss, 1 for binary with BCEWithLogitsLoss')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs (default: 50)')
    parser.add_argument('--model_name', type=str, default='GCN', help='model name from model/<GNN_name>.py file')
    parser.add_argument('--model_config', type=str, default='/home/isampaio/Desktop/Ines/DEPGNN/DEP-GNN/model_configs/gcn.yaml', help='model config file')
    parser.add_argument('--patience', type=int, default=50, help='num of epochs patience for early_stopping (default: 10)')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for training (default: 128)')
    parser.add_argument('--lr', type=float, default=[0.001, 0.00001, 0.0001], nargs='+', help='learning rate(s) for grid search')
    parser.add_argument('--weight_decay', type=float, default=None, help='weight decay (default: 5e-2)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer AdamW,(Adam)')

    """############ DEP Sampler  ############"""
    parser.add_argument('--alpha', type=float, default=[0.0001, 0.001, 0.01, 0.00001], nargs='+', help='alpha sparsity hyperparameter(s)')
    parser.add_argument('--beta', type=float, default=[0.0001, 0.001, 0.00001], nargs='+', help='beta sparsity hyperparameter(s)')
    parser.add_argument('--DEP_lr', type=float, default=None, help='use different lr for DEP training (default: None)')
    parser.add_argument('--curr_sp', type=float, default=0.0, help='current sparsity level (default: 0.05)')
    parser.add_argument('--iter_step', type=int, default=5,  help='num epoch to increase sp level') # nargs='+',
    parser.add_argument('--prune_sp', type=float, default=0.05, help='incremental prunning sparsity')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout (use 0.0 for None)')

    """############ Graph Dataset  ############"""
    parser.add_argument('--dataset_name', type=str,
                        default='PearC_EdgeW_PearC_Sp_fully_connected_raw_Sex', 
                        help='name of the dataset folder to use?')
    parser.add_argument('--data_dir', type=str,
                        default='/home/isampaio/Desktop/Ines/DEPGNN/Data/',
                        help='datasets_dir')
    parser.add_argument('--split', type=float,
                        nargs='+',
                        default=[0.8, 0.2],
                        help='Train/val/test proportions, e.g., for k fold set only train and test --split 0.9 0.1 0 / for holdout method --split 0.8 0.1 0.1')
    parser.add_argument('--num_repeats', type=int, default=1, help='num of repeats for k-fold (default: 1)')
    parser.add_argument('--k_folds', type=int, default=4, help='num of k folds (default: 4)')
    parser.add_argument('--num_workers', type=int, default=8, help='num_workers in dataloader')

    """############ OUTPUT SETTINGS ############"""
    parser.add_argument('--model_dir', type=str,
                        default='/home/isampaio/Desktop/Ines/DEPGNN/results/tune_results/',
                        help='folder to save all tuning results (combos CSV, results CSV, best model)')
    parser.add_argument('--disable_verb', type=bool,
                        default=True,
                        help='to not save models for all folds')

    ########## Parse parameters ####

    params = parser.parse_args()

    # Hardcode flags expected by TrainerDEPGNN (removed from CLI)
    params.freeze = False
    params.use_pretrained_sampler = False

    print(params)
    setup_seed(params.seed)

    #### Dataset  (loaded once, shared across all combos) #####
    hdf5_path = os.path.join(params.data_dir, f"{params.dataset_name}.h5")
    dataset_dir = os.path.join(params.data_dir, f"{params.dataset_name}_Dataset")
    dataset = BrainGraphDataset(root=Path(dataset_dir), hdf5_path=hdf5_path)

    train_size = int(len(dataset) * params.split[0])

    dataset = dataset.shuffle()
    train_dataset = dataset[:train_size]   # k-fold is performed on this portion
    test_dataset = dataset[train_size:]    # held-out test set

    # k-fold loaders (built once; same splits for every combo)
    sfk = KFold_DataLoader(params, stratify=True)
    train_loaders, val_loaders = sfk.get_nk_loaders(train_dataset)

    # Test loader (same for all folds / combos)
    test_loader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False)

    #### Model Setup #####
    node_feat_dim = dataset.num_node_features
    output_dim = params.y_dim

    # Import model class
    import importlib
    module_name = params.model_name
    module = importlib.import_module(f"models.{module_name}")
    ModelClass = getattr(module, module_name)
    model_config = read_yaml(params.model_config)[0]

    #### Build grid of hyperparameter combos #####
    grid_keys, grid_combos, fixed_params = build_grid(params)
    n_combos = len(grid_combos)

    print("\n" + "=" * 60)
    print(f"Grid Search: {n_combos} hyperparameter combinations")
    print(f"Varying parameters: {grid_keys}")
    print("=" * 60)

    # ---- Output folder (single folder for all 3 files) ----
    root_dir = params.model_dir
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)

    combos_csv_path  = os.path.join(root_dir, "hyperparameter_combos.csv")  # File 1
    results_csv_path = os.path.join(root_dir, "tuning_results.csv")         # File 2
    best_model_path  = os.path.join(root_dir, "best_model.pth")             # File 3

    # ---- Global best tracker (selection based on mean val F1 only) ----
    best_mean_val_f1 = -1.0
    best_combo_id = None

    #### Iterate over all combos #####
    for combo_id, combo in enumerate(grid_combos, start=1):
        print("\n" + "#" * 70)
        print(f"  COMBO {combo_id}/{n_combos}:  {combo}")
        print("#" * 70)

        # --- Apply this combo's hyper-params to params ---
        for key, value in combo.items():
            # Treat dropout=0.0 as None (no dropout)
            if key == 'dropout' and value == 0.0:
                setattr(params, key, None)
            else:
                setattr(params, key, value)

        # --- Save this combo to CSV ---
        save_combo_csv(combos_csv_path, combo_id, combo)

        # --- k-fold CV ---
        print(f"Training with {params.k_folds}-fold CV, {params.num_repeats} repeat(s)")
        print(f"Total training runs: {len(train_loaders)} (num_repeats x k_folds)")

        combo_t_start = time.perf_counter()  # time the full k-fold for this combo

        all_results = []
        fold_binary_masks = []  # collect binary edge masks for Jaccard
        # Track the best individual fold within this combo (to save its weights)
        best_fold_val_f1 = -1.0
        best_fold_model_state = None
        best_fold_sampler_state = None
        best_fold_sampler_min_sp = None

        for fold_idx, (train_loader, val_loader) in enumerate(zip(train_loaders, val_loaders)):
            repeat_num = (fold_idx // params.k_folds) + 1
            fold_num = (fold_idx % params.k_folds) + 1

            print(f"\n{'=' * 60}")
            print(f"Combo {combo_id} | Repeat {repeat_num}/{params.num_repeats}, "
                  f"Fold {fold_num}/{params.k_folds} (Run {fold_idx + 1}/{len(train_loaders)})")
            print("=" * 60)

            # Data loaders for this fold (include test for Trainer compatibility)
            data_loaders = {
                'train': train_loader,
                'val': val_loader,
                'test': test_loader,
            }

            # Reinitialize model and sampler for each fold (fresh start)
            fold_model = ModelClass(config=model_config, input_dim=node_feat_dim, output_dim=output_dim)
            fold_sampler = DEP(params, node_feat_dim, output_dim)

            # Point model_dir to a temp folder so Trainer can save there;
            # we delete it afterwards to avoid 36×4 folders of .pth files.
            tmp_fold_dir = os.path.join(root_dir, "_tmp_fold")
            params.model_dir = tmp_fold_dir

            # Train
            setup_seed(params.seed)  # reset seed per fold for reproducibility
            results = train_one_fold(params, data_loaders, fold_model, fold_sampler)
            all_results.append(results)

            # Clean up the per-fold .pth files saved by the Trainer
            if params.disable_verb== True:
                if os.path.isdir(tmp_fold_dir):
                    shutil.rmtree(tmp_fold_dir)

            # After training, fold_model and fold_sampler hold the best-epoch
            # weights (restored by the Trainer).
            # Run one forward pass to populate sampler.edge_mask with the
            # correct binary mask for the restored weights + current min_sp
            binary_mask = extract_binary_mask(fold_sampler, train_loader)
            fold_binary_masks.append(binary_mask)

            # Keep the best fold's weights.
            if results['best_val_f1'] > best_fold_val_f1:
                best_fold_val_f1 = results['best_val_f1']
                best_fold_model_state = copy.deepcopy(fold_model.state_dict())
                best_fold_sampler_state = copy.deepcopy(fold_sampler.state_dict())
                best_fold_sampler_min_sp = fold_sampler.min_sp

        combo_elapsed = (time.perf_counter() - combo_t_start) / 60

        # --- Jaccard index across fold masks ---
        jaccard_mean, jaccard_std = compute_mean_jaccard(fold_binary_masks)
        print(f"Edge-mask Jaccard Index: {jaccard_mean:.4f} +/- {jaccard_std:.4f}")
        print(f"Total k-fold training time: {combo_elapsed:.1f}min")

        # --- Aggregate k-fold results ---
        results_df = pd.DataFrame(all_results)
        # Inject Jaccard so it flows into the CSV alongside the other metrics
        results_df['jaccard'] = jaccard_mean  # same value per fold (it's a cross-fold metric)
        mean_results = results_df.mean()
        std_results = results_df.std()
        # Override std for jaccard (it's a single cross-fold value, std comes from pairs)
        std_results['jaccard'] = jaccard_std

        print("\n" + "=" * 60)
        print(f"K-Fold Summary for Combo {combo_id}")
        print("=" * 60)
        print(f"Best Val Loss:      {mean_results['best_val_loss']:.4f} +/- {std_results['best_val_loss']:.4f}")
        print(f"Best Val Accuracy:  {mean_results['best_val_acc']:.4f} +/- {std_results['best_val_acc']:.4f}")
        print(f"Best Val F1:        {mean_results['best_val_f1']:.4f} +/- {std_results['best_val_f1']:.4f}")
        print(f"Best Val Precision: {mean_results['best_val_precision']:.4f} +/- {std_results['best_val_precision']:.4f}")
        print(f"Best Val Recall:    {mean_results['best_val_recall']:.4f} +/- {std_results['best_val_recall']:.4f}")
        print(f"Best Val AUC:       {mean_results['best_val_auc']:.4f} +/- {std_results['best_val_auc']:.4f}")
        if 'final_sparsity' in results_df.columns:
            print(f"Final Sparsity:     {mean_results['final_sparsity']:.4f} +/- {std_results['final_sparsity']:.4f}")
        print(f"\nTest Accuracy:      {mean_results['test_acc']:.4f} +/- {std_results['test_acc']:.4f}")
        print(f"Test F1:            {mean_results['test_f1']:.4f} +/- {std_results['test_f1']:.4f}")
        print(f"Test Loss:          {mean_results['test_loss']:.4f} +/- {std_results['test_loss']:.4f}")

        # --- Append mean +/- std to the global tuning results CSV ---
        append_results_csv(results_csv_path, combo_id, mean_results, std_results, combo_elapsed)
        print(f"Results appended to {results_csv_path}")

        # --- Save best model (selection based on mean val F1 only, NOT test) ---
        current_mean_val_f1 = mean_results['best_val_f1']
        if current_mean_val_f1 > best_mean_val_f1:
            best_mean_val_f1 = current_mean_val_f1
            best_combo_id = combo_id

            checkpoint = {
                'model_state_dict': best_fold_model_state,
                'sampler_state_dict': best_fold_sampler_state,
                'sampler_min_sp': best_fold_sampler_min_sp,
                'combo_id': combo_id,
                'combo_params': combo,
                'mean_val_f1': float(mean_results['best_val_f1']),
                'std_val_f1': float(std_results['best_val_f1']),
                'mean_val_acc': float(mean_results['best_val_acc']),
                'mean_val_loss': float(mean_results['best_val_loss']),
                'mean_test_f1': float(mean_results['test_f1']),
                'mean_test_acc': float(mean_results['test_acc']),
            }
            torch.save(checkpoint, best_model_path)
            print(f"*** New best model saved! combo_id={combo_id}, "
                  f"mean_val_F1={current_mean_val_f1:.4f} ***")
        else:
            print(f"No improvement (current mean val F1 {current_mean_val_f1:.4f} "
                  f"<= best {best_mean_val_f1:.4f} from combo {best_combo_id})")

    # --- Final summary ---
    print("\n" + "#" * 70)
    print(f"Grid Search Complete!  {n_combos} combos evaluated.")
    print(f"Output folder:         {root_dir}")
    print(f"  1) Hyperparameter combos: {combos_csv_path}")
    print(f"  2) Tuning results:        {results_csv_path}")
    print(f"  3) Best model checkpoint: {best_model_path}")
    print("#" * 70)

    # Print the best combo by val F1
    if os.path.isfile(results_csv_path):
        summary = pd.read_csv(results_csv_path)
        best_row = summary.loc[summary['best_val_f1_mean'].idxmax()]
        print(f"\nBest combo (by mean val F1): combo_id={int(best_row['combo_id'])}")
        for k in grid_keys:
            print(f"  {k}: {best_row[k]}")
        print(f"  Val F1:  {best_row['best_val_f1_mean']:.4f} +/- {best_row['best_val_f1_std']:.4f}")
        print(f"  Test F1: {best_row['test_f1_mean']:.4f} +/- {best_row['test_f1_std']:.4f}  (not used for selection)")


if __name__ == '__main__':
    main()
