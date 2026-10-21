
import argparse
import copy
import csv
import json
import random
import os

import numpy as np
import torch
import sys
from pathlib import Path
from sklearn.metrics import roc_curve
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


# Add main directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from TrainerDEPGNN_new import Trainer

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
from models.DEP_new import DEP


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_pretrained_sampler(sampler, params):
    """Load DEP weights + pruning level. Handles the new checkpoint
    (dep_selected_checkpoint.pt: keys 'sampler', 'min_sp') and the two old formats."""
    if params.use_pretrained_sampler and os.path.exists(params.sampler_pretrain_dir):
        ckpt = torch.load(params.sampler_pretrain_dir, map_location='cpu')   # moved to GPU by the Trainer
        if isinstance(ckpt, dict) and 'sampler' in ckpt:            # new format
            sampler.load_state_dict(ckpt['sampler'])
            sampler.min_sp = ckpt['min_sp']
        elif isinstance(ckpt, dict) and 'state_dict' in ckpt:       # old format with min_sp
            sampler.load_state_dict(ckpt['state_dict'])
            sampler.min_sp = ckpt['min_sp']
        else:                                                       # old format, weights only
            sampler.load_state_dict(ckpt)
        print(f"  - Loaded pretrained DEP sampler, min_sp={sampler.min_sp:.4f}")
    return sampler
 
 
def train_one_fold(params, data_loaders, model, sampler):
    """Train a single fold. Returns results, the trained sampler and the trained model."""
    trainer = Trainer(params, data_loaders, model, sampler)
    return trainer.train_for_classification()
 
 
def _labels_from_graphs(dataset):
    """Read graph labels in dataset order."""
    return np.asarray([int(dataset[i].y.item()) for i in range(len(dataset))])
 
 
def transform_labels(dataset, y):
    """Normalize binary labels from [1, 2] to [0, 1], matching Baselines."""
    if np.array_equal(np.unique(y), np.array([1, 2])):
        y = y - 1
 
    graph_labels = _labels_from_graphs(dataset)
 
    def normalize_graph(data: Data) -> Data:
        data.y = data.y.to(dtype=torch.long) - 1
        return data
 
    if np.array_equal(np.unique(graph_labels), np.array([1, 2])):
        dataset.transform = normalize_graph
 
    if not np.array_equal(_labels_from_graphs(dataset), y):
        raise ValueError("Graph labels were not normalized consistently")
 
    return dataset, y
 
 
def load_persisted_splits(dataset, task, seed, batch_size, num_workers=0):
    """Build loaders from the persisted Baselines split for ``task`` (unchanged logic).
    Also returns the parsed payload so the dataset order can be checked against it."""
    split_path = Path(__file__).resolve().parents[1] / "splits" / f"{task}_seed{seed}.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file does not exist: {split_path}")
 
    payload = json.loads(split_path.read_text())
    if len(dataset) == 0:
        raise ValueError("Cannot build data loaders from an empty dataset")
 
    def indices(name):
        values = np.asarray(payload[name], dtype=int)
        if values.ndim != 1 or values.size == 0:
            raise ValueError(f"Split '{name}' must be a non-empty one-dimensional index list")
        if np.any(values < 0) or np.any(values >= len(dataset)):
            raise IndexError(f"Split '{name}' contains an index outside the dataset")
        if np.unique(values).size != values.size:
            raise ValueError(f"Split '{name}' contains duplicate indices")
        return values
 
    test_idx = indices("test_idx")
    folds = payload.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError(f"Split file must contain at least one fold: {split_path}")
 
    test_loader = DataLoader(torch.utils.data.Subset(dataset, test_idx.tolist()),
                             batch_size=batch_size, shuffle=False, num_workers=num_workers)
 
    train_loaders, val_loaders = [], []
    test_set = set(test_idx.tolist())
    for fold_number, fold in enumerate(folds, start=1):
        try:
            train_idx = np.asarray(fold["train_idx"], dtype=int)
            val_idx = np.asarray(fold["val_idx"], dtype=int)
        except KeyError as exc:
            raise ValueError(f"Fold {fold_number} is missing {exc.args[0]} indices") from exc
        if train_idx.ndim != 1 or val_idx.ndim != 1:
            raise ValueError(f"Fold {fold_number} indices must be one-dimensional")
        if train_idx.size == 0 or val_idx.size == 0:
            raise ValueError(f"Fold {fold_number} must have non-empty train and validation sets")
        if np.any(train_idx < 0) or np.any(train_idx >= len(dataset)):
            raise IndexError(f"Fold {fold_number} train indices are outside the dataset")
        if np.any(val_idx < 0) or np.any(val_idx >= len(dataset)):
            raise IndexError(f"Fold {fold_number} validation indices are outside the dataset")
        if np.intersect1d(train_idx, val_idx).size:
            raise ValueError(f"Fold {fold_number} train and validation sets overlap")
        if test_set.intersection(train_idx.tolist()) or test_set.intersection(val_idx.tolist()):
            raise ValueError(f"Fold {fold_number} overlaps the persisted test set")
 
        train_loaders.append(DataLoader(torch.utils.data.Subset(dataset, train_idx.tolist()),
                                        batch_size=batch_size, shuffle=True, num_workers=num_workers))
        val_loaders.append(DataLoader(torch.utils.data.Subset(dataset, val_idx.tolist()),
                                      batch_size=batch_size, shuffle=False, num_workers=num_workers))
 
    return train_loaders, val_loaders, test_loader, payload

def fixed_mask_run(params, data_loaders, model, H, subdir, node_feat_dim, output_dim):
    """Train `model` with a frozen mask H: no pruning, no L1/L2, mask parameters out of the optimizer,
    up to --ft_epochs epochs with early stopping on validation AUC (patience --patience),
    best epoch restored. Used for fine-tuning (B) and retraining from scratch (C)."""
    p = copy.copy(params)
    p.model_dir = os.path.join(params.model_dir, subdir)
    p.freeze = True                 # internal only: does not change the user's --freeze
    p.epochs = params.ft_epochs
    p.patience = params.patience
    os.makedirs(p.model_dir, exist_ok=True)
    H = torch.as_tensor(H, dtype=torch.bool)
    sampler = DEP(p, node_feat_dim, output_dim)
    sampler.set_fixed_mask(H)
    results, _, _ = train_one_fold(p, data_loaders, model, sampler)
    return results


def _numeric_summary(rows):
    """Return mean/std for numeric result fields across folds."""
    summary = {}
    if not rows:
        return summary
    for key in rows[0]:
        if key == "fold":
            continue
        values = []
        for row in rows:
            value = row.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
        if values:
            array = np.asarray(values, dtype=float)
            summary[key] = {
                "mean": float(np.nanmean(array)),
                "std": float(np.nanstd(array)),
            }
    return summary


def save_run_results(output_dir, params, fold_results, persisted_folds):
    """Save hyperparameters, per-fold results, and aggregate statistics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _numeric_summary(fold_results)
    payload = {
        "hyperparameters": vars(params).copy(),
        "persisted_folds": persisted_folds,
        "folds": fold_results,
        "summary": summary,
    }
    json_path = output_dir / "results.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str, allow_nan=True))

    metric_names = sorted(summary)
    csv_path = output_dir / "fold_results.csv"
    with csv_path.open("w", newline="") as handle:
        fieldnames = ["fold"] + sorted({key for row in fold_results for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for index, row in enumerate(fold_results, start=1):
            writer.writerow({"fold": index, **row})

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "mean", "std"])
        writer.writeheader()
        for metric in metric_names:
            writer.writerow({"metric": metric, **summary[metric]})

    print(f"Saved run results: {json_path}")
    print(f"Saved fold results: {csv_path}")
    print(f"Saved summary results: {summary_path}")
    return payload
 

def youden_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximising TPR - FPR. Fit on validation only."""
    fpr, tpr, thr = roc_curve(labels, scores)
    j = tpr - fpr
    best = thr[np.argmax(j)]
    return float(best) if np.isfinite(best) else 0.5   # roc_curve's first thr is +inf


def main():

    """############ GNN Model Training ############"""
    parser = argparse.ArgumentParser(description='DEP-GNN Training')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--loss', choices=('CrossEntropyLoss', 'BCEWithLogitsLoss'), default='CrossEntropyLoss')
    parser.add_argument('--y_dim', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=300,
                        help='upper bound for the DEP pruning phase; it stops after the last level')
    parser.add_argument('--model_name', type=str, default='GCN')
    parser.add_argument('--model_config', type=str,
                        default='/home/isampaio/Desktop/Ines/DEPGNN/DEP-GNN/model_configs/gcn.yaml',
                        help='must define the SAME GCN as the DenseGCN baseline for this task')
    parser.add_argument('--patience', type=int, default=20,
                        help='early-stopping patience (val AUC) for every frozen-mask phase: B, C, --freeze runs')
    parser.add_argument('--ft_epochs', type=int, default=200,
                        help='max epochs for every frozen-mask phase (same budget as the GCN baseline)')
    parser.add_argument('--batch_size', type=int, default=32, help='same as baseline')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='GCN only; DEP scores get 0')
    parser.add_argument('--optimizer', choices=('AdamW', 'Adam'), default='AdamW', help='same as baseline')
    parser.add_argument('--use_scheduler', action='store_true', help='cosine LR schedule (off = as baseline)')
    parser.add_argument('--clip_value', type=float, default=0.0)
 
    """############ DEP Sampler  ############"""
    parser.add_argument('--alpha', type=float, default=0.0, help='L1 on scores; 1e-5 already ~ the CE loss')
    parser.add_argument('--beta', type=float, default=0.0, help='L2 on scores; 1e-3 would be ~50x the CE loss')
    parser.add_argument('--DEP_lr', type=float, default=None)
    parser.add_argument('--curr_sp', type=float, default=0.0)
    parser.add_argument('--iter_step', type=int, default=5)
    parser.add_argument('--prune_sp', type=float, default=0.05)
    parser.add_argument('--warmup_epochs', type=int, default=50)
    parser.add_argument('--max_sp', type=float, default=0.999)
    parser.add_argument('--delta', type=float, default=0.01, help='val-AUC tolerance; fix before running')
 
    """############ Follow-up runs per fold ############"""
    parser.add_argument('--finetune', action=argparse.BooleanOptionalAction, default=True,
                        help='B (default on): after selecting the sparsity, continue training the same GCN '
                             'with the DEP mask frozen. --no-finetune to skip')
    parser.add_argument('--retrain', action=argparse.BooleanOptionalAction, default=False,
                        help='C: new GCN from scratch with the DEP mask frozen (mask-quality control)')
 
    """############ Graph Dataset  ############"""
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='FC-row dataset, e.g. PearC_EdgeW_PearC_Sp_fully_connected_raw_Sex')
    parser.add_argument('--task', choices=('Sex', 'FluidIntelligence'), default='Sex')
    parser.add_argument('--data_dir', type=str, default='/mnt/datafast/ines/DEPGNN/Data/')
    parser.add_argument('--baseline_npz', type=str, default=None,
                        help='baseline fc_upper_triangle_*.npz of this task, to check subject order')
    parser.add_argument('--num_workers', type=int, default=0)
 
    """############ CRITICAL DEP SETTINGS############"""
    parser.add_argument('--freeze', action='store_true',
                        help='NO DEP learning: train a GCN on the mask of a loaded sampler '
                             '(requires --use_pretrained_sampler). Not needed for B/C.')
    parser.add_argument('--use_pretrained_sampler', action='store_true',
                        help='load DEP scores + pruning level from --sampler_pretrain_dir')
    parser.add_argument('--sampler_pretrain_dir', type=str,
                        default='/home/isampaio/Desktop/Ines/DEPGNN/DEP_weights/sampler.pth')
    parser.add_argument('--model_dir', type=str, required=True, help='new folder per run (never reuse)')
 
    params = parser.parse_args()
 
    # ── flag consistency
    if params.freeze and not params.use_pretrained_sampler:
        parser.error("--freeze needs --use_pretrained_sampler (otherwise the mask is just the random "
                     "initialisation). For random-mask controls use set_fixed_mask with a matched density.")
    if params.freeze and (params.finetune or params.retrain):
        print("NOTE: --freeze already trains a GCN on a fixed mask; fine-tune/retrain are skipped.")
        params.finetune = params.retrain = False
    if params.use_pretrained_sampler and not params.freeze:
        print("WARNING: DEP continues learning from the loaded scores; the schedule starts at the loaded "
              "pruning level. This is not the protocol of the main experiments.")
    print(params)
    setup_seed(params.seed)


    ########## Parse parameters ####

    params = parser.parse_args()
    print(params)
    setup_seed(params.seed)

    #### Dataset #####
        # ── Load dataset ─────────────────────────────────────────────────────

    hdf5_path = os.path.join(params.data_dir, f"{params.dataset_name}.h5")
    dataset_dir = os.path.join(params.data_dir, f"{params.dataset_name}_Dataset")
    dataset = BrainGraphDataset(
        root=Path(dataset_dir),
        hdf5_path=hdf5_path,
    )
    print(f"Dataset: {len(dataset)} graphs")

    dataset, _ = transform_labels(dataset, _labels_from_graphs(dataset))

    # Inspect first graph to determine dimensions
    sample = dataset[0]
    num_features = sample.x.size(1)

    # ── Load the persisted Baselines split ──────────────────────────────
    train_loaders, val_loaders, test_loader = load_persisted_splits(
        dataset, params.task, params.seed, params.batch_size, params.num_workers)
    persisted_folds = len(train_loaders)
    print(f"Loaded persisted split: {persisted_folds} train/validation folds")


    #### Model Setup #####
    node_feat_dim = dataset.num_node_features
    output_dim = params.y_dim
    
    # Import model class
    import importlib
    module_name = params.model_name
    module = importlib.import_module(f"models.{module_name}")
    ModelClass = getattr(module, module_name)
    model_config = read_yaml(params.model_config)[0]

    #### Train!!! #####
    original_model_dir = params.model_dir
    
    os.makedirs(original_model_dir, exist_ok=True)
    with open(os.path.join(original_model_dir, "run_config.json"), "w") as f:
        json.dump({**vars(params), "model_config": model_config}, f, indent=2, default=str)
    

    all_results = []

    for fold_idx, (train_loader, val_loader) in enumerate(zip(train_loaders, val_loaders)):
        print(f"\n{'=' * 60}")
        print(f"Persisted fold {fold_idx + 1}/{persisted_folds}")
        print("=" * 60)

        data_loaders = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
        }
        fold_dir = os.path.join(original_model_dir, f"fold_{fold_idx + 1}")
        params.model_dir = fold_dir
        os.makedirs(fold_dir, exist_ok=True)

        # 1) DEP learns the mask and sparsity level. The trainer evaluates
        # validation and the shared test set for this fold.
        fold_model = ModelClass(config=model_config, input_dim=node_feat_dim, output_dim=output_dim)
        fold_sampler = DEP(params, node_feat_dim, output_dim)
        if params.use_pretrained_sampler:
            fold_sampler = load_pretrained_sampler(fold_sampler, params)
        results, sampler, model = train_one_fold(params, data_loaders, fold_model, fold_sampler)
        row = {"fold": fold_idx + 1, **results}

        H_sel = sampler.hard_mask().cpu() if sampler.fixed_mask is None else sampler.fixed_mask.cpu()
        np.save(os.path.join(fold_dir, "dep_selected_mask.npy"), H_sel.numpy())
        torch.save(H_sel, os.path.join(fold_dir, "dep_selected_mask.pt"))

        # 2) Variant B: fine-tune the same GCN with the selected mask frozen.
        # The validated H_sel is reused in memory; no duplicate frozen-mask
        # artifact is written in the follow-up directory.
        if params.finetune:
            ft = fixed_mask_run(
                params, data_loaders, model, H_sel, "finetune",
                node_feat_dim, output_dim,
            )
            row.update({f"ft_{k}": v for k, v in ft.items()})

        # 3) Variant C: retrain a new GCN from scratch with the same mask.
        if params.retrain:
            new_model = ModelClass(config=model_config, input_dim=node_feat_dim, output_dim=output_dim)
            rs = fixed_mask_run(
                params, data_loaders, new_model, H_sel, "retrain_scratch",
                node_feat_dim, output_dim,
            )
            row.update({f"rs_{k}": v for k, v in rs.items()})

        all_results.append(row)
        params.model_dir = original_model_dir

    if not all_results:
        raise RuntimeError("No folds were trained")

    print("\n" + "=" * 60)
    print("Cross-validation summary across persisted folds")
    print("=" * 60)
    summary = _numeric_summary(all_results)
    for metric in (
        "val_auc", "val_f1", "val_balanced_accuracy",
        "test_auc", "test_f1", "test_balanced_accuracy",
        "ft_val_auc", "ft_test_auc", "rs_val_auc", "rs_test_auc",
    ):
        if metric in summary:
            print(f"{metric}: {summary[metric]['mean']:.4f} ± {summary[metric]['std']:.4f}")

    save_run_results(original_model_dir, params, all_results, persisted_folds)

    print('\n' + "=" * 60)
    print('Training Complete!')
    print("=" * 60)

if __name__ == '__main__':
    main()
