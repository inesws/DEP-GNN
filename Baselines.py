"""Train reproducible logistic-regression and dense-GCN baselines.

The vector and graph datasets are never independently split: one persisted set
of subject indices is used by both pipelines and by every sparsity level.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from tqdm.auto import tqdm

sys.path.insert(0, "/home/isampaio/Desktop/Ines/DEPGNN/DEP-GNN/")

from utils.customDataset import BrainGraphDataset
from utils.model_utils import read_yaml

LOGGER = logging.getLogger("baselines")
SPLIT_REPEATS = 1 #2
SPLIT_FOLDS = 2  #5
LR_GRID = tuple(np.logspace(-4, 2, 1)) #tuple(np.logspace(-4, 2, 7))


@dataclass(frozen=True)
class Splits:
    """Persisted train/test and repeated train/validation indices."""

    test_idx: np.ndarray
    folds: tuple[tuple[np.ndarray, np.ndarray], ...]
    subject_ids: tuple[str, ...]


def setup_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (including CUDA) RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _as_subject_id(value: Any) -> str:
    """Convert NumPy/HDF5 subject identifiers to comparable strings."""
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def _labels_from_graphs(dataset: BrainGraphDataset) -> np.ndarray:
    """Read graph labels in dataset order."""
    return np.asarray([int(dataset[i].y.item()) for i in range(len(dataset))])


def load_or_create_splits(
    labels: np.ndarray, subject_ids: Sequence[Any], split_path: Path, seed: int
) -> Splits:
    """Load persisted indices or create one stratified 80/20 split and 2x5 folds."""
    ids = tuple(_as_subject_id(x) for x in subject_ids)
    if split_path.exists():
        payload = json.loads(split_path.read_text())
        loaded_ids = tuple(payload["subject_ids"])
        if loaded_ids != ids:
            raise ValueError(f"Subject order differs from persisted split {split_path}")
        folds = tuple(
            (np.asarray(pair["train_idx"], dtype=int), np.asarray(pair["val_idx"], dtype=int))
            for pair in payload["folds"]
        )
        return Splits(np.asarray(payload["test_idx"], dtype=int), folds, ids)

    all_idx = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        all_idx, test_size=0.2, random_state=seed, stratify=labels
    )
    # The splitter returns positions into the 80% array; translate them back to
    # original subject indices before persisting.
    splitter = RepeatedStratifiedKFold(
        n_splits=SPLIT_FOLDS, n_repeats=SPLIT_REPEATS, random_state=seed
    )
    folds = tuple(
        (train_idx[tr].astype(int), train_idx[val].astype(int))
        for tr, val in splitter.split(train_idx, labels[train_idx])
    )
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(
        json.dumps(
            {
                "test_idx": test_idx.tolist(),
                "folds": [
                    {"train_idx": tr.tolist(), "val_idx": val.tolist()} for tr, val in folds
                ],
                "subject_ids": list(ids),
            },
            indent=2,
        )
    )
    return Splits(test_idx, folds, ids)


def metric_values(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Return AUC and thresholded F1, with an explicit NaN for undefined AUC."""
    try:
        auc = float(roc_auc_score(labels, scores))
    except ValueError:
        auc = float("nan")
    return {"auc": auc, "f1": float(f1_score(labels, scores >= 0.5, zero_division=0))}


def mean_std(values: Iterable[float]) -> dict[str, float]:
    """Summarize finite fold metrics."""
    array = np.asarray(list(values), dtype=float)
    return {"mean": float(np.nanmean(array)), "std": float(np.nanstd(array))}


def run_logistic_regression(
    X: np.ndarray, y: np.ndarray, splits: Splits
) -> dict[str, Any]:
    """Tune and evaluate a leakage-free standardized logistic regression."""
    LOGGER.info(
        "Logistic regression: testing %d C values over %d CV folds",
        len(LR_GRID),
        len(splits.folds),
    )
    candidates: list[dict[str, Any]] = []
    for c_value in tqdm(LR_GRID, desc="LR hyperparameters", unit="C"):
        fold_metrics = []
        for fold_train, fold_val in splits.folds:
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", LogisticRegression(C=float(c_value), max_iter=2000, solver="liblinear")),
                ]
            )
            model.fit(X[fold_train], y[fold_train])
            fold_metrics.append(
                metric_values(y[fold_val], model.predict_proba(X[fold_val])[:, 1])
            )
        candidates.append({"C": float(c_value), "folds": fold_metrics})
        LOGGER.info(
            "LR C=%g: mean CV AUC=%.4f, F1=%.4f",
            c_value,
            np.nanmean([metric["auc"] for metric in fold_metrics]),
            np.nanmean([metric["f1"] for metric in fold_metrics]),
        )
    best = max(candidates, key=lambda item: np.nanmean([m["auc"] for m in item["folds"]]))
    LOGGER.info("LR best C=%g", best["C"])
    final_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=best["C"], max_iter=2000, solver="liblinear")),
        ]
    )
    train_idx = np.concatenate([tr for tr, _ in splits.folds])
    train_idx = np.unique(train_idx)
    final_model.fit(X[train_idx], y[train_idx])
    test_metrics = metric_values(y[splits.test_idx], final_model.predict_proba(X[splits.test_idx])[:, 1])
    LOGGER.info("LR test AUC=%.4f, F1=%.4f", test_metrics["auc"], test_metrics["f1"])
    return {
        "best_hyperparameters": {"C": best["C"]},
        "folds": best["folds"],
        "cv": {
            "auc": mean_std(m["auc"] for m in best["folds"]),
            "f1": mean_std(m["f1"] for m in best["folds"]),
        },
        "test": test_metrics,
    }


class DenseGCN(nn.Module):
    """Small dense (non-learned-sparsity) graph classifier."""

    def __init__(self, input_dim: int, hidden_dim: int, layers: int, dropout: float, use_edge_weight: bool):
        super().__init__()
        self.dropout = dropout
        self.use_edge_weight = use_edge_weight
        self.convs = nn.ModuleList(
            [GCNConv(input_dim, hidden_dim)]
            + [GCNConv(hidden_dim, hidden_dim) for _ in range(max(0, layers - 1))]
        )
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, data: Data) -> torch.Tensor:
        edge_weight = None
        if self.use_edge_weight and getattr(data, "edge_attr", None) is not None:
            # GCNConv normalizes with degree^(-1/2); signed weights can create
            # negative degrees and NaN values, so retain magnitudes only.
            edge_weight = data.edge_attr.abs()
            if edge_weight.ndim > 1:
                edge_weight = edge_weight.squeeze(-1)
        x = data.x
        for conv in self.convs:
            x = F.relu(conv(x, data.edge_index, edge_weight))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(global_mean_pool(x, data.batch))


def _loader(dataset: Any, indices: np.ndarray, batch_size: int, seed: int, shuffle: bool) -> DataLoader:
    """Create a deterministic PyG loader for a selected index array."""
    selected = dataset[indices.tolist()]
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(selected, batch_size=batch_size, shuffle=shuffle, generator=generator)


def train_gcn_fold(
    dataset: BrainGraphDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    input_dim: int,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[dict[str, float], int]:
    """Train one fold with validation-AUC early stopping."""
    setup_seed(seed)
    model = DenseGCN(
        input_dim, int(config["hidden_dim"]), int(config.get("num_layers", 2)),
        float(config["dropout"]), bool(config["use_edge_weight"])
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    criterion = nn.CrossEntropyLoss()
    train_loader = _loader(dataset, train_idx, int(config["batch_size"]), seed, True)
    val_loader = _loader(dataset, val_idx, int(config["batch_size"]), seed, False)
    best_auc, best_epoch, stale, best_state = -np.inf, 0, 0, None
    epoch_bar = tqdm(
        range(1, int(config["epochs"]) + 1),
        desc="GCN epochs",
        unit="epoch",
        leave=False,
    )
    for epoch in epoch_bar:
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch.y.view(-1))
            loss.backward()
            optimizer.step()
        metrics = evaluate_gcn(model, val_loader, device)
        epoch_bar.set_postfix(val_auc=f"{metrics['auc']:.4f}", val_f1=f"{metrics['f1']:.4f}")
        if metrics["auc"] > best_auc:
            best_auc, best_epoch, stale = metrics["auc"], epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= int(config["patience"]):
                LOGGER.info(
                    "GCN early stopping at epoch %d (best epoch=%d, val AUC=%.4f)",
                    epoch,
                    best_epoch,
                    best_auc,
                )
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return evaluate_gcn(model, val_loader, device), best_epoch


def evaluate_gcn(model: DenseGCN, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Evaluate a graph model and return AUC/F1."""
    model.eval()
    labels, scores = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch.to(device))
            labels.extend(batch.y.view(-1).cpu().numpy())
            scores.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return metric_values(np.asarray(labels), np.asarray(scores))


def config_grid(config_path: Path | None) -> list[dict[str, Any]]:
    """Normalize the repository YAML format into the GCN training settings."""
    raw = read_yaml(str(config_path)) if config_path else [{}]
    result = []
    for item in raw:
        result.append(
            {
                "hidden_dim": item.get("hidden_dim", 64),
                "num_layers": item.get("num_layers", 2),
                "dropout": item.get("dropout", 0.5),
                "lr": item.get("lr", 1e-3),
                "weight_decay": item.get("weight_decay", 1e-4),
                "batch_size": item.get("batch_size", 32),
                "epochs": item.get("epochs", 100),
                "patience": item.get("patience", 15),
                "use_edge_weight": item.get("use_edge_weight", item.get("edge_attr") is not None),
            }
        )
    return result


def run_gcn(
    dataset: BrainGraphDataset, splits: Splits, configs: list[dict[str, Any]],
    seed: int, device: torch.device,
) -> dict[str, Any]:
    """Tune GCN settings on the shared folds, then retrain on all 80% training subjects."""
    input_dim = int(dataset[0].x.shape[-1])
    candidates = []
    LOGGER.info(
        "GCN hyperparameter tuning: testing %d configurations over %d CV folds on %s",
        len(configs),
        len(splits.folds),
        device,
    )
    for config_number, config in enumerate(
        tqdm(configs, desc="GCN configurations", unit="config"), start=1
    ):
        LOGGER.info("GCN config %d/%d: %s", config_number, len(configs), config)
        fold_results, epochs = [], []
        fold_bar = tqdm(
            enumerate(splits.folds, start=1),
            total=len(splits.folds),
            desc=f"GCN config {config_number} folds",
            unit="fold",
            leave=False,
        )
        for fold_number, (train_idx, val_idx) in fold_bar:
            result, best_epoch = train_gcn_fold(dataset, train_idx, val_idx, input_dim, config, seed + fold_number, device)
            fold_results.append(result)
            epochs.append(best_epoch)
            fold_bar.set_postfix(val_auc=f"{result['auc']:.4f}", val_f1=f"{result['f1']:.4f}")
        candidates.append({"config": config, "folds": fold_results, "epochs": epochs})
        LOGGER.info(
            "GCN config %d: mean CV AUC=%.4f, F1=%.4f",
            config_number,
            np.nanmean([metric["auc"] for metric in fold_results]),
            np.nanmean([metric["f1"] for metric in fold_results]),
        )
    best = max(candidates, key=lambda item: np.nanmean([m["auc"] for m in item["folds"]]))
    LOGGER.info("GCN best configuration: %s", best["config"])
    train_idx = np.unique(np.concatenate([tr for tr, _ in splits.folds]))
    final_config = dict(best["config"])
    final_config["epochs"] = max(1, int(np.nanmean(best["epochs"])))
    model = DenseGCN(input_dim, int(final_config["hidden_dim"]), int(final_config["num_layers"]),
                     float(final_config["dropout"]), bool(final_config["use_edge_weight"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(final_config["lr"]), weight_decay=float(final_config["weight_decay"]))
    model.train()
    LOGGER.info(
        "GCN final training on %d subjects for %d epochs",
        len(train_idx),
        final_config["epochs"],
    )
    for _ in tqdm(range(final_config["epochs"]), desc="GCN final training", unit="epoch"):
        for batch in _loader(dataset, train_idx, int(final_config["batch_size"]), seed, True):
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(batch), batch.y.view(-1))
            loss.backward()
            optimizer.step()
    test = evaluate_gcn(model, _loader(dataset, splits.test_idx, int(final_config["batch_size"]), seed, False), device)
    LOGGER.info("GCN test AUC=%.4f, F1=%.4f", test["auc"], test["f1"])
    return {
        "best_hyperparameters": best["config"],
        "folds": best["folds"],
        "cv": {"auc": mean_std(m["auc"] for m in best["folds"]), "f1": mean_std(m["f1"] for m in best["folds"])},
        "test": test,
    }


def check_alignment(dataset: BrainGraphDataset, vector_ids: Sequence[Any], vector_labels: np.ndarray) -> None:
    #"""Assert graph/vector order using HDF5 keys, with labels as a fallback."""
    #graph_ids = tuple(_as_subject_id(key) for key in dataset.graph_keys)
    #vector_ids_normalized = tuple(_as_subject_id(value) for value in vector_ids)
    graph_labels = _labels_from_graphs(dataset)
    #if graph_ids != vector_ids_normalized:
    #    if len(graph_ids) != len(vector_ids_normalized) or not np.array_equal(graph_labels, vector_labels):
    #        raise ValueError("Graph and vector subject order/labels are not aligned")
    #    LOGGER.warning("No matching graph IDs; accepted equal length and equal label order.")
    if not np.array_equal(graph_labels, vector_labels):
        raise ValueError("Graph and vector labels differ despite matching subject IDs")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=str, default='Sex')
    parser.add_argument("--sparsity", choices=("full", "top1", "top10", "top30"), default='full')
    parser.add_argument("--data_dir", type=str,
                        default='/mnt/datafast/ines/DEPGNN/')
    parser.add_argument("--dataset_name", type=str,
                        default='PearC_EdgeW_PearC_Sp_fully_connected_raw_Sex')
    parser.add_argument("--npz_path", type=str,
                        default='/mnt/datafast/ines/DEPGNN/fc_upper_triangle_sp_fully_connected_Male.npz')
    parser.add_argument("--config", type=str, default='/home/isampaio/Desktop/Ines/DEPGNN/DEP-GNN/model_configs/baseline_gcn.yaml', help='model config files')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", choices=("lr", "gcn", "both"), default="both")
    parser.add_argument("--out_dir", default="results")
    return parser.parse_args()


def save_results(results: dict[str, Any], output_dir: Path) -> None:
    """Persist JSON and CSV results and verify both files were written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2, allow_nan=True))

    row = {"sparsity": results["sparsity"]}
    for name in ("lr", "gcn"):
        if name in results:
            row[f"{name}_auc_mean"] = results[name]["cv"]["auc"]["mean"]
            row[f"{name}_auc_std"] = results[name]["cv"]["auc"]["std"]
            row[f"{name}_f1_mean"] = results[name]["cv"]["f1"]["mean"]
            row[f"{name}_f1_std"] = results[name]["cv"]["f1"]["std"]
            row[f"{name}_test_auc"] = results[name]["test"]["auc"]
            row[f"{name}_test_f1"] = results[name]["test"]["f1"]

    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)

    for path in (results_path, csv_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise OSError(f"Result file was not written correctly: {path}")
        LOGGER.info("Saved results to %s", path.resolve())


def main() -> None:
    """Run selected baselines and write JSON/CSV results."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    setup_seed(args.seed)
    npz = np.load(args.npz_path, allow_pickle=True)
    X, y, subject_ids = np.asarray(npz["fc"]), np.asarray(npz["labels"]).astype(int), npz["subject_ids"]
    hdf5_path = os.path.join(args.data_dir, f"{args.dataset_name}.h5")
    dataset = BrainGraphDataset(root=Path(args.data_dir, f"{args.dataset_name}_Dataset"), hdf5_path=hdf5_path)
    if len(dataset) != len(X):
        raise ValueError("Graph and vector dataset lengths differ")
    check_alignment(dataset, subject_ids, y)
    split_file = Path("splits") / f"{args.task}_seed{args.seed}.json"
    splits = load_or_create_splits(y, subject_ids, split_file, args.seed)
    results: dict[str, Any] = {"task": args.task, "sparsity": args.sparsity, "seed": args.seed}
    if args.models in ("lr", "both"):
        results["lr"] = run_logistic_regression(X, y, splits)
    if args.models in ("gcn", "both"):
        results["gcn"] = run_gcn(dataset, splits, config_grid(Path(args.config) if args.config else None),
                                 args.seed, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = (Path(args.out_dir) / args.task / args.sparsity).resolve()
    LOGGER.info("Output directory: %s", output_dir)
    save_results(results, output_dir)
    LOGGER.info("Final summary")
    LOGGER.info("model | CV AUC (mean +/- std) | CV F1 (mean +/- std) | test AUC | test F1")
    for name in ("lr", "gcn"):
        if name in results:
            cv_auc, cv_f1 = results[name]["cv"]["auc"], results[name]["cv"]["f1"]
            test = results[name]["test"]
            LOGGER.info(
                "%s | %.4f +/- %.4f | %.4f +/- %.4f | %.4f | %.4f",
                name, cv_auc["mean"], cv_auc["std"], cv_f1["mean"], cv_f1["std"],
                test["auc"], test["f1"],
            )


if __name__ == "__main__":
    main()
