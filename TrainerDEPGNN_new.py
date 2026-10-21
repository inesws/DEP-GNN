import copy
import json
import os
from timeit import default_timer as timer

import numpy as np
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, balanced_accuracy_score
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Shared evaluation helpers (same as the LR / GCN / MoG pipelines)
# ─────────────────────────────────────────────────────────────────────────────
def youden_threshold(labels, scores):
    """Threshold maximising TPR - FPR. Fit on validation only."""
    fpr, tpr, thr = roc_curve(labels, scores)
    best = thr[np.argmax(tpr - fpr)]
    return float(best) if np.isfinite(best) else 0.5


def metric_values(labels, scores, threshold):
    try:
        auc = float(roc_auc_score(labels, scores))
    except ValueError:
        auc = float("nan")
    pred = scores >= threshold
    return {
        "auc": auc,
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "threshold": float(threshold),
    }


def to_cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


class Trainer(object):
    """
    DEP-GNN trainer (classification).

    Schedule
      1. warm-up: `warmup_epochs` epochs on the full graph (min_sp = curr_sp, usually 0)
      2. progressive pruning: every `iter_step` epochs the pruned fraction increases
             by prune_sp up to 0.90, by 0.01 up to 0.99, by 0.001 up to max_sp (0.999)
      3. after the last level has had its `iter_step` epochs, training stops
         (every level gets the same number of epochs, so no level is favoured)

    Checkpoints
      for every sparsity level, the epoch with the best validation AUC is kept (on CPU).

    Selection (tolerance rule)
      best = max validation AUC over all levels
      sp*  = the sparsest level whose validation AUC >= best - delta
      the model + mask of sp* are restored and evaluated on the test set.
    """

    def __init__(self, params, data_loader, model, sampler=None):
        self.params = params
        self.data_loader = data_loader
        self.model = model.cuda()
        self.sampler = sampler.cuda() if sampler is not None else None

        if params.loss == "CrossEntropyLoss":
            self.criterion = CrossEntropyLoss().cuda()
        else:
            self.criterion = BCEWithLogitsLoss().cuda()

        # ── DEP freeze / train mode
        if self.sampler is not None:
            for p in self.sampler.parameters():
                p.requires_grad = not params.freeze
            print("DEP Sampler in", "Freeze mode" if params.freeze else "Train mode")

        # ── Optimizer: separate groups, NO weight decay on the DEP scores
        sampler_params = list(self.sampler.parameters()) if (self.sampler is not None and not params.freeze) else []
        model_params = list(self.model.parameters())
        wd = params.weight_decay if params.weight_decay is not None else 0.0
        groups = [{"params": model_params, "lr": params.lr, "weight_decay": wd}]
        if sampler_params:
            groups.append({"params": sampler_params,
                           "lr": params.DEP_lr if params.DEP_lr is not None else params.lr,
                           "weight_decay": 0.0})
        Opt = torch.optim.Adam if params.optimizer == "Adam" else torch.optim.AdamW
        self.optimizer = Opt(groups)

        self.scheduler = None
        if getattr(params, "use_scheduler", False):
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=params.epochs, eta_min=1e-6)

        self.max_sp = getattr(params, "max_sp", 0.999)
        self.delta = getattr(params, "delta", 0.01)
        self.warmup_epochs = getattr(params, "warmup_epochs", 0)

        print(self.model)

    # ─────────────────────────────────────────────────────────────────────────
    # helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _forward(self, data):
        sampled, _ = self.sampler(data) if self.sampler is not None else (data, None)
        if sampled.edge_attr is not None:
            return self.model(sampled.x, sampled.edge_index, sampled.batch, sampled.edge_attr)
        return self.model(sampled.x, sampled.edge_index, sampled.batch)

    def _task_loss(self, out, y):
        if self.params.loss == "CrossEntropyLoss":
            return self.criterion(out, y.view(-1).long())
        return self.criterion(out.view(-1), y.view(-1).float())

    def _scores(self, out):
        """P(class 1) from the model output."""
        if self.params.loss == "CrossEntropyLoss":
            return torch.softmax(out, dim=1)[:, 1]
        return torch.sigmoid(out.view(-1))

    def _next_step(self, sp):
        """Pruning increment for the current level (lands exactly on 0.90, 0.99, max_sp)."""
        eps = 1e-9
        if sp < 0.90 - eps:
            return min(self.params.prune_sp, round(0.90 - sp, 5))
        if sp < 0.99 - eps:
            return min(0.01, round(0.99 - sp, 5))
        if sp < self.max_sp - eps:
            return min(0.001, round(self.max_sp - sp, 5))
        return 0.0

    @torch.no_grad()
    def predict(self, loader):
        """Labels, P(class 1), mean task loss (no regularisation), in loader order."""
        self.model.eval()
        if self.sampler is not None:
            self.sampler.eval()
        ys, ss, losses = [], [], []
        for data in loader:
            data = data.cuda()
            out = self._forward(data)
            losses.append(self._task_loss(out, data.y).item())
            ys.append(data.y.view(-1).cpu().numpy())
            ss.append(self._scores(out).cpu().numpy())
        return np.concatenate(ys), np.concatenate(ss), float(np.mean(losses))

    # ─────────────────────────────────────────────────────────────────────────
    # training
    # ─────────────────────────────────────────────────────────────────────────
    def train_for_classification(self):
        p = self.params
        level_ckpts = {}           # sparsity -> best checkpoint at that level
        level_curve = []           # (epoch, sparsity, val_auc) for every epoch
        epochs_at_level = 0

        # "prune": DEP learns the mask and the sparsity follows the schedule.
        # "fixed": no pruning schedule (no sampler, frozen sampler, or a fixed control mask);
        #          plain early stopping on validation AUC, exactly like the GCN baseline.
        prune_mode = (self.sampler is not None and not p.freeze and self.sampler.fixed_mask is None)
        epochs_no_improve, best_fixed_auc = 0, -np.inf

        if prune_mode:
            print(f"Starting pruning percentage: {self.sampler.min_sp * 100:.2f}%  "
                  f"(warm-up {self.warmup_epochs} epochs, max {self.max_sp * 100:.1f}%, delta {self.delta})")
        else:
            print(f"Fixed-mask mode: early stopping on validation AUC, patience {p.patience}")

        for epoch in range(p.epochs):
            start = timer()
            self.model.train()
            if self.sampler is not None:
                self.sampler.train()

            tr_losses = []
            for data in tqdm(self.data_loader["train"], mininterval=10):
                data = data.cuda()
                self.optimizer.zero_grad()
                out = self._forward(data)
                loss = self._task_loss(out, data.y)
                if prune_mode:
                    loss = loss + self.sampler.l1_norm() + self.sampler.l2_norm()
                loss.backward()

                clip = getattr(p, "clip_value", 0)
                if clip and clip > 0:      # clip model and sampler separately
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                    if self.sampler is not None:
                        torch.nn.utils.clip_grad_norm_(self.sampler.parameters(), clip)

                self.optimizer.step()
                tr_losses.append(loss.item())

            if self.scheduler is not None:
                self.scheduler.step()

            # ── validation at the current sparsity level
            y_val, s_val, v_loss = self.predict(self.data_loader["val"])
            v_auc = float(roc_auc_score(y_val, s_val))
            sp = round(self.sampler.min_sp, 5) if self.sampler is not None else 0.0
            level_curve.append((epoch + 1, sp, v_auc))

            if sp not in level_ckpts or v_auc > level_ckpts[sp]["v_auc"]:
                level_ckpts[sp] = {
                    "v_auc": v_auc, "epoch": epoch + 1, "v_loss": v_loss,
                    "model": to_cpu_state(self.model),
                    "sampler": to_cpu_state(self.sampler) if self.sampler is not None else None,
                }

            print(f"Epoch {epoch + 1:4d} | sp {sp * 100:6.2f}% | TrLoss {np.mean(tr_losses):.4f} | "
                  f"VaLoss {v_loss:.4f} | VaAUC {v_auc:.4f} | {(timer() - start) / 60:.2f} min")

            # ── fixed-mask mode: plain early stopping on validation AUC
            if not prune_mode:
                if v_auc > best_fixed_auc:
                    best_fixed_auc, epochs_no_improve = v_auc, 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= p.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
                continue

            # ── prune mode: advance the pruning schedule
            in_warmup = (epoch + 1) <= self.warmup_epochs
            if not in_warmup:
                epochs_at_level += 1
                if epochs_at_level >= p.iter_step:
                    step = self._next_step(self.sampler.min_sp)
                    if step > 0:
                        old = self.sampler.min_sp
                        self.sampler.update_prune(step)
                        print(f"  pruning {old * 100:.2f}% -> {self.sampler.min_sp * 100:.2f}%")
                        epochs_at_level = 0
                    else:
                        break                  # last level has had its iter_step epochs

        # ── tolerance-based selection
        best_auc = max(c["v_auc"] for c in level_ckpts.values())
        ok = [s for s, c in level_ckpts.items() if c["v_auc"] >= best_auc - self.delta]
        sp_star = max(ok)
        ck = level_ckpts[sp_star]
        self.model.load_state_dict(ck["model"])
        if self.sampler is not None:
            fixed = self.sampler.fixed_mask                    # load_state_dict does not touch it
            self.sampler.load_state_dict(ck["sampler"])
            self.sampler.min_sp = sp_star
            self.sampler.fixed_mask = fixed
        if prune_mode:
            print(f"Selected sparsity {sp_star * 100:.2f}% (epoch {ck['epoch']}, val AUC {ck['v_auc']:.4f}; "
                  f"best val AUC {best_auc:.4f}, delta {self.delta})")
        else:
            print(f"Frozen mask: best epoch {ck['epoch']}, val AUC {ck['v_auc']:.4f}")

        # ── final evaluation: threshold from validation, applied to test
        y_val, s_val, _ = self.predict(self.data_loader["val"])
        y_te, s_te, te_loss = self.predict(self.data_loader["test"])
        thr = youden_threshold(y_val, s_val)
        val_m = metric_values(y_val, s_val, thr)
        test_m = metric_values(y_te, s_te, thr)
        print(f"Test AUC {test_m['auc']:.4f} | BA {test_m['balanced_accuracy']:.4f} | F1 {test_m['f1']:.4f}")

        # ── save everything needed for tables, ensemble and mask analyses
        os.makedirs(p.model_dir, exist_ok=True)
        levels = sorted(level_ckpts)
        np.savez_compressed(
            os.path.join(p.model_dir, "dep_fold_outputs.npz"),
            val_y=y_val, val_scores=s_val, test_y=y_te, test_scores=s_te, threshold=thr,
            mask=self.sampler.get_edge_vector() if self.sampler is not None else np.array([]),
            sp_star=sp_star, delta=self.delta,
            levels=np.array(levels), level_best_val_auc=np.array([level_ckpts[s]["v_auc"] for s in levels]),
            level_best_epoch=np.array([level_ckpts[s]["epoch"] for s in levels]),
            curve=np.array(level_curve),     # columns: epoch, sparsity, val AUC
        )
        torch.save({"model": ck["model"], "sampler": ck["sampler"], "min_sp": sp_star},
                   os.path.join(p.model_dir, "dep_selected_checkpoint.pt"))

        results = {
            "best_epoch": ck["epoch"],
            "selected_sparsity": (sp_star if prune_mode else
                                  1.0 - float(self.sampler.get_edge_vector().mean()) if self.sampler is not None else 0.0),
            "mode": "prune" if prune_mode else "frozen_mask",
            "val_auc_selected": ck["v_auc"],
            "val_auc_best_any_level": best_auc,
            "val_auc": val_m["auc"], "val_balanced_accuracy": val_m["balanced_accuracy"], "val_f1": val_m["f1"],
            "test_auc": test_m["auc"], "test_balanced_accuracy": test_m["balanced_accuracy"],
            "test_f1": test_m["f1"], "test_loss": te_loss, "threshold": thr,
            "mask_density": float(self.sampler.get_edge_vector().mean()) if self.sampler is not None else 1.0,
        }
        with open(os.path.join(p.model_dir, "dep_fold_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        return results, self.sampler, self.model
