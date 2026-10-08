"""Clean training engine for EEG classification."""

from __future__ import annotations

import inspect
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import math

from torch import optim
from torch.optim.lr_scheduler import LambdaLR

from ..utils.metrics import (
    compute_metrics,
    compute_subject_metrics,
    fmt_metrics,
    prefix_metrics,
)
from ..utils.subject_votes import (
    append_vote_records,
    log_vote_report,
    vote_records_to_wandb_table,
    vote_summary_to_wandb,
)

logger = logging.getLogger(__name__)

_SEP = "─" * 90


def _get_cosine_with_warmup_schedule(
    optimizer: optim.Optimizer,
    warmup_epochs: int,
    n_epochs: int,
) -> LambdaLR:
    """Cosine decay with linear warmup. warmup_epochs=0 gives pure cosine decay."""
    def _lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = float(epoch - warmup_epochs) / float(max(1, n_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, _lr_lambda)


class Trainer:
    """Supervised EEG classification trainer."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        cfg: dict,
        device: torch.device,
        wandb_run=None,
        ch_names: list[str] | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.cfg = cfg
        self.device = device
        self.wandb_run = wandb_run
        self.ch_names = ch_names

        try:
            self._supports_ch_names = "ch_names" in inspect.signature(self.model.forward).parameters
        except (TypeError, ValueError):
            self._supports_ch_names = False

        # ── Optimizer ───────────────────────────────────────────────────
        lr = float(cfg.get("learning_rate", 1e-4))
        weight_decay = float(cfg.get("weight_decay", 1e-4))
        self.optimizer = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # ── Scheduler ───────────────────────────────────────────────────
        n_epochs = int(cfg.get("train_epochs", 100))
        warmup_epochs = int(cfg.get("warmup_epochs", 0))
        if warmup_epochs > 0:
            # Cosine decay with linear warmup (standard for fine-tuning)
            self.scheduler = _get_cosine_with_warmup_schedule(
                self.optimizer, warmup_epochs, n_epochs
            )
        else:
            # Pure cosine decay — existing default behaviour (no warmup)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=n_epochs
            )

        self.criterion = nn.CrossEntropyLoss()
        self.n_epochs = n_epochs
        self.patience = int(cfg.get("patience", 15))
        self.log_interval = int(cfg.get("log_interval", 50))
        self.log_batch = bool(cfg.get("log_batch", False))
        self.use_subject_vote = bool(cfg.get("use_subject_vote", True))
        self.max_grad_norm = float(cfg.get("max_grad_norm", 4.0))
        self.global_step = 0

        # ── Early Stopping ──────────────────────────────────────────────
        self.early_stopping_metric = str(cfg.get("early_stopping_metric", "val_loss")).lower()
        # Determine whether to minimize or maximize based on metric name
        if "loss" in self.early_stopping_metric:
            self.early_stopping_mode = "minimize"
        elif "f1" in self.early_stopping_metric or "accuracy" in self.early_stopping_metric:
            self.early_stopping_mode = "maximize"
        else:
            self.early_stopping_mode = "maximize"  # Default to maximize for unknown metrics
        logger.info(
            f"Early stopping: metric={self.early_stopping_metric}, mode={self.early_stopping_mode}"
        )

        # ── Checkpoint ──────────────────────────────────────────────────
        save_dir = Path(cfg.get("save_dir", "outputs/checkpoints"))
        save_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_path = save_dir / "best.pth"

    def _forward_model(self, X: torch.Tensor) -> torch.Tensor:
        if self._supports_ch_names and self.ch_names is not None:
            return self.model(X, ch_names=self.ch_names)
        return self.model(X)

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> tuple[float, dict, dict | None]:
        self.model.train()
        losses, all_logits, all_labels, all_sids = [], [], [], []
        running_correct = running_total = 0
        n_batches = len(self.train_loader)

        print()
        print(_SEP)
        print(f"  Epoch {epoch:03d}/{self.n_epochs:03d}")
        print(_SEP)

        for batch_idx, (X, labels, sids) in enumerate(self.train_loader, start=1):
            self.global_step += 1
            X = X.float().to(self.device)
            labels = labels.long().to(self.device)    # (B,)
            sids = sids.long()                         # keep on CPU for metrics

            self.optimizer.zero_grad()
            logits = self._forward_model(X)           # (B, n_classes)
            loss = self.criterion(logits, labels)
            loss.backward()

            grad_norm = _compute_grad_norm(self.model)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            losses.append(loss.item())
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_sids.append(sids)

            preds = logits.detach().argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

            should_log = (
                batch_idx == 1
                or batch_idx % self.log_interval == 0
                or batch_idx == n_batches
            )
            if should_log:
                running_loss = float(np.mean(losses))
                running_acc = running_correct / running_total
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"  [{batch_idx:04d}/{n_batches:04d}] "
                    f"loss={loss.item():.4f}  run_loss={running_loss:.4f}  "
                    f"run_acc={running_acc:.4f}  grad={grad_norm:.3f}  lr={lr:.2e}"
                )

                if self.log_batch and self.wandb_run is not None:
                    self.wandb_run.log(
                        {
                            "train/global_step": self.global_step,
                            "train/batch_loss": loss.item(),
                            "train/running_loss": running_loss,
                            "train/running_acc": running_acc,
                            "train/grad_norm": grad_norm,
                            "train/lr": lr,
                        },
                        step=self.global_step,
                    )

        train_loss = float(np.mean(losses))
        cat_logits = torch.cat(all_logits)
        cat_labels = torch.cat(all_labels)
        cat_sids = torch.cat(all_sids).numpy()

        sample_metrics, preds = compute_metrics(cat_logits, cat_labels)
        subject_metrics = None
        if self.use_subject_vote:
            probs = torch.softmax(cat_logits, dim=1).cpu().numpy()
            subject_metrics = compute_subject_metrics(
                preds, cat_labels.numpy(), cat_sids,
                num_classes=cat_logits.shape[1],
                probs=probs,
            )

        return train_loss, sample_metrics, subject_metrics

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self, loader, return_raw: bool = False
    ) -> tuple[float, dict, dict | None] | tuple[float, dict, dict | None, dict]:
        self.model.eval()
        losses, all_logits, all_labels, all_sids = [], [], [], []

        for X, labels, sids in loader:
            X = X.float().to(self.device)
            labels = labels.long().to(self.device)

            logits = self._forward_model(X)
            loss = self.criterion(logits, labels)

            losses.append(loss.item())
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_sids.append(sids)

        mean_loss = float(np.mean(losses)) if losses else 0.0
        cat_logits = torch.cat(all_logits)
        cat_labels = torch.cat(all_labels)
        cat_sids = torch.cat(all_sids).numpy()

        sample_metrics, preds = compute_metrics(cat_logits, cat_labels)
        probs = None
        if self.use_subject_vote or return_raw:
            probs = torch.softmax(cat_logits, dim=1).cpu().numpy()
        subject_metrics = None
        vote_records, vote_summary = None, None
        if self.use_subject_vote:
            subject_metrics, vote_records, vote_summary = compute_subject_metrics(
                preds, cat_labels.numpy(), cat_sids,
                num_classes=cat_logits.shape[1],
                probs=probs,
                return_details=True,
            )

        if return_raw:
            raw = {
                "preds": preds,
                "y_true": cat_labels.numpy(),
                "sids": cat_sids,
                "probs": probs,
                "vote_records": vote_records,
                "vote_summary": vote_summary,
            }
            return mean_loss, sample_metrics, subject_metrics, raw
        return mean_loss, sample_metrics, subject_metrics

    # ------------------------------------------------------------------
    # Main train loop
    # ------------------------------------------------------------------

    def train(self) -> float:
        # Initialize best_metric based on whether we minimize or maximize
        if self.early_stopping_mode == "minimize":
            best_metric = float("inf")
        else:
            best_metric = float("-inf")
        wait = 0

        for epoch in range(1, self.n_epochs + 1):
            tr_loss, tr_sam, tr_sub = self.train_epoch(epoch)
            val_loss, val_sam, val_sub = self.evaluate(self.val_loader)
            test_loss, test_sam, test_sub = self.evaluate(self.test_loader)
            self.scheduler.step()

            # ── Terminal summary ─────────────────────────────────────
            lr = self.scheduler.get_last_lr()[0]
            print(_SEP)
            print(
                f"  Epoch {epoch:03d} | LR={lr:.2e} | "
                f"train={tr_loss:.4f}  val={val_loss:.4f}  test={test_loss:.4f}"
            )
            print(f"  Sample  | train: {fmt_metrics(tr_sam)}")
            print(f"  Sample  | val  : {fmt_metrics(val_sam)}")
            print(f"  Sample  | test : {fmt_metrics(test_sam)}")
            if self.use_subject_vote and tr_sub:
                print(f"  Subject | train: {fmt_metrics(tr_sub)}")
                print(f"  Subject | val  : {fmt_metrics(val_sub)}")
                print(f"  Subject | test : {fmt_metrics(test_sub)}")

            # ── Early stopping / checkpoint ──────────────────────────
            # Extract the metric based on configuration
            current_metric = self._extract_metric(val_loss, val_sam, val_sub)
            
            # Check for improvement using the appropriate comparison operator
            is_improved = self._is_improvement(current_metric, best_metric)
            
            if is_improved:
                best_metric = current_metric
                wait = 0
                torch.save(self.model.state_dict(), self.ckpt_path)
                metric_name = self._get_metric_display_name()
                print(f"  ✓ New best saved → {self.ckpt_path}  ({metric_name}={best_metric:.4f})")
            else:
                wait += 1
                suffix = " → early stopping!" if wait >= self.patience else ""
                metric_name = self._get_metric_display_name()
                print(
                    f"  · No improvement {wait}/{self.patience} "
                    f"(best {metric_name}={best_metric:.4f}){suffix}"
                )

            # ── WandB epoch log ──────────────────────────────────────
            if self.wandb_run is not None:
                payload: dict = {
                    "epoch/idx": epoch,
                    "epoch/lr": lr,
                    "epoch/train/loss": tr_loss,
                    "epoch/val/loss": val_loss,
                    "epoch/test/loss": test_loss,
                    f"epoch/best/{self.early_stopping_metric}": best_metric,
                    "epoch/wait": wait,
                    "epoch/improved": int(wait == 0),
                }
                payload.update(prefix_metrics(tr_sam,   "epoch/train/sample"))
                payload.update(prefix_metrics(val_sam,  "epoch/val/sample"))
                payload.update(prefix_metrics(test_sam, "epoch/test/sample"))
                if self.use_subject_vote and tr_sub:
                    payload.update(prefix_metrics(tr_sub,   "epoch/train/subject"))
                    payload.update(prefix_metrics(val_sub,  "epoch/val/subject"))
                    payload.update(prefix_metrics(test_sub, "epoch/test/subject"))
                self.wandb_run.log(payload, step=self.global_step)
                self.wandb_run.summary[f"best_{self.early_stopping_metric}"] = best_metric
                self.wandb_run.summary["best_checkpoint"] = str(self.ckpt_path)

            if wait >= self.patience:
                print(f"\n  Early stopping after {epoch} epochs.")
                break

        # ── Final evaluation with best checkpoint ────────────────────
        if self.ckpt_path.exists():
            self.model.load_state_dict(
                torch.load(self.ckpt_path, map_location=self.device, weights_only=True)
            )
            logger.info(f"Loaded best checkpoint from {self.ckpt_path}")

        _, final_val_sam, final_val_sub, final_val_raw = self.evaluate(
            self.val_loader, return_raw=True
        )
        _, final_test_sam, final_test_sub, final_test_raw = self.evaluate(
            self.test_loader, return_raw=True
        )

        print()
        print("=" * 90)
        print("  FINAL RESULTS (best checkpoint)")
        print(f"  Sample | val : {fmt_metrics(final_val_sam)}")
        print(f"  Sample | test: {fmt_metrics(final_test_sam)}")
        if self.use_subject_vote and final_val_sub:
            print(f"  Subject| val : {fmt_metrics(final_val_sub)}")
            print(f"  Subject| test: {fmt_metrics(final_test_sub)}")
        print("=" * 90)

        # ── Per-subject vote breakdown ───────────────────────────────
        final_raws = {"val": final_val_raw, "test": final_test_raw}
        votes_csv = self.ckpt_path.parent / "final_subject_votes.csv"
        votes_csv.unlink(missing_ok=True)
        for split, raw in final_raws.items():
            if not raw.get("vote_records"):
                continue
            log_vote_report(f"final/{split}", raw["vote_records"], raw["vote_summary"])
            append_vote_records(
                votes_csv, raw["vote_records"],
                strategy="final", model=self.cfg.get("model_name", ""),
                dataset=self.cfg.get("dataset_name", ""),
                task=self.cfg.get("classify_choice", ""),
                split=split, run="final",
            )
        if votes_csv.exists():
            logger.info(f"Final per-subject votes saved → {votes_csv}")

        if self.wandb_run is not None:
            final: dict = {"epoch/idx": self.n_epochs + 1}
            final.update(prefix_metrics(final_val_sam,  "final/val/sample"))
            final.update(prefix_metrics(final_test_sam, "final/test/sample"))
            if self.use_subject_vote and final_val_sub:
                final.update(prefix_metrics(final_val_sub,  "final/val/subject"))
                final.update(prefix_metrics(final_test_sub, "final/test/subject"))
            for split, raw in final_raws.items():
                if not raw.get("vote_records"):
                    continue
                final.update(vote_summary_to_wandb(raw["vote_summary"], f"final/{split}"))
                table = vote_records_to_wandb_table(raw["vote_records"])
                if table is not None:
                    final[f"final/{split}/subject_votes"] = table
            self.wandb_run.log(final, step=self.global_step + 1)
            self.wandb_run.summary.update(final)
            if votes_csv.exists():
                self.wandb_run.summary["subject_votes_csv"] = str(votes_csv)

        return best_metric

    # ------------------------------------------------------------------
    # Helper methods for early stopping
    # ------------------------------------------------------------------

    def _extract_metric(self, val_loss: float, val_sam: dict, val_sub: dict | None) -> float:
        """Extract the metric value based on early_stopping_metric configuration."""
        # Check if metric is "val_loss" (direct loss value)
        if "loss" in self.early_stopping_metric:
            return val_loss

        # Try to extract from sample metrics, then subject metrics. Metric dicts
        # use CamelCase keys ("BalancedAccuracy", "AUPRC"), the config uses lower
        # case ("val_balancedaccuracy"), so match case-insensitively.
        metric_key = self._parse_metric_key(self.early_stopping_metric)
        value = _lookup_ci(val_sam, metric_key)
        if value is None and val_sub is not None:
            value = _lookup_ci(val_sub, metric_key)

        if value is not None:
            if value == -1.0:
                # Undefined on a single-class split (e.g. one LOSO validation subject).
                logger.warning(
                    f"Early-stopping metric '{self.early_stopping_metric}' is undefined "
                    f"(-1.0) on this validation split — it likely holds a single class. "
                    f"Falling back to val_loss for model selection this epoch."
                )
                return -val_loss if self.early_stopping_mode == "maximize" else val_loss
            return value

        # Fallback: return worst possible value
        logger.warning(
            f"Metric '{self.early_stopping_metric}' not found in val_sam={list(val_sam.keys())} "
            f"or val_sub={list(val_sub.keys()) if val_sub else None}. "
            f"Falling back to worst value."
        )
        return float("inf") if self.early_stopping_mode == "minimize" else float("-inf")

    def _is_improvement(self, current: float, best: float) -> bool:
        """Check if current metric value represents an improvement over best."""
        if self.early_stopping_mode == "minimize":
            return current < best
        else:
            return current > best

    def _get_metric_display_name(self) -> str:
        """Get a human-readable name for the early stopping metric."""
        parts = self.early_stopping_metric.split("_")
        return " ".join(parts)  # e.g., "val_loss" -> "val loss", "val_f1" -> "val f1"

    @staticmethod
    def _parse_metric_key(metric_name: str) -> str:
        """Strip the "val_" prefix (e.g. "val_auprc" → "auprc"); matching against
        the metric dicts is case-insensitive (see ``_lookup_ci``).
        """
        if metric_name.startswith("val_"):
            metric_name = metric_name[4:]
        return metric_name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lookup_ci(metrics: dict, key: str):
    """Case-insensitive dict lookup; None when the key is absent."""
    key = key.lower()
    for k, v in metrics.items():
        if k.lower() == key:
            return v
    return None


def _compute_grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm(2).item() ** 2
    return total ** 0.5
