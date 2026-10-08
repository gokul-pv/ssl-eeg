"""Pretraining engine for the BrainLM-EEG masked autoencoders."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)

_SEP = "─" * 90


# ---------------------------------------------------------------------------
# LR schedule helper
# ---------------------------------------------------------------------------


def _get_cosine_with_warmup_schedule(
    optimizer: optim.Optimizer,
    warmup_epochs: int,
    n_epochs: int,
) -> LambdaLR:
    """Cosine decay with linear warmup."""
    import math

    def _lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(
            max(1, n_epochs - warmup_epochs)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, _lr_lambda)


# ---------------------------------------------------------------------------
# Gradient norm helper
# ---------------------------------------------------------------------------


def _compute_grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm(2).item() ** 2
    return total**0.5


# ---------------------------------------------------------------------------
# PretrainTrainer
# ---------------------------------------------------------------------------


class PretrainTrainer:
    """Pretraining trainer for the BrainLM-EEG variants."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        cfg: dict,
        device: torch.device,
        val_loader=None,
        wandb_run=None,
        probe_loader_fn=None,
        common_ch_names: list[str] | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.wandb_run = wandb_run
        self.probe_loader_fn = probe_loader_fn
        self.common_ch_names: list[str] = common_ch_names or []

        # ── Training hyperparameters ──────────────────────────────────────
        self.n_epochs = int(cfg.get("train_epochs", 200))
        self.patience = int(cfg.get("patience", 30))
        self.log_interval = int(cfg.get("log_interval", 100))
        self.log_batch = bool(cfg.get("log_batch", False))
        self.max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
        self.global_step = 0

        # ── Optimizer ────────────────────────────────────────────────────
        # Linear scaling rule (MAE): lr = base_lr × batch_size / 256.
        base_lr = float(cfg.get("base_lr", cfg.get("learning_rate", 1.5e-4)))
        batch_size = int(cfg.get("batch_size", 256))
        lr = base_lr * batch_size / 256
        weight_decay = float(cfg.get("weight_decay", 0.05))

        # Weight decay exclusion: do NOT apply WD to biases or LayerNorm params
        # (MAE paper Appendix A; standard ViT/MAE practice).
        no_wd_keywords = ("bias", "norm")
        decay_params = [
            p for n, p in model.named_parameters()
            if not any(kw in n for kw in no_wd_keywords) and p.requires_grad
        ]
        no_decay_params = [
            p for n, p in model.named_parameters()
            if any(kw in n for kw in no_wd_keywords) and p.requires_grad
        ]
        self.optimizer = optim.AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=(0.9, 0.95),
        )

        # ── Mixed precision (AMP) ─────────────────────────────────────────
        use_amp = bool(cfg.get("use_amp", False)) and torch.cuda.is_available()
        self.use_amp = use_amp
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        # ── LR schedule ──────────────────────────────────────────────────
        warmup = int(cfg.get("warmup_epochs", 40))
        if bool(cfg.get("use_lr_schedule", True)):
            self.scheduler = _get_cosine_with_warmup_schedule(
                self.optimizer, warmup, self.n_epochs
            )
        else:
            # Constant LR — identity schedule, no warmup
            warmup = 0
            self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1.0)

        # ── Checkpointing ────────────────────────────────────────────────
        save_dir = Path(cfg.get("save_dir", "outputs/checkpoints/pretrain/mae"))
        save_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_full = save_dir / "best_pretrain_full.pth"
        self.ckpt_encoder = save_dir / "best_pretrain_encoder.pth"

        # ── Linear probe ─────────────────────────────────────────────────
        self.linear_probe = bool(cfg.get("linear_probe", False))
        self.probe_interval = int(cfg.get("probe_interval", 10))  # every N epochs
        if self.linear_probe and probe_loader_fn is None:
            logger.warning(
                "linear_probe=True but no probe_loader_fn provided. "
                "Linear probe evaluation will be skipped."
            )
            self.linear_probe = False

        logger.info(
            f"PretrainTrainer | epochs={self.n_epochs} | warmup={warmup} | "
            f"base_lr={base_lr:.2e} | effective_lr={lr:.2e} (×{batch_size}/256) | "
            f"wd={weight_decay} (excl. bias/norm) | amp={use_amp} | "
            f"patience={self.patience} | linear_probe={self.linear_probe}"
        )

    # -------------------------------------------------------------------------
    # Training epoch
    # -------------------------------------------------------------------------

    def _forward(self, batch):
        """Model forward on one pretraining batch ``(X,)`` or ``(X, X_unnormalized)``."""
        X = batch[0].float().to(self.device)                    # (B, C, T)
        if len(batch) > 1:                                      # BrainLM-EEG-Microstate
            return self.model(X, X_unnormalized=batch[1].float().to(self.device))
        return self.model(X)

    def train_epoch(
        self, epoch: int
    ) -> tuple[float, float, float | None, float | None, float | None]:
        """Run one training epoch."""
        self.model.train()
        losses: list[float] = []
        gnorms: list[float] = []
        recon_losses: list[float] = []
        aux_losses: list[float] = []
        aux_accs: list[float] = []
        n_batches = len(self.train_loader)

        print()
        print(_SEP)
        print(f"  Epoch {epoch:03d}/{self.n_epochs:03d}")
        print(_SEP)

        for batch_idx, batch in enumerate(self.train_loader, start=1):
            self.global_step += 1
            self.optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                loss, _pred, _mask = self._forward(batch)       # scalar, (B,N,D), (B,N)
            self.scaler.scale(loss).backward()
            # Compute true pre-clip grad norm (unscale first when AMP is on)
            self.scaler.unscale_(self.optimizer)
            gnorm = _compute_grad_norm(self.model)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            losses.append(loss.item())
            gnorms.append(gnorm)
            recon = getattr(self.model, "last_recon_loss", None)
            aux = getattr(self.model, "last_aux_loss", None)
            aux_acc = getattr(self.model, "last_aux_acc", None)
            if recon is not None:
                recon_losses.append(recon.item())
            if aux is not None:
                aux_losses.append(aux.item())
            if aux_acc is not None:
                aux_accs.append(aux_acc.item())

            should_log = (
                batch_idx == 1
                or batch_idx % self.log_interval == 0
                or batch_idx == n_batches
            )
            if should_log:
                run_loss = float(np.mean(losses))
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"  [{batch_idx:04d}/{n_batches:04d}] "
                    f"loss={loss.item():.4f}  run_loss={run_loss:.4f}  "
                    f"grad={gnorm:.3f}  lr={lr:.2e}"
                )

                if self.log_batch and self.wandb_run is not None:
                    self.wandb_run.log(
                        {
                            "pretrain/batch/loss": loss.item(),
                            "pretrain/batch/run_loss": run_loss,
                            "pretrain/batch/grad_norm": gnorm,
                            "pretrain/batch/lr": lr,
                            "pretrain/batch/global_step": self.global_step,
                        },
                        step=self.global_step,
                    )

        epoch_loss = float(np.mean(losses))
        epoch_gnorm = float(np.mean(gnorms))
        epoch_recon = float(np.mean(recon_losses)) if recon_losses else None
        epoch_aux = float(np.mean(aux_losses)) if aux_losses else None
        epoch_aux_acc = float(np.mean(aux_accs)) if aux_accs else None
        return epoch_loss, epoch_gnorm, epoch_recon, epoch_aux, epoch_aux_acc

    # -------------------------------------------------------------------------
    # Evaluation (reconstruction loss on val set)
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, loader) -> tuple[float, float | None, float | None, float | None]:
        """Compute mean total/recon/aux loss and aux accuracy over a DataLoader."""
        self.model.eval()
        losses: list[float] = []
        recon_losses: list[float] = []
        aux_losses: list[float] = []
        aux_accs: list[float] = []
        for batch in loader:
            loss, _, _ = self._forward(batch)
            losses.append(loss.item())
            recon = getattr(self.model, "last_recon_loss", None)
            aux = getattr(self.model, "last_aux_loss", None)
            aux_acc = getattr(self.model, "last_aux_acc", None)
            if recon is not None:
                recon_losses.append(recon.item())
            if aux is not None:
                aux_losses.append(aux.item())
            if aux_acc is not None:
                aux_accs.append(aux_acc.item())
        mean_loss = float(np.mean(losses)) if losses else 0.0
        mean_recon = float(np.mean(recon_losses)) if recon_losses else None
        mean_aux = float(np.mean(aux_losses)) if aux_losses else None
        mean_aux_acc = float(np.mean(aux_accs)) if aux_accs else None
        return mean_loss, mean_recon, mean_aux, mean_aux_acc

    # -------------------------------------------------------------------------
    # Linear probe evaluation
    # -------------------------------------------------------------------------

    def _run_linear_probe(self, epoch: int) -> dict[str, float] | None:
        """Freeze the MAE encoder, train a linear head on a labelled dataset,
        and return classification metrics.
        """
        if self.probe_loader_fn is None:
            return None

        from ..utils.metrics import compute_metrics

        logger.info(f"  [Probe] Running linear probe at epoch {epoch}...")

        train_loader, val_loader, n_classes = self.probe_loader_fn()
        encoder_dim = getattr(self.model, "encoder_dim", self.cfg.get("encoder_dim", 512))
        probe_epochs = int(self.cfg.get("probe_epochs", 30))
        probe_lr = float(self.cfg.get("probe_lr", 1e-3))

        # Simple linear head
        probe_head = nn.Linear(encoder_dim, n_classes).to(self.device)
        probe_opt = optim.Adam(probe_head.parameters(), lr=probe_lr)
        criterion = nn.CrossEntropyLoss()

        self.model.eval()
        probe_head.train()
        for _ in range(probe_epochs):
            for batch in train_loader:
                X, labels, _sids = batch
                X = X.float().to(self.device)
                labels = labels.long().to(self.device)
                # Encode (no mask)
                with torch.no_grad():
                    latent, _, _ = self.model.encode(X, mask_ratio=0.0)
                    features = (self.model.get_features(latent)
                                if hasattr(self.model, "get_features")
                                else latent.mean(dim=1))
                probe_opt.zero_grad()
                logits = probe_head(features)
                loss = criterion(logits, labels)
                loss.backward()
                probe_opt.step()

        # Evaluate
        probe_head.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                X, labels, _sids = batch
                X = X.float().to(self.device)
                latent, _, _ = self.model.encode(X, mask_ratio=0.0)
                features = (self.model.get_features(latent)
                            if hasattr(self.model, "get_features")
                            else latent.mean(dim=1))
                logits = probe_head(features)
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        cat_logits = torch.cat(all_logits)
        cat_labels = torch.cat(all_labels)
        metrics, _ = compute_metrics(cat_logits, cat_labels)
        logger.info(f"  [Probe] epoch={epoch}: {metrics}")
        return metrics

    # -------------------------------------------------------------------------
    # Save checkpoints
    # -------------------------------------------------------------------------

    def _save_checkpoints(self, epoch: int, best_val_loss: float) -> None:
        """Save both the full model and the encoder-only state dict with metadata."""
        meta = {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "common_ch_names": self.common_ch_names,
        }
        # Full model (encoder + decoder — for resuming pretraining)
        torch.save({"state_dict": self.model.state_dict(), **meta}, self.ckpt_full)

        # Encoder only (for supervised fine-tuning)
        _encoder_prefixes = getattr(
            self.model,
            "encoder_key_prefixes",
            ("patch_embed", "encoder_blocks", "encoder_norm"),
        )
        encoder_state = {
            k: v for k, v in self.model.state_dict().items()
            if k.startswith(_encoder_prefixes)
        }
        torch.save({"state_dict": encoder_state, **meta}, self.ckpt_encoder)

    # -------------------------------------------------------------------------
    # Main training loop
    # -------------------------------------------------------------------------

    def train(self) -> float:
        """Run the full pretraining loop."""
        best_val_loss = float("inf")
        wait = 0

        for epoch in range(1, self.n_epochs + 1):
            tr_loss, tr_gnorm, tr_recon, tr_aux, tr_aux_acc = self.train_epoch(epoch)
            self.scheduler.step()
            lr = self.scheduler.get_last_lr()[0]

            # ── Validation loss ──────────────────────────────────────────
            if self.val_loader is not None:
                val_loss, val_recon, val_aux, val_aux_acc = self.evaluate(self.val_loader)
                stopping_loss = val_loss
            else:
                val_loss, val_recon, val_aux, val_aux_acc = None, None, None, None
                stopping_loss = tr_loss

            # ── Terminal summary ─────────────────────────────────────────
            print(_SEP)
            tr_components = (
                f" (recon={tr_recon:.4f} aux={tr_aux:.4f}"
                + (f" aux_acc={tr_aux_acc:.3f}" if tr_aux_acc is not None else "")
                + ")"
                if tr_recon is not None and tr_aux is not None
                else ""
            )
            val_components = (
                f" (recon={val_recon:.4f} aux={val_aux:.4f}"
                + (f" aux_acc={val_aux_acc:.3f}" if val_aux_acc is not None else "")
                + ")"
                if val_recon is not None and val_aux is not None
                else ""
            )
            if val_loss is not None:
                print(
                    f"  Epoch {epoch:03d} | LR={lr:.2e} | "
                    f"train_loss={tr_loss:.4f}{tr_components}  "
                    f"val_loss={val_loss:.4f}{val_components}  "
                    f"grad_norm={tr_gnorm:.3f}"
                )
            else:
                print(
                    f"  Epoch {epoch:03d} | LR={lr:.2e} | "
                    f"train_loss={tr_loss:.4f}{tr_components}  grad_norm={tr_gnorm:.3f}"
                )

            # ── Early stopping ───────────────────────────────────────────
            if stopping_loss < best_val_loss:
                best_val_loss = stopping_loss
                wait = 0
                self._save_checkpoints(epoch=epoch, best_val_loss=best_val_loss)
                print(
                    f"  ✓ New best saved → {self.ckpt_full}  "
                    f"(val_loss={best_val_loss:.4f})"
                )
            else:
                wait += 1
                suffix = " → early stopping!" if wait >= self.patience else ""
                print(
                    f"  · No improvement {wait}/{self.patience} "
                    f"(best val_loss={best_val_loss:.4f}){suffix}"
                )

            # ── Optional linear probe ────────────────────────────────────
            probe_metrics = None
            if self.linear_probe and epoch % self.probe_interval == 0:
                probe_metrics = self._run_linear_probe(epoch)

            # ── WandB logging ────────────────────────────────────────────
            if self.wandb_run is not None:
                payload: dict = {
                    "pretrain/epoch/idx": epoch,
                    "pretrain/epoch/lr": lr,
                    "pretrain/epoch/train_loss": tr_loss,
                    "pretrain/epoch/grad_norm": tr_gnorm,
                    "pretrain/epoch/best_val_loss": best_val_loss,
                    "pretrain/epoch/wait": wait,
                    "pretrain/epoch/improved": int(wait == 0),
                }
                if val_loss is not None:
                    payload["pretrain/epoch/val_loss"] = val_loss
                if tr_recon is not None:
                    payload["pretrain/epoch/train_recon_loss"] = tr_recon
                if tr_aux is not None:
                    payload["pretrain/epoch/train_aux_loss"] = tr_aux
                if tr_aux_acc is not None:
                    payload["pretrain/epoch/train_aux_acc"] = tr_aux_acc
                if val_recon is not None:
                    payload["pretrain/epoch/val_recon_loss"] = val_recon
                if val_aux is not None:
                    payload["pretrain/epoch/val_aux_loss"] = val_aux
                if val_aux_acc is not None:
                    payload["pretrain/epoch/val_aux_acc"] = val_aux_acc
                if probe_metrics is not None:
                    for k, v in probe_metrics.items():
                        payload[f"pretrain/probe/{k.lower()}"] = v
                self.wandb_run.log(payload, step=self.global_step)
                self.wandb_run.summary["pretrain_best_val_loss"] = best_val_loss
                self.wandb_run.summary["pretrain_best_encoder_ckpt"] = str(
                    self.ckpt_encoder
                )

            if wait >= self.patience:
                print(f"\n  Early stopping after {epoch} epochs.")
                break

        # ── Final summary ────────────────────────────────────────────────
        print()
        print("=" * 90)
        print("  PRETRAINING COMPLETE")
        print(f"  Best val_loss       : {best_val_loss:.4f}")
        print(f"  Full checkpoint     : {self.ckpt_full}")
        print(f"  Encoder checkpoint  : {self.ckpt_encoder}")
        print("=" * 90)

        return best_val_loss
