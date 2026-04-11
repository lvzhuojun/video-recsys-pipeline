"""Generic Trainer: training loop, early stopping, checkpointing, TensorBoard."""

from pathlib import Path
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    """Model-agnostic trainer with early stopping and TensorBoard support.

    The Trainer delegates all loss computation to a user-supplied
    ``loss_fn`` and all evaluation to a user-supplied ``eval_fn``.  This
    keeps the Trainer reusable for both retrieval (Two-Tower) and ranking
    (DeepFM / DIN) models without any subclassing.

    Args:
        model: PyTorch model to train.
        optimizer: Optimiser instance (e.g. ``torch.optim.Adam``).
        device: ``torch.device`` to move tensors to.
        scheduler: Optional LR scheduler called once per epoch.
        writer: Optional ``SummaryWriter`` for TensorBoard logging.
        grad_clip: Max gradient L2 norm for clipping (0 = disabled).

    Example:
        >>> trainer = Trainer(model, optimizer, device)
        >>> trainer.fit(train_loader, loss_fn, eval_fn,
        ...             n_epochs=20, checkpoint_path="experiments/checkpoints/best.pt")
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: Optional[object] = None,
        writer: Optional[object] = None,
        grad_clip: float = 1.0,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.writer = writer
        self.grad_clip = grad_clip
        self.global_step = 0
        self.history: list = []   # per-epoch metrics: [{epoch, train_loss, ...}, ...]

    # ------------------------------------------------------------------
    # Core training step
    # ------------------------------------------------------------------

    def train_one_epoch(
        self,
        loader: DataLoader,
        loss_fn: Callable[[nn.Module, Dict[str, torch.Tensor]], torch.Tensor],
    ) -> float:
        """Run one full pass over ``loader`` and return mean batch loss.

        Args:
            loader: DataLoader yielding dict batches.
            loss_fn: Callable ``(model, batch) -> scalar loss``.

        Returns:
            Average loss across all batches in the epoch.
        """
        self.model.train()
        total_loss = 0.0

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.optimizer.zero_grad()
            loss = loss_fn(self.model, batch)
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            total_loss += loss.item()
            self.global_step += 1

        return total_loss / max(len(loader), 1)

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        loss_fn: Callable,
        eval_fn: Callable[["Trainer"], Dict[str, float]],
        n_epochs: int = 20,
        monitor: str = "loss",
        higher_is_better: bool = True,
        patience: int = 5,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, float]:
        """Train model with early stopping, checkpointing, and TensorBoard.

        Args:
            train_loader: Training DataLoader.
            loss_fn: ``(model, batch) -> scalar loss``.
            eval_fn: ``(trainer) -> dict`` returning validation metrics.
                     Must include the key specified by ``monitor``.
            n_epochs: Maximum training epochs.
            monitor: Metric key to watch for early stopping.
            higher_is_better: True for recall/AUC; False for loss.
            patience: Epochs without improvement before early stopping.
            checkpoint_path: Where to save the best model.

        Returns:
            Dict of best validation metrics.
        """
        sentinel = -float("inf") if higher_is_better else float("inf")
        best_metric = sentinel
        best_metrics: Dict[str, float] = {}
        no_improve = 0

        logger.info(
            f"Training start | device={self.device} | epochs={n_epochs} | "
            f"monitor={monitor} | patience={patience}"
        )

        for epoch in range(1, n_epochs + 1):
            train_loss = self.train_one_epoch(train_loader, loss_fn)
            metrics = eval_fn(self)
            metrics["train_loss"] = train_loss
            self.history.append({"epoch": epoch, **metrics})

            if self.writer:
                self.writer.add_scalar("train/loss", train_loss, epoch)
                for k, v in metrics.items():
                    if k != "train_loss":
                        self.writer.add_scalar(f"val/{k}", v, epoch)

            current = metrics.get(monitor, sentinel)
            improved = current > best_metric if higher_is_better else current < best_metric

            if improved:
                best_metric = current
                best_metrics = dict(metrics)
                no_improve = 0
                if checkpoint_path:
                    self._save_checkpoint(checkpoint_path, epoch, metrics)
                marker = "*"
            else:
                no_improve += 1
                marker = ""

            m_str = "  ".join(
                f"{k}={v:.4f}" for k, v in metrics.items() if k != "train_loss"
            )
            logger.info(
                f"Epoch {epoch:3d}/{n_epochs} | loss={train_loss:.4f} | "
                f"{m_str} {marker}"
            )

            if self.scheduler is not None:
                self.scheduler.step()

            if no_improve >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

        logger.info(
            f"Training complete | best {monitor}={best_metric:.4f}"
        )
        return best_metrics

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self, path: str, epoch: int, metrics: Dict[str, float]
    ) -> None:
        """Save model + optimiser state to disk.

        Args:
            path: File path for the checkpoint (.pt).
            epoch: Current epoch number (stored for reference).
            metrics: Validation metrics at checkpoint time.
        """
        ckpt_path = Path(path)
        try:
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "metrics": metrics,
                },
                ckpt_path,
            )
            logger.info(f"Checkpoint saved -> {ckpt_path}")
        except OSError as e:
            logger.error(f"Could not save checkpoint: {e}")

    def load_checkpoint(self, path: str) -> Dict:
        """Load model + optimiser state from a checkpoint file.

        Args:
            path: Path to the .pt checkpoint file.

        Returns:
            The full checkpoint dict (epoch, metrics, etc.).
        """
        ckpt_path = Path(path)
        try:
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            logger.info(
                f"Checkpoint loaded from {ckpt_path} "
                f"(epoch {ckpt.get('epoch', '?')})"
            )
            return ckpt
        except (OSError, KeyError) as e:
            logger.error(f"Could not load checkpoint: {e}")
            raise
