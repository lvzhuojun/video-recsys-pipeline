"""Train MMoE multi-task ranking model for joint watch_ratio and like prediction.

Usage (from project root):
    python src/training/train_multitask.py

Pipeline:
    1. Load processed data + metadata
    2. Build RankingDataset(mtl_mode=True) for train / val / test
    3. Train with MTL loss: watch_loss + like_loss (weighted)
    4. Evaluate: watch_auc, like_auc, watch_gauc, like_gauc, watch_logloss
    5. Load best checkpoint and report test metrics
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.dataset import RankingDataset, load_meta, load_split
from src.evaluation.metrics import compute_auc, compute_gauc, compute_logloss
from src.models.multitask import MMoE
from src.training.trainer import Trainer
from src.utils.gpu_utils import get_device, log_memory_stats, set_seed
from src.utils.logger import get_logger

logger = get_logger(
    __name__, log_file=str(_ROOT / "experiments" / "logs" / "train_multitask.log")
)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_configs() -> Tuple[dict, dict]:
    def _read(path):
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    return (
        _read(_ROOT / "configs" / "base_config.yaml"),
        _read(_ROOT / "configs" / "multitask_config.yaml"),
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_multitask(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Compute watch_auc, like_auc, watch_gauc, like_gauc, watch_logloss.

    Args:
        model: Trained MMoE model.
        loader: DataLoader for the eval split (mtl_mode=True).
        device: Compute device.

    Returns:
        Dict with keys:
          watch_auc, like_auc, watch_gauc, like_gauc, watch_logloss
    """
    model.eval()

    all_watch_preds, all_like_preds = [], []
    all_labels, all_watch_raw, all_like_labels = [], [], []
    all_uids = []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        watch_pred, like_pred = model(batch)

        all_watch_preds.append(watch_pred.cpu().numpy())
        all_like_preds.append(like_pred.cpu().numpy())
        all_labels.append(batch["label"].cpu().numpy())
        all_watch_raw.append(batch["watch_ratio_raw"].cpu().numpy())
        all_like_labels.append(batch["like_label"].cpu().numpy())
        all_uids.append(batch["user_id"].cpu().numpy())

    watch_preds   = np.concatenate(all_watch_preds)
    like_preds    = np.concatenate(all_like_preds)
    labels        = np.concatenate(all_labels)
    watch_raw     = np.concatenate(all_watch_raw)
    like_labels   = np.concatenate(all_like_labels)
    uids          = np.concatenate(all_uids)

    # Watch AUC/GAUC uses binary label (>=0.7 threshold), logloss also
    watch_auc   = compute_auc(labels, watch_preds)
    watch_gauc  = compute_gauc(uids, labels, watch_preds)
    watch_logloss = compute_logloss(labels, watch_preds)

    # Like AUC/GAUC uses binary like_labels
    like_auc    = compute_auc(like_labels, like_preds)
    like_gauc   = compute_gauc(uids, like_labels, like_preds)

    return {
        "watch_auc":    watch_auc,
        "watch_gauc":   watch_gauc,
        "watch_logloss": watch_logloss,
        "like_auc":     like_auc,
        "like_gauc":    like_gauc,
    }


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def make_mtl_loss_fn(
    watch_loss_weight: float,
    like_loss_weight: float,
    pos_weight: float,
    device: torch.device,
):
    """Return the MTL loss function for MMoE.

    MTL loss = watch_loss_weight * BCE(watch_pred, label)
             + like_loss_weight  * BCE(like_pred,  like_label)

    Both tasks use pos_weight to compensate for class imbalance.

    Args:
        watch_loss_weight: Scalar weight for watch task loss.
        like_loss_weight: Scalar weight for like task loss.
        pos_weight: Positive class weight for weighted BCE.
        device: Compute device.

    Returns:
        loss_fn(model, batch) -> scalar loss
    """
    pw = torch.tensor(pos_weight, device=device)

    def _weighted_bce(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        weights = torch.where(labels == 1.0, pw, torch.ones_like(pw))
        return (
            weights * nn.functional.binary_cross_entropy(preds, labels, reduction="none")
        ).mean()

    def loss_fn(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        watch_pred, like_pred = model(batch)

        # Watch task: use binary label (>= 0.7 threshold)
        watch_loss = _weighted_bce(watch_pred, batch["label"])
        # Like task: use like_label (binary)
        like_loss  = _weighted_bce(like_pred,  batch["like_label"])

        return watch_loss_weight * watch_loss + like_loss_weight * like_loss

    return loss_fn


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train_multitask(
    base_cfg: dict,
    mtl_cfg: dict,
    meta: dict,
    train_data: dict,
    val_data: dict,
    test_data: dict,
    device: torch.device,
    seed: int,
) -> Dict[str, float]:
    """Full MTL training + evaluation for MMoE.

    Returns:
        Test metrics dict.
    """
    logger.info(f"\n{'='*50}")
    logger.info("Training MMoE (Multi-task Learning)")
    logger.info(f"{'='*50}")

    # Datasets with mtl_mode=True to get watch_ratio_raw and like_label
    train_ds = RankingDataset(train_data, meta, mtl_mode=True)
    val_ds   = RankingDataset(val_data,   meta, mtl_mode=True)
    test_ds  = RankingDataset(test_data,  meta, mtl_mode=True)

    tc = mtl_cfg["training"]
    ec = mtl_cfg["evaluation"]

    train_loader = DataLoader(
        train_ds, batch_size=tc["batch_size"], shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=ec["batch_size"], shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=ec["batch_size"], shuffle=False, num_workers=0
    )

    # Model
    model = MMoE(meta, mtl_cfg).to(device)
    logger.info(f"\n{model}")

    # Optimizer + scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=tc["lr"],
        weight_decay=tc["weight_decay"],
    )
    scheduler = None
    if tc.get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tc["n_epochs"]
        )

    log_dir  = _ROOT / base_cfg["logging"]["log_dir"]
    ckpt_dir = _ROOT / base_cfg["logging"]["checkpoint_dir"]
    ckpt_path = str(ckpt_dir / "mmoe_best.pt")

    writer = SummaryWriter(log_dir=str(log_dir / "mmoe"))

    # Compute pos_weight from training data
    n_pos = int((train_data["labels"] == 1.0).sum())
    n_neg = len(train_data["labels"]) - n_pos
    pos_weight = n_neg / max(n_pos, 1)

    loss_fn = make_mtl_loss_fn(
        watch_loss_weight=tc.get("watch_loss_weight", 1.0),
        like_loss_weight=tc.get("like_loss_weight", 0.5),
        pos_weight=pos_weight,
        device=device,
    )

    def eval_fn(trainer: Trainer) -> Dict[str, float]:
        return evaluate_multitask(trainer.model, val_loader, device)

    trainer = Trainer(model, optimizer, device, scheduler=scheduler, writer=writer)
    trainer.fit(
        train_loader,
        loss_fn=loss_fn,
        eval_fn=eval_fn,
        n_epochs=tc["n_epochs"],
        monitor=tc["monitor"],
        higher_is_better=True,
        patience=tc["early_stop_patience"],
        checkpoint_path=ckpt_path,
    )
    writer.close()

    # Test evaluation with best checkpoint
    logger.info("Loading best checkpoint for MMoE test evaluation …")
    trainer.load_checkpoint(ckpt_path)
    test_metrics = evaluate_multitask(model, test_loader, device)

    logger.info("\nMMoE — Test Results")
    logger.info("=" * 40)
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    return test_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    base_cfg, mtl_cfg = _load_configs()
    seed = base_cfg["project"]["seed"]
    set_seed(seed)
    device = get_device()
    log_memory_stats()

    proc_dir = _ROOT / base_cfg["data"]["processed_dir"]
    meta       = load_meta(proc_dir)
    train_data = load_split(proc_dir, "train")
    val_data   = load_split(proc_dir, "val")
    test_data  = load_split(proc_dir, "test")

    results = train_multitask(
        base_cfg, mtl_cfg, meta,
        train_data, val_data, test_data,
        device, seed,
    )

    results_dir = _ROOT / base_cfg["logging"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    import json
    with open(results_dir / "multitask_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_dir / 'multitask_results.json'}")


if __name__ == "__main__":
    main()
