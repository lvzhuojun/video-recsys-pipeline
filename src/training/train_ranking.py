"""Train DeepFM and DIN ranking models and evaluate with AUC/GAUC.

Usage (from project root):
    python src/training/train_ranking.py --model deepfm
    python src/training/train_ranking.py --model din
    python src/training/train_ranking.py --model all   # trains both sequentially

Pipeline:
    1. Load processed data + metadata
    2. Build RankingDataset for train / val / test
    3. Train with weighted BCE loss (handles class imbalance)
    4. Evaluate: AUC, GAUC, LogLoss per epoch
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
from src.models.deepfm import DeepFM
from src.models.din import DIN
from src.training.trainer import Trainer
from src.utils.gpu_utils import get_device, log_memory_stats, set_seed
from src.utils.logger import get_logger

logger = get_logger(
    __name__, log_file=str(_ROOT / "experiments" / "logs" / "train_ranking.log")
)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_configs() -> Tuple[dict, dict]:
    def _read(path):
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    return _read(_ROOT / "configs" / "base_config.yaml"), \
           _read(_ROOT / "configs" / "ranking_config.yaml")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_ranking(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Compute AUC, GAUC, and LogLoss on a data split.

    Args:
        model: Trained ranking model (DeepFM or DIN).
        loader: DataLoader for the eval split.
        device: Compute device.

    Returns:
        Dict with keys ``auc``, ``gauc``, ``logloss``.
    """
    model.eval()
    all_labels, all_scores, all_uids = [], [], []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        scores = model(batch).cpu().numpy()
        all_scores.append(scores)
        all_labels.append(batch["label"].cpu().numpy())
        all_uids.append(batch["user_id"].cpu().numpy())

    labels = np.concatenate(all_labels)
    scores = np.concatenate(all_scores)
    uids   = np.concatenate(all_uids)

    return {
        "auc":     compute_auc(labels, scores),
        "gauc":    compute_gauc(uids, labels, scores),
        "logloss": compute_logloss(labels, scores),
    }


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def make_ranking_loss_fn(pos_weight: float, device: torch.device):
    """Return a weighted BCE loss function for ranking.

    Weighting positive samples compensates for the ~14% positive rate.

    Args:
        pos_weight: Weight for positive samples (≈ neg/pos ratio).
        device: Compute device for the weight tensor.
    """
    criterion = nn.BCELoss(
        weight=None  # per-sample weight applied below
    )
    pw = torch.tensor(pos_weight, device=device)

    def loss_fn(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        preds = model(batch)              # (B,)
        labels = batch["label"]           # (B,)
        # Apply pos_weight: positives are penalised more
        weights = torch.where(labels == 1.0, pw, torch.ones_like(pw))
        loss = (weights * nn.functional.binary_cross_entropy(preds, labels, reduction="none")).mean()
        return loss

    return loss_fn


# ---------------------------------------------------------------------------
# Train one model
# ---------------------------------------------------------------------------

def train_model(
    model_name: str,
    base_cfg: dict,
    rank_cfg: dict,
    meta: dict,
    train_data: dict,
    val_data: dict,
    test_data: dict,
    device: torch.device,
    seed: int,
) -> Dict[str, float]:
    """Full training + evaluation for a single ranking model.

    Args:
        model_name: ``"deepfm"`` or ``"din"``.

    Returns:
        Test metrics dict.
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"Training {model_name.upper()}")
    logger.info(f"{'='*50}")

    # Datasets
    train_ds = RankingDataset(train_data, meta)
    val_ds   = RankingDataset(val_data,   meta)
    test_ds  = RankingDataset(test_data,  meta)

    tc = rank_cfg["training"]
    ec = rank_cfg["evaluation"]

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
    if model_name == "deepfm":
        model = DeepFM(meta, rank_cfg).to(device)
    elif model_name == "din":
        model = DIN(meta, rank_cfg).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")
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

    log_dir = _ROOT / base_cfg["logging"]["log_dir"]
    ckpt_dir = _ROOT / base_cfg["logging"]["checkpoint_dir"]
    ckpt_path = str(ckpt_dir / f"{model_name}_best.pt")

    writer = SummaryWriter(log_dir=str(log_dir / model_name))
    loss_fn = make_ranking_loss_fn(tc["pos_weight"], device)

    def eval_fn(trainer: Trainer) -> Dict[str, float]:
        return evaluate_ranking(trainer.model, val_loader, device)

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
    logger.info(f"Loading best checkpoint for {model_name} test evaluation …")
    trainer.load_checkpoint(ckpt_path)
    test_metrics = evaluate_ranking(model, test_loader, device)

    logger.info(f"\n{model_name.upper()} — Test Results")
    logger.info("=" * 40)
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    return test_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(model_arg: str = "all") -> None:
    base_cfg, rank_cfg = _load_configs()
    seed = base_cfg["project"]["seed"]
    set_seed(seed)
    device = get_device()
    log_memory_stats()

    proc_dir = _ROOT / base_cfg["data"]["processed_dir"]
    meta       = load_meta(proc_dir)
    train_data = load_split(proc_dir, "train")
    val_data   = load_split(proc_dir, "val")
    test_data  = load_split(proc_dir, "test")

    models_to_train = ["deepfm", "din"] if model_arg == "all" else [model_arg]
    results = {}

    for name in models_to_train:
        results[name] = train_model(
            name, base_cfg, rank_cfg, meta,
            train_data, val_data, test_data,
            device, seed,
        )

    logger.info("\n" + "=" * 50)
    logger.info("Final Comparison")
    logger.info("=" * 50)
    for model_name, metrics in results.items():
        m_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        logger.info(f"{model_name.upper():10s}: {m_str}")

    # Save summary
    results_dir = _ROOT / base_cfg["logging"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    import json
    with open(results_dir / "ranking_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_dir / 'ranking_results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["deepfm", "din", "all"], default="all",
        help="Which ranking model to train"
    )
    args = parser.parse_args()
    main(model_arg=args.model)
