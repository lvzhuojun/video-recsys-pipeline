"""Train the Two-Tower retrieval model and evaluate with Faiss.

Usage (from project root):
    python src/training/train_retrieval.py
    python src/training/train_retrieval.py --neg_mode random

Pipeline:
    1. Load processed data + metadata
    2. Build RetrievalDataset (positive pairs only)
    3. Train TwoTowerModel with in-batch negative InfoNCE loss
    4. Evaluate: encode all items → build Faiss index
                 encode all val users → search → Recall@K, NDCG@K
    5. Load best checkpoint and evaluate on test set
    6. Save final item embeddings for downstream Faiss serving
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.dataset import RankingDataset, RetrievalDataset, load_meta, load_split
from src.evaluation.metrics import compute_retrieval_metrics
from src.models.two_tower import TwoTowerModel
from src.retrieval.faiss_index import FaissIndex
from src.training.trainer import Trainer
from src.utils.gpu_utils import get_device, log_memory_stats, set_seed
from src.utils.logger import get_logger

logger = get_logger(
    __name__, log_file=str(_ROOT / "experiments" / "logs" / "train_retrieval.log")
)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_configs() -> Tuple[dict, dict]:
    def _read(path):
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    base = _read(_ROOT / "configs" / "base_config.yaml")
    ret  = _read(_ROOT / "configs" / "retrieval_config.yaml")
    return base, ret


# ---------------------------------------------------------------------------
# Item / User encoding helpers
# ---------------------------------------------------------------------------

def build_item_feature_lookup(
    *data_dicts: dict,
) -> Dict[int, dict]:
    """Collect item features from one or more data splits.

    Takes the first occurrence of each item_id (features are the same
    across interactions for the same item).

    Returns:
        Dict mapping item_id → {item_dense, item_category, item_dur_bkt}.
    """
    lookup: Dict[int, dict] = {}
    for d in data_dicts:
        for i, vid in enumerate(d["item_ids"]):
            vid = int(vid)
            if vid not in lookup:
                lookup[vid] = {
                    "item_dense":    d["item_dense"][i],
                    "item_category": int(d["item_category"][i]),
                    "item_dur_bkt":  int(d["item_dur_bkt"][i]),
                }
    return lookup


@torch.no_grad()
def encode_all_items(
    model: TwoTowerModel,
    item_lookup: Dict[int, dict],
    device: torch.device,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode every known item through ItemTower.

    Args:
        model: Trained TwoTowerModel.
        item_lookup: Feature dict from ``build_item_feature_lookup``.
        device: Compute device.
        batch_size: Items to encode per forward pass.

    Returns:
        (item_ids, embeddings) — numpy arrays of shape (N,) and (N, D).
    """
    model.eval()
    all_ids = sorted(item_lookup.keys())
    all_embs: List[np.ndarray] = []

    for start in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[start : start + batch_size]
        feat = item_lookup

        item_id_t = torch.tensor(batch_ids, dtype=torch.long, device=device)
        item_dense_t = torch.tensor(
            np.array([feat[i]["item_dense"] for i in batch_ids], dtype=np.float32),
            device=device,
        )
        item_cat_t = torch.tensor(
            [feat[i]["item_category"] for i in batch_ids], dtype=torch.long, device=device
        )
        item_dur_t = torch.tensor(
            [feat[i]["item_dur_bkt"] for i in batch_ids], dtype=torch.long, device=device
        )

        emb = model.encode_item(item_id_t, item_dense_t, item_cat_t, item_dur_t)
        all_embs.append(emb.cpu().numpy())

    return np.array(all_ids, dtype=np.int32), np.vstack(all_embs)


@torch.no_grad()
def encode_users_for_eval(
    model: TwoTowerModel,
    data: dict,
    device: torch.device,
    batch_size: int = 512,
) -> Tuple[List[int], np.ndarray]:
    """Encode unique users from a data split (use last interaction per user).

    Args:
        model: Trained TwoTowerModel.
        data: Processed data dict for one split.
        device: Compute device.
        batch_size: Users per forward pass.

    Returns:
        (user_id_list, user_embeddings) of shapes (U,) and (U, D).
    """
    model.eval()
    # Use the last row per user — most complete sequence history
    user_last: Dict[int, int] = {}
    for i, uid in enumerate(data["user_ids"]):
        user_last[int(uid)] = i

    user_ids_sorted = sorted(user_last.keys())
    indices = [user_last[uid] for uid in user_ids_sorted]
    all_embs: List[np.ndarray] = []

    for start in range(0, len(user_ids_sorted), batch_size):
        batch_idx = indices[start : start + batch_size]
        batch = {
            "user_id":    torch.tensor(data["user_ids"][batch_idx],    dtype=torch.long,    device=device),
            "user_dense": torch.tensor(data["user_dense"][batch_idx],  dtype=torch.float32, device=device),
            "history_seq":torch.tensor(data["history_seqs"][batch_idx],dtype=torch.long,    device=device),
            "history_len":torch.tensor(data["history_lens"][batch_idx],dtype=torch.long,    device=device),
        }
        emb = model.encode_user(batch)
        all_embs.append(emb.cpu().numpy())

    return user_ids_sorted, np.vstack(all_embs)


# ---------------------------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------------------------

def evaluate_retrieval(
    model: TwoTowerModel,
    eval_data: dict,
    item_lookup: Dict[int, dict],
    faiss_cfg: dict,
    k_list: List[int],
    device: torch.device,
    encode_batch: int = 512,
) -> Dict[str, float]:
    """Full retrieval evaluation: encode → index → search → metrics.

    Args:
        model: (Possibly trained) TwoTowerModel in eval mode.
        eval_data: Val or test data dict.
        item_lookup: Item feature dict.
        faiss_cfg: Faiss config section.
        k_list: List of K values for Recall@K etc.
        device: Compute device.
        encode_batch: Batch size for encoding.

    Returns:
        Dict of metric names → float values.
    """
    # 1. Encode all items and build Faiss index
    item_ids_arr, item_embs = encode_all_items(
        model, item_lookup, device, encode_batch
    )
    faiss_idx = FaissIndex(
        dim=item_embs.shape[1],
        index_type=faiss_cfg.get("index_type", "flat"),
        n_lists=faiss_cfg.get("n_lists", 50),
        n_probe=faiss_cfg.get("n_probe", 10),
    )
    faiss_idx.build(item_embs, item_ids_arr)

    # 2. Encode users
    user_ids_list, user_embs = encode_users_for_eval(
        model, eval_data, device, encode_batch
    )

    # 3. Search Faiss for each user
    max_k = max(k_list)
    _, retrieved_ids = faiss_idx.search(user_embs, top_k=max_k)  # (U, max_k)
    user_retrieved = {
        uid: retrieved_ids[i] for i, uid in enumerate(user_ids_list)
    }

    # 4. Build ground-truth: positive items per user
    user_relevant: Dict[int, set] = defaultdict(set)
    for uid, iid, label in zip(
        eval_data["user_ids"], eval_data["item_ids"], eval_data["labels"]
    ):
        if label == 1.0:
            user_relevant[int(uid)].add(int(iid))

    return compute_retrieval_metrics(user_retrieved, user_relevant, k_list)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def retrieval_loss_fn(
    model: TwoTowerModel, batch: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """In-batch InfoNCE loss for the Two-Tower model."""
    user_emb, item_emb = model(batch)
    return model.in_batch_loss(user_emb, item_emb)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(neg_mode: str = "in_batch", seq_model: str = "mean_pool") -> None:
    base_cfg, ret_cfg = _load_configs()

    # Allow CLI to override the seq_model setting from config
    ret_cfg["model"]["seq_model"] = seq_model

    seed = base_cfg["project"]["seed"]
    set_seed(seed)
    device = get_device()
    log_memory_stats()

    proc_dir = _ROOT / base_cfg["data"]["processed_dir"]
    ckpt_dir = _ROOT / base_cfg["logging"]["checkpoint_dir"]
    log_dir  = _ROOT / base_cfg["logging"]["log_dir"]

    # Use seq_model name in checkpoint filename so both variants can coexist
    ckpt_name = f"two_tower_{seq_model}_best.pt"
    ckpt_path = str(ckpt_dir / ckpt_name)

    # Load data
    meta        = load_meta(proc_dir)
    train_data  = load_split(proc_dir, "train")
    val_data    = load_split(proc_dir, "val")
    test_data   = load_split(proc_dir, "test")

    item_lookup = build_item_feature_lookup(train_data, val_data, test_data)

    # Datasets & loaders
    train_ds = RetrievalDataset(train_data, meta, neg_mode=neg_mode, seed=seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=ret_cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=True,  # Required for in-batch: keep batch size constant
    )

    # Model
    model = TwoTowerModel(meta, ret_cfg).to(device)
    logger.info(f"\n{model}")

    # Optimizer + scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=ret_cfg["training"]["lr"],
        weight_decay=ret_cfg["training"]["weight_decay"],
    )
    scheduler = None
    if ret_cfg["training"].get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=ret_cfg["training"]["n_epochs"]
        )

    # TensorBoard — separate run dir per seq_model for clean comparison
    writer = SummaryWriter(log_dir=str(log_dir / f"two_tower_{seq_model}"))

    k_list = ret_cfg["evaluation"]["top_k"]
    faiss_cfg = ret_cfg["faiss"]
    encode_batch = ret_cfg["evaluation"]["encode_batch_size"]

    def eval_fn(trainer: Trainer) -> Dict[str, float]:
        return evaluate_retrieval(
            trainer.model, val_data, item_lookup,
            faiss_cfg, k_list, device, encode_batch,
        )

    # Train
    trainer = Trainer(
        model, optimizer, device,
        scheduler=scheduler, writer=writer,
    )
    trainer.fit(
        train_loader,
        loss_fn=retrieval_loss_fn,
        eval_fn=eval_fn,
        n_epochs=ret_cfg["training"]["n_epochs"],
        monitor=ret_cfg["training"]["monitor"],
        higher_is_better=True,
        patience=ret_cfg["training"]["early_stop_patience"],
        checkpoint_path=ckpt_path,
    )
    writer.close()

    # Final test evaluation
    logger.info("Loading best checkpoint for test evaluation …")
    trainer.load_checkpoint(ckpt_path)
    test_metrics = evaluate_retrieval(
        model, test_data, item_lookup, faiss_cfg, k_list, device, encode_batch,
    )
    logger.info("=" * 50)
    logger.info("Test Set Results")
    logger.info("=" * 50)
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # Save final item embeddings for serving
    emb_dir = _ROOT / "experiments" / "results"
    emb_dir.mkdir(parents=True, exist_ok=True)
    item_ids_arr, item_embs = encode_all_items(
        model, item_lookup, device, encode_batch
    )
    np.save(str(emb_dir / "item_ids.npy"), item_ids_arr)
    np.save(str(emb_dir / "item_embeddings.npy"), item_embs)
    logger.info(f"Item embeddings saved to {emb_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--neg_mode", choices=["in_batch", "random"], default="in_batch",
        help="Negative sampling strategy for retrieval training"
    )
    parser.add_argument(
        "--seq_model", choices=["mean_pool", "sasrec"], default="mean_pool",
        help="Sequence encoder in UserTower: mean_pool (default) or sasrec"
    )
    args = parser.parse_args()
    main(neg_mode=args.neg_mode, seq_model=args.seq_model)
