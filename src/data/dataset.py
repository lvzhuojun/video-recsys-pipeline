"""PyTorch Dataset classes for the two-stage recommendation pipeline.

Two datasets are provided:
- RetrievalDataset  – used to train the Two-Tower recall model.
  Supports both random negative sampling and in-batch negative mode.
- RankingDataset    – used to train DeepFM / DIN ranking models.
  Returns all features + binary label.
"""

import pickle
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_split(processed_dir: Path, name: str) -> Dict[str, np.ndarray]:
    """Load a processed data split from disk.

    Args:
        processed_dir: Path to data/processed/.
        name: One of ``"train"``, ``"val"``, ``"test"``.

    Returns:
        Dict mapping feature names to numpy arrays.
    """
    path = processed_dir / f"{name}_data.pkl"
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        logger.error(f"Processed data not found at {path}. Run download_data.py first.")
        raise


def load_meta(processed_dir: Path) -> dict:
    """Load dataset metadata (vocab sizes, feature dims).

    Args:
        processed_dir: Path to data/processed/.

    Returns:
        Metadata dict as saved by FeatureEngineer.
    """
    path = processed_dir / "meta.pkl"
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        logger.error(f"Metadata not found at {path}. Run download_data.py first.")
        raise


# ---------------------------------------------------------------------------
# Retrieval Dataset  (Two-Tower training)
# ---------------------------------------------------------------------------

class RetrievalDataset(Dataset):
    """Dataset for training the Two-Tower dual-encoder recall model.

    Each sample is a (user, positive_item) pair.  The ``neg_mode`` argument
    controls how negatives are handled:

    - ``"in_batch"``   – only positive pairs are returned; the training loop
      treats other items in the same batch as negatives (efficient, realistic).
    - ``"random"``     – one random negative item is sampled per positive pair
      and returned alongside it (simpler, used for ablation comparisons).

    Args:
        data: Dict of numpy arrays as produced by FeatureEngineer.
        meta: Dataset metadata dict.
        neg_mode: ``"in_batch"`` (default) or ``"random"``.
        seed: RNG seed for random negative sampling.

    Returns (per __getitem__):
        Dict with keys:
          user_id, user_dense,
          pos_item_id, pos_item_dense, pos_item_category, pos_item_dur_bkt,
          history_seq, history_len
          [+ neg_* keys when neg_mode == "random"]
    """

    def __init__(
        self,
        data: Dict[str, np.ndarray],
        meta: dict,
        neg_mode: str = "in_batch",
        seed: int = 42,
    ) -> None:
        assert neg_mode in ("in_batch", "random"), \
            f"neg_mode must be 'in_batch' or 'random', got '{neg_mode}'"
        self.data = data
        self.meta = meta
        self.neg_mode = neg_mode
        self.n_items = meta["n_items"]
        self._rng = np.random.default_rng(seed)

        # Only keep positive interactions for the retrieval task
        pos_mask = data["labels"] == 1.0
        self._indices = np.where(pos_mask)[0]
        logger.info(
            f"RetrievalDataset [{neg_mode}] | "
            f"positive pairs: {len(self._indices):,} / {len(data['labels']):,}"
        )

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        i = self._indices[idx]
        d = self.data

        sample = {
            "user_id":          torch.tensor(d["user_ids"][i],     dtype=torch.long),
            "user_dense":       torch.tensor(d["user_dense"][i],   dtype=torch.float32),
            "pos_item_id":      torch.tensor(d["item_ids"][i],     dtype=torch.long),
            "pos_item_dense":   torch.tensor(d["item_dense"][i],   dtype=torch.float32),
            "pos_item_category":torch.tensor(d["item_category"][i],dtype=torch.long),
            "pos_item_dur_bkt": torch.tensor(d["item_dur_bkt"][i], dtype=torch.long),
            "history_seq":      torch.tensor(d["history_seqs"][i], dtype=torch.long),
            "history_len":      torch.tensor(d["history_lens"][i], dtype=torch.long),
        }

        if self.neg_mode == "random":
            neg_id = self._sample_random_negative(int(d["user_ids"][i]))
            sample.update({
                "neg_item_id":      torch.tensor(neg_id,                          dtype=torch.long),
                "neg_item_dense":   torch.zeros(self.meta["item_dense_dim"],      dtype=torch.float32),
                "neg_item_category":torch.tensor(0,                               dtype=torch.long),
                "neg_item_dur_bkt": torch.tensor(0,                               dtype=torch.long),
            })
        return sample

    def _sample_random_negative(self, user_id: int) -> int:
        """Uniformly sample a random item id (not guaranteed non-interacted,
        but collision probability is low with large item catalogues)."""
        return int(self._rng.integers(0, self.n_items))

    def __repr__(self) -> str:
        return (
            f"RetrievalDataset("
            f"n_positive={len(self._indices):,}, "
            f"neg_mode='{self.neg_mode}', "
            f"n_items={self.n_items:,})"
        )


# ---------------------------------------------------------------------------
# Ranking Dataset  (DeepFM / DIN training)
# ---------------------------------------------------------------------------

class RankingDataset(Dataset):
    """Dataset for training CTR ranking models (DeepFM / DIN).

    Returns all user and item features together with a binary label.

    Args:
        data: Dict of numpy arrays as produced by FeatureEngineer.
        meta: Dataset metadata dict.

    Returns (per __getitem__):
        Dict with keys:
          user_id, item_id, user_dense, item_dense,
          item_category, item_dur_bkt,
          history_seq, history_len, label
    """

    def __init__(self, data: Dict[str, np.ndarray], meta: dict, mtl_mode: bool = False) -> None:
        self.data = data
        self.meta = meta
        self.mtl_mode = mtl_mode
        n_pos = int((data["labels"] == 1.0).sum())
        logger.info(
            f"RankingDataset | "
            f"samples: {len(data['labels']):,}  "
            f"positive: {n_pos:,}  "
            f"pos_rate: {n_pos/len(data['labels'])*100:.1f}%"
            + (f"  [MTL mode]" if mtl_mode else "")
        )

    def __len__(self) -> int:
        return len(self.data["labels"])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        d = self.data
        sample = {
            "user_id":       torch.tensor(d["user_ids"][idx],      dtype=torch.long),
            "item_id":       torch.tensor(d["item_ids"][idx],      dtype=torch.long),
            "user_dense":    torch.tensor(d["user_dense"][idx],    dtype=torch.float32),
            "item_dense":    torch.tensor(d["item_dense"][idx],    dtype=torch.float32),
            "item_category": torch.tensor(d["item_category"][idx], dtype=torch.long),
            "item_dur_bkt":  torch.tensor(d["item_dur_bkt"][idx],  dtype=torch.long),
            "history_seq":   torch.tensor(d["history_seqs"][idx],  dtype=torch.long),
            "history_len":   torch.tensor(d["history_lens"][idx],  dtype=torch.long),
            "label":         torch.tensor(d["labels"][idx],        dtype=torch.float32),
        }
        if self.mtl_mode:
            sample["watch_ratio_raw"] = torch.tensor(
                d["watch_ratio_raw"][idx], dtype=torch.float32
            )
            sample["like_label"] = torch.tensor(
                d["like_labels"][idx], dtype=torch.float32
            )
        return sample

    def __repr__(self) -> str:
        n = len(self.data["labels"])
        pos = int((self.data["labels"] == 1.0).sum())
        return (
            f"RankingDataset("
            f"n_samples={n:,}, "
            f"pos_rate={pos/n*100:.1f}%)"
        )
