"""Feature engineering pipeline for the video recommendation system.

Reads raw KuaiRec-schema interactions and outputs numpy arrays ready for
PyTorch datasets.  All statistics are computed on training data only to
prevent data leakage into validation / test splits.

Processed output schema (saved per split as dict-of-arrays .pkl):
    user_ids          int32  (N,)       user index for embedding lookup
    item_ids          int32  (N,)       video index for embedding lookup
    item_category     int32  (N,)       category index for embedding lookup
    item_duration_bkt int32  (N,)       duration bucket [0-4] for embedding
    user_dense        float32 (N, 25)   activity stats + category prefs
    item_dense        float32 (N, 3)    hist_ctr + like_rate + log_popularity
    history_seqs      int32  (N, L)     past L video_ids (1-indexed; 0=pad)
    history_lens      int32  (N,)       actual sequence length (<=L)
    labels            float32 (N,)      1 if watch_ratio >= 0.7 else 0
"""

import pickle
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Duration bucket boundaries in seconds
_DURATION_BINS = [15, 30, 60, 120]   # 5 buckets: <15, 15-30, 30-60, 60-120, >120


class FeatureEngineer:
    """End-to-end feature computation pipeline.

    Args:
        config: Parsed base_config.yaml dict.

    Example:
        >>> engineer = FeatureEngineer(cfg)
        >>> engineer.process_and_save()
    """

    def __init__(self, config: dict) -> None:
        self.cfg = config
        self.data_cfg = config["data"]
        self.feat_cfg = config["features"]
        self.raw_dir = _ROOT / self.data_cfg["raw_dir"]
        self.proc_dir = _ROOT / self.data_cfg["processed_dir"]
        self.seq_len: int = self.data_cfg["sequence_length"]
        self.pos_thresh: float = self.data_cfg["pos_threshold"]
        self.n_categories: int = self.data_cfg["n_categories"]

        # Filled during fit()
        self._user_stats: Dict[int, np.ndarray] = {}
        self._item_stats: Dict[int, np.ndarray] = {}
        self._default_user_feat = np.zeros(self.feat_cfg["user_dense_dim"], dtype=np.float32)
        self._default_item_feat = np.zeros(self.feat_cfg["item_dense_dim"], dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_and_save(self) -> None:
        """Full pipeline: load → split → fit → transform → save."""
        df = self._load_raw()
        train_df, val_df, test_df = self._temporal_split(df)

        logger.info("Computing user / item statistics from training set …")
        self._fit_user_stats(train_df)
        self._fit_item_stats(train_df)

        logger.info("Building user interaction sequences …")
        # Build sequences from the full sorted df (temporal order preserved)
        user_seq_map = self._build_user_sequences(df)

        self.proc_dir.mkdir(parents=True, exist_ok=True)

        for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            logger.info(f"Transforming {name} split ({len(split_df):,} rows) …")
            data = self._transform(split_df, user_seq_map)
            self._save_split(data, name)

        # Save metadata needed by models (vocab sizes, feature dims)
        meta = {
            "n_users":          self.data_cfg["n_users"],
            "n_items":          self.data_cfg["n_items"],
            "n_categories":     self.n_categories,
            "n_duration_bkts":  self.feat_cfg["n_duration_buckets"],
            "user_dense_dim":   self.feat_cfg["user_dense_dim"],
            "item_dense_dim":   self.feat_cfg["item_dense_dim"],
            "seq_len":          self.seq_len,
            "pos_thresh":       self.pos_thresh,
            "train_size":       len(train_df),
            "val_size":         len(val_df),
            "test_size":        len(test_df),
        }
        self._save_meta(meta)
        logger.info("Feature engineering complete.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_raw(self) -> pd.DataFrame:
        path = self.raw_dir / "interactions.csv"
        try:
            df = pd.read_csv(path)
            df = df.sort_values("timestamp").reset_index(drop=True)
            logger.info(f"Loaded {len(df):,} interactions from {path}")
            return df
        except FileNotFoundError:
            logger.error(f"Raw data not found at {path}. Run download_data.py first.")
            raise

    def _temporal_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split by timestamp to avoid future data leaking into training."""
        n = len(df)
        val_r = self.data_cfg["val_ratio"]
        test_r = self.data_cfg["test_ratio"]
        train_end = int(n * (1.0 - val_r - test_r))
        val_end = int(n * (1.0 - test_r))
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        logger.info(
            f"Split sizes — train: {len(train_df):,}  "
            f"val: {len(val_df):,}  test: {len(test_df):,}"
        )
        return train_df, val_df, test_df

    def _fit_user_stats(self, train_df: pd.DataFrame) -> None:
        """Compute per-user dense features from training data only.

        Feature layout (25 dims):
          [0]   avg_watch_ratio
          [1]   like_rate
          [2]   follow_rate
          [3]   comment_rate
          [4]   share_rate
          [5:25] category preference distribution (normalised)
        """
        grp = train_df.groupby("user_id")
        for uid, g in grp:
            cat_counts = np.zeros(self.n_categories, dtype=np.float32)
            for cat_id, cnt in g["video_category"].value_counts().items():
                cat_counts[int(cat_id)] = cnt
            cat_prefs = cat_counts / (cat_counts.sum() + 1e-9)

            feat = np.array([
                g["watch_ratio"].mean(),
                g["like"].mean(),
                g["follow"].mean(),
                g["comment"].mean(),
                g["share"].mean(),
            ], dtype=np.float32)
            self._user_stats[int(uid)] = np.concatenate([feat, cat_prefs])

        # Default vector for unseen users: global mean
        all_feats = np.stack(list(self._user_stats.values()))
        self._default_user_feat = all_feats.mean(axis=0)

    def _fit_item_stats(self, train_df: pd.DataFrame) -> None:
        """Compute per-item dense features from training data only.

        Feature layout (3 dims):
          [0] historical_ctr  (avg watch_ratio)
          [1] like_rate
          [2] log_popularity  (log10(1 + n_interactions), normalised to [0,1])
        """
        grp = train_df.groupby("video_id")
        counts = grp.size()
        max_log = np.log10(1 + counts.max())

        for vid, g in grp:
            log_pop = np.log10(1 + len(g)) / (max_log + 1e-9)
            self._item_stats[int(vid)] = np.array([
                g["watch_ratio"].mean(),
                g["like"].mean(),
                log_pop,
            ], dtype=np.float32)

        all_item_feats = np.stack(list(self._item_stats.values()))
        self._default_item_feat = all_item_feats.mean(axis=0)

    def _build_user_sequences(
        self, df: pd.DataFrame
    ) -> Dict[int, list]:
        """Build cumulative watch history per user (temporal order).

        Returns a dict mapping user_id → list of video_ids in order of
        increasing timestamp.  video_ids are stored 1-indexed so that 0
        can serve as the padding token in embedding layers.
        """
        seq_map: Dict[int, list] = {}
        for uid, g in df.groupby("user_id"):
            seq_map[int(uid)] = (g["video_id"].values + 1).tolist()
        return seq_map

    def _get_history_at(
        self,
        user_id: int,
        row_pos_in_user_history: int,
        seq_map: Dict[int, list],
    ) -> Tuple[np.ndarray, int]:
        """Return the sequence of the last ``seq_len`` items before row_pos.

        Args:
            user_id: User identifier.
            row_pos_in_user_history: How many interactions this user has had
                *before* the current one.
            seq_map: Pre-built user sequence map (1-indexed video_ids).

        Returns:
            (padded_seq, actual_length) where padded_seq has shape (seq_len,).
        """
        full_seq = seq_map.get(user_id, [])
        hist = full_seq[:row_pos_in_user_history][-self.seq_len:]
        actual_len = len(hist)
        padded = np.zeros(self.seq_len, dtype=np.int32)
        if actual_len > 0:
            padded[:actual_len] = hist  # left-align; zeros on the right (causal-attention safe)
        return padded, actual_len

    @staticmethod
    def _duration_bucket(duration_seconds: int) -> int:
        """Map video duration to a bucket index [0, 4].

        Buckets: 0 → <15s,  1 → 15-30s,  2 → 30-60s,
                 3 → 60-120s,  4 → ≥120s
        """
        return int(np.searchsorted(_DURATION_BINS, duration_seconds))

    def _transform(
        self, split_df: pd.DataFrame, seq_map: Dict[int, list]
    ) -> Dict[str, np.ndarray]:
        """Apply all features to a split DataFrame.

        Args:
            split_df: DataFrame for one split (train/val/test).
            seq_map: Mapping from user_id to full chronological video sequence.

        Returns:
            Dict of numpy arrays, one entry per feature.
        """
        n = len(split_df)
        user_ids        = np.empty(n, dtype=np.int32)
        item_ids        = np.empty(n, dtype=np.int32)
        item_category   = np.empty(n, dtype=np.int32)
        item_dur_bkt    = np.empty(n, dtype=np.int32)
        user_dense      = np.empty((n, self.feat_cfg["user_dense_dim"]), dtype=np.float32)
        item_dense      = np.empty((n, self.feat_cfg["item_dense_dim"]), dtype=np.float32)
        history_seqs    = np.zeros((n, self.seq_len), dtype=np.int32)
        history_lens    = np.zeros(n, dtype=np.int32)
        labels          = np.empty(n, dtype=np.float32)
        watch_ratio_raw = np.empty(n, dtype=np.float32)
        like_labels     = np.empty(n, dtype=np.float32)

        # Track how many interactions each user has been processed so far
        # (to build the correct "before this interaction" sequence)
        user_cursor: Dict[int, int] = {}

        for idx, row in enumerate(split_df.itertuples(index=False)):
            uid = int(row.user_id)
            vid = int(row.video_id)

            user_ids[idx]        = uid
            item_ids[idx]        = vid
            item_category[idx]   = int(row.video_category)
            item_dur_bkt[idx]    = self._duration_bucket(int(row.video_duration))
            user_dense[idx]      = self._user_stats.get(uid, self._default_user_feat)
            item_dense[idx]      = self._item_stats.get(vid, self._default_item_feat)
            labels[idx]          = 1.0 if row.watch_ratio >= self.pos_thresh else 0.0
            watch_ratio_raw[idx] = float(row.watch_ratio)
            like_labels[idx]     = float(row.like)

            pos_in_hist = user_cursor.get(uid, 0)
            seq, seq_len = self._get_history_at(uid, pos_in_hist, seq_map)
            history_seqs[idx] = seq
            history_lens[idx] = seq_len
            user_cursor[uid]  = pos_in_hist + 1

        return {
            "user_ids":         user_ids,
            "item_ids":         item_ids,
            "item_category":    item_category,
            "item_dur_bkt":     item_dur_bkt,
            "user_dense":       user_dense,
            "item_dense":       item_dense,
            "history_seqs":     history_seqs,
            "history_lens":     history_lens,
            "labels":           labels,
            "watch_ratio_raw":  watch_ratio_raw,
            "like_labels":      like_labels,
        }

    def _save_split(self, data: Dict[str, np.ndarray], name: str) -> None:
        path = self.proc_dir / f"{name}_data.pkl"
        try:
            with open(path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved {name} split → {path}")
        except OSError as e:
            logger.error(f"Failed to save {name} split: {e}")
            raise

    def _save_meta(self, meta: dict) -> None:
        path = self.proc_dir / "meta.pkl"
        try:
            with open(path, "wb") as f:
                pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved metadata → {path}")
        except OSError as e:
            logger.error(f"Failed to save meta: {e}")
            raise
