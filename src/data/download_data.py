"""Mock data generator that reproduces the KuaiRec interaction schema.

KuaiRec official dataset: https://kuairec.com/
This script generates synthetic data with the same field structure so the
full pipeline can be developed and tested without downloading the real dataset.

Run from the project root:
    python src/data/download_data.py
"""

import sys
from pathlib import Path

# Make sure the project root is importable regardless of CWD
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import yaml

from src.utils.logger import get_logger
from src.utils.gpu_utils import set_seed

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# KuaiRec field documentation
# ---------------------------------------------------------------------------
# user_id          int   unique user identifier (0 … n_users-1)
# video_id         int   unique video identifier (0 … n_items-1)
# watch_ratio      float fraction of video watched; >1.0 means re-watched
# like             int   1 if user liked the video, else 0
# follow           int   1 if user followed the creator, else 0
# comment          int   1 if user commented, else 0
# share            int   1 if user shared, else 0
# timestamp        int   Unix timestamp (seconds)
# video_category   int   content category id (0 … n_categories-1)
# video_duration   int   video length in seconds


def _load_config() -> dict:
    config_path = _ROOT / "configs" / "base_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_mock_data(
    n_users: int = 500,
    n_items: int = 1000,
    n_interactions: int = 15000,
    n_categories: int = 20,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate a synthetic dataset that mirrors KuaiRec's schema.

    Statistical properties mimic real short-video platforms:
    - User activity follows a power-law (few heavy users, many casual ones).
    - Video popularity follows a power-law (long-tail distribution).
    - watch_ratio is right-skewed (most views are partial).
    - Engagement actions (like/follow/comment/share) are sparse and
      positively correlated with watch_ratio.

    Args:
        n_users: Number of unique users.
        n_items: Number of unique videos.
        n_interactions: Total number of user–video interaction records.
        n_categories: Number of video content categories.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (interactions_df, videos_df, users_df).
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1. Video catalogue (static video metadata)
    # ------------------------------------------------------------------
    video_ids = np.arange(n_items)
    # Categories with a skewed distribution (some categories dominate)
    cat_probs = rng.dirichlet(np.ones(n_categories) * 0.5)
    video_categories = rng.choice(n_categories, size=n_items, p=cat_probs)
    # Duration: short (5-30s), medium (30-120s), long (120-300s)
    dur_weights = [0.45, 0.40, 0.15]
    dur_ranges = [(5, 30), (30, 120), (120, 300)]
    video_durations = np.array([
        int(rng.integers(*dur_ranges[rng.choice(3, p=dur_weights)]))
        for _ in range(n_items)
    ])
    videos_df = pd.DataFrame({
        "video_id": video_ids,
        "video_category": video_categories,
        "video_duration": video_durations,
    })

    # ------------------------------------------------------------------
    # 2. User catalogue
    # ------------------------------------------------------------------
    users_df = pd.DataFrame({"user_id": np.arange(n_users)})

    # ------------------------------------------------------------------
    # 3. Interaction sampling with power-law popularity
    # ------------------------------------------------------------------
    # User activity: power-law — some users are very active
    user_activity = rng.pareto(1.5, size=n_users) + 1.0
    user_probs = user_activity / user_activity.sum()

    # Item popularity: Zipf-like power-law
    item_popularity = rng.pareto(1.2, size=n_items) + 1.0
    item_probs = item_popularity / item_popularity.sum()

    sampled_users = rng.choice(n_users, size=n_interactions, p=user_probs)
    sampled_items = rng.choice(n_items, size=n_interactions, p=item_probs)

    # Timestamps: 30-day window starting 2024-01-01
    t_start = int(pd.Timestamp("2024-01-01").timestamp())
    t_end = int(pd.Timestamp("2024-01-31").timestamp())
    timestamps = rng.integers(t_start, t_end, size=n_interactions)

    # ------------------------------------------------------------------
    # 4. watch_ratio: beta distribution (right-skewed, occasionally > 1)
    # ------------------------------------------------------------------
    watch_ratio = rng.beta(a=1.2, b=2.5, size=n_interactions)
    # ~8% of interactions are re-watches (ratio > 1.0)
    rewatch_mask = rng.random(n_interactions) < 0.08
    watch_ratio[rewatch_mask] = rng.uniform(1.0, 1.8, size=rewatch_mask.sum())
    watch_ratio = np.clip(watch_ratio, 0.0, 2.0).astype(np.float32)

    # ------------------------------------------------------------------
    # 5. Engagement actions (sparse, correlated with watch_ratio)
    # ------------------------------------------------------------------
    # P(action | watch_ratio) is a sigmoid-like function
    def _action_prob(ratio: np.ndarray, base: float, scale: float) -> np.ndarray:
        return np.clip(base * (ratio ** scale), 0.0, 1.0)

    like    = (rng.random(n_interactions) < _action_prob(watch_ratio, 0.12, 1.5)).astype(np.int8)
    follow  = (rng.random(n_interactions) < _action_prob(watch_ratio, 0.02, 2.0)).astype(np.int8)
    comment = (rng.random(n_interactions) < _action_prob(watch_ratio, 0.015, 2.0)).astype(np.int8)
    share   = (rng.random(n_interactions) < _action_prob(watch_ratio, 0.008, 2.5)).astype(np.int8)

    # ------------------------------------------------------------------
    # 6. Attach video metadata to interactions
    # ------------------------------------------------------------------
    vid_cat = videos_df.set_index("video_id")["video_category"].to_dict()
    vid_dur = videos_df.set_index("video_id")["video_duration"].to_dict()

    interactions_df = pd.DataFrame({
        "user_id":        sampled_users.astype(np.int32),
        "video_id":       sampled_items.astype(np.int32),
        "watch_ratio":    watch_ratio,
        "like":           like,
        "follow":         follow,
        "comment":        comment,
        "share":          share,
        "timestamp":      timestamps.astype(np.int64),
        "video_category": pd.Series(sampled_items).map(vid_cat).astype(np.int32).values,
        "video_duration": pd.Series(sampled_items).map(vid_dur).astype(np.int32).values,
    })

    # Sort by time (important for temporal train/val/test split)
    interactions_df = interactions_df.sort_values("timestamp").reset_index(drop=True)

    return interactions_df, videos_df, users_df


def save_raw_data(
    interactions_df: pd.DataFrame,
    videos_df: pd.DataFrame,
    users_df: pd.DataFrame,
    raw_dir: Path,
) -> None:
    """Persist raw data to CSV files.

    Args:
        interactions_df: Interaction records.
        videos_df: Video metadata.
        users_df: User metadata.
        raw_dir: Directory to write into.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        interactions_df.to_csv(raw_dir / "interactions.csv", index=False)
        videos_df.to_csv(raw_dir / "videos.csv", index=False)
        users_df.to_csv(raw_dir / "users.csv", index=False)
        logger.info(f"Raw data saved to {raw_dir}")
    except OSError as e:
        logger.error(f"Failed to save raw data: {e}")
        raise


def print_stats(interactions_df: pd.DataFrame) -> None:
    """Print dataset statistics to the logger."""
    n = len(interactions_df)
    pos = (interactions_df["watch_ratio"] >= 0.7).sum()
    logger.info("=" * 50)
    logger.info("Dataset Statistics")
    logger.info("=" * 50)
    logger.info(f"Total interactions : {n:,}")
    logger.info(f"Unique users       : {interactions_df['user_id'].nunique():,}")
    logger.info(f"Unique videos      : {interactions_df['video_id'].nunique():,}")
    logger.info(f"Positive samples   : {pos:,}  ({pos/n*100:.1f}%)")
    logger.info(f"Negative samples   : {n-pos:,}  ({(n-pos)/n*100:.1f}%)")
    logger.info(f"Avg watch_ratio    : {interactions_df['watch_ratio'].mean():.3f}")
    logger.info(f"Like rate          : {interactions_df['like'].mean()*100:.2f}%")
    logger.info(f"Follow rate        : {interactions_df['follow'].mean()*100:.2f}%")
    logger.info(f"Sparsity           : "
                f"{1 - n / (interactions_df['user_id'].nunique() * interactions_df['video_id'].nunique()):.4f}")
    logger.info("=" * 50)
    logger.info("KuaiRec official dataset: https://kuairec.com/")
    logger.info("To use the real dataset, download from the link above")
    logger.info("and place interactions.csv / videos.csv in data/raw/")


def main() -> None:
    cfg = _load_config()
    set_seed(cfg["project"]["seed"])

    raw_dir = _ROOT / cfg["data"]["raw_dir"]

    logger.info("Generating KuaiRec-schema mock dataset …")
    interactions_df, videos_df, users_df = generate_mock_data(
        n_users=cfg["data"]["n_users"],
        n_items=cfg["data"]["n_items"],
        n_interactions=cfg["data"]["n_interactions"],
        n_categories=cfg["data"]["n_categories"],
        seed=cfg["project"]["seed"],
    )

    save_raw_data(interactions_df, videos_df, users_df, raw_dir)
    print_stats(interactions_df)

    # Immediately run feature engineering so processed data is ready
    logger.info("Running feature engineering on raw data …")
    from src.data.feature_engineering import FeatureEngineer
    engineer = FeatureEngineer(cfg)
    engineer.process_and_save()
    logger.info("Done. Processed data written to data/processed/")


if __name__ == "__main__":
    main()
