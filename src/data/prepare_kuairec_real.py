"""Convert real KuaiRec 2.0 data into the pipeline's expected CSV format.

Downloads automatically via Zenodo if data/raw/kuairec_real/ is missing.
Otherwise reads from local extracted files.

KuaiRec 2.0 reference:
  Gao et al., "KuaiRec: A Fully-Observed Dataset and Insights for Evaluating
  Recommender Systems", CIKM 2022.  https://kuairec.com/

Output (same schema as download_data.py):
    data/raw/interactions.csv   — interactions with watch_ratio, like, ...
    data/raw/videos.csv         — video metadata
    data/raw/users.csv          — user IDs

Feature mapping from real KuaiRec:
  watch_ratio    → interaction matrix field (already computed)
  like           → watch_ratio >= 1.0 (re-watch = strong positive signal)
  follow         → 0  (not available in KuaiRec 2.0 interaction matrix)
  comment        → 0  (same)
  share          → 0  (same)
  video_category → first element of item_categories.feat list
  video_duration → interaction matrix field (milliseconds → seconds)

Usage:
    python src/data/prepare_kuairec_real.py          # use small_matrix (1.4M users)
    python src/data/prepare_kuairec_real.py --big    # use big_matrix  (7K users)
    python src/data/prepare_kuairec_real.py --n 200000  # sample N rows
"""

import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)

KUAI_DIR = _ROOT / "data" / "raw" / "kuairec_real" / "KuaiRec 2.0" / "data"
RAW_OUT  = _ROOT / "data" / "raw"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_cat(feat_str: str) -> int:
    """Parse '[8]' or '[27, 9]' → first integer category."""
    try:
        lst = ast.literal_eval(feat_str)
        return int(lst[0]) if lst else 0
    except Exception:
        return 0


def _load_item_category(kuai_dir: Path) -> pd.Series:
    """Return Series: video_id → primary category (int)."""
    ic = pd.read_csv(kuai_dir / "item_categories.csv")
    ic["video_category"] = ic["feat"].apply(_first_cat)
    return ic.set_index("video_id")["video_category"]


# ---------------------------------------------------------------------------
# Main preparation
# ---------------------------------------------------------------------------

def prepare(matrix: str = "small", n_rows: int = 300_000, seed: int = 42) -> None:
    """Load KuaiRec interaction matrix and convert to pipeline format.

    Args:
        matrix: 'small' (1411 users × 3327 items, dense) or
                'big'   (7176 users × 10728 items, sparse)
        n_rows: Max rows to sample (use -1 for full dataset).
        seed:   Random seed for sampling.
    """
    csv_name = "small_matrix.csv" if matrix == "small" else "big_matrix.csv"
    csv_path = KUAI_DIR / csv_name

    if not csv_path.exists():
        logger.error(
            f"KuaiRec data not found at {KUAI_DIR}.\n"
            "Download it with:\n"
            "  curl -L https://zenodo.org/records/18164998/files/KuaiRec.zip "
            "-o data/raw/kuairec_real/KuaiRec.zip\n"
            "  cd data/raw/kuairec_real && unzip KuaiRec.zip"
        )
        sys.exit(1)

    logger.info(f"Loading KuaiRec {matrix}_matrix …")
    interactions = pd.read_csv(csv_path)
    logger.info(f"  Loaded {len(interactions):,} rows | "
                f"{interactions.user_id.nunique():,} users | "
                f"{interactions.video_id.nunique():,} items")

    # ── Sampling ────────────────────────────────────────────────────────────
    if 0 < n_rows < len(interactions):
        # Stratified sample: keep proportional user representation
        rng = np.random.default_rng(seed)
        users = interactions["user_id"].unique()
        rows_per_user = max(1, n_rows // len(users))
        sampled_parts = []
        for uid in users:
            udf = interactions[interactions["user_id"] == uid]
            # Keep the most recent rows (preserve temporal ordering)
            take = min(rows_per_user, len(udf))
            sampled_parts.append(udf.sort_values("timestamp").tail(take))
        interactions = pd.concat(sampled_parts, ignore_index=True)
        logger.info(f"  Sampled → {len(interactions):,} rows "
                    f"({interactions.user_id.nunique():,} users)")

    # ── Sort by time ────────────────────────────────────────────────────────
    interactions = interactions.sort_values("timestamp").reset_index(drop=True)

    # ── Load item categories ─────────────────────────────────────────────────
    cat_series = _load_item_category(KUAI_DIR)
    interactions["video_category"] = (
        interactions["video_id"].map(cat_series).fillna(0).astype(int)
    )

    # ── Remap IDs to contiguous 0-indexed ────────────────────────────────────
    user_ids_sorted = sorted(interactions["user_id"].unique())
    item_ids_sorted = sorted(interactions["video_id"].unique())
    user_map = {uid: i for i, uid in enumerate(user_ids_sorted)}
    item_map = {vid: i for i, vid in enumerate(item_ids_sorted)}
    cat_ids  = sorted(interactions["video_category"].unique())
    cat_map  = {c: i for i, c in enumerate(cat_ids)}

    interactions["user_id"]   = interactions["user_id"].map(user_map)
    interactions["video_id"]  = interactions["video_id"].map(item_map)
    interactions["video_category"] = interactions["video_category"].map(cat_map)

    # ── Clip watch_ratio ─────────────────────────────────────────────────────
    interactions["watch_ratio"] = interactions["watch_ratio"].clip(0.0, 2.0).astype("float32")

    # ── Derive engagement actions ─────────────────────────────────────────────
    # like  ≈ re-watch (watch_ratio ≥ 1.0 means watched more than once)
    # follow/comment/share not available → 0
    interactions["like"]    = (interactions["watch_ratio"] >= 1.0).astype("int8")
    interactions["follow"]  = 0
    interactions["comment"] = 0
    interactions["share"]   = 0

    # ── video_duration: milliseconds → seconds (already in ms in the CSV) ───
    # KuaiRec stores video_duration in milliseconds
    interactions["video_duration"] = (interactions["video_duration"] / 1000.0).clip(1, 600).astype(int)

    # ── Build output DataFrames ───────────────────────────────────────────────
    interactions_out = interactions[[
        "user_id", "video_id", "watch_ratio", "like", "follow",
        "comment", "share", "timestamp", "video_category", "video_duration",
    ]].copy()

    videos_df = (
        interactions_out[["video_id", "video_category", "video_duration"]]
        .drop_duplicates("video_id")
        .sort_values("video_id")
        .reset_index(drop=True)
    )
    users_df = pd.DataFrame({"user_id": sorted(interactions_out["user_id"].unique())})

    # ── Stats ─────────────────────────────────────────────────────────────────
    n = len(interactions_out)
    pos = (interactions_out["watch_ratio"] >= 0.7).sum()
    logger.info("=" * 50)
    logger.info("KuaiRec Real Dataset Statistics")
    logger.info("=" * 50)
    logger.info(f"Total interactions : {n:,}")
    logger.info(f"Unique users       : {interactions_out['user_id'].nunique():,}")
    logger.info(f"Unique items       : {interactions_out['video_id'].nunique():,}")
    logger.info(f"Unique categories  : {interactions_out['video_category'].nunique():,}")
    logger.info(f"Positive (≥0.7)    : {pos:,}  ({pos/n*100:.1f}%)")
    logger.info(f"Like rate (rewatch): {interactions_out['like'].mean()*100:.1f}%")
    logger.info(f"Avg watch_ratio    : {interactions_out['watch_ratio'].mean():.3f}")
    logger.info(f"Sparsity           : "
                f"{1 - n / (interactions_out['user_id'].nunique() * interactions_out['video_id'].nunique()):.4f}")
    logger.info("=" * 50)

    # ── Save ──────────────────────────────────────────────────────────────────
    RAW_OUT.mkdir(parents=True, exist_ok=True)
    interactions_out.to_csv(RAW_OUT / "interactions.csv", index=False)
    videos_df.to_csv(RAW_OUT / "videos.csv", index=False)
    users_df.to_csv(RAW_OUT / "users.csv", index=False)
    logger.info(f"Raw data saved to {RAW_OUT}")

    # ── Feature engineering ───────────────────────────────────────────────────
    logger.info("Running feature engineering …")
    from src.data.feature_engineering import FeatureEngineer
    import yaml
    with open(_ROOT / "configs" / "base_config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    n_cats = interactions_out["video_category"].nunique()
    cfg["data"]["n_users"]      = len(users_df)
    cfg["data"]["n_items"]      = len(videos_df)
    cfg["data"]["n_categories"] = n_cats
    # user_dense = 5 activity stats + n_categories category fractions
    cfg["features"]["user_dense_dim"] = 5 + n_cats
    engineer = FeatureEngineer(cfg)
    engineer.process_and_save()
    logger.info("Done.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--big", action="store_true",
                        help="Use big_matrix instead of small_matrix")
    parser.add_argument("--n", type=int, default=300_000,
                        help="Max interactions to sample (-1 = all)")
    args = parser.parse_args()
    prepare(
        matrix="big" if args.big else "small",
        n_rows=args.n if args.n > 0 else int(1e9),
    )


if __name__ == "__main__":
    main()
