"""KuaiRec preprocessor for real KuaiRec dataset.

KuaiRec is a fully-observed recommendation dataset from Kuaishou.
Dataset page: https://kuairec.com/

To use real data:
  1. Download from https://kuairec.com/ (requires registration)
  2. Place files in data/raw/kuairec/:
       - big_matrix.csv   (12M interactions)
       - small_matrix.csv (4.7M interactions, fully observed)
       - video_features_basic.csv (optional, for category/duration)
  3. Run: python src/data/kuairec_preprocessor.py

The preprocessor maps the KuaiRec schema to the project's mock schema
(interactions.csv) and updates base_config.yaml with real data sizes.
"""

import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class KuaiRecPreprocessor:
    """Preprocessor for real KuaiRec dataset.

    Reads raw KuaiRec CSVs and converts them to the project's interaction
    schema (interactions.csv) compatible with FeatureEngineer.

    Args:
        cfg: Parsed base_config.yaml dict.

    Example:
        >>> from src.data.kuairec_preprocessor import KuaiRecPreprocessor
        >>> preprocessor = KuaiRecPreprocessor(cfg)
        >>> if preprocessor.check_data_available():
        ...     cfg = preprocessor.preprocess(use_small=True)
        ... else:
        ...     preprocessor.print_download_instructions()
    """

    # KuaiRec column mappings
    _KUAIREC_INTERACTION_COLS = {
        "big_matrix": {
            "user_id": "user_id",
            "item_id": "video_id",
            "watch_ratio": "watch_ratio",
            "like": "like",
            "comment": "comment",
            "follow": "follow",
            "share": "forward",    # KuaiRec uses 'forward' for share
            "timestamp": "timestamp",
        },
        "small_matrix": {
            "user_id": "user_id",
            "item_id": "video_id",
            "watch_ratio": "watch_ratio",
            "like": "like",
            "comment": "comment",
            "follow": "follow",
            "share": "forward",
            "timestamp": "timestamp",
        },
    }

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.data_cfg = cfg["data"]
        self.raw_dir     = _ROOT / self.data_cfg["raw_dir"] / "kuairec"
        self.out_raw_dir = _ROOT / self.data_cfg["raw_dir"]
        self.proc_dir    = _ROOT / self.data_cfg["processed_dir"]

    def check_data_available(self) -> bool:
        """Check if KuaiRec data files are present.

        Returns:
            True if big_matrix.csv or small_matrix.csv exists.
        """
        big_matrix   = self.raw_dir / "big_matrix.csv"
        small_matrix = self.raw_dir / "small_matrix.csv"
        available = big_matrix.exists() or small_matrix.exists()
        if available:
            logger.info(f"KuaiRec data found in {self.raw_dir}")
        else:
            logger.info(f"KuaiRec data NOT found in {self.raw_dir}")
        return available

    def print_download_instructions(self) -> None:
        """Print instructions for downloading the KuaiRec dataset."""
        instructions = """
========================================================
  KuaiRec Dataset Download Instructions
========================================================

KuaiRec is a real-world, fully-observed recommendation dataset
from Kuaishou (the Chinese short-video platform).

Step 1: Visit https://kuairec.com/ and register for access.

Step 2: Download the dataset files:
  - small_matrix.csv   (~4.7M interactions, 1411 users, 3327 items)
  - big_matrix.csv     (~12M interactions, 7176 users, 10728 items)
  - video_features_basic.csv  (video metadata: category, duration)

Step 3: Place the files here:
  {raw_dir}/
    small_matrix.csv
    big_matrix.csv
    video_features_basic.csv   (optional)

Step 4: Run the preprocessor:
  python src/data/kuairec_preprocessor.py --use-small

Expected CSV columns (small_matrix / big_matrix):
  user_id, video_id, watch_ratio, like, comment, follow, forward, timestamp

Expected CSV columns (video_features_basic):
  video_id, feat  (feat is a list of category tags)
  or alternatively: video_id, video_type, upload_time, duration (in ms)

Note: watch_ratio values > 1.0 are allowed (rewatches) and are clipped to
      the range [0, 1] during preprocessing.
========================================================
""".format(raw_dir=self.raw_dir)
        print(instructions)
        logger.info("Download instructions printed.")

    def preprocess(self, use_small: bool = True) -> dict:
        """Read and convert KuaiRec data to project schema.

        Args:
            use_small: If True, use small_matrix.csv (faster); else big_matrix.csv.

        Returns:
            Updated cfg dict with corrected n_users, n_items, n_categories.

        Raises:
            FileNotFoundError: If the selected matrix file is not found.
        """
        matrix_file = "small_matrix.csv" if use_small else "big_matrix.csv"
        matrix_path = self.raw_dir / matrix_file

        if not matrix_path.exists():
            raise FileNotFoundError(
                f"KuaiRec matrix not found at {matrix_path}. "
                "Run print_download_instructions() for download help."
            )

        logger.info(f"Loading KuaiRec interactions from {matrix_path} …")
        df = pd.read_csv(matrix_path)
        logger.info(f"Loaded {len(df):,} raw interactions.")

        # ── Column harmonisation ──
        # Map KuaiRec column names to project schema
        col_map = self._KUAIREC_INTERACTION_COLS[
            "small_matrix" if use_small else "big_matrix"
        ]
        rename = {}
        for proj_col, kuai_col in col_map.items():
            if kuai_col in df.columns and kuai_col != proj_col:
                rename[kuai_col] = proj_col
        df = df.rename(columns=rename)

        # Ensure required columns exist; fill defaults if missing
        required = ["user_id", "video_id", "watch_ratio"]
        for col in required:
            if col not in df.columns:
                raise ValueError(
                    f"Required column '{col}' not found in {matrix_path}. "
                    f"Available columns: {list(df.columns)}"
                )

        for optional_col, default in [
            ("like", 0), ("comment", 0), ("follow", 0), ("share", 0),
        ]:
            if optional_col not in df.columns:
                df[optional_col] = default
                logger.info(f"Column '{optional_col}' not found — filled with {default}.")

        if "timestamp" not in df.columns:
            # Use row index as pseudo-timestamp (preserves relative order)
            df["timestamp"] = np.arange(len(df))
            logger.info("'timestamp' column not found — using row order.")

        # ── Clip watch_ratio to [0, 1] (rewatches can exceed 1.0) ──
        df["watch_ratio"] = df["watch_ratio"].clip(0.0, 1.0).astype(np.float32)

        # ── Label-encode user_id and video_id → 0-indexed ──
        user_encoder = LabelEncoder()
        item_encoder = LabelEncoder()

        df["user_id"]   = user_encoder.fit_transform(df["user_id"])
        df["video_id"]  = item_encoder.fit_transform(df["video_id"])

        n_users = int(df["user_id"].max()) + 1
        n_items = int(df["video_id"].max()) + 1
        logger.info(f"Encoded: {n_users:,} users, {n_items:,} items.")

        # ── Video features (category / duration) ──
        video_feat_path = self.raw_dir / "video_features_basic.csv"
        n_categories = self.cfg["data"].get("n_categories", 20)
        df["video_category"] = 0   # default
        df["video_duration"]  = 30  # default: 30 seconds bucket

        if video_feat_path.exists():
            logger.info(f"Loading video features from {video_feat_path} …")
            vf = pd.read_csv(video_feat_path)
            vf["video_id"] = item_encoder.transform(
                vf["video_id"].astype(df["video_id"].dtype)
                if vf["video_id"].dtype == df["video_id"].dtype
                else vf["video_id"]
            ) if "video_id" in vf.columns else np.arange(len(vf))

            if "feat" in vf.columns:
                # feat column may be a list string like "[1, 3, 5]"
                # Take the first tag as category
                def _first_feat(val):
                    try:
                        tags = str(val).strip("[]").split(",")
                        return int(tags[0].strip()) if tags else 0
                    except (ValueError, IndexError):
                        return 0

                vf["video_category"] = vf["feat"].apply(_first_feat)
                n_categories = max(n_categories, int(vf["video_category"].max()) + 1)

            if "duration" in vf.columns:
                # KuaiRec duration is in ms → convert to seconds
                vf["video_duration"] = (vf["duration"] / 1000).clip(lower=0).astype(int)
            else:
                vf["video_duration"] = 30

            # Merge features onto interactions
            merge_cols = ["video_id"]
            if "video_category" in vf.columns:
                merge_cols.append("video_category")
            if "video_duration" in vf.columns:
                merge_cols.append("video_duration")

            vf_subset = vf[merge_cols].drop_duplicates("video_id")
            df = df.merge(vf_subset, on="video_id", how="left", suffixes=("", "_vf"))

            # Fill merged columns, prefer the joined ones
            for col in ["video_category", "video_duration"]:
                joined = col + "_vf"
                if joined in df.columns:
                    df[col] = df[joined].fillna(df[col])
                    df = df.drop(columns=[joined])

            df["video_category"] = df["video_category"].fillna(0).astype(int)
            df["video_duration"]  = df["video_duration"].fillna(30).astype(int)
            logger.info(f"Video features merged. n_categories={n_categories}.")
        else:
            logger.info("video_features_basic.csv not found — using default category=0, duration=30s.")
            df["video_category"] = 0
            df["video_duration"]  = 30

        # Clip category to valid range
        df["video_category"] = df["video_category"].clip(0, n_categories - 1)

        # ── Sort by timestamp ──
        df = df.sort_values("timestamp").reset_index(drop=True)

        # ── Ensure binary columns are int ──
        for col in ["like", "comment", "follow", "share"]:
            df[col] = df[col].fillna(0).astype(int).clip(0, 1)

        # ── Save interactions.csv (project schema) ──
        self.out_raw_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.out_raw_dir / "interactions.csv"
        out_cols = [
            "user_id", "video_id", "watch_ratio", "like",
            "follow", "comment", "share", "timestamp",
            "video_category", "video_duration",
        ]
        df[out_cols].to_csv(out_path, index=False)
        logger.info(f"Saved {len(df):,} interactions → {out_path}")

        # ── Save ID mappings ──
        id_mappings = {
            "user_encoder":   user_encoder,
            "item_encoder":   item_encoder,
            "n_users":        n_users,
            "n_items":        n_items,
            "n_categories":   n_categories,
        }
        self.proc_dir.mkdir(parents=True, exist_ok=True)
        mapping_path = self.proc_dir / "id_mappings.pkl"
        with open(mapping_path, "wb") as f:
            pickle.dump(id_mappings, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved ID mappings → {mapping_path}")

        # ── Update config with real data sizes ──
        updated_cfg = dict(self.cfg)
        updated_cfg["data"] = dict(self.cfg["data"])
        updated_cfg["data"]["n_users"]      = n_users
        updated_cfg["data"]["n_items"]      = n_items
        updated_cfg["data"]["n_categories"] = n_categories
        updated_cfg["data"]["n_interactions"] = len(df)

        logger.info(
            f"KuaiRec preprocessing complete.\n"
            f"  Users      : {n_users:,}\n"
            f"  Items      : {n_items:,}\n"
            f"  Categories : {n_categories}\n"
            f"  Interactions: {len(df):,}\n"
            f"  Output     : {out_path}"
        )
        return updated_cfg


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(
        description="Preprocess real KuaiRec dataset into project schema."
    )
    parser.add_argument(
        "--use-small", action="store_true", default=True,
        help="Use small_matrix.csv (default). Use --no-use-small for big_matrix."
    )
    parser.add_argument(
        "--no-use-small", dest="use_small", action="store_false",
        help="Use big_matrix.csv instead of small_matrix."
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Only check if data is available; print instructions if not."
    )
    args = parser.parse_args()

    cfg_path = _ROOT / "configs" / "base_config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    preprocessor = KuaiRecPreprocessor(cfg)

    if not preprocessor.check_data_available():
        preprocessor.print_download_instructions()
    elif not args.check_only:
        updated_cfg = preprocessor.preprocess(use_small=args.use_small)
        print(f"\nUpdated data sizes:")
        print(f"  n_users     : {updated_cfg['data']['n_users']:,}")
        print(f"  n_items     : {updated_cfg['data']['n_items']:,}")
        print(f"  n_categories: {updated_cfg['data']['n_categories']}")
        print(f"\nIMPORTANT: Update configs/base_config.yaml with these sizes "
              "before running feature_engineering.py!")
