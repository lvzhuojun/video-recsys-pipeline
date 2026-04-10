"""Full two-stage recommendation pipeline: recall → rank → top-K results.

Usage:
    python main.py --user_id 42
    python main.py --user_id 42 --ranker din --recall_k 100 --top_k 10
    python main.py --user_id 42 --no_train   # skip training, use existing checkpoints

Pipeline:
    1. [Optional] Train Two-Tower + ranking model if checkpoints not found
    2. Encode user → Faiss search → recall_k candidates (Two-Tower)
    3. Score each candidate with ranking model (DeepFM/DIN)
    4. Return top_k results with scores and item metadata
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import load_meta, load_split
from src.models.deepfm import DeepFM
from src.models.din import DIN
from src.models.two_tower import TwoTowerModel
from src.retrieval.faiss_index import FaissIndex
from src.training.train_retrieval import (
    build_item_feature_lookup,
    encode_all_items,
)
from src.utils.gpu_utils import get_device, set_seed
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _read_cfg(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_trained(
    ckpt_dir: Path,
    ranker: str,
    neg_mode: str,
    no_train: bool,
) -> None:
    """Train models if checkpoints are missing (unless --no_train)."""
    recall_ckpt = ckpt_dir / "two_tower_best.pt"
    rank_ckpt   = ckpt_dir / f"{ranker}_best.pt"

    need_recall = not recall_ckpt.exists()
    need_rank   = not rank_ckpt.exists()

    if no_train and (need_recall or need_rank):
        missing = []
        if need_recall: missing.append(str(recall_ckpt))
        if need_rank:   missing.append(str(rank_ckpt))
        raise FileNotFoundError(
            f"--no_train specified but checkpoints not found:\n"
            + "\n".join(f"  {p}" for p in missing)
            + "\nRun without --no_train to train first."
        )

    if need_recall:
        logger.info("Two-Tower checkpoint not found — training now...")
        from src.training.train_retrieval import main as train_retrieval
        train_retrieval(neg_mode=neg_mode)
    else:
        logger.info(f"Two-Tower checkpoint found: {recall_ckpt}")

    if need_rank:
        logger.info(f"{ranker.upper()} checkpoint not found — training now...")
        from src.training.train_ranking import main as train_ranking
        train_ranking(model_arg=ranker)
    else:
        logger.info(f"{ranker.upper()} checkpoint found: {rank_ckpt}")


def _load_two_tower(
    meta: dict,
    ret_cfg: dict,
    ckpt_path: str,
    device: torch.device,
) -> TwoTowerModel:
    model = TwoTowerModel(meta, ret_cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


def _load_ranker(
    name: str,
    meta: dict,
    rank_cfg: dict,
    ckpt_path: str,
    device: torch.device,
) -> torch.nn.Module:
    if name == "deepfm":
        model = DeepFM(meta, rank_cfg).to(device)
    elif name == "din":
        model = DIN(meta, rank_cfg).to(device)
    else:
        raise ValueError(f"Unknown ranker: {name}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


def _get_user_features(
    user_id: int,
    data: dict,
) -> Dict:
    """Return the last interaction row for user_id (most complete history)."""
    user_last: Dict[int, int] = {}
    for i, uid in enumerate(data["user_ids"]):
        if int(uid) == user_id:
            user_last[user_id] = i
    if user_id not in user_last:
        valid_sample = sorted(set(int(u) for u in data["user_ids"]))[:10]
        raise ValueError(
            f"user_id={user_id} not found in test data. "
            f"Sample of valid IDs: {valid_sample} ..."
        )
    idx = user_last[user_id]
    return {
        "user_id":    int(data["user_ids"][idx]),
        "user_dense": data["user_dense"][idx],
        "history_seq": data["history_seqs"][idx],
        "history_len": int(data["history_lens"][idx]),
    }


@torch.no_grad()
def recall(
    retrieval_model: TwoTowerModel,
    item_lookup: Dict[int, dict],
    user_feat: Dict,
    recall_k: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stage 1: Two-Tower → Faiss → top-recall_k item IDs.

    Returns:
        (item_ids, scores) numpy arrays of shape (recall_k,).
    """
    # Encode all items and build Faiss
    item_ids_arr, item_embs = encode_all_items(
        retrieval_model, item_lookup, device, batch_size=512
    )
    idx = FaissIndex(dim=item_embs.shape[1], index_type="flat")
    idx.build(item_embs, item_ids_arr)

    # Encode user
    uid_t   = torch.tensor([user_feat["user_id"]], dtype=torch.long, device=device)
    den_t   = torch.tensor(user_feat["user_dense"][np.newaxis], dtype=torch.float32, device=device)
    hist_t  = torch.tensor(user_feat["history_seq"][np.newaxis], dtype=torch.long, device=device)
    hlen_t  = torch.tensor([user_feat["history_len"]], dtype=torch.long, device=device)
    user_emb = retrieval_model.encode_user(
        {"user_id": uid_t, "user_dense": den_t, "history_seq": hist_t, "history_len": hlen_t}
    ).cpu().numpy()

    scores, candidate_ids = idx.search(user_emb, top_k=recall_k)
    return candidate_ids[0], scores[0]


@torch.no_grad()
def rank(
    ranker: torch.nn.Module,
    candidate_ids: np.ndarray,
    item_lookup: Dict[int, dict],
    user_feat: Dict,
    device: torch.device,
) -> np.ndarray:
    """Stage 2: ranking model scores for each candidate.

    Returns:
        scores numpy array of shape (len(candidate_ids),).
    """
    ranker.eval()
    B = len(candidate_ids)
    uid   = user_feat["user_id"]
    u_den = user_feat["user_dense"]
    hist  = user_feat["history_seq"]
    hlen  = user_feat["history_len"]

    batch = {
        "user_id":       torch.tensor([uid]   * B, dtype=torch.long, device=device),
        "user_dense":    torch.tensor(np.tile(u_den, (B, 1)), dtype=torch.float32, device=device),
        "history_seq":   torch.tensor(np.tile(hist,  (B, 1)), dtype=torch.long, device=device),
        "history_len":   torch.tensor([hlen]  * B, dtype=torch.long, device=device),
        "item_id":       torch.tensor(candidate_ids.astype(np.int64), dtype=torch.long, device=device),
        "item_dense":    torch.tensor(
            np.array([item_lookup[int(i)]["item_dense"] for i in candidate_ids], dtype=np.float32),
            device=device,
        ),
        "item_category": torch.tensor(
            [item_lookup[int(i)]["item_category"] for i in candidate_ids],
            dtype=torch.long, device=device,
        ),
        "item_dur_bkt":  torch.tensor(
            [item_lookup[int(i)]["item_dur_bkt"] for i in candidate_ids],
            dtype=torch.long, device=device,
        ),
        "label": torch.zeros(B, dtype=torch.float32, device=device),  # placeholder
    }

    scores = ranker(batch).cpu().numpy()
    return scores


def recommend(
    user_id: int,
    recall_k: int = 100,
    top_k: int = 10,
    ranker_name: str = "deepfm",
    no_train: bool = False,
    neg_mode: str = "in_batch",
) -> List[Dict]:
    """End-to-end recommendation for a single user.

    Args:
        user_id: Integer user ID (0-based, must exist in test split).
        recall_k: Number of candidates to retrieve from Faiss.
        top_k: Number of final recommendations to return.
        ranker_name: ``"deepfm"`` or ``"din"``.
        no_train: If True, raise if checkpoints are missing instead of training.
        neg_mode: Negative sampling strategy (used only when training).

    Returns:
        List of dicts, each with keys: item_id, recall_score, rank_score,
        item_category, item_dur_bkt, item_dense_norm.
    """
    set_seed(42)
    device = get_device()

    base_cfg  = _read_cfg(ROOT / "configs/base_config.yaml")
    ret_cfg   = _read_cfg(ROOT / "configs/retrieval_config.yaml")
    rank_cfg  = _read_cfg(ROOT / "configs/ranking_config.yaml")

    proc_dir  = ROOT / base_cfg["data"]["processed_dir"]
    ckpt_dir  = ROOT / base_cfg["logging"]["checkpoint_dir"]

    _ensure_trained(ckpt_dir, ranker_name, neg_mode, no_train)

    meta       = load_meta(proc_dir)
    train_data = load_split(proc_dir, "train")
    val_data   = load_split(proc_dir, "val")
    test_data  = load_split(proc_dir, "test")
    item_lookup = build_item_feature_lookup(train_data, val_data, test_data)

    retrieval_model = _load_two_tower(
        meta, ret_cfg, str(ckpt_dir / "two_tower_best.pt"), device
    )
    ranker = _load_ranker(
        ranker_name, meta, rank_cfg, str(ckpt_dir / f"{ranker_name}_best.pt"), device
    )

    user_feat = _get_user_features(user_id, test_data)

    # ── Stage 1: Recall ──────────────────────────────────────────────────────
    logger.info(f"Stage 1: Two-Tower recall (top {recall_k} from {len(item_lookup)} items)")
    candidate_ids, recall_scores = recall(
        retrieval_model, item_lookup, user_feat, recall_k, device
    )

    # ── Stage 2: Ranking ─────────────────────────────────────────────────────
    logger.info(f"Stage 2: {ranker_name.upper()} ranking ({recall_k} candidates → top {top_k})")
    rank_scores = rank(ranker, candidate_ids, item_lookup, user_feat, device)

    # ── Merge and sort ───────────────────────────────────────────────────────
    order = np.argsort(-rank_scores)[:top_k]
    results = []
    for rank_pos, idx in enumerate(order, start=1):
        iid = int(candidate_ids[idx])
        feat = item_lookup[iid]
        results.append({
            "rank":            rank_pos,
            "item_id":         iid,
            "recall_score":    float(recall_scores[idx]),
            "rank_score":      float(rank_scores[idx]),
            "item_category":   feat["item_category"],
            "item_dur_bkt":    feat["item_dur_bkt"],
            "item_dense_norm": float(np.linalg.norm(feat["item_dense"])),
        })

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-stage recommendation pipeline"
    )
    parser.add_argument("--user_id",   type=int, default=42,
                        help="User ID to recommend for (must exist in test split; not all user IDs are in test)")
    parser.add_argument("--recall_k",  type=int, default=100,
                        help="Number of candidates from Two-Tower Faiss recall")
    parser.add_argument("--top_k",     type=int, default=10,
                        help="Number of final recommendations to show")
    parser.add_argument("--ranker",    choices=["deepfm", "din"], default="deepfm",
                        help="Ranking model to use")
    parser.add_argument("--neg_mode",  choices=["in_batch", "random"], default="in_batch",
                        help="Negative sampling for Two-Tower (used if training needed)")
    parser.add_argument("--no_train",  action="store_true",
                        help="Do not train; raise if checkpoints are missing")
    args = parser.parse_args()

    results = recommend(
        user_id=args.user_id,
        recall_k=args.recall_k,
        top_k=args.top_k,
        ranker_name=args.ranker,
        no_train=args.no_train,
        neg_mode=args.neg_mode,
    )

    print(f"\n{'='*60}")
    print(f"  Top-{args.top_k} Recommendations for User {args.user_id}")
    print(f"  Recall: Two-Tower → Faiss top-{args.recall_k}")
    print(f"  Ranker: {args.ranker.upper()}")
    print(f"{'='*60}")
    print(f"{'Rank':>4}  {'ItemID':>6}  {'RecallScore':>11}  {'RankScore':>9}  {'Cat':>3}  {'Dur':>3}")
    print(f"{'-'*60}")
    for r in results:
        print(
            f"{r['rank']:>4}  {r['item_id']:>6}  "
            f"{r['recall_score']:>11.4f}  {r['rank_score']:>9.4f}  "
            f"{r['item_category']:>3}  {r['item_dur_bkt']:>3}"
        )
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
