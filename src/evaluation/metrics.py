"""Evaluation metrics for both retrieval and ranking stages.

Retrieval metrics (used by Two-Tower / Faiss):
    recall_at_k    — fraction of relevant items retrieved in top-K
    ndcg_at_k      — Normalised Discounted Cumulative Gain
    hit_rate_at_k  — binary: 1 if any relevant item in top-K
    compute_retrieval_metrics — batch version over all test users

Ranking metrics (used by DeepFM / DIN):
    compute_auc    — overall ROC-AUC
    compute_gauc   — group (per-user) weighted AUC; preferred in industry
    compute_logloss — binary cross-entropy loss
"""

from collections import defaultdict
from typing import Dict, List, Set

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

from ..utils.logger import get_logger

logger = get_logger(__name__)


# ===========================================================================
# Retrieval metrics
# ===========================================================================

def recall_at_k(retrieved: np.ndarray, relevant: Set[int], k: int) -> float:
    """Recall@K for a single user.

    Args:
        retrieved: Ordered array of retrieved item IDs.
        relevant: Set of ground-truth positive item IDs.
        k: Cut-off rank.

    Returns:
        Float in [0, 1].
    """
    if not relevant:
        return 0.0
    top_k = set(int(x) for x in retrieved[:k])
    return len(top_k & relevant) / len(relevant)


def ndcg_at_k(retrieved: np.ndarray, relevant: Set[int], k: int) -> float:
    """NDCG@K for a single user.

    DCG uses binary relevance (0/1).  IDCG is the score of a perfect
    ranking that places all relevant items at the top.

    Args:
        retrieved: Ordered array of retrieved item IDs.
        relevant: Set of ground-truth positive item IDs.
        k: Cut-off rank.

    Returns:
        Float in [0, 1].
    """
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / np.log2(rank + 2)
        for rank, item in enumerate(retrieved[:k])
        if int(item) in relevant
    )
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(retrieved: np.ndarray, relevant: Set[int], k: int) -> float:
    """Hit Rate@K for a single user (1 if any relevant item in top-K).

    Args:
        retrieved: Ordered array of retrieved item IDs.
        relevant: Set of ground-truth positive item IDs.
        k: Cut-off rank.

    Returns:
        1.0 or 0.0.
    """
    if not relevant:
        return 0.0
    top_k = set(int(x) for x in retrieved[:k])
    return 1.0 if top_k & relevant else 0.0


def compute_retrieval_metrics(
    user_retrieved: Dict[int, np.ndarray],
    user_relevant: Dict[int, Set[int]],
    k_list: List[int] = (10, 50),
) -> Dict[str, float]:
    """Compute retrieval metrics averaged across all users.

    Only users with at least one relevant item are included.

    Args:
        user_retrieved: ``{user_id: sorted_item_id_array}``.
        user_relevant:  ``{user_id: set_of_positive_item_ids}``.
        k_list: List of K values to evaluate.

    Returns:
        Dict like ``{'recall@10': 0.23, 'ndcg@10': 0.19, 'hit@10': 0.41, ...}``.
    """
    results: Dict[str, List[float]] = defaultdict(list)

    for uid, relevant in user_relevant.items():
        if not relevant or uid not in user_retrieved:
            continue
        retrieved = user_retrieved[uid]
        for k in k_list:
            results[f"recall@{k}"].append(recall_at_k(retrieved, relevant, k))
            results[f"ndcg@{k}"].append(ndcg_at_k(retrieved, relevant, k))
            results[f"hit@{k}"].append(hit_rate_at_k(retrieved, relevant, k))

    if not results:
        logger.warning("No users with relevant items found — returning zeros.")
        return {f"recall@{k}": 0.0 for k in k_list}

    return {k: float(np.mean(v)) for k, v in results.items()}


# ===========================================================================
# Ranking metrics
# ===========================================================================

def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Overall ROC-AUC score.

    Args:
        labels: Binary ground-truth array (0 / 1).
        scores: Predicted probability or score array.

    Returns:
        AUC value in [0, 1].
    """
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError as e:
        logger.warning(f"AUC computation failed: {e}")
        return 0.0


def compute_gauc(
    user_ids: np.ndarray, labels: np.ndarray, scores: np.ndarray
) -> float:
    """Group AUC (GAUC) — per-user AUC weighted by number of interactions.

    GAUC is preferred over global AUC in industry because it measures
    intra-user ranking quality; a model that always recommends globally
    popular items can have high AUC but low GAUC.

    Args:
        user_ids: Array of user identifiers aligned with labels/scores.
        labels: Binary ground-truth array.
        scores: Predicted probability or score array.

    Returns:
        Weighted mean per-user AUC.
    """
    user_aucs: List[float] = []
    user_weights: List[int] = []

    for uid in np.unique(user_ids):
        mask = user_ids == uid
        u_labels = labels[mask]
        u_scores = scores[mask]

        # Skip users with only one class — AUC is undefined
        if len(np.unique(u_labels)) < 2:
            continue

        try:
            auc = float(roc_auc_score(u_labels, u_scores))
            user_aucs.append(auc)
            user_weights.append(int(mask.sum()))
        except ValueError:
            continue

    if not user_aucs:
        logger.warning("GAUC: no valid users found — returning 0.")
        return 0.0

    total_w = sum(user_weights)
    return float(sum(a * w for a, w in zip(user_aucs, user_weights)) / total_w)


def compute_logloss(labels: np.ndarray, scores: np.ndarray) -> float:
    """Binary cross-entropy log-loss (lower is better).

    Args:
        labels: Binary ground-truth array.
        scores: Predicted probability array.

    Returns:
        Log-loss value.
    """
    scores_clipped = np.clip(scores, 1e-7, 1 - 1e-7)
    try:
        return float(log_loss(labels, scores_clipped))
    except ValueError as e:
        logger.warning(f"LogLoss computation failed: {e}")
        return float("inf")
