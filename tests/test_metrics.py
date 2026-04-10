"""Unit tests for evaluation metrics."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pytest

from src.evaluation.metrics import (
    recall_at_k, ndcg_at_k, hit_rate_at_k,
    compute_retrieval_metrics,
    compute_auc, compute_gauc, compute_logloss,
)


class TestRetrievalMetrics:
    def test_recall_perfect(self):
        retrieved = np.array([1, 2, 3, 4, 5])
        relevant = {1, 2, 3}
        assert recall_at_k(retrieved, relevant, k=5) == 1.0

    def test_recall_partial(self):
        retrieved = np.array([1, 10, 11, 12, 13])
        relevant = {1, 2, 3}
        # Only item 1 is in top-5; recall = 1/3
        assert abs(recall_at_k(retrieved, relevant, k=5) - 1 / 3) < 1e-9

    def test_recall_zero(self):
        retrieved = np.array([10, 11, 12])
        relevant = {1, 2, 3}
        assert recall_at_k(retrieved, relevant, k=3) == 0.0

    def test_recall_empty_relevant(self):
        retrieved = np.array([1, 2, 3])
        assert recall_at_k(retrieved, set(), k=3) == 0.0

    def test_ndcg_perfect_order(self):
        # All relevant items at top — NDCG should be 1.0
        retrieved = np.array([1, 2, 3, 99, 100])
        relevant = {1, 2, 3}
        assert abs(ndcg_at_k(retrieved, relevant, k=5) - 1.0) < 1e-6

    def test_ndcg_worst_order(self):
        # All relevant items below k — NDCG@3 = 0
        retrieved = np.array([10, 11, 12, 1, 2])
        relevant = {1, 2, 3}
        assert ndcg_at_k(retrieved, relevant, k=3) == 0.0

    def test_hit_rate_found(self):
        retrieved = np.array([99, 1, 98])
        relevant = {1}
        assert hit_rate_at_k(retrieved, relevant, k=3) == 1.0

    def test_hit_rate_not_found(self):
        retrieved = np.array([99, 98, 97])
        relevant = {1}
        assert hit_rate_at_k(retrieved, relevant, k=3) == 0.0

    def test_compute_retrieval_metrics_keys(self):
        user_retrieved = {0: np.array([1, 2, 3, 10, 11])}
        user_relevant  = {0: {1, 2}}
        metrics = compute_retrieval_metrics(user_retrieved, user_relevant, k_list=[3, 5])
        assert "recall@3" in metrics
        assert "ndcg@3" in metrics
        assert "hit@3" in metrics
        assert "recall@5" in metrics

    def test_compute_retrieval_metrics_values(self):
        user_retrieved = {0: np.array([1, 99, 2])}
        user_relevant  = {0: {1, 2}}
        metrics = compute_retrieval_metrics(user_retrieved, user_relevant, k_list=[3])
        assert abs(metrics["recall@3"] - 1.0) < 1e-9


class TestRankingMetrics:
    def test_auc_perfect(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        assert compute_auc(labels, scores) == 1.0

    def test_auc_random(self):
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 2, size=100)
        scores = rng.random(100)
        auc = compute_auc(labels, scores)
        assert 0.0 <= auc <= 1.0

    def test_gauc_equals_auc_single_user(self):
        user_ids = np.zeros(4, dtype=int)
        labels   = np.array([0, 0, 1, 1])
        scores   = np.array([0.1, 0.2, 0.8, 0.9])
        gauc = compute_gauc(user_ids, labels, scores)
        assert abs(gauc - 1.0) < 1e-9

    def test_gauc_skip_single_class_user(self):
        """Users with only positives or only negatives are skipped."""
        user_ids = np.array([0, 0, 1, 1])
        labels   = np.array([1, 1, 0, 1])   # user 0: only pos → skip
        scores   = np.array([0.8, 0.9, 0.1, 0.8])
        gauc = compute_gauc(user_ids, labels, scores)
        assert 0.0 <= gauc <= 1.0   # only user 1 contributes

    def test_logloss_perfect(self):
        labels = np.array([1.0, 0.0])
        scores = np.array([1.0 - 1e-7, 1e-7])
        loss = compute_logloss(labels, scores)
        assert loss < 0.01

    def test_logloss_worst(self):
        labels = np.array([1.0, 0.0])
        scores = np.array([0.0 + 1e-7, 1.0 - 1e-7])
        loss = compute_logloss(labels, scores)
        assert loss > 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
