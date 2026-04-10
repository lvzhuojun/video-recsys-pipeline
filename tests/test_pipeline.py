"""Integration tests for the end-to-end recommendation pipeline.

These tests verify that data can flow through the full pipeline:
download_data → feature_engineering → dataset → model → faiss → metrics
"""

import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import pytest

MOCK_META = {
    "n_users": 20,
    "n_items": 50,
    "n_categories": 5,
    "n_duration_bkts": 5,
    "user_dense_dim": 25,
    "item_dense_dim": 3,
    "seq_len": 5,
    "pos_thresh": 0.7,
}

RETRIEVAL_CFG = {
    "model": {
        "embed_dim": 8,
        "seq_embed_dim": 8,
        "cat_embed_dim": 4,
        "dur_embed_dim": 4,
        "dense_hidden": 8,
        "output_dim": 16,
        "temperature": 0.07,
        "dropout": 0.0,
    }
}

DEVICE = torch.device("cpu")


def _make_fake_data(n=100, pos_rate=0.2, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    n_users, n_items = MOCK_META["n_users"], MOCK_META["n_items"]
    seq_len = MOCK_META["seq_len"]
    return {
        "user_ids":      rng.integers(0, n_users, n).astype(np.int32),
        "item_ids":      rng.integers(0, n_items, n).astype(np.int32),
        "item_category": rng.integers(0, 5, n).astype(np.int32),
        "item_dur_bkt":  rng.integers(0, 5, n).astype(np.int32),
        "user_dense":    rng.random((n, 25)).astype(np.float32),
        "item_dense":    rng.random((n, 3)).astype(np.float32),
        "history_seqs":  rng.integers(0, n_items + 1, (n, seq_len)).astype(np.int32),
        "history_lens":  rng.integers(1, seq_len + 1, n).astype(np.int32),
        "labels":        (rng.random(n) < pos_rate).astype(np.float32),
    }


class TestDatasetPipeline:
    """Test that Dataset classes work with fake data."""

    def test_retrieval_dataset_in_batch(self):
        from src.data.dataset import RetrievalDataset
        data = _make_fake_data(n=50, pos_rate=0.5)
        ds = RetrievalDataset(data, MOCK_META, neg_mode="in_batch")
        assert len(ds) == int((data["labels"] == 1).sum())
        sample = ds[0]
        assert "user_id" in sample and "pos_item_id" in sample
        assert "neg_item_id" not in sample   # in_batch doesn't add neg keys

    def test_retrieval_dataset_random_neg(self):
        from src.data.dataset import RetrievalDataset
        data = _make_fake_data(n=50, pos_rate=0.5)
        ds = RetrievalDataset(data, MOCK_META, neg_mode="random")
        sample = ds[0]
        assert "neg_item_id" in sample

    def test_ranking_dataset(self):
        from src.data.dataset import RankingDataset
        data = _make_fake_data(n=100)
        ds = RankingDataset(data, MOCK_META)
        assert len(ds) == 100
        sample = ds[0]
        assert "label" in sample
        assert sample["user_dense"].shape == (25,)
        assert sample["history_seq"].shape == (5,)


class TestModelPipeline:
    """Test that models produce valid outputs from dataset batches."""

    def _collate(self, dataset, n=4):
        samples = [dataset[i] for i in range(n)]
        return {k: torch.stack([s[k] for s in samples]) for k in samples[0]}

    def test_two_tower_forward(self):
        from src.data.dataset import RetrievalDataset
        from src.models.two_tower import TwoTowerModel
        data = _make_fake_data(n=50, pos_rate=0.5)
        ds = RetrievalDataset(data, MOCK_META, neg_mode="in_batch")
        batch = self._collate(ds)
        model = TwoTowerModel(MOCK_META, RETRIEVAL_CFG)
        user_emb, item_emb = model(batch)
        assert user_emb.shape == (4, 16)
        # L2 norms should be 1
        assert torch.allclose(torch.norm(user_emb, dim=-1), torch.ones(4), atol=1e-5)

    def test_deepfm_forward(self):
        from src.data.dataset import RankingDataset
        from src.models.deepfm import DeepFM
        cfg = {
            "model": {
                "user_embed_dim": 8, "item_embed_dim": 8,
                "cat_embed_dim": 4, "dur_embed_dim": 4,
                "deepfm": {"fm_embed_dim": 8, "mlp_hidden_dims": [16], "dropout": 0.0},
            }
        }
        data = _make_fake_data(n=50)
        ds = RankingDataset(data, MOCK_META)
        batch = self._collate(ds)
        model = DeepFM(MOCK_META, cfg)
        preds = model(batch)
        assert preds.shape == (4,)
        assert ((preds >= 0) & (preds <= 1)).all()


class TestFaissIndexPipeline:
    """Test Faiss index build + search."""

    def test_build_and_search(self):
        from src.retrieval.faiss_index import FaissIndex
        dim = 16
        n_items = 100
        embs = np.random.randn(n_items, dim).astype(np.float32)
        # L2 normalise
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
        embs /= norms
        item_ids = np.arange(n_items, dtype=np.int32)

        idx = FaissIndex(dim=dim, index_type="flat")
        idx.build(embs, item_ids)

        query = embs[0:1]
        scores, retrieved = idx.search(query, top_k=5)
        assert retrieved.shape == (1, 5)
        # Top result should be the query itself
        assert retrieved[0, 0] == 0

    def test_save_and_load(self):
        from src.retrieval.faiss_index import FaissIndex
        dim = 8
        embs = np.random.randn(20, dim).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
        embs /= norms
        ids = np.arange(20, dtype=np.int32)

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "test_index")
            idx1 = FaissIndex(dim=dim)
            idx1.build(embs, ids)
            idx1.save(path)

            idx2 = FaissIndex(dim=dim)
            idx2.load(path)
            _, r1 = idx1.search(embs[0], top_k=3)
            _, r2 = idx2.search(embs[0], top_k=3)
            np.testing.assert_array_equal(r1, r2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
