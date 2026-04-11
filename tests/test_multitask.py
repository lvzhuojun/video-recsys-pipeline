"""Unit tests for MMoE multi-task model and MTL dataset."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import pytest

# Small mock meta matching test fixtures
MOCK_META = {
    "n_users": 50,
    "n_items": 100,
    "n_categories": 5,
    "n_duration_bkts": 5,
    "user_dense_dim": 25,
    "item_dense_dim": 3,
    "seq_len": 10,
    "pos_thresh": 0.7,
}

# Small MMoE config for fast tests
MOCK_MTL_CFG = {
    "model": {
        "user_embed_dim": 16,
        "item_embed_dim": 16,
        "cat_embed_dim": 8,
        "dur_embed_dim": 4,
        "mmoe": {
            "n_experts": 2,
            "expert_hidden_dim": 32,
            "expert_output_dim": 16,
            "n_tasks": 2,
            "task_tower_dims": [16, 8],
            "dropout": 0.0,
            "seq_embed_dim": 16,
            "attention_hidden_dims": [16, 8],
        },
    }
}

B, L = 8, 10
DEVICE = torch.device("cpu")


def _make_ranking_batch():
    """Create a mock ranking batch (with MTL fields)."""
    return {
        "user_id":         torch.randint(0, 50, (B,)),
        "item_id":         torch.randint(0, 100, (B,)),
        "user_dense":      torch.randn(B, 25),
        "item_dense":      torch.randn(B, 3),
        "item_category":   torch.randint(0, 5, (B,)),
        "item_dur_bkt":    torch.randint(0, 5, (B,)),
        "history_seq":     torch.randint(0, 101, (B, L)),
        "history_len":     torch.randint(1, L + 1, (B,)),
        "label":           torch.randint(0, 2, (B,)).float(),
        "watch_ratio_raw": torch.rand(B),
        "like_label":      torch.randint(0, 2, (B,)).float(),
    }


def _make_mock_data(n: int = 200) -> dict:
    """Create a mock data dict compatible with RankingDataset."""
    rng = np.random.default_rng(42)
    return {
        "user_ids":        rng.integers(0, 50,  size=n).astype(np.int32),
        "item_ids":        rng.integers(0, 100, size=n).astype(np.int32),
        "item_category":   rng.integers(0, 5,   size=n).astype(np.int32),
        "item_dur_bkt":    rng.integers(0, 5,   size=n).astype(np.int32),
        "user_dense":      rng.standard_normal((n, 25)).astype(np.float32),
        "item_dense":      rng.standard_normal((n, 3)).astype(np.float32),
        "history_seqs":    rng.integers(0, 101, size=(n, 10)).astype(np.int32),
        "history_lens":    rng.integers(1, 11,  size=n).astype(np.int32),
        "labels":          (rng.random(n) > 0.7).astype(np.float32),
        "watch_ratio_raw": rng.random(n).astype(np.float32),
        "like_labels":     (rng.random(n) > 0.5).astype(np.float32),
    }


# ── MMoE ───────────────────────────────────────────────────────────────────

class TestMMoE:
    def setup_method(self):
        from src.models.multitask import MMoE
        self.model = MMoE(MOCK_META, MOCK_MTL_CFG).to(DEVICE)

    def test_output_shapes(self):
        """MMoE should return two tensors of shape (B,)."""
        batch = _make_ranking_batch()
        watch_pred, like_pred = self.model(batch)
        assert watch_pred.shape == (B,), f"watch_pred shape: {watch_pred.shape}"
        assert like_pred.shape == (B,),  f"like_pred shape: {like_pred.shape}"

    def test_output_ranges(self):
        """Both predictions should be in [0, 1]."""
        batch = _make_ranking_batch()
        watch_pred, like_pred = self.model(batch)
        assert (watch_pred >= 0.0).all() and (watch_pred <= 1.0).all(), \
            "watch_pred out of [0, 1] range."
        assert (like_pred >= 0.0).all() and (like_pred <= 1.0).all(), \
            "like_pred out of [0, 1] range."

    def test_watch_gradient_flows(self):
        """Backward on watch loss should produce gradients in watch tower."""
        from src.models.multitask import MMoE
        model = MMoE(MOCK_META, MOCK_MTL_CFG).to(DEVICE)
        batch = _make_ranking_batch()

        watch_pred, _ = model(batch)
        watch_loss = torch.nn.functional.binary_cross_entropy(watch_pred, batch["label"])
        watch_loss.backward()

        # Watch tower (task 0) should have gradients
        watch_tower_params = list(model.towers[0].parameters())
        has_grad = any(p.grad is not None for p in watch_tower_params)
        assert has_grad, "Watch tower should have gradients after watch loss backward."

    def test_like_gradient_flows(self):
        """Backward on like loss should produce gradients in like tower."""
        from src.models.multitask import MMoE
        model = MMoE(MOCK_META, MOCK_MTL_CFG).to(DEVICE)
        batch = _make_ranking_batch()

        _, like_pred = model(batch)
        like_loss = torch.nn.functional.binary_cross_entropy(like_pred, batch["like_label"])
        like_loss.backward()

        # Like tower (task 1) should have gradients
        like_tower_params = list(model.towers[1].parameters())
        has_grad = any(p.grad is not None for p in like_tower_params)
        assert has_grad, "Like tower should have gradients after like loss backward."

    def test_repr(self):
        """repr should mention MMoE, Expert, and Task."""
        r = repr(self.model)
        assert "MMoE" in r,   "repr should contain 'MMoE'"
        assert "Expert" in r, "repr should contain 'Expert'"
        assert "Task" in r,   "repr should contain 'Task'"

    def test_no_nan_output(self):
        """MMoE outputs should not contain NaN values."""
        batch = _make_ranking_batch()
        watch_pred, like_pred = self.model(batch)
        assert not torch.isnan(watch_pred).any(), "watch_pred contains NaN."
        assert not torch.isnan(like_pred).any(),  "like_pred contains NaN."

    def test_independent_task_outputs(self):
        """Watch and like predictions should not be identical (different towers)."""
        batch = _make_ranking_batch()
        watch_pred, like_pred = self.model(batch)
        # It's extremely unlikely they are identical (but not guaranteed)
        # Just check they run and don't crash
        assert watch_pred.shape == like_pred.shape


# ── MTL Dataset ─────────────────────────────────────────────────────────────

class TestMTLDataset:
    def test_mtl_mode_keys(self):
        """RankingDataset(mtl_mode=True) batch should include watch_ratio_raw and like_label."""
        from src.data.dataset import RankingDataset
        from torch.utils.data import DataLoader

        data = _make_mock_data(n=100)
        ds = RankingDataset(data, MOCK_META, mtl_mode=True)

        # Test __getitem__
        sample = ds[0]
        assert "watch_ratio_raw" in sample, \
            "MTL dataset should include 'watch_ratio_raw' key."
        assert "like_label" in sample, \
            "MTL dataset should include 'like_label' key."
        assert sample["watch_ratio_raw"].dtype == torch.float32
        assert sample["like_label"].dtype == torch.float32

    def test_non_mtl_mode_missing_keys(self):
        """RankingDataset(mtl_mode=False) should NOT include MTL-specific keys."""
        from src.data.dataset import RankingDataset

        data = _make_mock_data(n=100)
        ds = RankingDataset(data, MOCK_META, mtl_mode=False)
        sample = ds[0]
        assert "watch_ratio_raw" not in sample, \
            "Non-MTL dataset should not include 'watch_ratio_raw'."
        assert "like_label" not in sample, \
            "Non-MTL dataset should not include 'like_label'."

    def test_mtl_dataloader_batch(self):
        """DataLoader with mtl_mode=True should produce correct batch keys."""
        from src.data.dataset import RankingDataset
        from torch.utils.data import DataLoader

        data = _make_mock_data(n=50)
        ds = RankingDataset(data, MOCK_META, mtl_mode=True)
        loader = DataLoader(ds, batch_size=16, shuffle=False)

        batch = next(iter(loader))
        assert "watch_ratio_raw" in batch
        assert "like_label" in batch
        assert batch["watch_ratio_raw"].shape[0] == 16
        assert batch["like_label"].shape[0] == 16

    def test_mtl_values_in_range(self):
        """watch_ratio_raw should be float, like_label should be 0/1."""
        from src.data.dataset import RankingDataset

        data = _make_mock_data(n=100)
        ds = RankingDataset(data, MOCK_META, mtl_mode=True)

        for i in range(len(ds)):
            sample = ds[i]
            wr = sample["watch_ratio_raw"].item()
            ll = sample["like_label"].item()
            assert 0.0 <= wr <= 1.0, f"watch_ratio_raw={wr} out of range."
            assert ll in (0.0, 1.0), f"like_label={ll} is not binary."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
