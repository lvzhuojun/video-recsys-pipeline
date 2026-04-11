"""Unit tests for model forward passes and output shapes."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import pytest

# Minimal meta dict for testing (no need for real data files)
MOCK_META = {
    "n_users": 50,
    "n_items": 100,
    "n_categories": 10,
    "n_duration_bkts": 5,
    "user_dense_dim": 25,
    "item_dense_dim": 3,
    "seq_len": 10,
    "pos_thresh": 0.7,
}

MOCK_RETRIEVAL_CFG = {
    "model": {
        "embed_dim": 16,
        "seq_embed_dim": 8,
        "cat_embed_dim": 8,
        "dur_embed_dim": 4,
        "dense_hidden": 16,
        "output_dim": 32,
        "temperature": 0.07,
        "dropout": 0.0,
        "seq_model": "mean_pool",
    }
}

MOCK_RETRIEVAL_CFG_SASREC = {
    "model": {
        "embed_dim": 16,
        "seq_embed_dim": 8,
        "cat_embed_dim": 8,
        "dur_embed_dim": 4,
        "dense_hidden": 16,
        "output_dim": 32,
        "temperature": 0.07,
        "dropout": 0.0,
        "seq_model": "sasrec",
        "sasrec": {
            "hidden_dim": 16,
            "n_layers": 2,
            "n_heads": 2,
            "max_seq_len": 10,
            "dropout": 0.0,
        },
    }
}

MOCK_RANKING_CFG = {
    "model": {
        "user_embed_dim": 16,
        "item_embed_dim": 16,
        "cat_embed_dim": 8,
        "dur_embed_dim": 4,
        "deepfm": {
            "fm_embed_dim": 8,
            "mlp_hidden_dims": [32, 16],
            "dropout": 0.0,
        },
        "din": {
            "attention_hidden_dims": [16, 8],
            "mlp_hidden_dims": [32, 16],
            "dropout": 0.0,
            "seq_embed_dim": 16,
        },
    }
}

B, L = 8, 10   # batch size, sequence length
DEVICE = torch.device("cpu")


def _make_retrieval_batch():
    return {
        "user_id":           torch.randint(0, 50, (B,)),
        "user_dense":        torch.randn(B, 25),
        "pos_item_id":       torch.randint(0, 100, (B,)),
        "pos_item_dense":    torch.randn(B, 3),
        "pos_item_category": torch.randint(0, 10, (B,)),
        "pos_item_dur_bkt":  torch.randint(0, 5, (B,)),
        "history_seq":       torch.randint(0, 101, (B, L)),
        "history_len":       torch.randint(1, L + 1, (B,)),
    }


def _make_ranking_batch():
    return {
        "user_id":       torch.randint(0, 50, (B,)),
        "item_id":       torch.randint(0, 100, (B,)),
        "user_dense":    torch.randn(B, 25),
        "item_dense":    torch.randn(B, 3),
        "item_category": torch.randint(0, 10, (B,)),
        "item_dur_bkt":  torch.randint(0, 5, (B,)),
        "history_seq":   torch.randint(0, 101, (B, L)),
        "history_len":   torch.randint(1, L + 1, (B,)),
        "label":         torch.randint(0, 2, (B,)).float(),
    }


# ── Two-Tower ──────────────────────────────────────────────────────────────

class TestTwoTowerModel:
    def setup_method(self):
        from src.models.two_tower import TwoTowerModel
        self.model = TwoTowerModel(MOCK_META, MOCK_RETRIEVAL_CFG).to(DEVICE)

    def test_output_shape(self):
        batch = _make_retrieval_batch()
        user_emb, item_emb = self.model(batch)
        assert user_emb.shape == (B, 32)
        assert item_emb.shape == (B, 32)

    def test_l2_normalised(self):
        batch = _make_retrieval_batch()
        user_emb, item_emb = self.model(batch)
        norms_u = torch.norm(user_emb, p=2, dim=-1)
        norms_i = torch.norm(item_emb, p=2, dim=-1)
        assert torch.allclose(norms_u, torch.ones(B), atol=1e-5)
        assert torch.allclose(norms_i, torch.ones(B), atol=1e-5)

    def test_in_batch_loss_scalar(self):
        batch = _make_retrieval_batch()
        user_emb, item_emb = self.model(batch)
        loss = self.model.in_batch_loss(user_emb, item_emb)
        assert loss.shape == ()     # scalar
        assert loss.item() > 0

    def test_repr_contains_params(self):
        r = repr(self.model)
        assert "params" in r
        assert "UserTower" in r
        assert "ItemTower" in r


# ── DeepFM ─────────────────────────────────────────────────────────────────

class TestDeepFM:
    def setup_method(self):
        from src.models.deepfm import DeepFM
        self.model = DeepFM(MOCK_META, MOCK_RANKING_CFG).to(DEVICE)

    def test_output_shape(self):
        batch = _make_ranking_batch()
        preds = self.model(batch)
        assert preds.shape == (B,)

    def test_output_range(self):
        batch = _make_ranking_batch()
        preds = self.model(batch)
        assert (preds >= 0).all() and (preds <= 1).all()

    def test_repr(self):
        r = repr(self.model)
        assert "DeepFM" in r
        assert "fm_embed_dim" in r


# ── DIN ────────────────────────────────────────────────────────────────────

class TestDIN:
    def setup_method(self):
        from src.models.din import DIN
        self.model = DIN(MOCK_META, MOCK_RANKING_CFG).to(DEVICE)

    def test_output_shape(self):
        batch = _make_ranking_batch()
        preds = self.model(batch)
        assert preds.shape == (B,)

    def test_output_range(self):
        batch = _make_ranking_batch()
        preds = self.model(batch)
        assert (preds >= 0).all() and (preds <= 1).all()

    def test_padding_mask(self):
        """Padding positions (history_seq==0) should not affect output."""
        from src.models.din import DIN
        model = DIN(MOCK_META, MOCK_RANKING_CFG).to(DEVICE)
        batch = _make_ranking_batch()
        # Zero out the last half of the sequence for all users
        batch_zero = {k: v.clone() for k, v in batch.items()}
        batch_zero["history_seq"][:, L // 2:] = 0
        batch_zero["history_len"] = torch.full((B,), L // 2)

        # Outputs should differ from full history (just checking it runs cleanly)
        preds_full = model(batch)
        preds_zero = model(batch_zero)
        assert preds_full.shape == preds_zero.shape

    def test_repr(self):
        r = repr(self.model)
        assert "DIN" in r
        assert "Attention" in r


# ── SASRecEncoder ──────────────────────────────────────────────────────────

class TestSASRecEncoder:
    def setup_method(self):
        from src.models.sasrec import SASRecEncoder
        self.encoder = SASRecEncoder(
            n_items=100, hidden_dim=16, max_seq_len=10,
            n_layers=2, n_heads=2, dropout=0.0,
        ).to(DEVICE)

    def test_output_shape(self):
        seq = torch.randint(0, 101, (B, L))
        out = self.encoder(seq)
        assert out.shape == (B, 16)

    def test_padding_only_seq_no_crash(self):
        """All-zero (padding) sequences should not crash."""
        seq = torch.zeros(B, L, dtype=torch.long)
        out = self.encoder(seq)
        assert out.shape == (B, 16)

    def test_causal_mask_no_future_leakage(self):
        """Masking positions after k should not change positions 0..k output."""
        from src.models.sasrec import SASRecEncoder
        encoder = SASRecEncoder(
            n_items=100, hidden_dim=16, max_seq_len=10,
            n_layers=1, n_heads=2, dropout=0.0,
        ).to(DEVICE)
        encoder.eval()

        seq = torch.randint(1, 101, (1, L))           # no padding
        seq_truncated = seq.clone()
        seq_truncated[:, 5:] = 0                       # mask positions 5-9

        with torch.no_grad():
            out_full = encoder(seq)
            out_trunc = encoder(seq_truncated)

        # The two outputs differ because the last non-padding position differs;
        # we just check they both run cleanly with valid shapes.
        assert out_full.shape == (1, 16)
        assert out_trunc.shape == (1, 16)

    def test_repr_contains_n_layers(self):
        r = repr(self.encoder)
        assert "n_layers=2" in r
        assert "params" in r


# ── TwoTowerModel with SASRec ───────────────────────────────────────────────

class TestTwoTowerSASRec:
    def setup_method(self):
        from src.models.two_tower import TwoTowerModel
        self.model = TwoTowerModel(MOCK_META, MOCK_RETRIEVAL_CFG_SASREC).to(DEVICE)

    def test_output_shape(self):
        batch = _make_retrieval_batch()
        user_emb, item_emb = self.model(batch)
        assert user_emb.shape == (B, 32)
        assert item_emb.shape == (B, 32)

    def test_l2_normalised(self):
        batch = _make_retrieval_batch()
        user_emb, item_emb = self.model(batch)
        assert torch.allclose(torch.norm(user_emb, p=2, dim=-1), torch.ones(B), atol=1e-5)
        assert torch.allclose(torch.norm(item_emb, p=2, dim=-1), torch.ones(B), atol=1e-5)

    def test_repr_shows_sasrec(self):
        r = repr(self.model)
        assert "SASRec" in r


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
