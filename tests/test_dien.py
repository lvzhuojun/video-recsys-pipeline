"""Unit tests for DIEN (Deep Interest Evolution Network) model."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import pytest

# Small mock config matching DIEN's config section
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

MOCK_DIEN_CFG = {
    "model": {
        "user_embed_dim": 16,
        "item_embed_dim": 16,
        "cat_embed_dim": 8,
        "dur_embed_dim": 4,
        "dien": {
            "seq_embed_dim": 16,       # same as item_embed_dim for clean attention
            "mlp_hidden_dims": [32, 16],
            "dropout": 0.0,
            "aux_loss_weight": 0.1,
        },
    }
}

B, L = 8, 10   # batch size, sequence length
DEVICE = torch.device("cpu")


def _make_ranking_batch(all_padding: bool = False):
    """Create a mock ranking batch."""
    if all_padding:
        history = torch.zeros(B, L, dtype=torch.long)
    else:
        history = torch.randint(0, 101, (B, L))
    return {
        "user_id":       torch.randint(0, 50, (B,)),
        "item_id":       torch.randint(0, 100, (B,)),
        "user_dense":    torch.randn(B, 25),
        "item_dense":    torch.randn(B, 3),
        "item_category": torch.randint(0, 5, (B,)),
        "item_dur_bkt":  torch.randint(0, 5, (B,)),
        "history_seq":   history,
        "history_len":   torch.randint(1, L + 1, (B,)),
        "label":         torch.randint(0, 2, (B,)).float(),
    }


# ── AUGRUCell ──────────────────────────────────────────────────────────────

class TestAUGRUCell:
    def setup_method(self):
        from src.models.dien import AUGRUCell
        self.cell = AUGRUCell(input_dim=16, hidden_dim=16).to(DEVICE)

    def test_output_shape(self):
        """AUGRUCell output should be (B, hidden_dim)."""
        x_t    = torch.randn(B, 16)
        h_prev = torch.randn(B, 16)
        e_t    = torch.rand(B)
        h_new  = self.cell(x_t, h_prev, e_t)
        assert h_new.shape == (B, 16), f"Expected (B, 16), got {h_new.shape}"

    def test_zero_attention_gates_update(self):
        """When attention e_t=0, update gate u' = 0 → h_new should equal h_prev."""
        from src.models.dien import AUGRUCell
        cell = AUGRUCell(input_dim=16, hidden_dim=16).to(DEVICE)
        cell.eval()

        x_t    = torch.randn(B, 16)
        h_prev = torch.randn(B, 16)
        e_t    = torch.zeros(B)   # zero attention → no update

        with torch.no_grad():
            h_new = cell(x_t, h_prev, e_t)

        # When e_t=0: u' = 0 * u = 0 → h_new = (1-0)*h_prev + 0*h_tilde = h_prev
        assert torch.allclose(h_new, h_prev, atol=1e-5), (
            "With e_t=0, AUGRUCell should not change h_prev."
        )


# ── DIEN ───────────────────────────────────────────────────────────────────

class TestDIEN:
    def setup_method(self):
        from src.models.dien import DIEN
        self.model = DIEN(MOCK_META, MOCK_DIEN_CFG).to(DEVICE)

    def test_output_shape(self):
        """DIEN forward should return (B,) tensor."""
        batch = _make_ranking_batch()
        preds = self.model(batch)
        assert preds.shape == (B,), f"Expected ({B},), got {preds.shape}"

    def test_output_range(self):
        """DIEN predictions should be in [0, 1]."""
        batch = _make_ranking_batch()
        preds = self.model(batch)
        assert (preds >= 0.0).all() and (preds <= 1.0).all(), (
            "DIEN predictions should be in [0, 1]."
        )

    def test_aux_loss_positive(self):
        """compute_aux_loss() should return a scalar > 0."""
        batch = _make_ranking_batch()
        # Must call forward first to cache _h_states
        _ = self.model(batch)
        aux_loss = self.model.compute_aux_loss(batch)
        assert aux_loss.shape == (), f"Aux loss should be scalar, got {aux_loss.shape}"
        assert aux_loss.item() > 0.0, "Auxiliary loss should be positive."

    def test_all_padding_no_crash(self):
        """All-zero history_seq (all padding) should not crash."""
        batch = _make_ranking_batch(all_padding=True)
        preds = self.model(batch)
        assert preds.shape == (B,), "Should handle all-padding history without crash."
        assert not torch.isnan(preds).any(), "Predictions should not be NaN."

    def test_repr(self):
        """repr should mention DIEN, InterestExtractor, InterestEvolving."""
        r = repr(self.model)
        assert "DIEN" in r, "repr should contain 'DIEN'"
        assert "InterestExtractor" in r, "repr should contain 'InterestExtractor'"
        assert "InterestEvolving" in r, "repr should contain 'InterestEvolving'"

    def test_aux_loss_requires_forward_first(self):
        """compute_aux_loss without forward should raise RuntimeError."""
        from src.models.dien import DIEN
        fresh_model = DIEN(MOCK_META, MOCK_DIEN_CFG).to(DEVICE)
        batch = _make_ranking_batch()
        with pytest.raises(RuntimeError):
            fresh_model.compute_aux_loss(batch)

    def test_backward_pass(self):
        """Gradients should flow through both main loss and aux loss."""
        from src.models.dien import DIEN
        model = DIEN(MOCK_META, MOCK_DIEN_CFG).to(DEVICE)
        batch = _make_ranking_batch()

        preds = model(batch)
        main_loss = torch.nn.functional.binary_cross_entropy(preds, batch["label"])
        aux_loss  = model.compute_aux_loss(batch)
        total_loss = main_loss + 0.1 * aux_loss
        total_loss.backward()

        # Check some gradients exist
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in model.parameters()
        )
        assert has_grad, "No gradients found after backward pass."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
