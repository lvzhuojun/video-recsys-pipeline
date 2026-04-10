"""DeepFM: A Factorization-Machine Based Neural Network for CTR Prediction.

Paper: Guo et al., IJCAI 2017.  https://arxiv.org/abs/1703.04247

Architecture:
  Input features → shared sparse embeddings
        │
    ┌───┴───────────────────┐
    │                       │
  Linear part            FM part + Deep part
  (1st-order)        (2nd-order + high-order)
    │                       │
    └──────── add ──────────┘
                  │
              sigmoid → CTR probability

The key insight: DeepFM shares the same feature embedding between the FM
component (for 2nd-order interactions) and the Deep component (for
high-order interactions), so there is no manual feature engineering.

FM second-order interaction trick:
  Instead of O(n²·k) explicit pairwise products, FM uses:
  0.5 * sum[(Σ vᵢ)² - Σ vᵢ²]   →   O(n·k) computation
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.logger import get_logger

logger = get_logger(__name__)


class DeepFM(nn.Module):
    """DeepFM CTR prediction model.

    Feature schema (aligned with RankingDataset output):
      - user_id         (B,)     → Embedding
      - item_id         (B,)     → Embedding
      - item_category   (B,)     → Embedding
      - item_dur_bkt    (B,)     → Embedding
      - user_dense      (B, 25)  → Linear projection
      - item_dense      (B, 3)   → Linear projection

    All categorical features share the same FM embedding dimension (fm_embed_dim)
    so that the FM interaction trick is applicable.

    Args:
        meta: Dataset metadata dict from ``load_meta()``.
        cfg: Parsed ``ranking_config.yaml``.

    Returns:
        Sigmoid-activated CTR probability, shape ``(B,)``.
    """

    def __init__(self, meta: dict, cfg: dict) -> None:
        super().__init__()
        mc = cfg["model"]
        fm_dim = mc["deepfm"]["fm_embed_dim"]
        mlp_dims: List[int] = mc["deepfm"]["mlp_hidden_dims"]
        dropout = mc["deepfm"].get("dropout", 0.1)

        # ── Categorical feature embeddings (shared between FM and Deep) ──
        self.user_embed = nn.Embedding(meta["n_users"],       fm_dim)
        self.item_embed = nn.Embedding(meta["n_items"],       fm_dim)
        self.cat_embed  = nn.Embedding(meta["n_categories"],  fm_dim)
        self.dur_embed  = nn.Embedding(meta["n_duration_bkts"], fm_dim)
        # 4 categorical fields
        self.n_cat_fields = 4

        # ── Dense feature projections onto fm_dim (for FM consistency) ──
        self.user_dense_proj = nn.Linear(meta["user_dense_dim"], fm_dim)
        self.item_dense_proj = nn.Linear(meta["item_dense_dim"], fm_dim)
        # 2 dense fields
        self.n_dense_fields = 2

        self.n_fields = self.n_cat_fields + self.n_dense_fields  # 6

        # ── Linear (1st-order) part ──
        # One scalar bias per field value; we use the same embedding and
        # project each fm_dim vector down to a scalar.
        self.linear_proj = nn.Linear(fm_dim, 1, bias=False)

        # ── Deep (MLP) part ──
        deep_in = self.n_fields * fm_dim
        layer_dims = [deep_in] + list(mlp_dims) + [1]
        deep_layers: List[nn.Module] = []
        for i in range(len(layer_dims) - 1):
            deep_layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                deep_layers.append(nn.ReLU())
                deep_layers.append(nn.Dropout(dropout))
        self.deep = nn.Sequential(*deep_layers)

        # Global bias
        self.bias = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self) -> None:
        for emb in (self.user_embed, self.item_embed, self.cat_embed, self.dur_embed):
            nn.init.normal_(emb.weight, std=0.01)
        nn.init.xavier_uniform_(self.user_dense_proj.weight)
        nn.init.xavier_uniform_(self.item_dense_proj.weight)

    def _get_field_embeddings(
        self, batch: dict
    ) -> torch.Tensor:
        """Stack all field embeddings into (B, n_fields, fm_dim).

        Each row in the return tensor is one "feature field" embedding.
        FM treats each field equally regardless of its origin type.
        """
        e_user = self.user_embed(batch["user_id"])        # (B, fm_dim)
        e_item = self.item_embed(batch["item_id"])        # (B, fm_dim)
        e_cat  = self.cat_embed(batch["item_category"])  # (B, fm_dim)
        e_dur  = self.dur_embed(batch["item_dur_bkt"])   # (B, fm_dim)
        # Project dense features to fm_dim
        e_udense = F.relu(self.user_dense_proj(batch["user_dense"]))   # (B, fm_dim)
        e_idense = F.relu(self.item_dense_proj(batch["item_dense"]))   # (B, fm_dim)

        # (B, n_fields=6, fm_dim)
        return torch.stack([e_user, e_item, e_cat, e_dur, e_udense, e_idense], dim=1)

    def forward(self, batch: dict) -> torch.Tensor:
        """Compute CTR probability.

        Args:
            batch: Dict from RankingDataset with keys matching RankingDataset.

        Returns:
            (B,) sigmoid CTR probabilities.
        """
        embs = self._get_field_embeddings(batch)  # (B, F, K)

        # ── Linear part ── sum of per-field scalars
        linear_out = self.linear_proj(embs).squeeze(-1).sum(dim=1)  # (B,)

        # ── FM 2nd-order part ── O(F·K) trick:
        # 0.5 * (||Σ vᵢ||² - Σ||vᵢ||²)
        sum_of_emb   = embs.sum(dim=1)                          # (B, K)
        sq_of_sum    = (sum_of_emb ** 2).sum(dim=-1)            # (B,)
        sum_of_sq    = (embs ** 2).sum(dim=-1).sum(dim=-1)      # (B,)
        fm_out = 0.5 * (sq_of_sum - sum_of_sq)                  # (B,)

        # ── Deep part ──
        deep_in = embs.view(embs.size(0), -1)                   # (B, F*K)
        deep_out = self.deep(deep_in).squeeze(-1)               # (B,)

        logit = linear_out + fm_out + deep_out + self.bias
        return torch.sigmoid(logit)

    def __repr__(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        lin   = sum(p.numel() for p in self.linear_proj.parameters())
        deep  = sum(p.numel() for p in self.deep.parameters())
        embeds = total - lin - deep - 1  # -1 for bias
        return (
            f"DeepFM(\n"
            f"  Embeddings: {embeds:,} params\n"
            f"  Linear(1st-order): {lin:,} params\n"
            f"  Deep(MLP): {deep:,} params\n"
            f"  Total: {total:,} params\n"
            f"  n_fields={self.n_fields}, fm_embed_dim={self.user_embed.embedding_dim}\n"
            f")"
        )
