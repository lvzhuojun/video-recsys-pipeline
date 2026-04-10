"""DIN: Deep Interest Network for Click-Through Rate Prediction.

Paper: Zhou et al., KDD 2018.  https://arxiv.org/abs/1706.06978

Key insight: Rather than compressing a user's full history into a fixed
vector (Mean Pooling), DIN uses **attention** to weight each historical
item by its relevance to the *target* item being scored.  This lets the
model focus on the parts of user history that are actually relevant to
the current candidate.

Attention mechanism (differs from Transformer self-attention):
  For each past item vⱼ in user history and target item vₜ:
    score(vⱼ, vₜ) = MLP([vⱼ, vₜ, vⱼ⊙vₜ, vⱼ-vₜ])   # element-wise ops
  User interest = Σⱼ softmax(score_j) · vⱼ

Why different from Transformer self-attention?
  - DIN: target item attends to history (cross-attention)
  - Transformer SA: history attends to history (self-attention)
  - DIN uses a learned MLP score function instead of Q·Kᵀ/√d
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.logger import get_logger

logger = get_logger(__name__)


class DINAttention(nn.Module):
    """Attention score network between a target item and history items.

    Input for each (history_item, target_item) pair:
        [h_emb, t_emb, h_emb ⊙ t_emb, h_emb - t_emb]  → (4 * embed_dim,)

    Args:
        embed_dim: Embedding dimension of item vectors.
        hidden_dims: Hidden layer sizes of the attention MLP.
    """

    def __init__(self, embed_dim: int, hidden_dims: List[int]) -> None:
        super().__init__()
        in_dim = embed_dim * 4   # concat of 4 interaction signals
        layer_dims = [in_dim] + hidden_dims + [1]
        layers: List[nn.Module] = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        target_emb: torch.Tensor,  # (B, D)
        history_emb: torch.Tensor, # (B, L, D)
    ) -> torch.Tensor:
        """Compute attention scores.

        Args:
            target_emb: Target item embedding (B, D).
            history_emb: History item embeddings (B, L, D).

        Returns:
            (B, L) raw attention scores (before masking and softmax).
        """
        B, L, D = history_emb.shape
        t = target_emb.unsqueeze(1).expand(-1, L, -1)  # (B, L, D)

        # 4-way interaction features between history and target
        interaction = torch.cat([
            history_emb,          # (B, L, D)
            t,                    # (B, L, D)
            history_emb * t,      # element-wise product
            history_emb - t,      # element-wise difference
        ], dim=-1)                # (B, L, 4D)

        flat = interaction.view(B * L, -1)
        scores = self.net(flat).view(B, L)                # (B, L)
        return scores


class DIN(nn.Module):
    """Deep Interest Network CTR prediction model.

    Feature schema (same as DeepFM for fair comparison):
      - user_id         (B,)     → Embedding
      - item_id         (B,)     → Embedding  (target)
      - item_category   (B,)     → Embedding
      - item_dur_bkt    (B,)     → Embedding
      - user_dense      (B, 25)
      - item_dense      (B, 3)
      - history_seq     (B, L)   → Embedding  (history item IDs, 1-indexed)
      - history_len     (B,)     → attention mask

    Args:
        meta: Dataset metadata dict from ``load_meta()``.
        cfg: Parsed ``ranking_config.yaml``.

    Returns:
        Sigmoid-activated CTR probability, shape ``(B,)``.
    """

    def __init__(self, meta: dict, cfg: dict) -> None:
        super().__init__()
        mc = cfg["model"]
        item_dim = mc["item_embed_dim"]
        seq_dim  = mc["din"]["seq_embed_dim"]
        attn_hidden = mc["din"]["attention_hidden_dims"]
        mlp_dims: List[int] = mc["din"]["mlp_hidden_dims"]
        dropout = mc["din"].get("dropout", 0.1)

        # ── Feature embeddings ──
        self.user_embed = nn.Embedding(meta["n_users"],        mc["user_embed_dim"])
        self.item_embed = nn.Embedding(meta["n_items"],        item_dim)
        self.cat_embed  = nn.Embedding(meta["n_categories"],   mc["cat_embed_dim"])
        self.dur_embed  = nn.Embedding(meta["n_duration_bkts"], mc["dur_embed_dim"])
        # History sequence embedding (1-indexed; 0 = padding)
        self.hist_embed = nn.Embedding(meta["n_items"] + 1,    seq_dim, padding_idx=0)

        # ── DIN attention ──
        self.attention = DINAttention(seq_dim, attn_hidden)

        # ── MLP ──
        # Concat: user_emb + target_item_emb + cat_emb + dur_emb
        #       + user_dense + item_dense + hist_pooled
        concat_dim = (
            mc["user_embed_dim"]
            + item_dim
            + mc["cat_embed_dim"]
            + mc["dur_embed_dim"]
            + meta["user_dense_dim"]
            + meta["item_dense_dim"]
            + seq_dim
        )
        layer_dims = [concat_dim] + list(mlp_dims) + [1]
        mlp_layers: List[nn.Module] = []
        for i in range(len(layer_dims) - 1):
            mlp_layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                mlp_layers.append(nn.ReLU())
                mlp_layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*mlp_layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for emb in (self.user_embed, self.item_embed,
                    self.cat_embed, self.dur_embed, self.hist_embed):
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self, batch: dict) -> torch.Tensor:
        """Compute CTR probability.

        Args:
            batch: Dict from RankingDataset.

        Returns:
            (B,) sigmoid CTR probabilities.
        """
        # ── Target item embeddings ──
        i_emb   = self.item_embed(batch["item_id"])       # (B, item_dim)
        u_emb   = self.user_embed(batch["user_id"])       # (B, user_dim)
        c_emb   = self.cat_embed(batch["item_category"])  # (B, cat_dim)
        d_emb   = self.dur_embed(batch["item_dur_bkt"])   # (B, dur_dim)

        # ── History sequence → attention-pooled representation ──
        hist_emb = self.hist_embed(batch["history_seq"])  # (B, L, seq_dim)

        # Project target item to seq_dim for attention comparison
        # (use a simple linear if item_dim != seq_dim)
        target_for_attn = i_emb
        if i_emb.size(-1) != hist_emb.size(-1):
            # Fallback: average target dims down; kept simple for clarity
            target_for_attn = i_emb[:, : hist_emb.size(-1)]

        attn_scores = self.attention(target_for_attn, hist_emb)  # (B, L)

        # Mask padding positions (history_seq == 0)
        pad_mask = (batch["history_seq"] == 0)          # (B, L)
        attn_scores = attn_scores.masked_fill(pad_mask, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)        # (B, L)

        # Weighted pooling over history
        hist_pooled = (attn_weights.unsqueeze(-1) * hist_emb).sum(dim=1)  # (B, seq_dim)

        # ── Concat all features and pass through MLP ──
        x = torch.cat([
            u_emb,
            i_emb,
            c_emb,
            d_emb,
            batch["user_dense"],
            batch["item_dense"],
            hist_pooled,
        ], dim=-1)

        logit = self.mlp(x).squeeze(-1)
        return torch.sigmoid(logit)

    def __repr__(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        emb   = sum(p.numel() for p in list(self.user_embed.parameters())
                    + list(self.item_embed.parameters())
                    + list(self.cat_embed.parameters())
                    + list(self.dur_embed.parameters())
                    + list(self.hist_embed.parameters()))
        attn  = sum(p.numel() for p in self.attention.parameters())
        mlp_p = sum(p.numel() for p in self.mlp.parameters())
        return (
            f"DIN(\n"
            f"  Embeddings : {emb:,} params\n"
            f"  Attention  : {attn:,} params\n"
            f"  MLP        : {mlp_p:,} params\n"
            f"  Total      : {total:,} params\n"
            f")"
        )
