"""SASRec: Self-Attentive Sequential Recommendation.

Based on: Kang & McAuley, "Self-Attentive Sequential Recommendation", ICDM 2018.
https://arxiv.org/abs/1808.09781

SASRec uses a transformer decoder-style architecture to model user interaction
sequences, capturing both long-range dependencies and recency bias via causal
(left-to-right) self-attention.

Key differences from mean pooling:
- Order-aware: positional embeddings capture recency signals
- Causal attention: each item only attends to earlier items (no future leakage)
- Output = representation at the last non-padding position (most recent context)
"""

import torch
import torch.nn as nn


class SASRecBlock(nn.Module):
    """Single transformer block: Pre-LN → CausalAttention → Pre-LN → FFN.

    Uses Pre-Layer-Norm (norm before attention/FFN) for training stability.
    Residual connections wrap both sub-layers.

    Args:
        hidden_dim: Attention embedding dimension.
        n_heads: Number of attention heads. Must divide hidden_dim evenly.
        dropout: Dropout applied after attention and inside FFN.
    """

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        # Point-wise FFN: expand 4x with GELU, then project back
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, H) input sequence.
            causal_mask: (L, L) upper-triangular -inf mask (prevents attending to future).
            key_padding_mask: (B, L) bool mask, True at padding positions.
        Returns:
            (B, L, H) updated sequence.
        """
        # Pre-LN → self-attention → residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
        )
        x = x + self.dropout(attn_out)

        # Pre-LN → FFN → residual
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class SASRecEncoder(nn.Module):
    """SASRec sequence encoder for the User Tower.

    Encodes a padded item interaction history into a single fixed-size vector
    representing the user's current interest state.  The output is the hidden
    state at the **last non-padding position** — i.e., the most recent item's
    contextualised representation.

    Args:
        n_items:     Vocabulary size (items are 1-indexed; 0 = padding).
        hidden_dim:  Embedding and transformer hidden dimension.
        max_seq_len: Maximum sequence length.
        n_layers:    Number of stacked SASRecBlocks.
        n_heads:     Attention heads per block. Must divide hidden_dim.
        dropout:     Dropout probability.
    """

    def __init__(
        self,
        n_items: int,
        hidden_dim: int,
        max_seq_len: int,
        n_layers: int,
        n_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # Item embedding: 1-indexed, padding_idx=0 keeps padding as zero vector
        self.item_embed = nn.Embedding(n_items + 1, hidden_dim, padding_idx=0)
        # Learnable positional embedding (positions 0 .. max_seq_len-1)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)

        self.blocks = nn.ModuleList([
            SASRecBlock(hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.item_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        for block in self.blocks:
            nn.init.xavier_uniform_(block.attn.in_proj_weight)
            nn.init.zeros_(block.attn.in_proj_bias)
            nn.init.xavier_uniform_(block.attn.out_proj.weight)
            nn.init.zeros_(block.attn.out_proj.bias)

    def forward(self, history_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history_seq: (B, L) item ID sequence, 0 = padding (right-aligned).
        Returns:
            (B, hidden_dim) sequence representation at the last non-padding position.
        """
        B, L = history_seq.shape
        device = history_seq.device

        # Item embeddings + positional embeddings
        item_emb = self.item_embed(history_seq)                        # (B, L, H)
        positions = torch.arange(L, device=device).unsqueeze(0)        # (1, L)
        pos_emb = self.pos_embed(positions)                             # (1, L, H)
        x = self.dropout(item_emb + pos_emb)                           # (B, L, H)

        # Causal mask: position j cannot attend to position k > j
        # Upper triangular filled with -inf; diagonal and below = 0
        causal_mask = torch.triu(
            torch.full((L, L), float("-inf"), device=device), diagonal=1
        )

        # Padding mask: True at positions where history_seq == 0
        key_padding_mask = (history_seq == 0)                          # (B, L)
        # Guard: if a row is entirely padding, unmask position 0 so softmax
        # never sees all -inf (which produces NaN in PyTorch MHA).
        all_pad = key_padding_mask.all(dim=1, keepdim=True)            # (B, 1)
        key_padding_mask = key_padding_mask & ~all_pad.expand_as(key_padding_mask)
        # Zero out the embedding at position 0 for those rows so it doesn't add signal
        x = torch.where(all_pad.unsqueeze(-1), torch.zeros_like(x), x)

        for block in self.blocks:
            x = block(x, causal_mask, key_padding_mask)

        x = self.norm(x)                                               # (B, L, H)

        # Extract the hidden state at the last non-padding position
        seq_lens = (history_seq > 0).sum(dim=1)                        # (B,)
        last_pos = (seq_lens - 1).clamp(min=0)                         # (B,) — safe for empty seqs
        out = x[torch.arange(B, device=device), last_pos]              # (B, H)
        return out

    def __repr__(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return (
            f"SASRecEncoder("
            f"n_layers={len(self.blocks)}, "
            f"hidden_dim={self.hidden_dim}, "
            f"max_seq_len={self.max_seq_len}, "
            f"params={n:,})"
        )
