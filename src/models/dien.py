"""DIEN: Deep Interest Evolution Network.
Paper: Zhou et al., AAAI 2019. https://arxiv.org/abs/1809.03672

Architecture:
  1. Interest Extractor (GRU): hidden states h = [h_1...h_L] capturing cumulative interests
  2. Interest Evolving (AUGRU): attention-modulated GRU, target-item-aware evolution
  3. Auxiliary loss: predict next-item from extractor states to supervise extraction quality
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.logger import get_logger

logger = get_logger(__name__)


class AUGRUCell(nn.Module):
    """Attention-modulated GRU cell.

    update gate = e_t (attention score) * sigma(W_u[x_t, h_{t-1}])
    This makes interest evolution focus on items relevant to target.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden state dimension.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        # Reset gate
        self.W_r = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=True)
        # Update gate
        self.W_u = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=True)
        # Candidate hidden
        self.W_h = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=True)

    def forward(
        self,
        x_t: torch.Tensor,     # (B, input_dim)
        h_prev: torch.Tensor,  # (B, hidden_dim)
        e_t: torch.Tensor,     # (B,) attention score for this step
    ) -> torch.Tensor:
        """Compute next hidden state with attention-modulated update gate.

        Returns:
            h_new: (B, hidden_dim)
        """
        combined = torch.cat([x_t, h_prev], dim=-1)  # (B, input_dim + hidden_dim)
        r = torch.sigmoid(self.W_r(combined))                        # (B, hidden_dim)
        u = torch.sigmoid(self.W_u(combined))                        # (B, hidden_dim)
        u_prime = e_t.unsqueeze(-1) * u                              # (B, hidden_dim)
        h_tilde = torch.tanh(self.W_h(torch.cat([x_t, r * h_prev], dim=-1)))  # (B, hidden_dim)
        h_new = (1.0 - u_prime) * h_prev + u_prime * h_tilde        # (B, hidden_dim)
        return h_new


class InterestExtractor(nn.Module):
    """GRU-based interest extractor. Produces all hidden states for auxiliary loss.

    Args:
        seq_dim: Sequence/item embedding dimension.
        dropout: Dropout rate applied on GRU output.
    """

    def __init__(self, seq_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.gru = nn.GRU(seq_dim, seq_dim, batch_first=True)
        # Auxiliary predictor: concat(h_t, next_item_emb) → scalar logit
        self.aux_pred = nn.Linear(seq_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run GRU on history embedding sequence.

        Args:
            x: (B, L, seq_dim) history item embeddings.

        Returns:
            h_states: (B, L, seq_dim) all hidden states.
        """
        h_states, _ = self.gru(x)  # (B, L, seq_dim)
        h_states = self.dropout(h_states)
        return h_states

    def auxiliary_loss(
        self,
        h_states: torch.Tensor,   # (B, L, seq_dim) cached extractor hidden states
        next_emb: torch.Tensor,   # (B, L, seq_dim) next-item embeddings (positives)
        neg_emb: torch.Tensor,    # (B, L, seq_dim) negative-item embeddings
        mask: torch.Tensor,       # (B, L) bool, True = valid position
    ) -> torch.Tensor:
        """Compute auxiliary BCE loss supervising the extractor.

        For each valid time step t, predict whether next_emb is the real
        next item (positive) vs neg_emb (random negative).

        Returns:
            Scalar auxiliary loss.
        """
        # (B, L, seq_dim*2) → (B, L, 1) → (B, L)
        pos_input = torch.cat([h_states, next_emb], dim=-1)
        neg_input = torch.cat([h_states, neg_emb],  dim=-1)

        pos_logits = self.aux_pred(pos_input).squeeze(-1)  # (B, L)
        neg_logits = self.aux_pred(neg_input).squeeze(-1)  # (B, L)

        # Labels: positives=1, negatives=0
        pos_labels = torch.ones_like(pos_logits)
        neg_labels = torch.zeros_like(neg_logits)

        # BCE per-element, then mask and mean
        pos_loss = F.binary_cross_entropy_with_logits(pos_logits, pos_labels, reduction="none")
        neg_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_labels, reduction="none")

        aux_loss = pos_loss + neg_loss  # (B, L)

        mask_float = mask.float()
        n_valid = mask_float.sum().clamp(min=1.0)
        return (aux_loss * mask_float).sum() / n_valid


class InterestEvolving(nn.Module):
    """AUGRU-based interest evolving module.

    Runs AUGRU step-by-step, modulating update gates by target-item attention.

    Args:
        seq_dim: Hidden dimension (same as extractor output).
    """

    def __init__(self, seq_dim: int) -> None:
        super().__init__()
        self.seq_dim = seq_dim
        self.augru_cell = AUGRUCell(seq_dim, seq_dim)

    def forward(
        self,
        h_states: torch.Tensor,      # (B, L, seq_dim) extractor hidden states
        target_emb_proj: torch.Tensor,  # (B, seq_dim) projected target item embedding
        mask: torch.Tensor,          # (B, L) bool, True = valid history position
    ) -> torch.Tensor:
        """Evolve interest states with target-aware attention.

        Args:
            h_states: All hidden states from interest extractor.
            target_emb_proj: Target item embedding projected to seq_dim.
            mask: Valid positions mask (True = real item, False = padding).

        Returns:
            (B, seq_dim) final evolved hidden state.
        """
        B, L, D = h_states.shape

        # Compute scaled dot-product attention scores
        # (B, L, D) × (B, D, 1) → (B, L)
        target_expanded = target_emb_proj.unsqueeze(1).expand(-1, L, -1)  # (B, L, D)
        scores = (h_states * target_expanded).sum(-1) / (D ** 0.5)        # (B, L)

        # Mask padding positions before softmax
        scores = scores.masked_fill(~mask, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)  # (B, L)

        # Run AUGRU step by step
        h = torch.zeros(B, D, device=h_states.device, dtype=h_states.dtype)

        for t in range(L):
            x_t = h_states[:, t, :]        # (B, D)
            e_t = attn_weights[:, t]       # (B,)
            h_new = self.augru_cell(x_t, h, e_t)  # (B, D)

            # Only update hidden state at valid (non-padding) positions
            valid = mask[:, t].unsqueeze(-1).float()  # (B, 1)
            h = valid * h_new + (1.0 - valid) * h

        return h  # (B, seq_dim)


class DIEN(nn.Module):
    """Deep Interest Evolution Network CTR ranking model.

    Feature schema identical to DIN (fair comparison):
      - user_id         (B,)     → Embedding
      - item_id         (B,)     → Embedding  (target)
      - item_category   (B,)     → Embedding
      - item_dur_bkt    (B,)     → Embedding
      - user_dense      (B, 25)
      - item_dense      (B, 3)
      - history_seq     (B, L)   → Embedding  (history item IDs, 1-indexed)
      - history_len     (B,)

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
        seq_dim  = mc["dien"]["seq_embed_dim"]
        mlp_dims: List[int] = mc["dien"]["mlp_hidden_dims"]
        dropout = mc["dien"].get("dropout", 0.1)

        # ── Feature embeddings (identical to DIN) ──
        self.user_embed = nn.Embedding(meta["n_users"],         mc["user_embed_dim"])
        self.item_embed = nn.Embedding(meta["n_items"],         item_dim)
        self.cat_embed  = nn.Embedding(meta["n_categories"],    mc["cat_embed_dim"])
        self.dur_embed  = nn.Embedding(meta["n_duration_bkts"], mc["dur_embed_dim"])
        # 1-indexed; 0 = padding
        self.hist_embed = nn.Embedding(meta["n_items"] + 1,    seq_dim, padding_idx=0)

        # Project target item to seq_dim for attention (handles item_dim != seq_dim)
        self.target_proj = nn.Linear(item_dim, seq_dim, bias=False)

        # ── DIEN modules ──
        self.extractor = InterestExtractor(seq_dim, dropout)
        self.evolving  = InterestEvolving(seq_dim)

        # ── MLP ──
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

        # Cached extractor hidden states (set during forward, used in compute_aux_loss)
        self._h_states: torch.Tensor | None = None
        self._meta = meta

        self._init_weights()

    def _init_weights(self) -> None:
        for emb in (self.user_embed, self.item_embed,
                    self.cat_embed, self.dur_embed, self.hist_embed):
            nn.init.normal_(emb.weight, std=0.01)
        nn.init.xavier_uniform_(self.target_proj.weight)

    def forward(self, batch: dict) -> torch.Tensor:
        """Compute CTR probability.

        Args:
            batch: Dict from RankingDataset.

        Returns:
            (B,) sigmoid CTR probabilities.
        """
        # ── Target item embeddings ──
        i_emb = self.item_embed(batch["item_id"])       # (B, item_dim)
        u_emb = self.user_embed(batch["user_id"])       # (B, user_dim)
        c_emb = self.cat_embed(batch["item_category"])  # (B, cat_dim)
        d_emb = self.dur_embed(batch["item_dur_bkt"])   # (B, dur_dim)

        # ── History sequence ──
        hist_emb = self.hist_embed(batch["history_seq"])  # (B, L, seq_dim)
        mask = (batch["history_seq"] > 0)                 # (B, L) True = valid

        # ── Interest Extractor ──
        h_states = self.extractor(hist_emb)              # (B, L, seq_dim)
        self._h_states = h_states                        # cache for aux loss

        # ── Interest Evolving ──
        target_proj = self.target_proj(i_emb)            # (B, seq_dim)
        evolved = self.evolving(h_states, target_proj, mask)  # (B, seq_dim)

        # ── Concat all features → MLP → sigmoid ──
        x = torch.cat([
            u_emb,
            i_emb,
            c_emb,
            d_emb,
            batch["user_dense"],
            batch["item_dense"],
            evolved,
        ], dim=-1)

        logit = self.mlp(x).squeeze(-1)
        return torch.sigmoid(logit)

    def compute_aux_loss(self, batch: dict) -> torch.Tensor:
        """Compute auxiliary next-item prediction loss on extractor states.

        Should be called after forward() to use cached _h_states.

        Args:
            batch: Same batch passed to forward().

        Returns:
            Scalar auxiliary loss.
        """
        if self._h_states is None:
            raise RuntimeError("Call forward() before compute_aux_loss().")

        history_seq = batch["history_seq"]   # (B, L) 1-indexed
        B, L = history_seq.shape
        n_items = self._meta["n_items"]

        # Build next-item IDs: shift right by 1, fill last with item_id + 1 (clamped)
        # next_ids[b, t] = history_seq[b, t+1] for t < L-1
        #                  item_id[b] + 1 (as a proxy) for t == L-1
        next_ids = torch.zeros_like(history_seq)
        if L > 1:
            next_ids[:, :-1] = history_seq[:, 1:]
        # For the last position, use the target item (shifted to 1-indexed)
        next_ids[:, -1] = (batch["item_id"] + 1).clamp(max=n_items)

        # Random negatives in [1, n_items]
        neg_ids = torch.randint(1, n_items + 1, (B, L),
                                device=history_seq.device, dtype=history_seq.dtype)

        # Embeddings
        next_emb = self.hist_embed(next_ids.clamp(max=n_items))  # (B, L, seq_dim)
        neg_emb  = self.hist_embed(neg_ids)                       # (B, L, seq_dim)

        # Mask: only valid history positions
        mask = (history_seq > 0)  # (B, L)

        return self.extractor.auxiliary_loss(self._h_states, next_emb, neg_emb, mask)

    def __repr__(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        emb = sum(
            p.numel()
            for emb_module in (
                self.user_embed, self.item_embed, self.cat_embed,
                self.dur_embed, self.hist_embed
            )
            for p in emb_module.parameters()
        )
        ext  = sum(p.numel() for p in self.extractor.parameters())
        evol = sum(p.numel() for p in self.evolving.parameters())
        mlp_ = sum(p.numel() for p in self.mlp.parameters())
        proj = sum(p.numel() for p in self.target_proj.parameters())
        return (
            f"DIEN(\n"
            f"  Embeddings      : {emb:,} params\n"
            f"  TargetProj      : {proj:,} params\n"
            f"  InterestExtractor: {ext:,} params\n"
            f"  InterestEvolving : {evol:,} params\n"
            f"  MLP             : {mlp_:,} params\n"
            f"  Total           : {total:,} params\n"
            f")"
        )
