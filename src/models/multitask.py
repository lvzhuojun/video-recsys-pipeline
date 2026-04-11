"""MMoE: Multi-gate Mixture of Experts for Multi-task CTR Prediction.
Paper: Ma et al., KDD 2018. https://dl.acm.org/doi/10.1145/3219819.3220007

Tasks:
  Task 0 (watch_ratio): sigmoid prediction of continuous engagement [0, 1]
  Task 1 (like):        binary classification

Architecture:
  Shared feature encoding (same as DIN: embeddings + DIN attention history pooling)
  n_experts Expert Networks (each: FFN with ReLU)
  Per-task Gate Networks (softmax over n_experts)
  Task-specific towers (MLP → sigmoid)
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.logger import get_logger
from .din import DINAttention

logger = get_logger(__name__)


class ExpertNetwork(nn.Module):
    """Expert FFN: input_dim → expert_output_dim with ReLU.

    Args:
        input_dim: Dimension of shared feature input.
        expert_output_dim: Output dimension of this expert.
        hidden_dim: Hidden layer width (one hidden layer).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        expert_output_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, expert_output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, input_dim)
        Returns:
            (B, expert_output_dim)
        """
        return self.net(x)


class GatingNetwork(nn.Module):
    """Softmax gate: input_dim → n_experts.

    Produces a probability distribution over experts for each sample.

    Args:
        input_dim: Shared feature dimension.
        n_experts: Number of expert networks.
    """

    def __init__(self, input_dim: int, n_experts: int) -> None:
        super().__init__()
        self.gate = nn.Linear(input_dim, n_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, input_dim)
        Returns:
            (B, n_experts) softmax weights
        """
        return F.softmax(self.gate(x), dim=-1)


class MMoE(nn.Module):
    """Multi-gate Mixture of Experts for joint watch_ratio and like prediction.

    Feature schema (same as DIN for fair comparison):
      - user_id         (B,)     → Embedding
      - item_id         (B,)     → Embedding  (target)
      - item_category   (B,)     → Embedding
      - item_dur_bkt    (B,)     → Embedding
      - user_dense      (B, 25)
      - item_dense      (B, 3)
      - history_seq     (B, L)   → DIN-style attention pooling
      - history_len     (B,)

    Args:
        meta: Dataset metadata dict from ``load_meta()``.
        cfg: Parsed ``multitask_config.yaml``.

    Returns:
        Tuple of (watch_pred, like_pred), each shape ``(B,)`` in [0, 1].
    """

    def __init__(self, meta: dict, cfg: dict) -> None:
        super().__init__()
        mc = cfg["model"]
        mmoe_cfg = mc["mmoe"]

        item_dim    = mc["item_embed_dim"]
        seq_dim     = mmoe_cfg["seq_embed_dim"]
        n_experts   = mmoe_cfg["n_experts"]
        expert_hid  = mmoe_cfg["expert_hidden_dim"]
        expert_out  = mmoe_cfg["expert_output_dim"]
        n_tasks     = mmoe_cfg.get("n_tasks", 2)
        tower_dims: List[int] = mmoe_cfg["task_tower_dims"]
        dropout     = mmoe_cfg.get("dropout", 0.1)
        attn_hidden: List[int] = mmoe_cfg.get("attention_hidden_dims", [64, 16])

        # ── Feature embeddings (same as DIN) ──
        self.user_embed = nn.Embedding(meta["n_users"],         mc["user_embed_dim"])
        self.item_embed = nn.Embedding(meta["n_items"],         item_dim)
        self.cat_embed  = nn.Embedding(meta["n_categories"],    mc["cat_embed_dim"])
        self.dur_embed  = nn.Embedding(meta["n_duration_bkts"], mc["dur_embed_dim"])
        # 1-indexed; 0 = padding
        self.hist_embed = nn.Embedding(meta["n_items"] + 1, seq_dim, padding_idx=0)

        # ── DIN Attention for history pooling ──
        self.attention = DINAttention(seq_dim, attn_hidden)

        # Project target item to seq_dim (in case item_dim != seq_dim)
        self.target_proj = nn.Linear(item_dim, seq_dim, bias=False)

        # ── Shared input dimension ──
        # user_emb + item_emb + cat_emb + dur_emb + user_dense + item_dense + hist_pooled
        self.concat_dim = (
            mc["user_embed_dim"]
            + item_dim
            + mc["cat_embed_dim"]
            + mc["dur_embed_dim"]
            + meta["user_dense_dim"]
            + meta["item_dense_dim"]
            + seq_dim
        )

        # ── Expert Networks ──
        self.experts = nn.ModuleList([
            ExpertNetwork(self.concat_dim, expert_out, expert_hid, dropout)
            for _ in range(n_experts)
        ])

        # ── Gating Networks (one per task) ──
        self.gates = nn.ModuleList([
            GatingNetwork(self.concat_dim, n_experts)
            for _ in range(n_tasks)
        ])

        # ── Task-specific Towers ──
        # Task 0: watch_ratio, Task 1: like
        self.towers = nn.ModuleList()
        for _ in range(n_tasks):
            layer_dims = [expert_out] + list(tower_dims) + [1]
            tower_layers: List[nn.Module] = []
            for i in range(len(layer_dims) - 1):
                tower_layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
                if i < len(layer_dims) - 2:
                    tower_layers.append(nn.ReLU())
                    tower_layers.append(nn.Dropout(dropout))
            self.towers.append(nn.Sequential(*tower_layers))

        self.n_experts = n_experts
        self.n_tasks   = n_tasks

        self._init_weights()

    def _init_weights(self) -> None:
        for emb in (self.user_embed, self.item_embed,
                    self.cat_embed, self.dur_embed, self.hist_embed):
            nn.init.normal_(emb.weight, std=0.01)
        nn.init.xavier_uniform_(self.target_proj.weight)

    def _encode_features(self, batch: dict) -> torch.Tensor:
        """Build shared feature representation.

        Returns:
            (B, concat_dim) concatenated feature vector.
        """
        i_emb = self.item_embed(batch["item_id"])       # (B, item_dim)
        u_emb = self.user_embed(batch["user_id"])       # (B, user_dim)
        c_emb = self.cat_embed(batch["item_category"])  # (B, cat_dim)
        d_emb = self.dur_embed(batch["item_dur_bkt"])   # (B, dur_dim)

        # History → DIN attention pooling
        hist_emb = self.hist_embed(batch["history_seq"])  # (B, L, seq_dim)
        pad_mask = (batch["history_seq"] == 0)            # (B, L)

        # Project target to seq_dim for attention
        target_for_attn = self.target_proj(i_emb)          # (B, seq_dim)

        attn_scores = self.attention(target_for_attn, hist_emb)  # (B, L)
        attn_scores = attn_scores.masked_fill(pad_mask, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)        # (B, L)
        hist_pooled = (attn_weights.unsqueeze(-1) * hist_emb).sum(dim=1)  # (B, seq_dim)

        shared_input = torch.cat([
            u_emb,
            i_emb,
            c_emb,
            d_emb,
            batch["user_dense"],
            batch["item_dense"],
            hist_pooled,
        ], dim=-1)  # (B, concat_dim)

        return shared_input

    def forward(self, batch: dict) -> tuple:
        """Compute watch_ratio and like predictions.

        Args:
            batch: Dict from RankingDataset (mtl_mode=True compatible).

        Returns:
            (watch_pred, like_pred): each (B,) in [0, 1].
        """
        shared_input = self._encode_features(batch)  # (B, concat_dim)

        # ── Expert outputs ──
        expert_outputs = [expert(shared_input) for expert in self.experts]
        # Stack: (B, n_experts, expert_output_dim)
        expert_stack = torch.stack(expert_outputs, dim=1)

        # ── Per-task predictions ──
        predictions = []
        for task_idx in range(self.n_tasks):
            gate_weights = self.gates[task_idx](shared_input)  # (B, n_experts)
            # Weighted sum of experts: (B, n_experts, 1) × (B, n_experts, D) → (B, D)
            task_input = (gate_weights.unsqueeze(-1) * expert_stack).sum(dim=1)
            logit = self.towers[task_idx](task_input).squeeze(-1)  # (B,)
            pred = torch.sigmoid(logit)
            predictions.append(pred)

        return predictions[0], predictions[1]  # (watch_pred, like_pred)

    def __repr__(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        emb_params = sum(
            p.numel()
            for m in (self.user_embed, self.item_embed, self.cat_embed,
                      self.dur_embed, self.hist_embed)
            for p in m.parameters()
        )
        expert_params = sum(p.numel() for m in self.experts for p in m.parameters())
        gate_params   = sum(p.numel() for m in self.gates   for p in m.parameters())
        tower_params  = sum(p.numel() for m in self.towers  for p in m.parameters())
        return (
            f"MMoE(\n"
            f"  Embeddings   : {emb_params:,} params\n"
            f"  Expert x{self.n_experts}    : {expert_params:,} params\n"
            f"  Gate x{self.n_tasks}      : {gate_params:,} params\n"
            f"  Task Tower x{self.n_tasks}: {tower_params:,} params\n"
            f"  Total        : {total:,} params\n"
            f")"
        )
