"""Two-Tower (Dual Encoder) model for large-scale item retrieval.

Architecture overview:
  UserTower : user_id embedding  +  history sequence mean-pooling
              +  user dense stats  →  MLP  →  L2-normalized (D,) vector
  ItemTower : item_id embedding  +  category embedding  +  duration embedding
              +  item dense stats  →  MLP  →  L2-normalized (D,) vector

At serving time only the ItemTower needs to run offline to pre-compute item
embeddings; user embeddings are computed online per request.  Inner product
(= cosine similarity after L2 normalisation) drives the Faiss retrieval.
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared building block
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Fully-connected feed-forward network with ReLU activations.

    Args:
        layer_dims: List of integers specifying input and output sizes of
            each linear layer.  E.g., ``[128, 64, 32]`` creates two layers.
        dropout: Dropout probability applied after every hidden activation.
    """

    def __init__(self, layer_dims: List[int], dropout: float = 0.1) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            is_last = i == len(layer_dims) - 2
            if not is_last:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.net(x)


# ---------------------------------------------------------------------------
# User Tower
# ---------------------------------------------------------------------------

class UserTower(nn.Module):
    """Encodes a user into a fixed-size L2-normalised embedding.

    Inputs:
        user_id        (B,)      user index for embedding lookup
        user_dense     (B, 25)   activity stats + category preferences
        history_seq    (B, L)    past video IDs, 1-indexed (0 = padding)
        history_len    (B,)      actual sequence length

    Output:
        (B, output_dim)  L2-normalised user embedding
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int,
        seq_embed_dim: int,
        user_dense_dim: int,
        dense_hidden: int,
        output_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.user_embed = nn.Embedding(n_users, embed_dim)
        # seq_embed uses n_items+1 rows so that 0 (padding) has its own zero row
        self.seq_embed = nn.Embedding(n_items + 1, seq_embed_dim, padding_idx=0)
        self.dense_proj = nn.Linear(user_dense_dim, dense_hidden)

        in_dim = embed_dim + seq_embed_dim + dense_hidden
        self.mlp = MLP([in_dim, 256, 128, output_dim], dropout=dropout)
        self.output_dim = output_dim

        nn.init.normal_(self.user_embed.weight, std=0.01)
        nn.init.normal_(self.seq_embed.weight, std=0.01)

    def forward(
        self,
        user_id: torch.Tensor,
        user_dense: torch.Tensor,
        history_seq: torch.Tensor,
        history_len: torch.Tensor,
    ) -> torch.Tensor:
        u_emb = self.user_embed(user_id)                          # (B, E)

        # Masked mean pooling — ignore padding tokens (value 0)
        seq_emb = self.seq_embed(history_seq)                     # (B, L, SE)
        mask = (history_seq > 0).float().unsqueeze(-1)            # (B, L, 1)
        seq_pooled = (seq_emb * mask).sum(1) / (mask.sum(1) + 1e-9)  # (B, SE)

        d_feat = F.relu(self.dense_proj(user_dense))              # (B, DH)

        out = self.mlp(torch.cat([u_emb, seq_pooled, d_feat], dim=-1))
        return F.normalize(out, p=2, dim=-1)                      # (B, out_dim)

    def __repr__(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return f"UserTower(params={n:,}, output_dim={self.output_dim})"


# ---------------------------------------------------------------------------
# Item Tower
# ---------------------------------------------------------------------------

class ItemTower(nn.Module):
    """Encodes a video into a fixed-size L2-normalised embedding.

    Inputs:
        item_id        (B,)     video index for embedding lookup
        item_dense     (B, 3)   historical CTR, like-rate, log-popularity
        item_category  (B,)     category index
        item_dur_bkt   (B,)     duration bucket index [0-4]

    Output:
        (B, output_dim)  L2-normalised item embedding
    """

    def __init__(
        self,
        n_items: int,
        n_categories: int,
        n_dur_bkts: int,
        embed_dim: int,
        cat_embed_dim: int,
        dur_embed_dim: int,
        item_dense_dim: int,
        dense_hidden: int,
        output_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.item_embed = nn.Embedding(n_items, embed_dim)
        self.cat_embed  = nn.Embedding(n_categories, cat_embed_dim)
        self.dur_embed  = nn.Embedding(n_dur_bkts, dur_embed_dim)
        self.dense_proj = nn.Linear(item_dense_dim, dense_hidden)

        in_dim = embed_dim + cat_embed_dim + dur_embed_dim + dense_hidden
        self.mlp = MLP([in_dim, 256, 128, output_dim], dropout=dropout)
        self.output_dim = output_dim

        for emb in (self.item_embed, self.cat_embed, self.dur_embed):
            nn.init.normal_(emb.weight, std=0.01)

    def forward(
        self,
        item_id: torch.Tensor,
        item_dense: torch.Tensor,
        item_category: torch.Tensor,
        item_dur_bkt: torch.Tensor,
    ) -> torch.Tensor:
        i_emb = self.item_embed(item_id)       # (B, E)
        c_emb = self.cat_embed(item_category)  # (B, CE)
        d_emb = self.dur_embed(item_dur_bkt)   # (B, DE)
        d_feat = F.relu(self.dense_proj(item_dense))  # (B, DH)

        out = self.mlp(torch.cat([i_emb, c_emb, d_emb, d_feat], dim=-1))
        return F.normalize(out, p=2, dim=-1)   # (B, out_dim)

    def __repr__(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return f"ItemTower(params={n:,}, output_dim={self.output_dim})"


# ---------------------------------------------------------------------------
# Two-Tower Model
# ---------------------------------------------------------------------------

class TwoTowerModel(nn.Module):
    """Full Two-Tower dual-encoder retrieval model.

    Wraps UserTower and ItemTower with a shared training interface.
    Supports in-batch negative sampling (InfoNCE loss) and random negative
    sampling (BPR-style loss).

    Args:
        meta: Dataset metadata dict from ``load_meta()``.
        cfg: Parsed ``retrieval_config.yaml``.
    """

    def __init__(self, meta: dict, cfg: dict) -> None:
        super().__init__()
        mc = cfg["model"]
        self.output_dim = mc["output_dim"]
        self.temperature = mc["temperature"]

        self.user_tower = UserTower(
            n_users=meta["n_users"],
            n_items=meta["n_items"],
            embed_dim=mc["embed_dim"],
            seq_embed_dim=mc["seq_embed_dim"],
            user_dense_dim=meta["user_dense_dim"],
            dense_hidden=mc["dense_hidden"],
            output_dim=mc["output_dim"],
            dropout=mc.get("dropout", 0.1),
        )
        self.item_tower = ItemTower(
            n_items=meta["n_items"],
            n_categories=meta["n_categories"],
            n_dur_bkts=meta["n_duration_bkts"],
            embed_dim=mc["embed_dim"],
            cat_embed_dim=mc["cat_embed_dim"],
            dur_embed_dim=mc["dur_embed_dim"],
            item_dense_dim=meta["item_dense_dim"],
            dense_hidden=mc["dense_hidden"],
            output_dim=mc["output_dim"],
            dropout=mc.get("dropout", 0.1),
        )

    def encode_user(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode a batch through UserTower. Returns (B, D) L2-normalised."""
        return self.user_tower(
            batch["user_id"], batch["user_dense"],
            batch["history_seq"], batch["history_len"],
        )

    def encode_item(
        self,
        item_id: torch.Tensor,
        item_dense: torch.Tensor,
        item_category: torch.Tensor,
        item_dur_bkt: torch.Tensor,
    ) -> torch.Tensor:
        """Encode items through ItemTower. Returns (B, D) L2-normalised."""
        return self.item_tower(item_id, item_dense, item_category, item_dur_bkt)

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode user and positive item. Returns (user_emb, item_emb)."""
        user_emb = self.encode_user(batch)
        item_emb = self.encode_item(
            batch["pos_item_id"], batch["pos_item_dense"],
            batch["pos_item_category"], batch["pos_item_dur_bkt"],
        )
        return user_emb, item_emb

    def in_batch_loss(
        self, user_emb: torch.Tensor, item_emb: torch.Tensor
    ) -> torch.Tensor:
        """InfoNCE loss with in-batch negatives.

        Computes the (B, B) cosine-similarity matrix and treats the diagonal
        as the positive pair for each sample.  This is mathematically
        equivalent to contrastive loss with B-1 negatives per sample.

        Args:
            user_emb: (B, D) L2-normalised user embeddings.
            item_emb: (B, D) L2-normalised item embeddings.

        Returns:
            Scalar cross-entropy loss.
        """
        # Both are L2-normalised → inner product = cosine similarity
        sim = user_emb @ item_emb.T / self.temperature  # (B, B)
        labels = torch.arange(sim.size(0), device=sim.device)
        return F.cross_entropy(sim, labels)

    def random_neg_loss(
        self,
        user_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        neg_emb: torch.Tensor,
    ) -> torch.Tensor:
        """BPR (Bayesian Personalised Ranking) loss for random negatives.

        Maximises the margin between positive and negative scores.

        Args:
            user_emb: (B, D)
            pos_emb: (B, D)
            neg_emb: (B, D)

        Returns:
            Scalar BPR loss.
        """
        pos_scores = (user_emb * pos_emb).sum(dim=-1)   # (B,)
        neg_scores = (user_emb * neg_emb).sum(dim=-1)   # (B,)
        return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-9).mean()

    def __repr__(self) -> str:
        u = sum(p.numel() for p in self.user_tower.parameters())
        i = sum(p.numel() for p in self.item_tower.parameters())
        return (
            f"TwoTowerModel(\n"
            f"  UserTower : {u:,} params\n"
            f"  ItemTower : {i:,} params\n"
            f"  Total     : {u + i:,} params\n"
            f"  output_dim={self.output_dim}, temperature={self.temperature}\n"
            f")"
        )
