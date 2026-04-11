"""FastAPI serving application for the video recommendation system.

Provides online serving of the two-stage recommendation pipeline:
  1. Retrieval: Two-Tower model + Faiss index → top-K candidates
  2. Ranking:   DIN model → re-ranked scores

Windows-compatible: single worker, no multiprocessing (uvicorn default).

Usage:
    python src/serving/serve.py
    # or with uvicorn:
    uvicorn src.serving.serve:app --host 0.0.0.0 --port 8000 --workers 1

API Endpoints:
    GET  /health          → Service health check
    POST /recommend       → Get recommendations for a user
    POST /feedback        → Log user feedback (click, watch, like)
    GET  /stats           → Service statistics
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Optional FastAPI import (not installed in base env)
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    logger.warning(
        "FastAPI/uvicorn not installed. "
        "Install with: pip install fastapi uvicorn[standard]"
    )
    # Provide stubs so the module can be imported without FastAPI
    class BaseModel:  # type: ignore
        pass
    def Field(*args, **kwargs):  # type: ignore
        return None


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class RecommendRequest(BaseModel):
    """Request schema for /recommend endpoint."""
    user_id: int = Field(..., description="User identifier (0-indexed).")
    n_retrieve: int = Field(50, ge=1, le=500, description="Candidates from Faiss.")
    n_rank: int = Field(10, ge=1, le=100, description="Final re-ranked items to return.")
    context: Optional[Dict] = Field(None, description="Optional context features.")


class RecommendItem(BaseModel):
    """Single recommended item with score."""
    item_id: int
    retrieval_score: float
    ranking_score: float


class RecommendResponse(BaseModel):
    """Response schema for /recommend endpoint."""
    user_id: int
    items: List[RecommendItem]
    latency_ms: float


class FeedbackRequest(BaseModel):
    """Request schema for /feedback endpoint."""
    user_id: int = Field(..., description="User identifier.")
    item_id: int = Field(..., description="Item that was interacted with.")
    event: str = Field(..., description="Event type: 'click', 'watch', 'like', 'skip'.")
    watch_ratio: Optional[float] = Field(None, ge=0.0, le=1.0)


class FeedbackResponse(BaseModel):
    """Response schema for /feedback endpoint."""
    status: str
    message: str


class HealthResponse(BaseModel):
    """Response schema for /health endpoint."""
    status: str
    models_loaded: bool
    index_built: bool
    n_items_indexed: int
    uptime_seconds: float


class StatsResponse(BaseModel):
    """Response schema for /stats endpoint."""
    total_requests: int
    total_feedback_events: int
    avg_latency_ms: float
    uptime_seconds: float
    n_active_users: int


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

@dataclass
class AppState:
    """Holds all loaded models and data for the serving app."""
    # Models
    retrieval_model: Optional[object] = None
    ranking_model: Optional[object] = None

    # Faiss index + item embeddings
    faiss_index: Optional[object] = None
    item_embeddings: Optional[np.ndarray] = None
    item_ids: Optional[np.ndarray] = None

    # Item feature cache for ranking
    item_features: Optional[Dict] = None

    # User session: recent history (for sequence features)
    user_sessions: Dict[int, List[int]] = field(default_factory=dict)

    # Runtime stats
    start_time: float = field(default_factory=time.time)
    request_count: int = 0
    feedback_count: int = 0
    latencies: List[float] = field(default_factory=list)

    # Config
    meta: Optional[dict] = None
    device: Optional[torch.device] = None
    models_loaded: bool = False
    index_built: bool = False


# Global state instance
_state = AppState()


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class Predictor:
    """Two-stage recommendation predictor.

    Wraps the retrieval + ranking pipeline for online serving.

    Args:
        state: AppState with loaded models and Faiss index.
    """

    def __init__(self, state: AppState) -> None:
        self.state = state

    def recommend(
        self,
        user_id: int,
        n_retrieve: int = 50,
        n_rank: int = 10,
    ) -> List[RecommendItem]:
        """Generate recommendations for a user.

        Pipeline:
          1. Build user embedding from Two-Tower model
          2. Faiss search → top-n_retrieve candidates
          3. DIN ranking → re-score candidates
          4. Return top-n_rank items sorted by ranking score

        Args:
            user_id: 0-indexed user ID.
            n_retrieve: Number of candidates to retrieve.
            n_rank: Final number of items to return.

        Returns:
            List of RecommendItem sorted by ranking_score descending.
        """
        state = self.state
        device = state.device
        meta   = state.meta

        if not state.models_loaded:
            raise RuntimeError("Models not loaded yet.")

        # ── Build user features ──
        user_history = state.user_sessions.get(user_id, [])
        seq_len = meta.get("seq_len", 20)

        # Pad or truncate history to seq_len
        hist = user_history[-seq_len:]
        hist_padded = [0] * (seq_len - len(hist)) + [vid + 1 for vid in hist]

        batch = {
            "user_id":    torch.tensor([user_id], dtype=torch.long, device=device),
            "user_dense": torch.zeros(1, meta["user_dense_dim"], device=device),
            "history_seq": torch.tensor([hist_padded], dtype=torch.long, device=device),
            "history_len": torch.tensor([len(hist)], dtype=torch.long, device=device),
            # dummy item features (not used by user tower)
            "pos_item_id":       torch.tensor([0], dtype=torch.long, device=device),
            "pos_item_dense":    torch.zeros(1, meta["item_dense_dim"], device=device),
            "pos_item_category": torch.tensor([0], dtype=torch.long, device=device),
            "pos_item_dur_bkt":  torch.tensor([0], dtype=torch.long, device=device),
        }

        # ── Stage 1: Retrieval ──
        with torch.no_grad():
            user_emb, _ = state.retrieval_model(batch)

        user_emb_np = user_emb.cpu().numpy().astype(np.float32)

        if state.faiss_index is not None:
            scores, indices = state.faiss_index.search(user_emb_np, k=n_retrieve)
            candidate_ids = state.item_ids[indices[0]].tolist()
            retrieval_scores = scores[0].tolist()
        else:
            # Fallback: random candidates if no Faiss index
            n_items = meta.get("n_items", 1000)
            candidate_ids = list(range(min(n_retrieve, n_items)))
            retrieval_scores = [1.0] * len(candidate_ids)

        # ── Stage 2: Ranking ──
        if state.ranking_model is not None and len(candidate_ids) > 0:
            rank_items = self._rank_candidates(
                user_id, candidate_ids, retrieval_scores, device, meta
            )
        else:
            rank_items = [
                RecommendItem(
                    item_id=iid,
                    retrieval_score=float(rs),
                    ranking_score=float(rs),
                )
                for iid, rs in zip(candidate_ids, retrieval_scores)
            ]

        # Sort by ranking score and return top-n_rank
        rank_items.sort(key=lambda x: x.ranking_score, reverse=True)
        return rank_items[:n_rank]

    def _rank_candidates(
        self,
        user_id: int,
        candidate_ids: List[int],
        retrieval_scores: List[float],
        device: torch.device,
        meta: dict,
    ) -> List[RecommendItem]:
        """Score candidates with the ranking model.

        Args:
            user_id: User identifier.
            candidate_ids: List of candidate item IDs.
            retrieval_scores: Faiss retrieval scores (aligned with candidate_ids).
            device: Compute device.
            meta: Dataset metadata.

        Returns:
            List of RecommendItem with ranking_score populated.
        """
        state = self.state
        B = len(candidate_ids)
        seq_len = meta.get("seq_len", 20)

        user_history = state.user_sessions.get(user_id, [])
        hist = user_history[-seq_len:]
        hist_padded = [0] * (seq_len - len(hist)) + [vid + 1 for vid in hist]

        # Build batch
        rank_batch = {
            "user_id":       torch.tensor([user_id] * B, dtype=torch.long, device=device),
            "item_id":       torch.tensor(candidate_ids, dtype=torch.long, device=device),
            "user_dense":    torch.zeros(B, meta["user_dense_dim"], device=device),
            "item_dense":    torch.zeros(B, meta["item_dense_dim"], device=device),
            "item_category": torch.zeros(B, dtype=torch.long, device=device),
            "item_dur_bkt":  torch.zeros(B, dtype=torch.long, device=device),
            "history_seq":   torch.tensor([hist_padded] * B, dtype=torch.long, device=device),
            "history_len":   torch.tensor([len(hist)] * B, dtype=torch.long, device=device),
        }

        with torch.no_grad():
            rank_scores = state.ranking_model(rank_batch).cpu().numpy()

        return [
            RecommendItem(
                item_id=iid,
                retrieval_score=float(rs),
                ranking_score=float(rank_s),
            )
            for iid, rs, rank_s in zip(candidate_ids, retrieval_scores, rank_scores)
        ]


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

def _load_models_and_index() -> None:
    """Load retrieval/ranking models and build Faiss index at startup.

    If checkpoints don't exist, logs a warning (caller can trigger training).
    """
    global _state

    cfg_path = _ROOT / "configs" / "base_config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    retrieval_cfg_path = _ROOT / "configs" / "retrieval_config.yaml"
    with open(retrieval_cfg_path, encoding="utf-8") as f:
        retrieval_cfg = yaml.safe_load(f)

    ranking_cfg_path = _ROOT / "configs" / "ranking_config.yaml"
    with open(ranking_cfg_path, encoding="utf-8") as f:
        ranking_cfg = yaml.safe_load(f)

    from src.utils.gpu_utils import get_device

    device = get_device()
    _state.device = device
    _state.start_time = time.time()

    # Load metadata
    proc_dir = _ROOT / base_cfg["data"]["processed_dir"]
    meta_path = proc_dir / "meta.pkl"
    if meta_path.exists():
        import pickle
        with open(meta_path, "rb") as f:
            _state.meta = pickle.load(f)
        logger.info(f"Metadata loaded: {_state.meta}")
    else:
        logger.warning(
            f"Metadata not found at {meta_path}. "
            "Run download_data.py + feature_engineering.py first."
        )
        # Use placeholder meta so the app can still start
        _state.meta = {
            "n_users": 500, "n_items": 1000, "n_categories": 20,
            "n_duration_bkts": 5, "user_dense_dim": 25, "item_dense_dim": 3,
            "seq_len": 20,
        }

    meta = _state.meta
    ckpt_dir = _ROOT / base_cfg["logging"]["checkpoint_dir"]

    # ── Load retrieval model ──
    retrieval_ckpt = ckpt_dir / "two_tower_best.pt"
    if retrieval_ckpt.exists():
        from src.models.two_tower import TwoTowerModel
        retrieval_model = TwoTowerModel(meta, retrieval_cfg)
        ckpt = torch.load(retrieval_ckpt, map_location=device, weights_only=False)
        retrieval_model.load_state_dict(ckpt["model_state_dict"])
        retrieval_model.to(device).eval()
        _state.retrieval_model = retrieval_model
        logger.info(f"Retrieval model loaded from {retrieval_ckpt}")
    else:
        logger.warning(
            f"Retrieval checkpoint not found at {retrieval_ckpt}. "
            "Recommendations will use random candidates."
        )

    # ── Load ranking model ──
    ranking_ckpt = ckpt_dir / "din_best.pt"
    if ranking_ckpt.exists():
        from src.models.din import DIN
        ranking_model = DIN(meta, ranking_cfg)
        ckpt = torch.load(ranking_ckpt, map_location=device, weights_only=False)
        ranking_model.load_state_dict(ckpt["model_state_dict"])
        ranking_model.to(device).eval()
        _state.ranking_model = ranking_model
        logger.info(f"Ranking model loaded from {ranking_ckpt}")
    else:
        logger.warning(
            f"Ranking checkpoint not found at {ranking_ckpt}. "
            "Retrieval scores will be used directly."
        )

    # ── Build Faiss index from item embeddings ──
    if _state.retrieval_model is not None:
        _build_faiss_index(meta, retrieval_cfg, device)

    _state.models_loaded = True
    logger.info("AppState initialised. Service ready.")


def _build_faiss_index(meta: dict, retrieval_cfg: dict, device: torch.device) -> None:
    """Build Faiss index by encoding all items with the item tower.

    Args:
        meta: Dataset metadata.
        retrieval_cfg: Retrieval model config.
        device: Compute device.
    """
    global _state

    try:
        from src.retrieval.faiss_index import FaissIndex

        n_items = meta["n_items"]
        logger.info(f"Building Faiss index for {n_items:,} items …")

        # Encode all items using item tower
        model = _state.retrieval_model
        batch_size = 256
        all_embeddings = []

        for start in range(0, n_items, batch_size):
            end = min(start + batch_size, n_items)
            ids = torch.arange(start, end, dtype=torch.long, device=device)
            item_batch = {
                "pos_item_id":       ids,
                "pos_item_dense":    torch.zeros(len(ids), meta["item_dense_dim"], device=device),
                "pos_item_category": torch.zeros(len(ids), dtype=torch.long, device=device),
                "pos_item_dur_bkt":  torch.zeros(len(ids), dtype=torch.long, device=device),
            }
            with torch.no_grad():
                _, item_emb = model(
                    {
                        "user_id":    torch.zeros(len(ids), dtype=torch.long, device=device),
                        "user_dense": torch.zeros(len(ids), meta["user_dense_dim"], device=device),
                        "history_seq": torch.zeros(len(ids), meta.get("seq_len", 20), dtype=torch.long, device=device),
                        "history_len": torch.zeros(len(ids), dtype=torch.long, device=device),
                        **item_batch,
                    }
                )
            all_embeddings.append(item_emb.cpu().numpy())

        item_embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
        item_ids = np.arange(n_items)

        output_dim = retrieval_cfg["model"]["output_dim"]
        faiss_idx = FaissIndex(dim=output_dim, index_type="flat")
        faiss_idx.build(item_embeddings, item_ids)

        _state.faiss_index    = faiss_idx
        _state.item_embeddings = item_embeddings
        _state.item_ids        = item_ids
        _state.index_built     = True
        logger.info(f"Faiss index built with {n_items:,} items.")

    except Exception as e:
        logger.error(f"Failed to build Faiss index: {e}")
        _state.index_built = False


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

if _FASTAPI_AVAILABLE:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """FastAPI lifespan: load models on startup, cleanup on shutdown."""
        logger.info("Starting up recommendation service …")
        try:
            _load_models_and_index()
        except Exception as e:
            logger.error(f"Startup failed: {e}")
        yield
        logger.info("Shutting down recommendation service …")

    app = FastAPI(
        title="Video Recommendation API",
        description="Two-stage (Retrieval + Ranking) recommendation service.",
        version="1.0.0",
        lifespan=lifespan,
    )

    _predictor: Optional[Predictor] = None

    def _get_predictor() -> Predictor:
        global _predictor
        if _predictor is None:
            _predictor = Predictor(_state)
        return _predictor

    # ── Endpoints ──

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(
            status="ok" if _state.models_loaded else "initialising",
            models_loaded=_state.models_loaded,
            index_built=_state.index_built,
            n_items_indexed=len(_state.item_ids) if _state.item_ids is not None else 0,
            uptime_seconds=time.time() - _state.start_time,
        )

    @app.post("/recommend", response_model=RecommendResponse)
    async def recommend(request: RecommendRequest) -> RecommendResponse:
        """Get personalised video recommendations for a user.

        Args:
            request: RecommendRequest with user_id and retrieval/ranking params.

        Returns:
            RecommendResponse with ranked item list and latency.
        """
        t0 = time.time()
        _state.request_count += 1

        try:
            predictor = _get_predictor()
            items = predictor.recommend(
                user_id=request.user_id,
                n_retrieve=request.n_retrieve,
                n_rank=request.n_rank,
            )
        except Exception as e:
            logger.error(f"Recommendation failed for user {request.user_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        latency_ms = (time.time() - t0) * 1000
        _state.latencies.append(latency_ms)

        return RecommendResponse(
            user_id=request.user_id,
            items=items,
            latency_ms=round(latency_ms, 2),
        )

    @app.post("/feedback", response_model=FeedbackResponse)
    async def feedback(request: FeedbackRequest) -> FeedbackResponse:
        """Log user feedback event (click, watch, like, skip).

        Updates the user's session history for future recommendations.

        Args:
            request: FeedbackRequest with user_id, item_id, and event type.

        Returns:
            FeedbackResponse confirming receipt.
        """
        _state.feedback_count += 1

        # Update user session history for positive interactions
        if request.event in ("click", "watch", "like"):
            if request.user_id not in _state.user_sessions:
                _state.user_sessions[request.user_id] = []
            _state.user_sessions[request.user_id].append(request.item_id)
            # Keep only last seq_len items
            seq_len = _state.meta.get("seq_len", 20) if _state.meta else 20
            _state.user_sessions[request.user_id] = (
                _state.user_sessions[request.user_id][-seq_len:]
            )

        logger.debug(
            f"Feedback: user={request.user_id} item={request.item_id} "
            f"event={request.event}"
        )

        return FeedbackResponse(
            status="ok",
            message=f"Feedback '{request.event}' recorded for user {request.user_id}.",
        )

    @app.get("/stats", response_model=StatsResponse)
    async def stats() -> StatsResponse:
        """Service statistics endpoint.

        Returns:
            StatsResponse with request counts, latency, and uptime.
        """
        avg_latency = (
            float(np.mean(_state.latencies)) if _state.latencies else 0.0
        )
        return StatsResponse(
            total_requests=_state.request_count,
            total_feedback_events=_state.feedback_count,
            avg_latency_ms=round(avg_latency, 2),
            uptime_seconds=round(time.time() - _state.start_time, 1),
            n_active_users=len(_state.user_sessions),
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not _FASTAPI_AVAILABLE:
        print("FastAPI and uvicorn are required. Install with:")
        print("  pip install fastapi uvicorn[standard]")
        sys.exit(1)

    import uvicorn

    uvicorn.run(
        "src.serving.serve:app",
        host="0.0.0.0",
        port=8000,
        workers=1,       # Windows: single worker (no multiprocessing fork)
        reload=False,    # Disable reload in production
        log_level="info",
    )
