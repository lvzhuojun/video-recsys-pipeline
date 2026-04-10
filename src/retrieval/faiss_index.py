"""Faiss vector index wrapper for large-scale item retrieval.

Supports two index types:
  flat     — IndexFlatIP: exact brute-force inner-product search.
             Use for dev/evaluation; guaranteed optimal recall.
  ivfflat  — IndexIVFFlat: approximate search via inverted file.
             Use when the item catalogue is large (>100k items).

Both indexes assume L2-normalised embeddings so that inner product
equals cosine similarity.

Windows note: faiss-gpu is not available via pip on Windows.
This file uses faiss-cpu.  On Linux you can switch to faiss-gpu by
changing the build method to use faiss.StandardGpuResources().
"""

from pathlib import Path
from typing import Optional, Tuple

import faiss
import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


class FaissIndex:
    """Lightweight wrapper around a Faiss inner-product index.

    Args:
        dim: Embedding dimension (must match item embeddings).
        index_type: ``"flat"`` (exact) or ``"ivfflat"`` (approximate).
        n_lists: IVFFlat number of Voronoi cells (ignored for flat).
        n_probe: IVFFlat cells to visit at query time (ignored for flat).
            Higher ``n_probe`` → better recall, slower search.

    Example:
        >>> idx = FaissIndex(dim=64, index_type='flat')
        >>> idx.build(embeddings, item_ids)
        >>> scores, ids = idx.search(user_emb, top_k=50)
    """

    def __init__(
        self,
        dim: int,
        index_type: str = "flat",
        n_lists: int = 50,
        n_probe: int = 10,
    ) -> None:
        assert index_type in ("flat", "ivfflat"), \
            f"index_type must be 'flat' or 'ivfflat', got '{index_type}'"
        self.dim = dim
        self.index_type = index_type
        self.n_lists = n_lists
        self.n_probe = n_probe
        self._index: Optional[faiss.Index] = None
        self._item_ids: Optional[np.ndarray] = None

    # ------------------------------------------------------------------

    def build(self, item_embeddings: np.ndarray, item_ids: np.ndarray) -> None:
        """Build (and train if necessary) the Faiss index.

        Args:
            item_embeddings: Float32 array of shape ``(N, dim)``.
                Must be L2-normalised beforehand.
            item_ids: Int array of shape ``(N,)`` mapping position → item ID.
        """
        assert item_embeddings.ndim == 2 and item_embeddings.shape[1] == self.dim, \
            f"Expected embeddings shape (N, {self.dim}), got {item_embeddings.shape}"

        vecs = np.ascontiguousarray(item_embeddings, dtype=np.float32)
        n = len(vecs)

        if self.index_type == "flat":
            # Exact inner-product search — no training required
            self._index = faiss.IndexFlatIP(self.dim)

        elif self.index_type == "ivfflat":
            # Clip n_lists so it doesn't exceed the number of training vectors
            n_lists = min(self.n_lists, n // 10)
            quantizer = faiss.IndexFlatIP(self.dim)
            self._index = faiss.IndexIVFFlat(
                quantizer, self.dim, n_lists, faiss.METRIC_INNER_PRODUCT
            )
            logger.info(f"Training IVFFlat index with {n_lists} lists on {n} vectors …")
            self._index.train(vecs)
            self._index.nprobe = self.n_probe

        self._index.add(vecs)
        self._item_ids = item_ids.astype(np.int32).copy()

        logger.info(
            f"FaissIndex [{self.index_type}] built | "
            f"items={n:,} | dim={self.dim}"
        )

    # ------------------------------------------------------------------

    def search(
        self, query: np.ndarray, top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve the top-K nearest items for each query vector.

        Args:
            query: Float32 array of shape ``(D,)`` or ``(N, D)``.
                Must be L2-normalised.
            top_k: Number of results to return per query.

        Returns:
            Tuple ``(scores, item_ids)``:
              - ``scores``:   shape ``(top_k,)`` or ``(N, top_k)``
              - ``item_ids``: shape ``(top_k,)`` or ``(N, top_k)``
        """
        assert self._index is not None, "Call build() before search()."
        squeeze = query.ndim == 1
        q = np.atleast_2d(query).astype(np.float32)

        scores, indices = self._index.search(q, top_k)  # (N, top_k)
        # Map internal indices back to original item IDs
        item_ids = self._item_ids[indices]

        if squeeze:
            return scores[0], item_ids[0]
        return scores, item_ids

    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the Faiss index and item_ids array to disk.

        Args:
            path: File path (without extension). Creates ``<path>.index``
                and ``<path>_ids.npy``.
        """
        assert self._index is not None, "Nothing to save — build() not called."
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            faiss.write_index(self._index, str(p) + ".index")
            np.save(str(p) + "_ids.npy", self._item_ids)
            logger.info(f"Faiss index saved to {p}.index")
        except OSError as e:
            logger.error(f"Failed to save Faiss index: {e}")
            raise

    def load(self, path: str) -> None:
        """Load a previously saved index from disk.

        Args:
            path: Base path (without extension) matching what ``save()`` used.
        """
        p = Path(path)
        try:
            self._index = faiss.read_index(str(p) + ".index")
            self._item_ids = np.load(str(p) + "_ids.npy")
            if self.index_type == "ivfflat":
                self._index.nprobe = self.n_probe
            logger.info(f"Faiss index loaded from {p}.index")
        except OSError as e:
            logger.error(f"Failed to load Faiss index: {e}")
            raise

    def __repr__(self) -> str:
        n = len(self._item_ids) if self._item_ids is not None else 0
        return (
            f"FaissIndex(type='{self.index_type}', "
            f"dim={self.dim}, n_items={n:,})"
        )
