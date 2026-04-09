"""GPU detection, memory monitoring, and reproducibility utilities."""

import random
from typing import Dict

import numpy as np
import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_device() -> torch.device:
    """Detect and return the best available compute device.

    Prints GPU name and specs when CUDA is available; warns and falls back
    to CPU otherwise so training scripts never hard-crash on CPU-only machines.

    Returns:
        ``torch.device("cuda")`` or ``torch.device("cpu")``.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
        logger.info(f"VRAM: {props.total_memory / 1024 ** 3:.1f} GB")
    else:
        device = torch.device("cpu")
        logger.warning(
            "CUDA not available — falling back to CPU. "
            "Training will be significantly slower."
        )
    return device


def get_memory_stats() -> Dict[str, float]:
    """Return current GPU memory usage as a dict (all values in GB).

    Returns:
        Dict with keys ``allocated_gb``, ``reserved_gb``, ``total_gb``.
        Returns ``{"available": False}`` when no GPU is present.
    """
    if not torch.cuda.is_available():
        return {"available": False}
    return {
        "available": True,
        "allocated_gb": torch.cuda.memory_allocated() / 1024 ** 3,
        "reserved_gb": torch.cuda.memory_reserved() / 1024 ** 3,
        "total_gb": torch.cuda.get_device_properties(0).total_memory / 1024 ** 3,
    }


def log_memory_stats() -> None:
    """Log current GPU memory utilisation via the project logger."""
    stats = get_memory_stats()
    if stats.get("available"):
        logger.info(
            f"GPU Memory | allocated: {stats['allocated_gb']:.2f} GB  "
            f"reserved: {stats['reserved_gb']:.2f} GB  "
            f"total: {stats['total_gb']:.1f} GB"
        )


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for full reproducibility.

    Covers Python ``random``, NumPy, PyTorch CPU, and PyTorch CUDA RNGs.

    Args:
        seed: Integer seed value. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed fixed to {seed}")
