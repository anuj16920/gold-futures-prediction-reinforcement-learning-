"""
GPU setup and memory management.
"""
import logging
import torch

logger = logging.getLogger(__name__)


def setup_gpu():
    """Configure GPU settings for training."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        torch.backends.cudnn.benchmark = True
        mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
        logger.info(f"GPU memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
    else:
        device = torch.device("cpu")
        logger.info("GPU not available, using CPU")
    return device


def clear_gpu_cache():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared")
