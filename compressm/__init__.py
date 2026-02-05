"""
CompreSSM: In-Training Compression of State Space Models

This package implements balanced truncation-based compression for SSMs,
specifically the LRU (Linear Recurrent Unit) architecture.

Key modules:
- compressm.models: LRU model implementation
- compressm.reduction: Balanced truncation and HSV computation
- compressm.data: Dataset loaders (sMNIST, sCIFAR)
- compressm.training: Training loop and utilities
"""

__version__ = "1.0.0"

from compressm.models.lru import LRU, LRUBlock, LRULayer
from compressm.reduction.hsv import SelectionMode
from compressm.training.trainer import train, ReductionConfig

__all__ = [
    "LRU",
    "LRUBlock", 
    "LRULayer",
    "SelectionMode",
    "train",
    "ReductionConfig",
]
