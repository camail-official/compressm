"""Reduction algorithms for SSM compression."""

from compressm.reduction.hsv import (
    SelectionMode,
    hankel_singular_values_diagonal,
    reduction_analysis,
)
from compressm.reduction.balanced_truncation import (
    reduce_discrete_lti,
    lti_to_lru,
)

__all__ = [
    "SelectionMode",
    "hankel_singular_values_diagonal",
    "reduction_analysis",
    "reduce_discrete_lti",
    "lti_to_lru",
]
