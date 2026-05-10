"""Labeling utilities for predictor training.

Generic label generators used across the predictor stack (L1 alpha
targets, regime classification, future per-ticker direction
classifiers). Module is intentionally regime-agnostic — callers
parameterize the underlying log-return series and barrier widths
for their specific use case.
"""
from labeling.triple_barrier import (
    TRIPLE_BARRIER_SENTINEL,
    triple_barrier_alpha_labels,
    triple_barrier_class_labels,
)

__all__ = [
    "TRIPLE_BARRIER_SENTINEL",
    "triple_barrier_alpha_labels",
    "triple_barrier_class_labels",
]
