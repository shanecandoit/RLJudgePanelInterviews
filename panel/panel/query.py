"""Query selection: top-disagreement items plus a random anchor slice."""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def select_queries(scores: np.ndarray, n_query: int, anchor_frac: float,
                   rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Pick `n_query` candidate indices to resolve in the environment.

    ceil(anchor_frac * n_query) are uniform-random anchors (used for competence and
    audits); the rest are the highest-scoring of the remaining candidates.
    Returns (top_idx, anchor_idx), disjoint.
    """
    if n_query > len(scores):
        raise ValueError("fewer candidates than queries")
    n_anchor = min(n_query, math.ceil(anchor_frac * n_query))
    anchors = rng.choice(len(scores), n_anchor, replace=False)
    rest = np.setdiff1d(np.arange(len(scores)), anchors)
    top = rest[np.argsort(-scores[rest])[: n_query - n_anchor]]
    return top, anchors
