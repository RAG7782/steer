"""
STEER — Semantic Transformation for Embedding-space Exploration in Retrieval

A retrieval primitive for controllable semantic navigation in embedding spaces.

Usage:
    from steer import Steerer

    steerer = Steerer(model_name="all-MiniLM-L6-v2")
    results = steerer.search(query="VEGF inhibition", target="oncology", alpha=0.1)

Paper: https://arxiv.org/abs/XXXX.XXXXX  # TODO: update after submission
"""

__version__ = "0.1.0"

from steer.core import Steerer, steer, adaptive_alpha, rrf_fusion
from steer.operations import (
    rotate_toward,
    rotate_away,
    contrastive,
    amplify,
    diffuse,
    multi_view,
    gradient_walk,
)

__all__ = [
    "Steerer",
    "steer",
    "adaptive_alpha",
    "rrf_fusion",
    "rotate_toward",
    "rotate_away",
    "contrastive",
    "amplify",
    "diffuse",
    "multi_view",
    "gradient_walk",
]
