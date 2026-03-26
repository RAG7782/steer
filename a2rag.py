"""
A²RAG — Algebraic Retrieval for RAG Systems

Two operations for pre-retrieval query transformation:
1. rotate_toward: Embedding rotation for analogical retrieval
2. subtract_orthogonal: Orthogonal concept exclusion for semantic negation

Author: Renato Aparecido Gomes
License: MIT
"""

import numpy as np
from typing import Optional


def rotate_toward(
    source_emb: np.ndarray,
    target_emb: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Embedding rotation: interpolate source toward target by factor alpha.

    Enables analogical retrieval — querying domain A to discover
    what corresponds in domain B.

    Args:
        source_emb: Query embedding (normalized)
        target_emb: Target domain embedding (normalized)
        alpha: Rotation factor in [0, 1]. 0 = identity, 1 = full replacement

    Returns:
        Normalized rotated embedding
    """
    result = (1 - alpha) * source_emb + alpha * target_emb
    norm = np.linalg.norm(result)
    return result / norm if norm > 1e-10 else result


def subtract_orthogonal(
    base_emb: np.ndarray,
    exclude_emb: np.ndarray,
) -> np.ndarray:
    """
    Orthogonal concept exclusion: project out the unwanted semantic direction.

    Implements semantic negation — removing a concept from the query
    in embedding space without keyword filtering.

    Args:
        base_emb: Query embedding (normalized)
        exclude_emb: Concept to exclude (normalized)

    Returns:
        Normalized embedding orthogonal to exclude_emb
    """
    proj = np.dot(base_emb, exclude_emb) / (np.dot(exclude_emb, exclude_emb) + 1e-10)
    result = base_emb - proj * exclude_emb
    norm = np.linalg.norm(result)
    return result / norm if norm > 1e-10 else result


def algebraic_retrieve(
    query_emb: np.ndarray,
    corpus_embs: np.ndarray,
    top_k: int = 10,
    rotate_target: Optional[np.ndarray] = None,
    rotate_alpha: float = 0.4,
    exclude_concept: Optional[np.ndarray] = None,
) -> list[tuple[int, float]]:
    """
    Full algebraic retrieval pipeline.

    Applies optional rotation and/or subtraction to the query embedding,
    then retrieves top-k documents by cosine similarity.

    Args:
        query_emb: Query embedding (normalized)
        corpus_embs: Matrix of corpus embeddings (N x D, normalized)
        top_k: Number of results to return
        rotate_target: If provided, rotate query toward this target
        rotate_alpha: Rotation factor
        exclude_concept: If provided, subtract this concept from query

    Returns:
        List of (index, similarity_score) tuples, sorted by score descending
    """
    transformed = query_emb.copy()

    if exclude_concept is not None:
        transformed = subtract_orthogonal(transformed, exclude_concept)

    if rotate_target is not None:
        transformed = rotate_toward(transformed, rotate_target, rotate_alpha)

    similarities = corpus_embs @ transformed
    top_indices = np.argsort(similarities)[::-1][:top_k]

    return [(int(idx), float(similarities[idx])) for idx in top_indices]
