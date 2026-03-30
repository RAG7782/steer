"""
STEER extended operations — all 16 operations as standalone functions.
"""

import numpy as np
from typing import List, Optional
from steer.core import normalize_vec, normalize_rows, steer, adaptive_alpha, rrf_fusion


def rotate_toward(q: np.ndarray, t: np.ndarray, alpha: float) -> np.ndarray:
    """Steer query toward target domain."""
    return steer(q, t, alpha)


def rotate_away(q: np.ndarray, t: np.ndarray, alpha: float) -> np.ndarray:
    """Steer query away from target (bias correction)."""
    return steer(q, t, -abs(alpha))


def contrastive(
    q: np.ndarray,
    t_pos: np.ndarray,
    t_neg: np.ndarray,
    alpha: float = 0.1,
    beta: float = 0.2,
) -> np.ndarray:
    """Contrastive steer: push toward positive, away from negative."""
    return normalize_vec(q + alpha * t_pos - beta * t_neg)


def amplify(q: np.ndarray, centroid: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Steer toward corpus centroid (intensify specificity)."""
    return steer(q, centroid, alpha)


def diffuse(q: np.ndarray, centroid: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Steer away from corpus centroid (generalize, increase diversity)."""
    return steer(q, centroid, -abs(alpha))


def consensus(
    q: np.ndarray,
    targets: List[np.ndarray],
    alpha: float = 0.1,
) -> np.ndarray:
    """Steer toward mean of multiple targets (regularizer)."""
    mean_t = normalize_vec(np.mean(targets, axis=0))
    return steer(q, mean_t, alpha)


def multi_view(
    q: np.ndarray,
    targets: List[np.ndarray],
    corpus_embs: np.ndarray,
    alpha: float = 0.1,
    rrf_k: int = 60,
) -> np.ndarray:
    """
    Multi-view search: search q + T(q, t_i) for each target, fuse with RRF.

    Returns fused score array.
    """
    base_scores = q @ corpus_embs.T
    all_scores = [base_scores]
    for t in targets:
        q_s = steer(q, t, alpha)
        all_scores.append(q_s @ corpus_embs.T)
    return rrf_fusion(all_scores, k=rrf_k)


def gradient_walk(
    q: np.ndarray,
    t: np.ndarray,
    corpus_embs: np.ndarray,
    alphas: Optional[List[float]] = None,
) -> List[dict]:
    """
    Evaluate nDCG-proxy at multiple alpha values.

    Returns list of {alpha, scores} dicts.
    """
    if alphas is None:
        alphas = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

    results = []
    for a in alphas:
        if a == 0:
            scores = q @ corpus_embs.T
        else:
            q_s = steer(q, t, a)
            scores = q_s @ corpus_embs.T
        results.append({"alpha": a, "scores": scores})
    return results


def orbit(
    q: np.ndarray,
    targets: List[np.ndarray],
    corpus_embs: np.ndarray,
    alpha: float = 0.1,
) -> List[np.ndarray]:
    """
    Generate N steered result sets, one per target (no fusion).

    Returns list of score arrays.
    """
    return [steer(q, t, alpha) @ corpus_embs.T for t in targets]


def bridge(
    q: np.ndarray,
    source: np.ndarray,
    destination: np.ndarray,
    alpha: float = 0.1,
) -> np.ndarray:
    """Bridge: remove source domain influence, add destination."""
    direction = normalize_vec(destination - source)
    return normalize_vec(q + alpha * direction)


def triangulate(
    q: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    alpha1: float = 0.1,
    alpha2: float = 0.1,
) -> np.ndarray:
    """Sequential steering to two targets."""
    q_step1 = steer(q, t1, alpha1)
    return steer(q_step1, t2, alpha2)


def isotropy_correct(
    embeddings: np.ndarray,
    k: int = 1,
) -> tuple:
    """
    Remove top-k principal components to correct anisotropy.

    Returns (corrected_embeddings, components_removed).
    """
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(embeddings)
    result = embeddings.copy()
    comps = pca.components_[:k]
    for c in comps:
        result -= (result @ c).reshape(-1, 1) * c
    return normalize_rows(result), comps
