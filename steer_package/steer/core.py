"""
STEER core primitives and the Steerer class.
"""

import numpy as np
from typing import Optional, List, Union


def normalize_vec(v: np.ndarray) -> np.ndarray:
    """Normalize a single vector."""
    norm = np.linalg.norm(v)
    return v / max(norm, 1e-10)


def normalize_rows(X: np.ndarray) -> np.ndarray:
    """Normalize each row of a matrix."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-10)


def steer(q: np.ndarray, t: np.ndarray, alpha: float) -> np.ndarray:
    """
    The STEER primitive: normalize(q + alpha * t).

    Args:
        q: Query embedding (normalized).
        t: Target embedding (normalized).
        alpha: Steering intensity. Positive = toward, negative = away.

    Returns:
        Steered query embedding (normalized).
    """
    return normalize_vec(q + alpha * t)


def adaptive_alpha(
    q: np.ndarray,
    t: np.ndarray,
    alpha_max: float = 0.2,
    power: float = 2.0,
) -> float:
    """
    Compute adaptive alpha based on query-target similarity.

    Queries close to target get minimal rotation (already aligned).
    Queries far from target get stronger rotation.

    Args:
        q: Query embedding (normalized).
        t: Target embedding (normalized).
        alpha_max: Maximum alpha value.
        power: Exponent for the decay function. Default 2.0 (quadratic).

    Returns:
        Adaptive alpha value.
    """
    sim = float(np.dot(q, t))
    return alpha_max * (1.0 - sim) ** power


def rrf_fusion(
    scores_list: List[np.ndarray],
    k: int = 60,
) -> np.ndarray:
    """
    Reciprocal Rank Fusion of multiple score arrays.

    Args:
        scores_list: List of score arrays (each shape [n_docs]).
        k: RRF parameter (default 60).

    Returns:
        Fused scores array.
    """
    n = scores_list[0].shape[0]
    rrf = np.zeros(n)
    for s in scores_list:
        ranks = np.argsort(np.argsort(-s)) + 1
        rrf += 1.0 / (k + ranks)
    return rrf


class Steerer:
    """
    High-level STEER interface for semantic steering in retrieval.

    Example:
        steerer = Steerer(model_name="all-MiniLM-L6-v2")
        steerer.index(corpus_texts)
        results = steerer.search("VEGF inhibition", target="oncology", alpha=0.1)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")

        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.corpus_embs: Optional[np.ndarray] = None
        self.doc_ids: Optional[List[str]] = None

    def index(self, texts: List[str], ids: Optional[List[str]] = None, batch_size: int = 256):
        """Encode and index a corpus."""
        self.corpus_embs = np.array(
            self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
        )
        self.doc_ids = ids or [str(i) for i in range(len(texts))]
        self._centroid = normalize_vec(np.mean(self.corpus_embs, axis=0))

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text."""
        return np.array(self.model.encode(text, normalize_embeddings=True))

    def search(
        self,
        query: str,
        target: Optional[str] = None,
        alpha: float = 0.1,
        adaptive: bool = True,
        alpha_max: float = 0.2,
        multi_vector: bool = True,
        top_k: int = 10,
    ) -> List[dict]:
        """
        Search with optional steering.

        Args:
            query: Query text.
            target: Target domain text. If None, no steering (baseline search).
            alpha: Steering intensity (ignored if adaptive=True).
            adaptive: Use adaptive alpha based on query-target similarity.
            alpha_max: Maximum alpha for adaptive mode.
            multi_vector: Use multi-vector RRF fusion (recommended).
            top_k: Number of results to return.

        Returns:
            List of dicts with 'id', 'score', and 'steered' (bool) keys.
        """
        if self.corpus_embs is None:
            raise RuntimeError("Call .index() first.")

        q = self.encode(query)
        base_scores = q @ self.corpus_embs.T

        if target is None:
            # Baseline search
            top_idx = np.argsort(base_scores)[::-1][:top_k]
            return [{"id": self.doc_ids[i], "score": float(base_scores[i]), "steered": False} for i in top_idx]

        t = self.encode(target)

        if adaptive:
            alpha = adaptive_alpha(q, t, alpha_max=alpha_max)

        q_steered = steer(q, t, alpha)
        steered_scores = q_steered @ self.corpus_embs.T

        if multi_vector:
            fused = rrf_fusion([base_scores, steered_scores])
            top_idx = np.argsort(fused)[::-1][:top_k]
            return [{"id": self.doc_ids[i], "score": float(fused[i]), "steered": True} for i in top_idx]
        else:
            top_idx = np.argsort(steered_scores)[::-1][:top_k]
            return [{"id": self.doc_ids[i], "score": float(steered_scores[i]), "steered": True} for i in top_idx]

    def contrastive_search(
        self,
        query: str,
        positive_target: str,
        negative_target: str,
        alpha: float = 0.1,
        beta: float = 0.2,
        top_k: int = 10,
    ) -> List[dict]:
        """Search with contrastive steering (push toward positive, away from negative)."""
        if self.corpus_embs is None:
            raise RuntimeError("Call .index() first.")

        q = self.encode(query)
        t_pos = self.encode(positive_target)
        t_neg = self.encode(negative_target)
        q_contr = normalize_vec(q + alpha * t_pos - beta * t_neg)

        base_scores = q @ self.corpus_embs.T
        contr_scores = q_contr @ self.corpus_embs.T
        fused = rrf_fusion([base_scores, contr_scores])
        top_idx = np.argsort(fused)[::-1][:top_k]
        return [{"id": self.doc_ids[i], "score": float(fused[i]), "steered": True} for i in top_idx]

    def gradient_walk(
        self,
        query: str,
        target: str,
        alphas: Optional[List[float]] = None,
        top_k: int = 10,
    ) -> List[dict]:
        """
        Generate results at multiple alpha values to visualize the steering spectrum.

        Returns list of dicts, one per alpha, with 'alpha' and 'results' keys.
        """
        if self.corpus_embs is None:
            raise RuntimeError("Call .index() first.")

        if alphas is None:
            alphas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

        q = self.encode(query)
        t = self.encode(target)
        walk = []

        for a in alphas:
            if a == 0:
                scores = q @ self.corpus_embs.T
            else:
                q_s = steer(q, t, a)
                scores = q_s @ self.corpus_embs.T
            top_idx = np.argsort(scores)[::-1][:top_k]
            results = [{"id": self.doc_ids[i], "score": float(scores[i])} for i in top_idx]
            walk.append({"alpha": a, "results": results})

        return walk
