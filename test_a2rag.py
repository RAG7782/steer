"""
Tests for A²RAG algebraic retrieval operations.

Validates mathematical properties independent of any corpus or embedding model.
"""

import numpy as np
import pytest

from a2rag import rotate_toward, subtract_orthogonal, algebraic_retrieve


class TestRotation:
    def test_output_is_unit_vector(self):
        a = np.random.randn(384)
        a /= np.linalg.norm(a)
        b = np.random.randn(384)
        b /= np.linalg.norm(b)
        result = rotate_toward(a, b, alpha=0.5)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6

    def test_alpha_zero_is_identity(self):
        a = np.random.randn(384)
        a /= np.linalg.norm(a)
        b = np.random.randn(384)
        b /= np.linalg.norm(b)
        result = rotate_toward(a, b, alpha=0.0)
        assert np.allclose(result, a, atol=1e-6)

    def test_alpha_one_is_replacement(self):
        a = np.random.randn(384)
        a /= np.linalg.norm(a)
        b = np.random.randn(384)
        b /= np.linalg.norm(b)
        result = rotate_toward(a, b, alpha=1.0)
        assert np.allclose(result, b, atol=1e-6)

    def test_rotation_increases_similarity_to_target(self):
        a = np.random.randn(384)
        a /= np.linalg.norm(a)
        b = np.random.randn(384)
        b /= np.linalg.norm(b)
        sim_before = np.dot(a, b)
        rotated = rotate_toward(a, b, alpha=0.4)
        sim_after = np.dot(rotated, b)
        assert sim_after >= sim_before

    def test_monotonic_in_alpha(self):
        a = np.random.randn(384)
        a /= np.linalg.norm(a)
        b = np.random.randn(384)
        b /= np.linalg.norm(b)
        sims = []
        for alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            r = rotate_toward(a, b, alpha)
            sims.append(np.dot(r, b))
        for i in range(len(sims) - 1):
            assert sims[i + 1] >= sims[i] - 1e-6


class TestSubtraction:
    def test_output_is_unit_vector(self):
        a = np.random.randn(384)
        a /= np.linalg.norm(a)
        b = np.random.randn(384)
        b /= np.linalg.norm(b)
        result = subtract_orthogonal(a, b)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6

    def test_result_orthogonal_to_excluded(self):
        a = np.random.randn(384)
        a /= np.linalg.norm(a)
        b = np.random.randn(384)
        b /= np.linalg.norm(b)
        result = subtract_orthogonal(a, b)
        assert abs(np.dot(result, b)) < 1e-6

    def test_orthogonal_input_unchanged(self):
        a = np.zeros(384)
        a[0] = 1.0
        b = np.zeros(384)
        b[1] = 1.0
        result = subtract_orthogonal(a, b)
        assert np.allclose(result, a, atol=1e-6)

    def test_reduces_similarity_to_excluded(self):
        a = np.random.randn(384)
        a /= np.linalg.norm(a)
        b = np.random.randn(384)
        b /= np.linalg.norm(b)
        sim_before = abs(np.dot(a, b))
        result = subtract_orthogonal(a, b)
        sim_after = abs(np.dot(result, b))
        assert sim_after < sim_before


class TestAlgebraicRetrieve:
    def test_returns_top_k(self):
        query = np.random.randn(384)
        query /= np.linalg.norm(query)
        corpus = np.random.randn(100, 384)
        corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
        results = algebraic_retrieve(query, corpus, top_k=5)
        assert len(results) == 5

    def test_results_sorted_descending(self):
        query = np.random.randn(384)
        query /= np.linalg.norm(query)
        corpus = np.random.randn(100, 384)
        corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
        results = algebraic_retrieve(query, corpus, top_k=10)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_identity_without_transforms(self):
        query = np.random.randn(384)
        query /= np.linalg.norm(query)
        corpus = np.random.randn(50, 384)
        corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
        results = algebraic_retrieve(query, corpus, top_k=5)
        vanilla = corpus @ query
        top5_vanilla = np.argsort(vanilla)[::-1][:5]
        assert [r[0] for r in results] == list(top5_vanilla)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
