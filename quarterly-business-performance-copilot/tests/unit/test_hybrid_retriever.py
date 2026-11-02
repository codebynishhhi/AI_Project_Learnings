
import numpy as np
import pytest

from qbp_copilot.retrieval.hybrid_retriever import (
    tokenize,
    build_bm25_index,
    reciprocal_rank_fusion,
    hybrid_retrieve,
)


# --------------------------------------------------
# Test helper: fake embedding model
# --------------------------------------------------

class FakeEmbeddingModel:
    """
    Returns a deterministic query embedding.

    This avoids loading a real SentenceTransformer
    during unit tests.
    """

    def encode(
        self,
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ):
        return np.array(
            [[1.0, 0.0, 0.0]],
            dtype=np.float32,
        )


# --------------------------------------------------
# Sample document chunks
# --------------------------------------------------

@pytest.fixture
def sample_metadata():
    return [
        {
            "chunk_id": "chunk_001",
            "text": "NVIDIA quarterly revenue was 96 billion dollars.",
            "section": "Financial Results",
            "page_numbers": [1],
        },
        {
            "chunk_id": "chunk_002",
            "text": "Operating expenses increased during the quarter.",
            "section": "Income Statement",
            "page_numbers": [3],
        },
        {
            "chunk_id": "chunk_003",
            "text": "Data Center revenue increased year over year.",
            "section": "Business Performance",
            "page_numbers": [2],
        },
    ]


@pytest.fixture
def sample_vectors():
    # Each row represents one chunk embedding.
    # All vectors are normalized.
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


# --------------------------------------------------
# 1. Tokenization tests
# --------------------------------------------------

def test_tokenize_converts_text_to_lowercase_tokens():
    result = tokenize("NVIDIA Quarterly Revenue")

    assert result == [
        "nvidia",
        "quarterly",
        "revenue",
    ]


def test_tokenize_preserves_decimal_numbers():
    result = tokenize("Revenue was 96.2 billion")

    assert "96.2" in result


def test_tokenize_rejects_non_string_input():
    with pytest.raises(TypeError):
        tokenize(["revenue", "expenses"])


# --------------------------------------------------
# 2. BM25 index tests
# --------------------------------------------------

def test_build_bm25_index_creates_index(sample_metadata):
    bm25_index = build_bm25_index(sample_metadata)

    scores = bm25_index.get_scores(["revenue"])

    assert len(scores) == len(sample_metadata)
    assert np.isfinite(scores).all()


def test_build_bm25_index_rejects_empty_metadata():
    with pytest.raises(ValueError):
        build_bm25_index([])


def test_build_bm25_index_rejects_empty_chunk_text():
    metadata = [
        {
            "chunk_id": "chunk_001",
            "text": "   ",
        }
    ]

    with pytest.raises(ValueError):
        build_bm25_index(metadata)


# --------------------------------------------------
# 3. Reciprocal Rank Fusion tests
# --------------------------------------------------

def test_rrf_combines_multiple_rankings():
    rankings = [
        [0, 1, 2],
        [1, 2, 0],
    ]

    scores = reciprocal_rank_fusion(
        rankings=rankings,
        k=60,
    )

    # Document 1 is ranked first in the second list
    # and second in the first list.
    assert scores[1] > scores[0]
    assert scores[1] > scores[2]


def test_rrf_uses_one_based_ranks():
    scores = reciprocal_rank_fusion(
        rankings=[[0]],
        k=60,
    )

    assert scores[0] == pytest.approx(1 / 61)


def test_rrf_rejects_invalid_k():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(
            rankings=[[0, 1]],
            k=0,
        )


def test_rrf_returns_empty_dict_for_empty_rankings():
    scores = reciprocal_rank_fusion([])

    assert scores == {}


# --------------------------------------------------
# 4. Hybrid retrieval integration tests
# --------------------------------------------------

def test_hybrid_retrieval_returns_top_k_results(
    sample_metadata,
    sample_vectors,
):
    model = FakeEmbeddingModel()

    bm25_index = build_bm25_index(sample_metadata)

    results = hybrid_retrieve(
        query="NVIDIA quarterly revenue",
        model=model,
        vectors=sample_vectors,
        metadata=sample_metadata,
        bm25_index=bm25_index,
        top_k=2,
    )

    assert len(results) == 2

    assert all(
        "chunk_id" in result
        for result in results
    )

    assert all(
        "similarity_score" in result
        for result in results
    )

    assert all(
        "bm25_score" in result
        for result in results
    )

    assert all(
        "dense_rank" in result
        for result in results
    )

    assert all(
        "sparse_rank" in result
        for result in results
    )

    assert all(
        "rrf_score" in result
        for result in results
    )


def test_hybrid_retrieval_ranks_matching_evidence_first(
    sample_metadata,
    sample_vectors,
):
    model = FakeEmbeddingModel()

    bm25_index = build_bm25_index(sample_metadata)

    results = hybrid_retrieve(
        query="NVIDIA quarterly revenue",
        model=model,
        vectors=sample_vectors,
        metadata=sample_metadata,
        bm25_index=bm25_index,
        top_k=3,
    )

    assert results[0]["chunk_id"] == "chunk_001"


def test_hybrid_retrieval_rejects_empty_query(
    sample_metadata,
    sample_vectors,
):
    model = FakeEmbeddingModel()
    bm25_index = build_bm25_index(sample_metadata)

    with pytest.raises(ValueError):
        hybrid_retrieve(
            query=" ",
            model=model,
            vectors=sample_vectors,
            metadata=sample_metadata,
            bm25_index=bm25_index,
        )


def test_hybrid_retrieval_rejects_mismatched_vector_count(
    sample_metadata,
):
    model = FakeEmbeddingModel()
    bm25_index = build_bm25_index(sample_metadata)

    wrong_vectors = np.array(
        [[1.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        hybrid_retrieve(
            query="revenue",
            model=model,
            vectors=wrong_vectors,
            metadata=sample_metadata,
            bm25_index=bm25_index,
        )
