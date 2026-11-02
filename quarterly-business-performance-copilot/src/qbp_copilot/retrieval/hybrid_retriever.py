
import re

import numpy as np
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """
    Convert one text string into lowercase searchable tokens.

    Example:
        "NVIDIA Q2 revenue was $30.0 billion"
        -> ["nvidia", "q2", "revenue", "was", "30.0", "billion"]
    """
    if not isinstance(text, str):
        raise TypeError(
            f"Expected text to be a string, got {type(text).__name__}"
        )

    return re.findall(
        r"[a-z0-9]+(?:\.[0-9]+)?",
        text.lower(),
    )


def build_bm25_index(
    metadata: list[dict],
) -> BM25Okapi:
    """
    Build a BM25 index from the text of all document chunks.

    Input:
        List of chunk metadata dictionaries.

    Output:
        BM25 index that can score a search query against every chunk.
    """
    if not metadata:
        raise ValueError(
            "Cannot build BM25 index from empty metadata"
        )

    corpus = [
        tokenize(chunk["text"])
        for chunk in metadata
    ]

    if any(not tokens for tokens in corpus):
        raise ValueError(
            "At least one chunk contains no searchable tokens"
        )

    return BM25Okapi(corpus)


def reciprocal_rank_fusion(
    rankings: list[list[int]],
    k: int = 60,
) -> dict[int, float]:
    """
    Combine multiple ranked result lists using Reciprocal Rank Fusion.

    Each ranking contains document indices ordered from best to worst.

    RRF formula:
        score(document) = sum(1 / (k + rank))

    Rank starts at 1, not 0.
    """
    if k <= 0:
        raise ValueError("RRF k must be greater than zero")

    fused_scores: dict[int, float] = {}

    for ranking in rankings:
        for rank, document_index in enumerate(
            ranking,
            start=1,
        ):
            fused_scores[document_index] = (
                fused_scores.get(document_index, 0.0)
                + 1.0 / (k + rank)
            )

    return fused_scores


def hybrid_retrieve(
    metadata: list[dict],
    vectors: np.ndarray,
    bm25_index: BM25Okapi,
    model,
    query: str,
    top_k: int = 3,
    rrf_k: int = 60,
) -> list[dict]:
    """
    Retrieve relevant chunks using dense and sparse retrieval,
    then combine their rankings using Reciprocal Rank Fusion.

    Dense retrieval:
        Uses embedding similarity.

    Sparse retrieval:
        Uses BM25 keyword relevance.

    Fusion:
        Uses Reciprocal Rank Fusion (RRF).

    Returns:
        Ranked list of chunk metadata with retrieval scores.
    """

    # --------------------------------------------------
    # 1. Validate inputs
    # --------------------------------------------------

    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    if not metadata:
        return []

    vectors = np.asarray(vectors, dtype=np.float32)

    if vectors.ndim != 2:
        raise ValueError("Embedding vectors must be a 2D matrix")

    if vectors.shape[0] != len(metadata):
        raise ValueError(
            "Embedding count and metadata count do not match"
        )

    if not np.isfinite(vectors).all():
        raise ValueError("Embedding vectors contain invalid values")

    # --------------------------------------------------
    # 2. Dense retrieval
    # --------------------------------------------------

    query_vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    query_vector = np.asarray(
        query_vector,
        dtype=np.float32,
    )

    if query_vector.shape[0] != vectors.shape[1]:
        raise ValueError(
            "Query embedding dimension does not match document embeddings"
        )

    # Embeddings are normalized, so dot product = cosine similarity.
    dense_scores = vectors @ query_vector

    dense_ranking = np.argsort(
        -dense_scores,
        kind="stable",
    ).tolist()

    # --------------------------------------------------
    # 3. Sparse retrieval using BM25
    # --------------------------------------------------

    query_tokens = tokenize(query)

    if not query_tokens:
        raise ValueError("Query contains no searchable tokens")

    sparse_scores = bm25_index.get_scores(query_tokens)

    sparse_scores = np.asarray(
        sparse_scores,
        dtype=np.float32,
    )

    if len(sparse_scores) != len(metadata):
        raise ValueError(
            "BM25 scores and metadata count do not match"
        )

    # If BM25 finds no keyword matches, do not add an
    # arbitrary sparse ranking to the fusion process.
    if np.any(sparse_scores > 0):
        sparse_ranking = np.argsort(
            -sparse_scores,
            kind="stable",
        ).tolist()
    else:
        sparse_ranking = []

    # --------------------------------------------------
    # 4. Reciprocal Rank Fusion
    # --------------------------------------------------

    rankings_to_fuse = [dense_ranking]

    if sparse_ranking:
        rankings_to_fuse.append(sparse_ranking)

    fused_scores = reciprocal_rank_fusion(
        rankings=rankings_to_fuse,
        k=rrf_k,
    )

    # --------------------------------------------------
    # 5. Final ranking
    # --------------------------------------------------

    final_ranking = sorted(
        fused_scores,
        key=lambda document_index: fused_scores[document_index],
        reverse=True,
    )

    # --------------------------------------------------
    # 6. Build final results
    # --------------------------------------------------

    results = []

    for document_index in final_ranking[:top_k]:

        result = dict(metadata[document_index])

        result["similarity_score"] = float(
            dense_scores[document_index]
        )

        result["bm25_score"] = float(
            sparse_scores[document_index]
        )

        result["dense_rank"] = (
            dense_ranking.index(document_index) + 1
        )

        result["sparse_rank"] = (
            sparse_ranking.index(document_index) + 1
            if document_index in sparse_ranking
            else None
        )

        result["rrf_score"] = float(
            fused_scores[document_index]
        )

        results.append(result)

    return results
