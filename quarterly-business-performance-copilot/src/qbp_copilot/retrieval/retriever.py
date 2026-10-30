
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from qbp_copilot.ingestion.embedder import MODEL_NAME


def load_index(embeddings_dir: str | Path):
    """Load vectors and their corresponding chunk metadata."""
    embeddings_dir = Path(embeddings_dir)

    vectors = np.load(embeddings_dir / "chunk_embeddings.npy")
    metadata = json.loads(
        (embeddings_dir / "chunk_metadata.json").read_text(
            encoding="utf-8"
        )
    )

    if vectors.ndim != 2:
        raise ValueError("Expected a two-dimensional vector matrix")

    if vectors.shape[0] != len(metadata):
        raise ValueError("Vectors and metadata are misaligned")

    if vectors.shape[1] == 0 or not np.isfinite(vectors).all():
        raise ValueError("Invalid embedding matrix")

    if not metadata:
        raise ValueError("The retrieval index is empty")

    return vectors, metadata


def retrieve(
    query: str,
    model: SentenceTransformer,
    vectors: np.ndarray,
    metadata: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """Return the most semantically similar chunks."""
    if not query.strip():
        raise ValueError("Query cannot be empty")

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    if vectors.shape[0] != len(metadata):
        raise ValueError("Vectors and metadata are misaligned")

    query_vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    if query_vector.shape[0] != vectors.shape[1]:
        raise ValueError("Query and document embedding dimensions differ")

    # Because both sides are normalized, dot product equals cosine similarity.
    scores = vectors @ query_vector

    # Rank chunks from highest similarity to lowest.
    ranked_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for index in ranked_indices:
        result = metadata[int(index)].copy()
        result["similarity_score"] = float(scores[index])
        results.append(result)

    return results
