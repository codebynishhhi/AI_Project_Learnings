
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from qbp_copilot.ingestion.embedder import MODEL_NAME
from qbp_copilot.retrieval.retriever import load_index
from qbp_copilot.retrieval.hybrid_retriever import (
    build_bm25_index,
    hybrid_retrieve,
)


ROOT = Path(__file__).resolve().parents[1]

vectors, metadata = load_index(
    ROOT / "artifacts" / "embeddings"
)

model = SentenceTransformer(MODEL_NAME)

bm25_index = build_bm25_index(metadata)

questions = [
    "What was NVIDIA's actual quarterly revenue?",
    "What were the main operating expenses?",
    "What were GAAP and non-GAAP gross margins?",
]

for question in questions:
    print("\n" + "=" * 75)
    print("QUESTION:", question)

    results = hybrid_retrieve(
        query=question,
        model=model,
        vectors=vectors,
        metadata=metadata,
        bm25_index=bm25_index,
        top_k=3,
    )

    for rank, result in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print("Chunk:", result["chunk_id"])
        print("Section:", result.get("section"))
        print("Dense rank:", result["dense_rank"])
        print("Sparse rank:", result["sparse_rank"])
        print("RRF score:", round(result["rrf_score"], 5))
        print("Text:", result["text"][:350])
