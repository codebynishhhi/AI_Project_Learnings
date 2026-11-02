
from typing import Any

import numpy as np
from sentence_transformers import CrossEncoder

RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

def load_reranker(model_name : str = RERANKER_MODEL_NAME) -> CrossEncoder:
    """
    Load a cross-encoder model for query-document relevance scoring.
    """
    return CrossEncoder(
        model_name,
        max_length=512,
    )


def rerank_chunks(chunks:list[dict[str, any]],query : str, top_k:int = 3, model:str = RERANKER_MODEL_NAME) -> list[dict[str, Any]]:
    """
    Rerank hybrid retrieval chunks using a cross-encoder.

    Each query-document pair is jointly evaluated by the model.
    chunks are sorted by their reranker relevance scores.
    """
    print("query ", query)
    if not isinstance(query, str) or  not query.strip():
        raise ValueError("Invalid query!")
    
    if not chunks:
        raise ValueError("Chunks list cannot be empty!")
    
    if top_k <= 0:
        raise ValueError("Top k value must be greater than 1")
    
    # Build query-document pairs.
    # Each pair is evaluated jointly by the cross-encoder.
    sentence_pairs = []
    
    for each_chunk in chunks:
        # extract the text 
        text = each_chunk["text"]

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Chunk does not contain text!")
        
        sentence_pairs.append([query, text])

    scores = model.predict(
        sentence_pairs,
        batch_size = 16,
        show_progress_bar = False,
        convert_to_numpy = True
    )

    if len(scores) != len(chunks):
        raise RuntimeError("Number of reranker scores does not match chunks.")
    
    if not np.all(np.isfinite(scores)):
        raise ValueError("Reranker produced an infinite score ")

    # Attach scores without modifying the original candidate data.
    scored_chunks = []

    for candidate, score in zip(chunks, scores):
        result = candidate.copy()
        result["reranker_score"] = float(score)
        scored_chunks.append(result)

    # Highest relevance score comes first.
    scored_chunks.sort(
        key=lambda item: item["reranker_score"],
        reverse=True,
    )

    # Assign final reranker ranks.
    for rank, candidate in enumerate(scored_chunks, start=1):
        candidate["reranker_rank"] = rank

    return scored_chunks[:top_k]


