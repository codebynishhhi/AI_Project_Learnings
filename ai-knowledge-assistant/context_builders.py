def build_context(retrieved_chunks : list[tuple[str, float, dict]]) -> str:
    """
    Builds a context string from the retrieved chunks.
    Each result contains:
        - document: retrieved text
        - score: similarity score
        - metadata: additional information about the chunk
    Args:
        retrieved_chunks (list[tuple[str, float]]): A list of tuples containing the text chunks and their similarity scores.
    Returns:
        str: A formatted context string.
    """

    if not retrieved_chunks:
        return ""
    
    context_parts = []

    for rank, (document, score, metadata) in enumerate(retrieved_chunks, start=1):
        source = metadata.get("source", "unknown")
        chunk_id = metadata.get("chunk_id", "unknown")
        
        context_parts.append(
            f"Rank {rank}:\nSimilarity: {score:.4f}\nDocument: {document}\n | Source : {source} | Chunk ID: {chunk_id}"
        )
    return "".join(context_parts)


if __name__ == "__main__":
    sample_results = [
        (
            "Q3: What is the purpose of an activation function?\n"
            "A3: It introduces non-linear properties to the network.",
            0.7668,
        ),
        (
            "Q5: What is backpropagation used for?\n"
            "A5: It helps the network update its weights and learn.",
            0.3969,
        ),
    ]

    context = build_context(sample_results)

    print("===== GENERATED CONTEXT =====")
    print(context)

