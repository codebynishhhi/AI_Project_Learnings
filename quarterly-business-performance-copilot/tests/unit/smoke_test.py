from qbp_copilot.retrieval.reranker import (
    load_reranker,
    rerank_chunks,
)

model = load_reranker()

query = "What was NVIDIA's Q2 FY2027 GAAP gross margin?"

chunks = [
    {
        "chunk_id": "0000",
        "text": "NVIDIA reported a GAAP gross margin of 75.0% for Q2 FY2027.",
    },
    {
        "chunk_id": "0020",
        "text": "NVIDIA expects a GAAP gross margin of 74.0% for Q3 FY2027.",
    },
    {
        "chunk_id": "0011",
        "text": "NVIDIA reported its quarterly net income and operating expenses.",
    },
]

results = rerank_chunks(
    query=query,
    chunks=chunks,
    model=model,
    top_k=3,
)

for result in results:
    print(
        result["reranker_rank"],
        result["chunk_id"],
        result["reranker_score"],
    )