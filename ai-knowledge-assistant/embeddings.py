
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def generate_embedding(text: str) -> np.ndarray:
    return model.encode(text)


def cosine_similarity(
    vector_a: np.ndarray,
    vector_b: np.ndarray
) -> float:
    denominator = (
        np.linalg.norm(vector_a)
        * np.linalg.norm(vector_b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(vector_a, vector_b) / denominator
    )


if __name__ == "__main__":
    texts = [
        "Large language models help developers write code.",
        "LLMs assist programmers with software development.",
        "The weather is sunny today."
    ]

    embeddings = [generate_embedding(text) for text in texts]

    print(f"Dimensions: {len(embeddings[0])}")
    print(f"First five values: {embeddings[0][:5]}")

    print(
        "A <-> B similarity:",
        f"{cosine_similarity(embeddings[0], embeddings[1]):.4f}"
    )

    print(
        "A <-> C similarity:",
        f"{cosine_similarity(embeddings[0], embeddings[2]):.4f}"
    )
