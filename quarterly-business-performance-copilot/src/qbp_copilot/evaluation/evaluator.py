from pathlib import Path
import json
from sentence_transformers import SentenceTransformer

from qbp_copilot.retrieval.retriever import retrieve

def load_evaluation_set(eval_dataset_path:str | Path) -> list[dict]:
    path = Path(eval_dataset_path)

    # check if file is present in that path 
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation file not found: {path}")
    
    # now fetch the questions from the path using json loads
    questions = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(questions, list) or not questions:
        raise ValueError("Evaluation set must be a non-empty list")

    # check if the que is having a 
    for item in questions:
        if not item.get("question"):
            raise ValueError("An evaluation question is missing its text")

        if not item.get("relevant_chunk_ids"):
            raise ValueError(f"No relevant chunk id found for the question - {item.get("question_id")}")
        
    return questions

# now once we have the dataset of questions we evaluate retrieval using - recall@k, MRR, & Hit@1
def evaluate_retrieval(questions : list[dict], model:SentenceTransformer, vectors, metadata:list[dict], top_k :int = 3) -> dict:
    """Evaluate retrieval using Recall@K, MRR, and Hit Rate@1."""
    results = []
    for each_item in questions:
        retrieved = retrieve(
            query=each_item["question"],
            model=model,
            vectors=vectors,
            metadata=metadata,
            top_k=top_k,
        )

        # generating a set of all the relevant ids from the eval dataset
        relevant_ids = set(each_item["relevant_chunk_ids"])
        retrieved_ids = [chunk["chunk_id"] for chunk in retrieved]

        relevant_ranks = [
            rank
            for rank, chunk_id in enumerate(retrieved_ids, start=1)
            if chunk_id in relevant_ids
        ]

        # Recall@K: fraction of known relevant chunks retrieved.
        recall = len(set(retrieved_ids) & relevant_ids) / len(relevant_ids)

        # Reciprocal rank: inverse rank of the first relevant result.
        reciprocal_rank = (
            1 / relevant_ranks[0] if relevant_ranks else 0.0
        )

        # Hit@1: whether the first result is relevant.
        hit_at_1 = bool(retrieved_ids and retrieved_ids[0] in relevant_ids)

        results.append({
            "question_id": each_item["question_id"],
            "question": each_item["question"],
            "retrieved_chunk_ids": retrieved_ids,
            "relevant_ranks": relevant_ranks,
            f"recall_at_{top_k}": recall,
            "reciprocal_rank": reciprocal_rank,
            "hit_at_1": hit_at_1,
        })

    count = len(results)

    return {
        "total_questions": count,
        f"recall_at_{top_k}": sum(
            item[f"recall_at_{top_k}"] for item in results
        ) / count,
        "mrr": sum(
            item["reciprocal_rank"] for item in results
        ) / count,
        "hit_at_1": sum(
            item["hit_at_1"] for item in results
        ) / count,
        "question_results": results,
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]

    evaluation_path = (
        project_root / "evaluation" / "retrieval_eval.json"
    )
    embeddings_dir = project_root / "artifacts" / "embeddings"

    vectors_path = embeddings_dir / "chunk_embeddings.npy"
    metadata_path = embeddings_dir / "chunk_metadata.json"

    import numpy as np

    vectors = np.load(vectors_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    from qbp_copilot.ingestion.embedder import MODEL_NAME

    model = SentenceTransformer(MODEL_NAME)
    questions = load_evaluation_set(evaluation_path)

    report = evaluate_retrieval(
        questions=questions,
        model=model,
        vectors=vectors,
        metadata=metadata,
        top_k=3,
    )

    print("\n========== RETRIEVAL EVALUATION ==========")
    print("Questions:", report["total_questions"])
    print(f"Recall@3: {report['recall_at_3']:.2%}")
    print(f"MRR:      {report['mrr']:.2%}")
    print(f"Hit@1:    {report['hit_at_1']:.2%}")

    print("\n========== QUESTION-LEVEL RESULTS ==========")

    for result in report["question_results"]:
        print(f"\n{result['question_id']}: {result['question']}")
        print("Retrieved:", result["retrieved_chunk_ids"])
        print("Relevant ranks:", result["relevant_ranks"])
        print(f"Recall@3: {result['recall_at_3']:.2%}")
        print(f"Reciprocal rank: {result['reciprocal_rank']:.2f}")
        print("Hit@1:", result["hit_at_1"])


# Step 2 — Understand the three metrics
# Metric	What it measures
# Recall@3	How many of the known relevant chunks appeared in the top 3
# MRR	How highly the first relevant chunk ranked
# Hit@1	Whether the first retrieved chunk was relevant

# For example, if the correct chunk is ranked second:
# - Recall@3 = 100% (assuming one relevant chunk)
# - Reciprocal rank = 0.5
# - Hit@1 = 0%
# This distinction is important: finding the right evidence somewhere in the results is not the same as ranking it first.