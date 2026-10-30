from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from qbp_copilot.retrieval.retriever import retrieve
from qbp_copilot.retrieval.hybrid_retriever import (
    build_bm25_index,
    hybrid_retrieve,
)
from qbp_copilot.ingestion.embedder import MODEL_NAME


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
def evaluate_retrieval(questions : list[dict], model:SentenceTransformer, vectors, metadata:list[dict], bm25_index, top_k :int = 3) -> dict:
    """Evaluate retrieval using Recall@K, MRR, and Hit Rate@1."""
    results = []
    for each_item in questions:

        # retrieve is only for dense search i.e vector search
        # retrieved = retrieve(
        #     query=each_item["question"],
        #     model=model,
        #     vectors=vectors,
        #     metadata=metadata,
        #     top_k=top_k,
        # )

        # Evaluating hybrid retrieval
        retrieved = hybrid_retrieve(
            query = each_item["question"],
            model = model,
            vectors = vectors,
            metadata = metadata,
            bm25_index = bm25_index,
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


def save_evaluation_report(report, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    return output_path
    

if __name__ == "__main__":

    # --------------------------------------------------
    # 1. Resolve project paths
    # --------------------------------------------------

    project_root = Path(__file__).resolve().parents[3]

    evaluation_path = (
        project_root
        / "evaluation"
        / "retrieval_eval.json"
    )

    embeddings_dir = (
        project_root
        / "artifacts"
        / "embeddings"
    )

    vectors_path = embeddings_dir / "chunk_embeddings.npy"

    metadata_path = (
        embeddings_dir
        / "chunk_metadata.json"
    )

    output_path = (
        project_root
        / "evaluation"
        / "results"
        / "hybrid_v1.json"
    )

    # --------------------------------------------------
    # 2. Load evaluation questions
    # --------------------------------------------------

    questions = load_evaluation_set(evaluation_path)

    # --------------------------------------------------
    # 3. Load document embeddings and metadata
    # --------------------------------------------------

    vectors = np.load(vectors_path)

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    if len(metadata) != vectors.shape[0]:
        raise ValueError(
            "Embedding count and metadata count do not match"
        )

    # --------------------------------------------------
    # 4. Load embedding model
    # --------------------------------------------------

    model = SentenceTransformer(MODEL_NAME)

    # --------------------------------------------------
    # 5. Build sparse BM25 index
    # --------------------------------------------------

    bm25_index = build_bm25_index(metadata)

    print("BM25 index built successfully.")
    print("Document chunks:", len(metadata))

    # --------------------------------------------------
    # 6. Evaluate hybrid retrieval
    # --------------------------------------------------

    report = evaluate_retrieval(
        questions=questions,
        model=model,
        vectors=vectors,
        metadata=metadata,
        bm25_index=bm25_index,
        top_k=3,
    )

    # --------------------------------------------------
    # 7. Display aggregate metrics
    # --------------------------------------------------

    print("\n========== HYBRID RETRIEVAL EVALUATION ==========")

    print("Questions:", report["total_questions"])

    print(
        f"Recall@3: {report['recall_at_3']:.2%}"
    )

    print(
        f"MRR:      {report['mrr']:.2%}"
    )

    print(
        f"Hit@1:    {report['hit_at_1']:.2%}"
    )

    # --------------------------------------------------
    # 8. Display question-level results
    # --------------------------------------------------

    print("\n========== QUESTION-LEVEL RESULTS ==========")

    for result in report["question_results"]:

        print(
            f"\n{result['question_id']}: "
            f"{result['question']}"
        )

        print(
            "Retrieved:",
            result["retrieved_chunk_ids"],
        )

        print(
            "Relevant ranks:",
            result["relevant_ranks"],
        )

        print(
            f"Recall@3: {result['recall_at_3']:.2%}"
        )

        print(
            f"Reciprocal rank: "
            f"{result['reciprocal_rank']:.2f}"
        )

        print(
            "Hit@1:",
            result["hit_at_1"],
        )

    # --------------------------------------------------
    # 9. Persist the evaluation report
    # --------------------------------------------------

    saved_path = save_evaluation_report(
        report,
        output_path,
    )

    print("\nEvaluation report saved to:")
    print(saved_path)
