
from loader import load_pdf_file
from qa_chunker import split_qa_entries
from vector_store import VectorStore
from evaluation_dataset import EVALUATION_DATASET
from embeddings import generate_embedding

from rag_chain import MIN_RELEVANCE_SCORE
from retrieval_metrics import (
    calculate_rejection_rate,
    is_rejected_by_threshold,
)


PDF_PATH = (
    "/Users/nishiigupta/Documents/"
    "AI_Learnings_Projects/ai-knowledge-assistant/"
    "documents/neural_networks_sample.pdf"
)

TOP_K = 3


def contains_expected_chunk(document, expected_chunks):
    """Return True if the document contains an acceptable chunk marker."""
    return any(
        marker in document
        for marker in expected_chunks
    )


def main():
    # 1. Build the knowledge base.
    text = load_pdf_file(PDF_PATH)
    chunks = split_qa_entries(text)

    store = VectorStore()

    for chunk in chunks:
        store.add_text(chunk)

    # 2. Separate answerable and unanswerable test cases.
    answerable_cases = [
        case for case in EVALUATION_DATASET
        if case["answerable"]
    ]

    unanswerable_cases = [
        case for case in EVALUATION_DATASET
        if not case["answerable"]
    ]

    # 3. Initialize all evaluation counters.
    successful_queries = 0
    reciprocal_rank_sum = 0.0

    unanswerable_rejected = 0
    unanswerable_false_accepts = 0

    category_stats = {}

    # 4. Evaluate every query.
    for index, test_case in enumerate(
        EVALUATION_DATASET,
        start=1,
    ):
        question = test_case["question"]
        expected_chunks = test_case["expected_chunks"]
        answerable = test_case["answerable"]
        category = test_case["category"]

        query_embedding = generate_embedding(question)
        results = store.search(
            query_embedding,
            top_k=TOP_K,
        )

        print(f"\n{'=' * 70}")
        print(f"Case {index}: {category}")
        print(f"Question: {question}")

        if answerable:
            # 5. Find the rank of the first expected chunk.
            expected_rank = None

            for rank, (document, score, metadata) in enumerate(
                results,
                start=1,
            ):
                if contains_expected_chunk(
                    document,
                    expected_chunks,
                ):
                    expected_rank = rank
                    break

            hit = expected_rank is not None

            if hit:
                successful_queries += 1
                reciprocal_rank_sum += 1.0 / expected_rank

            category_stats.setdefault(
                category,
                {"queries": 0, "hits": 0},
            )

            category_stats[category]["queries"] += 1

            if hit:
                category_stats[category]["hits"] += 1

            status = "PASS" if hit else "FAIL"

            print(f"Expected chunks: {expected_chunks}")
            print(f"Retrieval result: {status}")
            print(f"Expected chunk rank: {expected_rank}")

        else:
            # 6. Evaluate threshold behavior for unanswerable queries.
            print("Expected chunks: none")
            print("Answerability: unsupported by the source document")

            if not results:
                rejected = is_rejected_by_threshold(
                    None,
                    MIN_RELEVANCE_SCORE,
                )
                print("No results retrieved.")
            else:
                top_score = results[0][1]

                rejected = is_rejected_by_threshold(
                    top_score,
                    MIN_RELEVANCE_SCORE,
                )

                print(f"Top similarity score: {top_score:.4f}")

            print(f"Relevance threshold: {MIN_RELEVANCE_SCORE:.2f}")

            if rejected:
                unanswerable_rejected += 1
                print("Threshold decision: REJECT")
            else:
                unanswerable_false_accepts += 1
                print("Threshold decision: ACCEPT (false accept)")

        # 7. Print the retrieved evidence.
        print("Retrieved results:")

        if not results:
            print("  No results returned.")

        for rank, (document, score, metadata) in enumerate(
            results,
            start=1,
        ):
            preview = document.splitlines()[0] if document else "(empty document)"

            print(
                f"  Rank {rank}: "
                f"{preview} "
                f"(similarity={score:.4f})"
            )

    # 8. Calculate metrics for answerable queries.
    total_answerable = len(answerable_cases)

    hit_rate = (
        successful_queries / total_answerable
        if total_answerable
        else 0.0
    )

    mrr = (
        reciprocal_rank_sum / total_answerable
        if total_answerable
        else 0.0
    )

    # 9. Calculate metrics for unanswerable queries.
    total_unanswerable = len(unanswerable_cases)

    rejection_rate = calculate_rejection_rate(
        unanswerable_rejected,
        total_unanswerable,
    )

    # 10. Print the final evaluation summary.
    print("\n" + "=" * 70)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 70)

    print(f"Total evaluation queries: {len(EVALUATION_DATASET)}")
    print(f"Answerable queries: {total_answerable}")
    print(f"Unanswerable queries: {total_unanswerable}")
    print(f"Successful retrievals: {successful_queries}")
    print(f"Hit Rate@{TOP_K}: {hit_rate:.2%}")
    print(f"MRR@{TOP_K}: {mrr:.4f}")

    print("\nUnanswerable-query threshold evaluation:")
    print(
        f"Unanswerable queries rejected: "
        f"{unanswerable_rejected}/{total_unanswerable}"
    )
    print(f"Unanswerable rejection rate: {rejection_rate:.2%}")
    print(f"Unanswerable false accepts: {unanswerable_false_accepts}")

    print("\nCategory-level retrieval performance:")

    for category, stats in category_stats.items():
        category_hit_rate = (
            stats["hits"] / stats["queries"]
            if stats["queries"]
            else 0.0
        )

        print(
            f"{category}: "
            f"{stats['hits']}/{stats['queries']} hits "
            f"({category_hit_rate:.2%})"
        )


if __name__ == "__main__":
    main()
