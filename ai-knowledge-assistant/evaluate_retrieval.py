from loader import load_pdf_file
from qa_chunker import split_qa_entries
from vector_store import VectorStore
from evaluation_dataset import EVALUATION_DATASET
from embeddings import generate_embedding


def main():

    pdf_path = "/Users/nishiigupta/Documents/AI_Learnings_Projects/ai-knowledge-assistant/documents/neural_networks_sample.pdf"
    # Build the knowledge base
    text = load_pdf_file(pdf_path)
    chunks = split_qa_entries(text)

    store = VectorStore()

    for chunk in chunks:
        store.add_text(chunk)

    # Evaluate retrieval
    total_queries = len(EVALUATION_DATASET)
    successful_queries = 0
    top_k = 3

    for test_case in EVALUATION_DATASET:
        question = test_case["question"]
        expected_chunk = test_case["expected_chunk"]

        query_embedding = generate_embedding(question)
        results = store.search(query_embedding, top_k=top_k)

        found = any(
            expected_chunk in document
            for document, score in results
        )

        if found:
            successful_queries += 1

        status = "PASS" if found else "FAIL"

        print(f"\n[{status}] {question}")
        print(f"Expected: {expected_chunk}")
        print("Retrieved:")

        for rank, (document, score) in enumerate(results, start=1):
            print(
                f"  Rank {rank}: "
                f"{document.splitlines()[0]} "
                f"(similarity={score:.4f})"
            )

    hit_rate = successful_queries / total_queries if total_queries else 0

    print("\n===== EVALUATION SUMMARY =====")
    print(f"Total queries: {total_queries}")
    print(f"Successful queries: {successful_queries}")
    print(f"Hit Rate@{top_k}: {hit_rate:.2%}")


if __name__ == "__main__":
    main()