from qa_chunker import split_qa_entries
from loader import load_pdf_file
from embeddings import generate_embedding
from vector_store import VectorStore
from context_builders import build_context


def main():

    pdf_path = "/Users/nishiigupta/Documents/AI_Learnings_Projects/ai-knowledge-assistant/documents/neural_networks_sample.pdf"

    # 1. load and clean the pdf
    text = load_pdf_file(pdf_path)

    # 2. chunk using the qa chunker for our specific Q&A format pdf document
    chunks = split_qa_entries(text)

    # 3. geenrate embeddings for each chunk and store in vector store
    store = VectorStore()

    for chunk in chunks :
        store.add_text(chunk)
    
    # 4. user query to retrieve relevant context
    question = input("Enter your question: ")
    if not question.strip():
        print("No question provided. Exiting.")
        return
    
    # 5. Retrieve relevant context from the vector store based on the user query
    query_embedding = generate_embedding(question)
    results = store.search(query_embedding=query_embedding, top_k=2)

    context = build_context(results)

    print("\n===== USER QUESTION =====")
    print(question)

    print("\n===== RETRIEVED CHUNKS =====")
    for rank, (document, score) in enumerate(results, start=1):
        print(f"\nRank: {rank} | Similarity: {score:.4f}")
        print(document)

    print("\n===== FINAL CONTEXT =====")
    print(context)


if __name__ == "__main__":
    main()

