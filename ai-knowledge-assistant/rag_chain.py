
from dotenv import load_dotenv
from groq import Groq

from loader import load_pdf_file
from qa_chunker import split_qa_entries
from vector_store import VectorStore
from context_builders import build_context
from rag_prompt import build_rag_prompt

from embeddings import generate_embedding
load_dotenv()

MODEL_NAME = "openai/gpt-oss-20b"
DOCUMENT_PATH = "documents/neural_networks_sample.pdf"


def build_knowledge_base() -> VectorStore:
    """Load the PDF and index its Q&A entries."""
    text = load_pdf_file(DOCUMENT_PATH)
    chunks = split_qa_entries(text)

    store = VectorStore()

    for chunk in chunks:
        store.add_text(chunk)

    return store


def answer_question(
    question: str,
    store: VectorStore,
    client: Groq,
) -> str:
    """Retrieve context and generate an answer using Groq."""

    # 1. Convert the question into an embedding.
    query_embedding = generate_embedding(question)

    # 2. Retrieve the most relevant chunks.
    results = store.search(query_embedding, top_k=3)

    print(f"\n===== RETRIEVAL RESULTS =====")
    for rank, (document, score) in enumerate(results, start =1):
        print(f"Ranks : {rank} | Similarity: {score:.4f}")
        print(f"Document: {document}\n")

    # 3. Handle an empty retrieval result.
    if not results:
        return "I couldn't find relevant information in the documents."

    # 4. Build the context from retrieved chunks.
    context = build_context(results)

    if not context.strip():
        return "I couldn't find relevant information in the documents."

    # 5. Build the RAG prompt.
    prompt = build_rag_prompt(
        question=question,
        context=context,
    )

    # 6. Ask the LLM to generate an answer.
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.2,
        max_completion_tokens=500,
    )

    answer = response.choices[0].message.content

    if not answer:
        return "The model did not return an answer."

    return answer.strip()


def main():
    client = Groq()

    print("Building the knowledge base...")
    store = build_knowledge_base()
    print(f"Knowledge base ready: {len(store.documents)} chunks")

    while True:
        question = input("\nAsk a question (or type 'exit'): ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Bye, pookie! 💗")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            answer = answer_question(question, store, client)
            print(f"\nAnswer:\n{answer}")
        except Exception as error:
            print(f"Something went wrong: {error}")


if __name__ == "__main__":
    main()
