
from dotenv import load_dotenv
from groq import Groq
import re
from loader import load_pdf_file
from qa_chunker import split_qa_entries
from vector_store import VectorStore
from context_builders import build_context
from rag_prompt import build_rag_prompt
from pathlib import Path
from embeddings import generate_embedding
load_dotenv()

MODEL_NAME = "openai/gpt-oss-20b"
DOCUMENT_PATH = "/Users/nishiigupta/Documents/AI_Learnings_Projects/ai-knowledge-assistant/documents/neural_networks_sample.pdf"
MIN_RELEVANCE_SCORE = 0.30

def build_knowledge_base() -> VectorStore:
    """Load the PDF and index its Q&A entries."""
    text = load_pdf_file(DOCUMENT_PATH)
    chunks = split_qa_entries(text)

    store = VectorStore()

    for index, chunk in enumerate(chunks, start = 1):
        question_match = re.match(r"Q(\d+):", chunk)
        chunk_id = question_match.group(1) if question_match else f"chunk_{index}"

    # dynamically identifuy the document name from path for source citation
    source_name = Path(DOCUMENT_PATH).name

    store.add_text(chunk, source=source_name, chunk_id=chunk_id)

    return store



import math
from numbers import Real


def answer_question(
    question: str,
    store: VectorStore,
    client: Groq,
) -> str:
    """Retrieve relevant context and generate a grounded answer."""

    # 1. Validate input
    if not isinstance(question, str) or not question.strip():
        return "Please enter a question."

    if len(question) > 200:
        return "Question is too long. Please keep it under 200 characters!"

    # 2. Generate the query embedding
    try:
        query_embedding = generate_embedding(question)
    except Exception:
        return "The retrieval service is temporarily unavailable."

    # 3. Retrieve relevant documents
    try:
        results = store.search(query_embedding, top_k=3)
    except Exception:
        return "The retrieval service is temporarily unavailable."

    if not results:
        return "I couldn't find relevant information in the documents."

    # Validate the retrieval result contract:
    # (document, similarity_score, metadata)
    if not isinstance(results, (list, tuple)):
        return "The retrieval service is temporarily unavailable."

    for result in results:
        if not isinstance(result, (list, tuple)) or len(result) != 3:
            return "The retrieval service is temporarily unavailable."

        document, score, metadata = result

        if not isinstance(document, str) or not isinstance(metadata, dict):
            return "The retrieval service is temporarily unavailable."

        if (
            isinstance(score, bool)
            or not isinstance(score, Real)
            or not math.isfinite(float(score))
        ):
            return "The retrieval service is temporarily unavailable."

    best_score = float(results[0][1])
    print(f"\nBest similarity score: {best_score:.4f}")

    if best_score < MIN_RELEVANCE_SCORE:
        return "I couldn't find relevant information in the documents."

    # Optional diagnostics: keep these for learning/debugging.
    print("\n===== RETRIEVAL RESULTS =====")
    for rank, (document, score, metadata) in enumerate(results, start=1):
        print(f"Rank {rank} | Score: {float(score):.4f}")
        print(f"Source: {metadata.get('source', 'unknown')}")
        print(f"Chunk ID: {metadata.get('chunk_id', 'unknown')}")
        print(document)

    # 4. Build the context
    try:
        context = build_context(results)
    except Exception:
        return "The retrieval service is temporarily unavailable."

    if not isinstance(context, str) or not context.strip():
        return "I couldn't find relevant information in the documents."

    # 5. Build the prompt
    prompt = build_rag_prompt(
        question=question,
        context=context,
    )

    # 6. Call the LLM
    try:
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
    except Exception:
        return "The AI service is temporarily unavailable."

    # 7. Safely extract the model's answer
    try:
        choices = response.choices

        if not choices:
            return "The AI service is temporarily unavailable."

        answer = choices[0].message.content

    except (AttributeError, IndexError, TypeError):
        return "The AI service is temporarily unavailable."

    # 8. Validate the answer
    if answer is None:
        return "The model did not return an answer."

    if not isinstance(answer, str):
        return "The AI service is temporarily unavailable."

    if not answer.strip():
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


