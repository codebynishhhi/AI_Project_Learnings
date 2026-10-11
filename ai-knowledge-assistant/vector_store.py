# storage only using lists & numpy arrays.

import numpy as np
from embeddings import generate_embedding
from loader import load_pdf_file
from chunker import split_into_paragraphs, create_chunks
from embeddings import generate_embedding

class VectorStore :
    def __init__(self):
        self.documents = []
        self.embeddings = []

    def add(self, text: str, embedding : np.ndarray):
        # documents stores the original text chunks.
        # embeddings stores the corresponding vectors.
        # add() inserts one text/vector pair.
        self.documents.append(text)
        self.embeddings.append(embedding)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        
        results = []
        for document, embedding in zip(self.documents, self.embeddings):
            denominator = (
                np.linalg.norm(query_embedding)
                * np.linalg.norm(embedding)
            )

            if denominator == 0:
                similarity = 0.0
            else:
                similarity = float(
                    np.dot(query_embedding, embedding) / denominator
                )
            results.append((document, similarity))

        # Sort by similarity (descending) and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def add_text(self, text: str) -> None:
        embedding = generate_embedding(text)
        self.add(text, embedding)


# TIEING UP PDF WITH VECTOR STORE TO - 
# 1. chunk the PDF into smaller pieces
# 2. generate embeddings for each chunk
# 3. store the chunks and their embeddings in the vector store.

if __name__ == "__main__":
    store = VectorStore()
    pdf_path = "/Users/nishiigupta/Documents/AI_Learnings_Projects/ai-knowledge-assistant/documents/neural_networks_sample.pdf"

    # 1. load and clean the pdf 
    text = load_pdf_file(pdf_path)

    # 2. split the text into paragraphs
    paragraphs = split_into_paragraphs(text)

    # 3. create size aware chunks and its embeddings and store in vector store
    chunks = create_chunks(paragraphs, max_characters=500, overlap_characters=20)
    print(f"Total chunks created: {len(chunks)}")

    for chunk in chunks:
        store.add_text(chunk)

    print(f"Store Documents: {len(store.documents)}")
    print(f"Stored Embeddings: {len(store.embeddings)}")


    # 5. Search the document
    query = "What is the purpose of an activation function?"
    query_embedding = generate_embedding(query)

    # similarity search in the vector store
    results = store.search(query_embedding=query_embedding, top_k=3)

    print(f"nishi results =======>", results)

    print(f"\nQuery: {query}")
    print("\n===== SEARCH RESULTS =====")

    for document, score in results:
        print(f"\nSimilarity: {score:.4f}")
        print(f"Document: {document}")





# =============================== TESTING VECTOR STORE Comments =========================================

#     texts = [
#         "Large language models help developers write code.",
#         "LLMs assist programmers with software development.",
#         "The weather is sunny today."
#     ]
 
#     store.add_text("A neural network learns patterns from data.")
#     for text in texts:
#         embedding = generate_embedding(text)
#         store.add(text, embedding)
    
#     query = "How can AI help software developers?"
#     query_embedding = generate_embedding(query)
#     results = store.search(query_embedding=query_embedding, top_k=2)

#     print(f"\nQuery: {query}")
#     print("\n===== SEARCH RESULTS =====")

#     for document, score in results:
#         print(f"\nSimilarity: {score:.4f}")
#         print(f"Document: {document}")

#     print(f"Stored {len(store.documents)} documents.")
#     print(f"Stored embeddings: {len(store.embeddings)}")

#     print(f"Stored {len(store.documents)} documents.")
#     print(f"First document: {store.documents[0]}")
#     print(f"First embedding (first 5 values): {store.embeddings[0][:5]}")

#     What you're learning: 
#     the vector store doesn't generate embeddings itself. 
#     It receives an embedding from the embedding model and stores it alongside the original text.