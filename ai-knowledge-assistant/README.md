
# AI Knowledge Assistant

A learning project to build a Retrieval-Augmented Generation (RAG)
assistant that answers questions using user-provided documents.

## Current capabilities

- Load text files
- Extract text from text-based PDFs using PyMuPDF
- Clean and normalize extracted text
- Split documents into paragraphs and sentences
- Create size-aware chunks with sentence-level overlap
- Generate local embeddings using sentence-transformers
- Compare embeddings using cosine similarity

## Tech stack

- Python
- PyMuPDF
- sentence-transformers
- all-MiniLM-L6-v2
- NumPy

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python chunker.py
python embeddings.py
```

## Roadmap

1. Document ingestion
2. Text cleaning and chunking
3. Embeddings
4. Vector storage
5. Similarity search
6. Retrieval evaluation
7. RAG answer generation
8. Source citations and insufficient-context handling
9. Intent routing
10. Guardrails, testing, and evaluation
