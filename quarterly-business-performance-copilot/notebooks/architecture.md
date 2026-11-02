
# Quarterly Business Performance Copilot

## 1. Objective

Build a conversational assistant that answers quarterly business performance questions using a financial report, with traceable evidence and reliable financial calculations.

## 2. Initial source

- Document: NVIDIA Q2 FY2027 Results
- Format: PDF
- Pages: 8

## 3. Architecture principles

- Preserve source page references.
- Handle narrative text and financial tables appropriately.
- Use local embeddings for the initial retrieval system.
- Calculate financial metrics deterministically in Python.
- Ground generated answers in retrieved evidence.
- Abstain when the source does not support an answer.
- Evaluate retrieval and answer quality using a fixed test set.

## 4. Initial technology choices

- Python
- PyMuPDF
- Pydantic
- Sentence Transformers
- NumPy
- Groq API
- pytest
- Streamlit

## 5. Current implementation status

- [x] Initial repository scaffold committed and pushed
- [ ] PDF extraction inspected and validated
- [ ] Document schema defined
- [ ] Chunking and retrieval implemented
- [ ] Evidence-grounded generation implemented
- [ ] Financial analytics implemented
- [ ] Evaluation suite passed
- [ ] MVP v1.0 released

## 6. Current limitations

The initial scaffold does not yet contain a working RAG pipeline. Extraction quality and table handling must be validated before chunking and indexing are implemented.
