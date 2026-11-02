# Quarterly Business Performance Copilot

**A retrieval-first AI system for understanding quarterly financial reports through document intelligence, semantic search, keyword retrieval, and hybrid ranking.**

| Project information | Details |
|---|---|
| Project | Quarterly Business Performance Copilot |
| Current milestone | Version 1 — Retrieval-first architecture |
| Language | Python 3.12 |
| Document processing | Docling |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Dense retrieval | Cosine similarity |
| Sparse retrieval | BM25 |
| Hybrid fusion | Reciprocal Rank Fusion (RRF) |
| Evaluation | Recall@K, MRR, Hit@1 |
| Testing | Pytest |
| Interface | Streamlit |

---

# 1. Project Overview

Quarterly financial reports contain large amounts of information distributed across financial statements, tables, management commentary, business performance metrics, and forward-looking guidance.

Extracting relevant information manually can be time-consuming, particularly when questions involve:

- Specific financial metrics.
- Comparisons across reporting periods.
- GAAP and non-GAAP financial measures.
- Revenue and operating expense analysis.
- Actual financial results versus future guidance.
- Evidence distributed across multiple sections or tables.

The **Quarterly Business Performance Copilot** aims to simplify financial document analysis through an intelligent document retrieval architecture.

The first version focuses on building a reliable retrieval foundation rather than immediately introducing an LLM.

The system processes financial documents, preserves their structural information, creates searchable chunks, generates embeddings, and retrieves relevant evidence using dense, sparse, and hybrid retrieval strategies.

**Version 1 objective:** Build and evaluate a reliable evidence retrieval pipeline that can serve as the foundation for a grounded financial question-answering system.

# 2. Problem Statement

Traditional keyword-based document search struggles with questions expressed using different terminology.

For example:

- Document: "Revenue of $96.2 billion."
- Query: "What were the company's quarterly sales?"

Exact keyword matching may fail to recognize that sales and revenue refer to related concepts.

Semantic retrieval can address this problem, but it may not always prioritize exact financial terminology, numerical values, or specific financial statement labels.

This creates a need for a retrieval architecture that combines:

1. Semantic understanding through dense retrieval.
2. Exact terminology matching through sparse retrieval.
3. Rank-level combination through hybrid retrieval.

The system uses this combination to improve the ranking of relevant financial evidence.

# 3. Current System Capabilities

## Document intelligence
- PDF ingestion using Docling.
- Structured document extraction.
- JSON and Markdown artifact generation.
- Document reading-order traversal.
- Table extraction and preservation.
- Section-aware chunking.
- Source-page and document-reference metadata.

## Retrieval
- Dense vector retrieval.
- Sparse keyword retrieval using BM25.
- Hybrid retrieval.
- Reciprocal Rank Fusion.
- Top-K evidence retrieval.
- Retrieval score and rank inspection.

## Evaluation and quality
- Labelled retrieval evaluation dataset.
- Recall@K calculation.
- Mean Reciprocal Rank calculation.
- Hit@1 calculation.
- Per-question evaluation reports.
- Automated unit and integration testing.

## Interface
- Streamlit-based retrieval interface.
- Interactive financial question input.
- Retrieved evidence display.

The current interface retrieves evidence. LLM-based answer generation is not yet implemented.

# 4. System Architecture

## 4.1 High-level architecture

```text
                  FINANCIAL PDF
                        |
                        v
               DOCUMENT INGESTION
                    Docling
                        |
                        v
              STRUCTURED DOCUMENT
                 JSON / Markdown
                        |
                        v
              STRUCTURE-AWARE
                  CHUNKING
                        |
                        v
                DOCUMENT CHUNKS
             Text + Metadata + Pages
                        |
              +---------+---------+
              |                   |
              v                   v
       DENSE EMBEDDING         BM25 INDEX
        GENERATION            CONSTRUCTION
              |                   |
              v                   |
         VECTOR INDEX              |
              |                   |
              +---------+---------+
                        |
                        v
                 HYBRID RETRIEVAL
               Dense + Sparse Search
                        |
                        v
              RECIPROCAL RANK FUSION
                        |
                        v
                  TOP-K EVIDENCE
                        |
                        v
                  STREAMLIT UI
```

## 4.2 Document processing flow

The document processing pipeline converts an unstructured financial PDF into structured and searchable content.

```text
PDF
 |
 v
Docling Parsing
 |
 v
Structured Document Representation
 |
 v
Reading Order and Element Extraction
 |
 v
Structure-Aware Chunking
 |
 v
Chunk Metadata and Provenance
 |
 v
Embedding Generation
 |
 v
Persisted Vector and Metadata Artifacts
```

## 4.3 Query-time retrieval flow

```text
User Query
    |
    v
Query Processing
    |
    +------------------------+
    |                        |
    v                        v
Dense Retrieval         Sparse Retrieval
    |                        |
    v                        v
Cosine Similarity        BM25 Scores
    |                        |
    v                        v
Dense Ranking            Sparse Ranking
    |                        |
    +-----------+------------+
                |
                v
       Reciprocal Rank Fusion
                |
                v
          Final Ranking
                |
                v
         Top-K Evidence
                |
                v
         Streamlit Display
```

# 5. Technology Stack and Design Decisions

| Technology | Responsibility | Why it was selected |
|---|---|---|
| Python | Application development | Ecosystem for document processing and AI |
| Docling | Document understanding | Structured parsing and table extraction |
| PyMuPDF | PDF inspection | Useful for debugging and PDF inspection |
| NumPy | Vector operations | Efficient numerical computation |
| Sentence Transformers | Embedding generation | Semantic representation of text |
| all-MiniLM-L6-v2 | Embedding model | Compact 384-dimensional embeddings |
| rank-bm25 | Sparse retrieval | Lightweight lexical retrieval |
| Streamlit | User interface | Rapid interactive prototype |
| Pytest | Automated testing | Repeatable validation of application behaviour |
| JSON | Metadata and reports | Human-readable persisted artifacts |
| NumPy NPY | Embedding storage | Efficient local vector persistence |

## Important design decisions and trade-offs

### 5.1 Docling instead of PDF-specific parsing logic

**Decision:** Use Docling as the document-understanding engine.

Why:

Financial documents contain different layouts, reading orders, tables, and sections. Implementing custom extraction logic for each PDF would make the system difficult to generalize.

Docling provides document parsing and structured representations.

Our application focuses on orchestration, chunking, metadata management, indexing, retrieval, and evaluation.

**Trade-off:**

- Advantage: Less custom layout and table extraction code.
- Limitation: Extraction quality still depends on the source document and parser output.
- Engineering responsibility: Validate extracted tables and source provenance instead of assuming parsing is always correct.

### 5.2 Structure-aware chunking instead of naive fixed-size chunking

**Decision:** Preserve document structure when creating chunks.

Financial information frequently depends on section headings, reporting periods, table headers, and explanatory context.

Splitting content at arbitrary character positions can separate a financial value from the heading or table context needed to interpret it.

The chunker therefore:

- Traverses document elements in reading order.
- Tracks heading hierarchy.
- Preserves table chunks.
- Includes relevant section context.
- Maintains source-page metadata.
- Splits oversized text at meaningful boundaries.

**Trade-off:**

- Advantage: Better preservation of document context and financial tables.
- Limitation: Large tables can create oversized chunks.
- Future improvement: Evaluate table-aware chunking and alternative chunk sizes against retrieval quality.

### 5.3 Local NumPy vector storage instead of a vector database

**Decision:** Store embeddings in NumPy arrays for Version 1.

The current document contains only 25 chunks. A local vector matrix is sufficient for experimenting with similarity calculations and retrieval behaviour.

**Trade-off:**

- Advantage: Simple, transparent, and easy to debug.
- Limitation: No distributed indexing, persistence management, or scalable ANN search.
- Future improvement: Evaluate FAISS or a production vector database when document volume and latency requirements justify it.

### 5.4 Compact embedding model

**Decision:** Use `all-MiniLM-L6-v2`.

The model generates 384-dimensional embeddings and provides a practical starting point for semantic retrieval.

**Trade-off:**

- Advantage: Relatively lightweight and suitable for a local prototype.
- Limitation: May not capture all domain-specific financial relationships or numerical reasoning.
- Future improvement: Benchmark alternative embedding models using the same evaluation dataset.

### 5.5 Hybrid retrieval instead of dense-only retrieval

**Decision:** Combine dense semantic search and BM25 keyword search.

Dense retrieval captures semantic similarity, while BM25 captures lexical relevance.

This is especially useful for financial reports, where both conceptual questions and exact terminology matter.

**Trade-off:**

- Advantage: Combines complementary retrieval signals.
- Limitation: Introduces additional indexing, scoring, and ranking complexity.
- Evaluation requirement: Hybrid retrieval must demonstrate measurable improvements over dense-only and sparse-only alternatives.

### 5.6 Reciprocal Rank Fusion instead of raw score addition

**Decision:** Use Reciprocal Rank Fusion to combine retrieval rankings.

BM25 scores and cosine similarity scores operate on different numerical scales. Directly adding their raw values can cause one scoring system to dominate the result.

RRF combines rankings rather than raw scores.

Formula:

\[
RRF(d)=\sum_{r}\frac{1}{k+rank_r(d)}
\]

Where:

- \(d\) is a document chunk.
- \(rank_r(d)\) is its position in a retrieval ranking.
- \(k\) is the RRF constant, currently 60.

**Trade-off:**

- Advantage: Does not require raw score calibration.
- Limitation: Uses ranking positions and does not preserve the magnitude of the original relevance scores.
- Future improvement: Evaluate top-N candidate fusion, score normalization, and alternative fusion strategies.

# 6. Project Structure

```text
quarterly-business-performance-copilot/
│
├── documents/
│   └── NVIDIA_Q2_FY2027_Results.pdf
│
├── artifacts/
│   ├── parsed/
│   │   ├── NVIDIA_Q2_FY2027_Results.json
│   │   └── NVIDIA_Q2_FY2027_Results.md
│   │
│   ├── chunks/
│   │   └── NVIDIA_Q2_FY2027_Results_chunks.json
│   │
│   └── embeddings/
│       ├── chunk_embeddings.npy
│       └── chunk_metadata.json
│
├── evaluation/
│   ├── retrieval_eval.json
│   └── results/
│       ├── dense_baseline_v1.json
│       └── hybrid_v1.json
│
├── src/
│   └── qbp_copilot/
│       ├── ingestion/
│       │   ├── document_parser.py
│       │   ├── chunker.py
│       │   ├── embedder.py
│       │   └── inspect_pdf.py
│       │
│       ├── retrieval/
│       │   ├── retriever.py
│       │   └── hybrid_retriever.py
│       │
│       └── evaluation/
│           └── evaluator.py
│
├── tests/
│   └── unit/
│       ├── test_hybrid.py
│       └── test_hybrid_retriever.py
│
├── app.py
├── requirements.txt
└── README.md
```

# 7. Document Ingestion and Chunking

## 7.1 Document parsing

Docling converts the source PDF into a structured document representation.

Implementation:

```text
src/qbp_copilot/ingestion/document_parser.py
```

Command:

```bash
PYTHONPATH=src python src/qbp_copilot/ingestion/document_parser.py
```

Current document:

`NVIDIA_Q2_FY2027_Results.pdf`

Observed parsing output:

| Attribute | Result |
|---|---:|
| Conversion status | SUCCESS |
| Pages | 8 |
| Tables | 12 |
| Text elements | 86 |
| Markdown output | Generated |
| JSON output | Generated |

These figures validate that the parser produced structured artifacts. They do not, by themselves, establish perfect extraction accuracy.

## 7.2 Structure-aware chunking

Implementation:

```text
src/qbp_copilot/ingestion/chunker.py
```

Command:

```bash
PYTHONPATH=src python src/qbp_copilot/ingestion/chunker.py
```

Observed output:

| Attribute | Result |
|---|---:|
| Ordered elements | 87 |
| Total chunks | 25 |
| Text chunks | 13 |
| Table chunks | 12 |
| Empty chunks | 0 |
| Missing page metadata | 0 |
| Missing source references | 0 |
| Duplicate chunk IDs | 0 |

Oversized text chunks were not observed in the validation run.

Tables were preserved intact, including larger financial reconciliation tables.

## 7.3 Chunk metadata

Each chunk contains information used for retrieval and evidence traceability, including:

- Chunk ID.
- Text.
- Section.
- Source pages.
- Document references.

The metadata is stored separately from the embedding matrix so that retrieved vector positions can be mapped back to their original document content.

# 8. Embedding and Indexing

Implementation:

```text
src/qbp_copilot/ingestion/embedder.py
```

Command:

```bash
PYTHONPATH=src python src/qbp_copilot/ingestion/embedder.py
```

Model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Observed embedding output:

| Attribute | Result |
|---|---:|
| Embedded chunks | 25 |
| Embedding dimensions | 384 |
| Matrix shape | (25, 384) |
| Data type | float32 |
| Embedding normalization | Enabled |
| Embedding alignment with metadata | Validated |

The embeddings are normalized, enabling cosine similarity to be calculated using the dot product.

Persisted artifacts:

```text
artifacts/embeddings/
├── chunk_embeddings.npy
└── chunk_metadata.json
```

# 9. Retrieval Implementation

## 9.1 Dense retrieval

Implementation:

```text
src/qbp_copilot/retrieval/retriever.py
```

The dense retriever:

1. Accepts a user query.
2. Generates its embedding using the same embedding model.
3. Calculates similarity against document embeddings.
4. Sorts chunks by descending similarity.
5. Returns the top-K results with similarity scores.

Because embeddings are normalized:

\[
cosine\_similarity(q,d)=q\cdot d
\]

## 9.2 Sparse retrieval using BM25

Implementation:

```text
src/qbp_copilot/retrieval/hybrid_retriever.py
```

BM25 indexes the tokenized content of document chunks.

At query time:

1. The query is tokenized.
2. BM25 calculates relevance scores against the indexed chunks.
3. Chunks are ranked according to their BM25 scores.

This provides lexical matching alongside semantic retrieval.

## 9.3 Hybrid retrieval

The hybrid retriever combines:

- Dense ranking.
- Sparse ranking.
- Reciprocal Rank Fusion.

Current implementation:

```text
src/qbp_copilot/retrieval/hybrid_retriever.py
```

The final result includes:

- Chunk metadata.
- Dense similarity score.
- BM25 score.
- Dense rank.
- Sparse rank.
- RRF score.

If BM25 produces no positive scores for a query, its arbitrary zero-score ranking is excluded from fusion.

This avoids allowing a ranking with no keyword matches to influence the final results.

# 10. Retrieval Evaluation

A labelled evaluation dataset is maintained at:

```text
evaluation/retrieval_eval.json
```

The current dataset contains six questions covering:

1. Quarterly revenue.
2. Operating expenses.
3. Data Center revenue.
4. GAAP and non-GAAP gross margins.
5. Net income.
6. Revenue outlook.

Each question has manually assigned relevant chunk IDs.

## 10.1 Metrics

### Recall@K

Measures the proportion of labelled relevant chunks retrieved within the top K results.

\[
Recall@K=
\frac{|Retrieved@K\cap Relevant|}
{|Relevant|}
\]

### Mean Reciprocal Rank

Measures the ranking position of the first relevant result across all questions.

\[
MRR=\frac{1}{N}\sum_{i=1}^{N}\frac{1}{rank_i}
\]

If no relevant chunk is retrieved, its reciprocal rank is zero.

### Hit@1

Measures whether the first retrieved result is relevant.

\[
Hit@1=
\frac{\text{Questions with relevant first result}}
{\text{Total questions}}
\]

## 10.2 Benchmark results

Current evaluation:

| Metric | Dense baseline | Hybrid retrieval |
|---|---:|---:|
| Recall@3 | 91.67% | 91.67% |
| MRR | 80.56% | 100.00% |
| Hit@1 | 66.67% | 100.00% |

### Interpretation

Hybrid retrieval improved the ranking of relevant evidence.

- Recall@3 remained unchanged.
- MRR increased by 19.44 percentage points.
- Hit@1 increased by 33.33 percentage points.

The Q1 revenue result improved from rank 2 to rank 1.

The Q5 net income result improved from rank 3 to rank 1.

Q4 still retrieves only one of its two labelled relevant chunks, resulting in 50% recall for that question.

**Important:** These results are preliminary. Six questions are insufficient to establish production-level retrieval reliability.

## 10.3 Evaluation reports

The evaluation harness is implemented in:

```text
src/qbp_copilot/evaluation/evaluator.py
```

Run:

```bash
PYTHONPATH=src python -m qbp_copilot.evaluation.evaluator
```

The harness:

1. Loads evaluation questions.
2. Loads document embeddings and metadata.
3. Builds the BM25 index.
4. Runs hybrid retrieval.
5. Calculates Recall@K, MRR and Hit@1.
6. Saves per-question and aggregate results.

Output:

```text
evaluation/results/hybrid_v1.json
```

The dense baseline is retained separately:

```text
evaluation/results/dense_baseline_v1.json
```

This enables comparisons without overwriting previous experimental results.

# 11. Automated Testing

Testing is implemented using Pytest.

Directory:

```text
tests/unit/
```

The tests cover two complementary levels.

## 11.1 Unit tests

File:

```text
tests/unit/test_hybrid_retriever.py
```

Coverage includes:

- Tokenization.
- Lowercase conversion.
- Decimal number tokenization.
- Invalid tokenization inputs.
- BM25 index construction.
- Empty metadata validation.
- Empty chunk-text validation.
- Reciprocal Rank Fusion calculations.
- One-based rank handling.
- Invalid RRF parameter validation.
- Empty ranking handling.
- Hybrid retrieval output structure.
- Hybrid evidence ranking.
- Empty query validation.
- Embedding and metadata alignment.

A fake embedding model is used for unit tests.

This makes the tests deterministic and avoids loading the actual SentenceTransformer model for every test.

## 11.2 Integration testing

File:

```text
tests/unit/test_hybrid.py
```

The integration test exercises hybrid retrieval against the actual document embeddings and metadata.

Unlike the unit tests, this test depends on the generated embedding artifacts being available.

The project-root path is resolved independently of the test's directory so artifact paths do not accidentally resolve under `tests/`.

## 11.3 Test execution

Run the hybrid unit tests:

```bash
PYTHONPATH=src pytest -v tests/unit/test_hybrid_retriever.py
```

Run the complete suite:

```bash
PYTHONPATH=src pytest -v
```

Latest observed test result:

```text
23 passed, 5 warnings
```

The test suite passed. The warnings remain to be inspected separately; they should not be treated as failures.

# 12. Running the Application

Start the Streamlit application:

```bash
PYTHONPATH=src streamlit run app.py
```

The interface currently supports:

- Entering financial questions.
- Retrieving relevant evidence.
- Inspecting document chunks.
- Reviewing retrieval results.

The current application is retrieval-focused and does not yet use an LLM to generate final answers.

# 13. Reproducibility and Artifact Management

The application separates source documents, parsed artifacts, chunks, embeddings, and evaluation reports.

This separation makes the processing stages independently inspectable.

The main persisted artifacts are:

| Artifact | Purpose |
|---|---|
| Parsed JSON | Structured Docling output |
| Parsed Markdown | Human-readable document representation |
| Chunk JSON | Searchable text and source metadata |
| Embedding NPY | Dense vector matrix |
| Chunk metadata JSON | Maps vectors to source evidence |
| Evaluation dataset JSON | Questions and expected relevant chunks |
| Dense baseline JSON | Reference evaluation |
| Hybrid evaluation JSON | Hybrid retrieval evaluation |

If the source document or chunking strategy changes, embeddings must be regenerated to preserve alignment between chunks and vectors.

For a reproducible benchmark, the document, chunking configuration, embedding model, evaluation questions, and relevance labels should be kept consistent.

# 14. Current Limitations

Version 1 is a working retrieval MVP, not a fully production-deployed system.

Known limitations:

1. Evaluation currently uses only six questions.
2. Generalization across multiple financial report formats has not yet been validated.
3. Financial table extraction accuracy requires further testing.
4. BM25-only retrieval has not yet been independently benchmarked.
5. Alternative embedding models have not yet been compared.
6. The current vector index is local NumPy storage.
7. Production-scale indexing and distributed retrieval are not implemented.
8. The Streamlit application is a prototype.
9. LLM answer generation is not implemented.
10. Answer faithfulness and factual correctness have not yet been evaluated.
11. Production monitoring, deployment, and operational alerting are future work.

# 15. Version 2 Roadmap

Version 2 will extend the retrieval-first architecture into a grounded financial question-answering application.

Planned development:

| Stage | Capability |
|---|---|
| 1 | Context assembly from retrieved evidence |
| 2 | Prompt construction and grounding instructions |
| 3 | LLM integration |
| 4 | Structured answer generation |
| 5 | Evidence-linked citations |
| 6 | Financial period and unit awareness |
| 7 | Actuals versus forecast distinction |
| 8 | Numerical reasoning and validation |
| 9 | Answer faithfulness evaluation |
| 10 | Answer correctness and citation evaluation |

The retrieval system will remain a separate component.

The LLM will consume retrieved evidence rather than replacing the retrieval engine.

# 16. Engineering Principles

The project follows the following engineering principles:

- Use general-purpose document understanding rather than PDF-specific hardcoded extraction.
- Preserve document structure and provenance.
- Separate ingestion, chunking, embedding, retrieval, and evaluation.
- Evaluate retrieval independently from answer generation.
- Compare retrieval approaches using the same labelled benchmark.
- Use automated tests to protect core retrieval behaviour.
- Persist intermediate artifacts for debugging and reproducibility.
- Prefer measurable improvements over assumptions.
- Build incrementally, validate each stage, and preserve stable milestones.

---

## Project Milestone

**Quarterly Business Performance Copilot — Version 1**

Current milestone: Retrieval-first architecture implemented and automated test suite passing.

Next milestone: Version 2 — Grounded LLM answer generation with source citations.

Repository: [AI_Project_Learnings](https://github.com/codebynishhhi/AI_Project_Learnings)