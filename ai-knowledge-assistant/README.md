# AI Knowledge Assistant

A hands-on learning project that implements a **Retrieval-Augmented Generation (RAG)** pipeline in Python. The assistant retrieves relevant information from a PDF knowledge base and uses an LLM to answer questions grounded in the retrieved context.

The goal of this project is to understand RAG from first principles: document ingestion, chunking, embeddings, vector search, context construction, prompt design, answer generation, failure handling, and evaluation.

## Features

- **PDF ingestion:** Extract text from PDFs using PyMuPDF.
- **Question-and-answer chunking:** Split the sample document into individual Q&A entries.
- **Local embeddings:** Generate 384-dimensional embeddings using Sentence Transformers and `all-MiniLM-L6-v2`.
- **Vector retrieval:** Store embeddings and retrieve relevant chunks using cosine similarity.
- **Context construction:** Build context from retrieved documents for the LLM.
- **Grounded answer generation:** Use Groq-hosted LLM inference to answer questions using retrieved context.
- **Insufficient-context handling:** Abstain when retrieved information does not adequately support an answer.
- **Input validation and error handling:** Handle invalid or excessive input, retrieval and embedding failures, and malformed or empty LLM responses.
- **Basic prompt-injection defenses:** Include prompt instructions and tests for selected injection scenarios.
- **Retrieval evaluation:** Evaluate retrieval on a small dataset covering baseline questions, paraphrases, conceptual queries, ambiguous questions, unanswerable questions, and adversarial cases.
- **Unit testing:** Use pytest to test retrieval metrics, answer-scoring logic, and evaluation-runner behavior.

## Architecture

```text
PDF Knowledge Base
       |
       v
   PDF Loader
       |
       v
  Q&A Chunker
       |
       v
 Embedding Model
       |
       v
   Vector Store
       |
       |                     User Question
       |                           |
       |                           v
       |                     Input Validation
       |                           |
       |                           v
       |                     Query Embedding
       |                           |
       +---------------------------+
                                   |
                                   v
                            Similarity Search
                                   |
                                   v
                            Relevance Threshold
                                   |
                                   v
                            Context Builder
                                   |
                                   v
                              RAG Prompt
                                   |
                                   v
                             Groq LLM API
                                   |
                                   v
                         Answer or Abstention
```

The knowledge base is indexed before queries are answered. At query time, the question is embedded, relevant chunks are retrieved, and the resulting context is passed to the LLM.

If retrieval does not meet the configured relevance threshold, the application can abstain rather than generate an unsupported answer. Passing that threshold does not guarantee that the retrieved context contains sufficient evidence, so answer-level evaluation remains important.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application logic |
| PyMuPDF | PDF text extraction |
| Sentence Transformers | Local embedding generation |
| `all-MiniLM-L6-v2` | Embedding model |
| NumPy | Numerical operations and similarity calculations |
| Groq API | LLM-based answer generation |
| pytest | Automated testing |

## Project Structure

```text
ai-knowledge-assistant/
├── documents/
│   └── neural_networks_sample.pdf
├── tests/
│   ├── test_rag.py
│   ├── test_retrieval_metrics.py
│   ├── test_answer_evaluator.py
│   └── test_evaluate_ans.py
├── loader.py
├── qa_chunker.py
├── embeddings.py
├── vector_store.py
├── context_builders.py
├── rag_prompt.py
├── rag_chain.py
├── evaluation_dataset.py
├── evaluate_retrieval.py
├── retrieval_metrics.py
├── answer_evaluation_dataset.py
├── answer_evaluator.py
├── evaluate_answers.py
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd ai-knowledge-assistant
```

### 2. Create and activate a virtual environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Groq API key

Set your API key as an environment variable. Never commit API keys to Git.

```bash
export GROQ_API_KEY="your-api-key"
```

The embedding model runs locally, while answer generation requires access to the configured Groq API and network connectivity.

## Running the Project

### Run the RAG application

```bash
python rag_chain.py
```

This builds the knowledge base from the configured PDF and runs the sample questions in the application’s main entry point.

### Run retrieval evaluation

```bash
python evaluate_retrieval.py
```

This evaluates the retrieval pipeline using the configured evaluation dataset and reports retrieval metrics and results by query category.

### Run the tests

```bash
pytest -q
```

## Retrieval Evaluation

The initial retrieval evaluation uses **20 test cases** across six categories.

| Metric | Baseline result |
|---|---:|
| Total evaluation queries | 20 |
| Answerable queries | 17 |
| Unanswerable queries | 3 |
| Successful retrievals for answerable queries | 17/17 |
| Hit Rate@3 | 100% |
| MRR@3 | 1.0000 |
| Unanswerable queries rejected by the relevance threshold | 2/3 |
| Unanswerable-query rejection rate | 66.67% |

The baseline achieved perfect Hit Rate@3 and MRR@3 on the answerable examples in this small dataset. However, one unanswerable query passed the relevance threshold because semantic similarity does not necessarily establish that the retrieved context contains the exact information requested.

These results are an initial learning baseline, not a statistically representative benchmark or evidence of production readiness.

## Answer-Quality Evaluation

The project also contains an answer-evaluation dataset, a scoring helper, and a runner with unit tests.

The scoring helper evaluates Boolean judgments of required facts and whether an unanswerable question was correctly rejected. It does not independently judge the generated answer text. The runner collects generated answers alongside reference answers and required facts.

**Status:** The evaluation components and runner have been unit-tested, but a complete automated answer-quality evaluation with fact-level judgments and aggregate answer-quality metrics is not yet implemented.

## Testing

The project has automated tests covering areas such as:

- RAG pipeline behavior and failure handling
- Retrieval rejection and metric validation
- Answer-scoring logic
- Answer-evaluation runner behavior

A previous full-suite run completed with **40 passed and 1 skipped**. A later focused run of the answer-evaluation runner tests completed with **3 passed**. Run the full test suite again before treating these as the final release results.

## Current Limitations

- The knowledge base and evaluation dataset are small.
- Retrieval evaluation results may not generalize to other documents or question types.
- Relevance thresholds are configurable heuristics, not guarantees of evidence sufficiency.
- Answer-quality scoring still requires fact-level judgments.
- The project does not yet implement a complete production evaluation and observability workflow.
- Hybrid search, reranking, deployment, and production-scale monitoring are outside the current scope.
- Source citations and intent routing have not been established as completed capabilities in this version.

## Learning Outcomes

Through this project, I practiced:

- Building a RAG pipeline end to end
- Understanding embeddings and semantic similarity
- Designing chunking and retrieval strategies
- Separating retrieval, context construction, prompting, and generation into modules
- Handling unsupported questions and common failure scenarios
- Writing unit tests for application behavior and evaluation logic
- Measuring retrieval quality and interpreting evaluation limitations

## Future Improvements

Potential next steps for a later version include:

1. Improve retrieval with hybrid search and reranking.
2. Add systematic answer-quality evaluation.
3. Experiment with chunk sizes, overlap, and relevance thresholds.
4. Add source-level citations and provenance.
5. Add tracing, latency measurement, and token-cost tracking.
6. Explore deployment and production monitoring.

---

**Project status:** Learning MVP implemented. Core RAG flow and baseline retrieval evaluation are in place; additional evaluation and production-hardening work remain future improvements.
