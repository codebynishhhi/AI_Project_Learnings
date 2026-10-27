# Quarterly Business Performance Copilot

A learning project to build a conversational assistant that answers questions about a company's quarterly performance using evidence from its published financial report.

The first document is the NVIDIA Q2 FY2027 financial results PDF in `documents/`. The project will be built incrementally, with an emphasis on clean module boundaries, document provenance, reliable numerical answers, and testable evaluation.

## Project status

**Step 1 — Repository scaffold and source document:** in progress.

No RAG pipeline has been implemented yet. The folder structure is a planned architecture, not a claim that these components already work.

## Planned architecture

```text
PDF report
   -> ingestion and extraction
   -> document schema and chunking
   -> embeddings and vector retrieval
   -> evidence/context assembly
   -> grounded answer generation
   -> answer with source pages and calculations
```

## Repository layout

```text
.
├── documents/                 # Source reports used by the project
├── evaluation/                # Evaluation questions and reference answers
├── notebooks/                 # Optional experiments; production logic stays in src/
├── src/qbp_copilot/
│   ├── ingestion/             # PDF extraction, table extraction, schema, chunking
│   ├── retrieval/             # Embeddings, index, similarity search
│   ├── analytics/             # Explicit financial calculations
│   ├── generation/            # Prompt construction and answer generation
│   └── evaluation/            # Metrics and evaluation runner
├── tests/
│   ├── unit/
│   └── integration/
├── artifacts/                 # Local generated outputs; ignored by Git
├── .env.example               # Names of required environment variables, no secrets
├── .gitignore
├── pyproject.toml
└── requirements.txt
```

## Initial setup

Python 3.11 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Add your own `GROQ_API_KEY` to `.env` only if/when the generation stage needs it. Never commit `.env` or API keys.

## Development principles

- Inspect extraction output before designing chunking rules.
- Preserve section, page, table headers, units, and reporting periods.
- Treat calculations as explicit, testable functions.
- Distinguish statements supported by the report from analytical hypotheses.
- Evaluate retrieval, numerical accuracy, citations, and abstention separately.
- Add dependencies only when a concrete project need justifies them.

## Source document

`documents/NVIDIA_Q2_FY2027_Results.pdf` is included as the working sample for this learning project. Keep the source document separate from generated extraction outputs and embeddings.

## License and document attribution

This repository is a learning project. Check the terms and attribution requirements of each dependency and the source document before redistributing or using the project commercially. PyMuPDF is offered under AGPL-3.0 or a commercial license; review its licensing terms for your intended use.
