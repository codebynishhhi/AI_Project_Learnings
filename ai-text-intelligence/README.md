# AI Text Intelligence CLI

A lightweight AI-powered command-line application for analyzing text using structured LLM outputs.

The application provides four AI capabilities:

* **Text Summarization**
* **Sentiment Analysis**
* **Key Point Extraction**
* **Action Item Extraction**

The project was built to practice production-oriented AI engineering fundamentals including LLM integration, prompt design, structured outputs, Pydantic validation, reusable LLM abstractions, error handling, logging, testing, and mocking.

---

## Features

### 1. Text Summarization

Generates a concise summary while preserving the important information from the input text.

Returns:

* Summary
* Key points

### 2. Sentiment Analysis

Analyzes the overall sentiment of the provided text.

Returns:

* Sentiment
* Confidence score
* Explanation

Supported sentiment categories:

* Positive
* Negative
* Neutral

### 3. Key Point Extraction

Extracts the most important points from a given piece of text.

### 4. Action Item Extraction

Identifies concrete tasks and next steps from meeting notes, emails, or other operational text.

---

## Architecture

```text
                    ┌─────────────────────┐
                    │      CLI / User     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      app.py         │
                    │  AI Operations + CLI │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
        ┌─────────────────┐        ┌─────────────────┐
        │   prompts.py    │        │   schemas.py    │
        │ Prompt Builders  │        │ Pydantic Models │
        └────────┬────────┘        └────────┬────────┘
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
                    ┌─────────────────────┐
                    │       llm.py        │
                    │ Reusable LLM Client │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Groq API       │
                    │   GPT-OSS-20B       │
                    └─────────────────────┘
```

---

## Project Structure

```text
ai-text-intelligence/
│
├── app.py                  # CLI and AI operations
├── config.py               # Environment configuration
├── llm.py                  # Reusable structured LLM helper
├── prompts.py              # Prompt builders
├── schemas.py              # Pydantic response schemas
│
├── tests/
│   ├── __init__.py
│   ├── test_schemas.py     # Schema validation tests
│   └── test_llm.py         # LLM layer + mocking tests
│
├── .env                    # Local API credentials
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Tech Stack

* **Python 3.9+**
* **Groq API**
* **GPT-OSS-20B**
* **Pydantic**
* **python-dotenv**
* **pytest**
* **unittest.mock**
* **Python logging**

---

## Getting Started

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd ai-text-intelligence
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

Activate it:

#### macOS / Linux

```bash
source .venv/bin/activate
```

#### Windows

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Do not commit `.env` or expose your API key publicly.

### 5. Run the application

```bash
python app.py
```

---

## Usage

When the application starts, you'll see:

```text
==========================================
       AI TEXT INTELLIGENCE
==========================================
1. Summarize Text
2. Analyze Sentiment
3. Extract Key Points
4. Extract Action Items
5. Exit
==========================================
Choose an option (1-5):
```

Select the desired operation and provide your text.

---

## Example

### Input

```text
The backend team needs to migrate the payment service by Friday.
Priya should update the API documentation.
Rahul needs to investigate the database latency issue.
The team should review the migration status next Monday.
```

### Action Item Extraction

```text
Action Items:
action_items=[
    'Migrate the payment service by Friday.',
    'Update the API documentation.',
    'Investigate the database latency issue.',
    'Review the migration status next Monday.'
]
```

---

## Structured Outputs

The application uses Pydantic models to validate LLM responses.

For example:

```python
class SummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    key_points: list[str]
```

The LLM is instructed to return a JSON schema matching the expected structure.

The response then follows:

```text
LLM Response
     ↓
JSON
     ↓
Pydantic Validation
     ↓
Typed Python Object
```

This prevents the rest of the application from blindly trusting arbitrary LLM output.

---

## Error Handling

The application handles failures at the LLM layer and returns `None` when the request or response processing fails.

Errors include scenarios such as:

* API request failure
* Invalid JSON
* Invalid structured output
* Pydantic validation failure

The application also uses Python's built-in logging framework for operational visibility.

Example:

```text
INFO | __main__ | Starting action item extraction
INFO | __main__ | Action item extraction completed successfully
```

---

## Testing

The project uses `pytest` for testing.

Run all tests:

```bash
pytest
```

The test suite covers:

* Valid Pydantic schemas
* Invalid schema inputs
* Structured response validation
* Mocked LLM responses
* Invalid JSON responses
* Invalid schema responses
* Verification of the mocked API call

### Mocking

External Groq API calls are mocked during testing.

This keeps tests:

* Fast
* Deterministic
* Independent of API availability
* Free from unnecessary API usage

The testing flow looks like:

```text
Test
 ↓
Mock Groq API
 ↓
Fake LLM Response
 ↓
JSON Parsing
 ↓
Pydantic Validation
 ↓
Assertion
```

---

## Engineering Concepts Practiced

This project focuses on several core AI engineering concepts:

### LLM Integration

Working directly with an LLM API and understanding request/response handling.

### Prompt Engineering

Separating prompts from application logic and using system/user message roles.

### Structured Generation

Using JSON Schema to constrain LLM responses.

### Data Validation

Using Pydantic to validate and type LLM-generated data.

### Separation of Concerns

The project separates:

* Application logic
* Prompt construction
* Configuration
* LLM interaction
* Response schemas

### Error Handling

Handling external API and model-output failures gracefully.

### Observability

Using structured application logging instead of relying entirely on `print()` statements.

### Testing AI Applications

Using mocks to test LLM-dependent code without making real API calls.

---

## Design Decisions

### Why Groq?

Groq provides a simple OpenAI-compatible API interface and fast inference, making it suitable for a lightweight AI application.

### Why Pydantic?

LLM output is probabilistic, while application code benefits from deterministic data structures.

Pydantic provides a validation boundary between:

```text
Probabilistic LLM Output
          ↓
Deterministic Application Logic
```

### Why Structured Outputs?

Instead of relying on the model to return arbitrary text, the application requests a predefined JSON structure.

This makes downstream processing more reliable.

### Why Mock the LLM?

Tests should not depend on:

* Network availability
* API availability
* API rate limits
* Model behavior
* API credentials

Mocking provides deterministic test behavior.

---

## Future Improvements

Possible extensions for this project include:

* [ ] Add a FastAPI interface
* [ ] Add batch text processing
* [ ] Add configurable LLM models
* [ ] Add token and latency tracking
* [ ] Add retry logic with exponential backoff
* [ ] Add structured application error types
* [ ] Add more comprehensive test coverage
* [ ] Add evaluation datasets for LLM outputs
* [ ] Add Langfuse-based observability
* [ ] Add a web UI
* [ ] Add support for multiple LLM providers

These are intentionally left outside the scope of the first version.

---

## What I Learned

This project helped establish the fundamentals required for building production-oriented LLM applications:

```text
Python
  ↓
LLM API
  ↓
Prompt Engineering
  ↓
Structured Outputs
  ↓
Pydantic Validation
  ↓
Reusable LLM Layer
  ↓
Error Handling
  ↓
Logging
  ↓
Testing + Mocking
  ↓
CLI Application
```

The main objective was not to build a complex AI system, but to understand the fundamentals of taking an LLM call and turning it into a reliable software component.

---

## Version

**v1.0.0**

A lightweight AI text intelligence application demonstrating structured LLM integration and basic production engineering practices.


Things I implemented in this project & Concepts I learned are 

| Area                      | What I learned                     |
| ------------------------- | ------------------------------------ |
| Python project setup      | `.venv`, requirements, `.gitignore`  |
| Environment configuration | `.env` + `python-dotenv`             |
| LLM integration           | Groq API                             |
| Prompt engineering        | System + user prompts                |
| Separation of concerns    | `prompts.py`, `schemas.py`, `llm.py` |
| Structured output         | JSON Schema + Pydantic               |
| Validation                | `ValidationError`, `extra="forbid"`  |
| Reusable LLM layer        | `ask_llm_structured()`               |
| Error handling            | Exceptions + graceful `None`         |
| Testing                   | pytest                               |
| Mocking                   | `unittest.mock.patch`                |
| Logging                   | Python `logging`                     |
| CLI design                | Menu-driven application              |
| Python modules            | `if __name__ == "__main__"`          |
| Debugging                 | Import, pytest, API/schema errors    |