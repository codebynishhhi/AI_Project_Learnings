
from pathlib import Path

import streamlit as st
from sentence_transformers import SentenceTransformer

from qbp_copilot.ingestion.embedder import MODEL_NAME
from qbp_copilot.retrieval.retriever import load_index, retrieve


PROJECT_ROOT = Path(__file__).resolve().parent
EMBEDDINGS_DIR = PROJECT_ROOT / "artifacts" / "embeddings"


st.set_page_config(
    page_title="Business Performance Copilot",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Business Performance Copilot")
st.caption(
    "Explore quarterly results through semantic retrieval. "
    "This version retrieves evidence; it does not generate answers."
)


@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)


@st.cache_resource
def load_retrieval_index():
    return load_index(EMBEDDINGS_DIR)


try:
    model = load_model()
    vectors, metadata = load_retrieval_index()
except Exception as exc:
    st.error(f"Could not load the retrieval system: {exc}")
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.subheader("Retrieval settings")
    top_k = st.slider(
        "Chunks to retrieve",
        min_value=1,
        max_value=min(10, len(metadata)),
        value=min(3, len(metadata)),
    )

    st.metric("Indexed chunks", len(metadata))
    st.metric("Embedding dimensions", vectors.shape[1])

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            for rank, result in enumerate(
                message.get("results", []), start=1
            ):
                with st.expander(
                    f"#{rank} · {result['chunk_id']} · "
                    f"Similarity {result['similarity_score']:.3f}"
                ):
                    st.write("**Section:**", result.get("section") or "Unknown")
                    st.write(
                        "**Source pages:**",
                        result.get("source_pages", []),
                    )
                    st.write(result["text"])


query = st.chat_input(
    "Ask about revenue, margins, cash flow, or other reported results..."
)

if query:
    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    try:
        results = retrieve(
            query=query,
            model=model,
            vectors=vectors,
            metadata=metadata,
            top_k=top_k,
        )

        if results:
            answer_text = (
                f"Retrieved {len(results)} potentially relevant "
                "evidence chunks. Review the source text below; "
                "these are not yet a generated answer."
            )
        else:
            answer_text = "No matching chunks were returned."

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer_text,
                "results": results,
            }
        )

        st.rerun()

    except Exception as exc:
        st.error(f"Retrieval failed: {exc}")


# The interface remembers the conversation within the current Streamlit session, but each question is independently searched against the document index. We are not implementing conversational memory or an LLM yet.

# ## Step 3 — Launch the app

# Activate your virtual environment if it isn't already active, then run this command from the project root:

# ```bash
# PYTHONPATH=src streamlit run app.py
# ```

# Streamlit should open a local browser page. If it doesn't, use the local URL printed in your terminal.

# Try these questions:

# - `What was the reported quarterly revenue?`
# - `What were the main operating expenses?`
# - `What does the cash flow statement report?`
# - `What were the GAAP and non-GAAP earnings per share?`

# For each question, inspect the returned text, similarity score, section, and source pages.

# ## Step 4 — Evaluate retrieval before adding an LLM

# For this first version, our goal isn't just to make the UI work. We want to establish whether retrieval is reliable.

# | Check | What we're looking for |
# |---|---|
# | Relevant evidence | Does the retrieved text contain the information needed to answer the question? |
# | Financial tables | Are the correct rows and figures retrieved together? |
# | Source traceability | Can we identify the originating page and chunk? |
# | Ranking quality | Does the most useful evidence appear near the top? |

# One important limitation: the current implementation uses **dense semantic retrieval only**. Exact financial figures, ticker symbols, and accounting labels can sometimes be better served by hybrid retrieval that combines semantic search with keyword matching. We should measure this before adding complexity.

# **Your next milestone:** get the UI running and inspect the results for the four questions. Then we'll build a small retrieval evaluation set, measure how often the right evidence appears in the top 3, and decide what needs improving before adding answer generation.