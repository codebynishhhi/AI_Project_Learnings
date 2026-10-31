from qbp_copilot.ingestion.chunker import chunk_document

def test_chunks_ids_are_unique():
    document = {
        "name": "test_document",
        "body": {"children": []},
        "texts": [],
        "tables": [],
        "groups": [],
    }

    chunks = chunk_document(document)
    ids = [chunk["id"] for chunk in chunks ]
    assert len(ids) == len(set(ids))

def test_empty_document_returns_no_chunk():
    document ={
        "name" : "test_doc",
        "body": {"children":[]},
        "texts": [],
        "tables" :[],
        "groups": []
    }

    assert chunk_document(document) == []
    
def test_text_split_respects_character_limit():
    from qbp_copilot.ingestion.chunker import (
        split_text_at_boundaries,
    )

    text = (
        "Revenue increased substantially. "
        "Operating income also increased. "
        "Net income improved during the period."
    )

    chunks = split_text_at_boundaries(text, max_chars=45)

    assert chunks
    assert all(len(chunk) <= 45 for chunk in chunks)
    assert "Revenue increased substantially." in " ".join(chunks)
    assert "Net income improved during the period." in " ".join(chunks)
