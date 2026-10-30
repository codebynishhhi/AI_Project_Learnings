
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# load the chunks from the chunks json file 
# chunk repr - {chunkid 1 :""} - so return is a dict
def load_chunks(chunk_file_path : str | Path) -> list[dict]:

    path = Path(chunk_file_path)
    # if the path has the file check
    if not path.is_file():
        raise FileNotFoundError("The chunk json file is not found!")

    # extract chunks uing json loads bcz the current chunk is in the form of json
    chunks = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(chunks, list):
        raise ValueError("Expected a JSON list of chunks") 
    
    # for each chunk check if they have valid id & text
    for chunk in chunks:
        if not chunk.get("chunk_id"):
            raise ValueError("A chunk is missing chunk_id")
        
        if not chunk.get("text","").strip():
            raise ValueError(
                f"Empty text in chunk {chunk.get('chunk_id')}"
            )
    return chunks


# once we have chunks loaded we create embeddings
def embed_chunks(chunks: list[dict],model_name: str = MODEL_NAME) -> np.ndarray:

    # check if we got the chunks
    if not chunks:
        raise ValueError("Cannot emed an empty chunk list")

    model = SentenceTransformer(model_name)

    texts = [chunk["text"] for chunk in chunks]

    # vector repr of the text
    vectors = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    vectors = np.asarray(vectors, dtype=np.float32)

    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise ValueError("Unexpected embedding matrix shape")

    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings contain invalid numbers")

    return vectors

def save_embeddings(
    chunks: list[dict],
    vectors: np.ndarray,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save vectors and their matching chunk metadata."""
    if len(chunks) != vectors.shape[0]:
        raise ValueError(
            "Each chunk must have exactly one embedding"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vectors_path = output_dir / "chunk_embeddings.npy"
    metadata_path = output_dir / "chunk_metadata.json"

    metadata = [
        {
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk.get("document_id"),
            "chunk_type": chunk.get("chunk_type"),
            "section": chunk.get("section"),
            "source_pages": chunk.get("source_pages", []),
            "element_refs": chunk.get("element_refs", []),
            "text": chunk["text"],
        }
        for chunk in chunks
    ]

    np.save(vectors_path, vectors)

    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "vectors_path": vectors_path,
        "metadata_path": metadata_path,
    }



if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]

    chunks_path = (
        project_root
        / "artifacts"
        / "chunks"
        / "NVIDIA_Q2_FY2027_Results_chunks.json"
    )

    output_dir = project_root / "artifacts" / "embeddings"

    chunks = load_chunks(chunks_path)
    vectors = embed_chunks(chunks)
    output_paths = save_embeddings(chunks, vectors, output_dir)

    print("Chunks embedded:", len(chunks))
    print("Embedding matrix shape:", vectors.shape)
    print("Vector dtype:", vectors.dtype)
    print("Vector norms:", np.linalg.norm(vectors, axis=1)[:5])
    print("Vectors saved to:", output_paths["vectors_path"])
    print("Metadata saved to:", output_paths["metadata_path"])
