
import json
import re  # NEW: sentence-boundary splitting
from pathlib import Path


HEADING_LABELS = {"section_header", "title"}
SUPPORTED_COLLECTIONS = {"texts", "tables", "groups"}

# Repeated page furniture is usually not useful for retrieval.
EXCLUDED_LABELS = {
    "page_footer",
    "page_header",
}


def load_document(json_path: str | Path) -> dict:
    """Load an existing Docling JSON artifact."""
    path = Path(json_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Parsed document not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        document = json.load(file)

    if not isinstance(document.get("body"), dict):
        raise ValueError(
            "Invalid Docling document: missing or invalid body"
        )

    return document


def resolve_reference(
    document: dict,
    reference: str,
) -> dict | None:
    """Resolve a Docling reference such as #/texts/3."""
    if not reference.startswith("#/"):
        return None

    parts = reference[2:].split("/")

    if len(parts) != 2:
        return None

    collection_name, index = parts

    if collection_name not in SUPPORTED_COLLECTIONS:
        return None

    collection = document.get(collection_name)

    if not isinstance(collection, list) or not index.isdigit():
        return None

    position = int(index)

    if position >= len(collection):
        return None

    element = collection[position]
    return element if isinstance(element, dict) else None


def get_page_numbers(element: dict) -> list[int]:
    """Return unique source page numbers for an element."""
    pages = {
        item["page_no"]
        for item in element.get("prov", [])
        if isinstance(item.get("page_no"), int)
    }

    return sorted(pages)


def render_table(element: dict) -> str:
    """
    Render a Docling table grid into row-oriented text.

    Preserve internal empty cells so values stay aligned
    with their original columns.
    """
    grid = element.get("data", {}).get("grid", [])

    if not grid:
        return ""

    rendered_rows = []

    for row in grid:
        cells = []

        for cell in row:
            if isinstance(cell, dict):
                value = cell.get("text", "")
            elif isinstance(cell, str):
                value = cell
            else:
                value = ""

            cells.append(str(value).strip())

        while cells and not cells[-1]:
            cells.pop()

        if cells:
            rendered_rows.append(" | ".join(cells))

    return "\n".join(rendered_rows)


def extract_ordered_elements(document: dict) -> list[dict]:
    """Traverse Docling references in document order."""
    ordered_elements = []
    visited = set()

    def visit(reference: str) -> None:
        if reference in visited:
            return

        element = resolve_reference(document, reference)

        if element is None:
            return

        visited.add(reference)

        if reference.startswith("#/groups/"):
            for child in element.get("children", []):
                child_ref = child.get("$ref")

                if child_ref:
                    visit(child_ref)

            return

        label = element.get("label", "text")

        if label in EXCLUDED_LABELS:
            return

        pages = get_page_numbers(element)

        if reference.startswith("#/texts/"):
            text = str(element.get("text", "")).strip()

            if not text:
                return

            ordered_elements.append({
                "type": "text",
                "text": text,
                "label": label,
                "level": element.get("level"),
                "pages": pages,
                "ref": reference,
            })

        elif reference.startswith("#/tables/"):
            table_text = render_table(element)

            if not table_text:
                return

            ordered_elements.append({
                "type": "table",
                "text": table_text,
                "label": "table",
                "pages": pages,
                "ref": reference,
            })

    for child in document["body"].get("children", []):
        reference = child.get("$ref")

        if reference:
            visit(reference)

    return ordered_elements


# NEW: Split oversized text at paragraph and sentence boundaries.
def split_text_at_boundaries(
    text: str,
    max_chars: int,
) -> list[str]:
    """
    Split text into pieces no longer than max_chars.

    Prefer paragraph boundaries, then sentence boundaries.
    Fall back to character splitting only when an individual
    sentence exceeds the configured limit.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be greater than zero")

    result = []
    current = ""

    def flush() -> None:
        nonlocal current

        if current:
            result.append(current)
            current = ""

    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if len(paragraph) <= max_chars:
            candidate = (
                f"{current}\n\n{paragraph}"
                if current
                else paragraph
            )

            if len(candidate) <= max_chars:
                current = candidate
            else:
                flush()
                current = paragraph

            continue

        # The paragraph is too long, so split it into sentences.
        flush()
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            if len(sentence) > max_chars:
                flush()

                # Last resort for a single oversized sentence.
                for start in range(0, len(sentence), max_chars):
                    result.append(
                        sentence[start:start + max_chars]
                    )

                continue

            candidate = (
                f"{current} {sentence}"
                if current
                else sentence
            )

            if len(candidate) <= max_chars:
                current = candidate
            else:
                flush()
                current = sentence

    flush()
    return result


def chunk_document(
    document: dict,
    max_chars: int = 1800,
) -> list[dict]:
    """
    Create structure-aware chunks.

    Text chunks are limited to max_chars after splitting.
    Tables remain independent to preserve their row structure.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be greater than zero")

    document_id = document.get("name", "unknown_document")
    elements = extract_ordered_elements(document)

    chunks = []
    text_buffer = []
    buffer_chars = 0
    buffer_pages = set()
    buffer_refs = []
    current_section = []

    def add_chunk(
        chunk_type: str,
        text: str,
        pages: list[int],
        refs: list[str],
        section: str | None,
    ) -> None:
        chunks.append({
            "chunk_id": (
                f"{document_id}_chunk_{len(chunks):04d}"
            ),
            "document_id": document_id,
            "chunk_type": chunk_type,
            "section": section,
            "text": text,
            "source_pages": sorted(set(pages)),
            "element_refs": refs,
        })

    def flush_text() -> None:
        nonlocal text_buffer, buffer_chars
        nonlocal buffer_pages, buffer_refs

        if not text_buffer:
            return

        add_chunk(
            chunk_type="text",
            text="\n\n".join(text_buffer),
            pages=list(buffer_pages),
            refs=list(buffer_refs),
            section=" > ".join(current_section) or None,
        )

        text_buffer = []
        buffer_chars = 0
        buffer_pages = set()
        buffer_refs = []

    def is_unit_note(text: str) -> bool:
        """Identify parenthesized financial-unit descriptions."""
        normalized = text.strip().lower()

        return (
            normalized.startswith("(")
            and normalized.endswith(")")
            and any(
                word in normalized
                for word in (
                    "million",
                    "billion",
                    "unaudited",
                    "except earnings per share",
                    "except per share data",
                )
            )
        )

    for index, element in enumerate(elements):
        label = element["label"]
        text = element["text"]

        if label in HEADING_LABELS:
            flush_text()

            level = max(1, element.get("level") or 1)
            current_section[:] = current_section[:level - 1]
            current_section.append(text)
            continue

        if element["type"] == "table":
            flush_text()
            context_lines = []

            # Preserve the unit note with its following table.
            if index > 0:
                previous = elements[index - 1]

                if (
                    previous["type"] == "text"
                    and is_unit_note(previous["text"])
                ):
                    context_lines.append(previous["text"])

            section = " > ".join(current_section) or None
            table_lines = []

            if section:
                table_lines.append(f"Section: {section}")

            if context_lines:
                table_lines.extend(context_lines)

            table_lines.append("Table data:")
            table_lines.append(text)

            add_chunk(
                chunk_type="table",
                text="\n".join(table_lines),
                pages=element["pages"],
                refs=[element["ref"]],
                section=section,
            )
            continue

        # Unit notes immediately before tables belong to the table.
        next_is_table = (
            index + 1 < len(elements)
            and elements[index + 1]["type"] == "table"
        )

        if is_unit_note(text) and next_is_table:
            continue

        separator = 2 if text_buffer else 0

        if (
            text_buffer
            and buffer_chars + len(text) + separator > max_chars
        ):
            flush_text()

        separator = 2 if text_buffer else 0
        text_buffer.append(text)
        buffer_chars += len(text) + separator
        buffer_pages.update(element["pages"])
        buffer_refs.append(element["ref"])

    flush_text()

    # CHANGED: Split oversized text chunks after initial chunking.
    # Tables are kept intact, even when they exceed max_chars.
    final_chunks = []

    for chunk in chunks:
        if (
            chunk["chunk_type"] != "text"
            or len(chunk["text"]) <= max_chars
        ):
            final_chunks.append(chunk)
            continue

        pieces = split_text_at_boundaries(
            chunk["text"],
            max_chars,
        )

        for piece in pieces:
            split_chunk = chunk.copy()
            split_chunk["text"] = piece
            final_chunks.append(split_chunk)

    # CHANGED: Assign unique sequential IDs after all splitting.
    for index, chunk in enumerate(final_chunks):
        chunk["chunk_id"] = (
            f"{document_id}_chunk_{index:04d}"
        )

    return final_chunks


def save_chunks(
    chunks: list[dict],
    output_path: str | Path,
) -> Path:
    """Persist the chunking result as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return path


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]

    parsed_path = (
        project_root
        / "artifacts"
        / "parsed"
        / "NVIDIA_Q2_FY2027_Results.json"
    )

    output_path = (
        project_root
        / "artifacts"
        / "chunks"
        / "NVIDIA_Q2_FY2027_Results_chunks.json"
    )

    document = load_document(parsed_path)
    elements = extract_ordered_elements(document)
    chunks = chunk_document(document, max_chars=1800)
    saved_path = save_chunks(chunks, output_path)

    print(f"Document: {document.get('name')}")
    print(f"Ordered elements: {len(elements)}")
    print(f"Chunks created: {len(chunks)}")
    print(
        "Text chunks:",
        sum(c["chunk_type"] == "text" for c in chunks),
    )
    print(
        "Table chunks:",
        sum(c["chunk_type"] == "table" for c in chunks),
    )
    print(f"Saved to: {saved_path}")
