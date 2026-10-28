
import json
from pathlib import Path

from docling.document_converter import DocumentConverter


def parse_document(source_path: str | Path):
    """Convert a document using Docling."""
    path = Path(source_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Document not found: {path}"
        )

    converter = DocumentConverter()
    result = converter.convert(str(path))

    if result.status.name != "SUCCESS":
        raise RuntimeError(
            f"Docling conversion failed: {result.status}"
        )

    return result


def save_parsed_document(
    result,
    output_dir: str | Path,
) -> dict:
    """Persist the structured document as JSON and Markdown."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    document = result.document
    stem = document.name

    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            document.export_to_dict(),
            file,
            ensure_ascii=False,
            indent=2,
        )

    markdown_path.write_text(
        document.export_to_markdown(),
        encoding="utf-8",
    )

    return {
        "json_path": json_path,
        "markdown_path": markdown_path,
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]

    source_path = (
        project_root
        / "documents"
        / "NVIDIA_Q2_FY2027_Results.pdf"
    )

    output_dir = project_root / "artifacts" / "parsed"

    result = parse_document(source_path)
    output_paths = save_parsed_document(result, output_dir)

    print(f"Status: {result.status}")
    print(f"Document: {result.document.name}")
    print(f"Pages: {len(result.document.pages)}")
    print(f"Tables: {len(result.document.tables)}")
    print(f"Text elements: {len(result.document.texts)}")
    print(f"JSON saved to: {output_paths['json_path']}")
    print(f"Markdown saved to: {output_paths['markdown_path']}")
