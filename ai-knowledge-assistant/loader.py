
from pathlib import Path
import re

import fitz


def clean_text(text: str) -> str:
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        cleaned_lines.append(line.strip())

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


def load_text_file(file_path: str) -> str:
    path = Path(file_path)

    with path.open("r", encoding="utf-8") as file:
        text = file.read()

    return clean_text(text)


def load_pdf_file(file_path: str) -> str:
    pages = []

    with fitz.open(file_path) as document:
        for page in document:
            text = page.get_text()

            if text:
                pages.append(text)

    return clean_text("\n".join(pages))
