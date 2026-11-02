# making sure our inspection function keeps working when the input is valid, missing, or incorrect.

from pathlib import Path

import pytest

from qbp_copilot.ingestion.inspect_pdf import inspect_pdf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF_PATH = (
    PROJECT_ROOT
    / "documents"
    / "NVIDIA_Q2_FY2027_Results.pdf"
)


def test_inspects_all_eight_pages():
    report = inspect_pdf(PDF_PATH)

    assert report["page_count"] == 8
    assert len(report["pages"]) == 8


def test_page_numbers_start_at_one():
    report = inspect_pdf(PDF_PATH)

    page_numbers = [
        page["page_number"]
        for page in report["pages"]
    ]

    assert page_numbers == list(range(1, 9))


def test_every_page_has_extracted_text():
    report = inspect_pdf(PDF_PATH)

    assert all(
        page["character_count"] > 0
        for page in report["pages"]
    )


def test_table_count_matches_extracted_tables():
    report = inspect_pdf(PDF_PATH)

    for page in report["pages"]:
        assert page["table_count"] == len(page["tables"])


def test_missing_pdf_raises_error(tmp_path):
    missing_path = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        inspect_pdf(missing_path)


def test_non_pdf_file_raises_error(tmp_path):
    text_file = tmp_path / "not_a_pdf.txt"
    text_file.write_text("This is not a PDF.")

    with pytest.raises(ValueError):
        inspect_pdf(text_file)
