from pathlib import Path
import pymupdf

# function to inspect the pdf - 
# 1. find the if pdf path is valid 
# 2. extract text and table
# 3. return structured dictonary - each page is a dict.  Later, tests and reporting code can consume this output without needing to repeat the PDF extraction
def inspect_pdf(pdf_path : str | Path) -> dict:
    """ Inspect a pdf , extract its text and tables and return a structured dict """

    path = Path(pdf_path)

    # Step 1: Validate the input. 
    # - a) file exists on path or not. b) format is pdf, or the ones we need  
    if not path.is_file() :
        raise FileNotFoundError(f"PDF not found: {path}") 
    
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.name}")
    
    pages = []

    # Step 2: Open the document safely.
    with pymupdf.open(path) as doc:
        for page_number, page in enumerate(doc, start= 1):

            # step 3: extract text from each page 
            text = page.get_text("text")

            # step 4 : extract the tables from each page if any
            table_finder = page.find_tables()
            tables = table_finder.tables

            # store the preview of the table
            table_previews = []

            for table_index, table in enumerate(tables, start = 1):
                # extract rows from tables
                rows = table.extract()
                table_previews.append({
                    "table_number" : table_index,
                    "row_count" : len(rows),
                    # # Full extracted table
                    "rows":rows,
                    "column_count": max(
                (len(row) for row in rows),
                default=0,
            ),
                    "preview_rows" : rows[:5]
                })


            # Step 5: Record a structured summary
            summary = {
                "page_number" : page_number, 
                "text_preview":text,
                "character_count" : len(text),
                "table_count" : len(tables),
                "tables": table_previews
            }

            # append the summary to our pages(output) array
            pages.append(summary)

    return {
        "file_name":path.name,
        "page_count" : len(pages),
        "pages" : pages
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    pdf_path = (
        project_root
        / "documents"
        / "NVIDIA_Q2_FY2027_Results.pdf"
    )

    report = inspect_pdf(pdf_path)

    print(f"File: {report['file_name']}")
    print(f"Total pages: {report['page_count']}")

    for page in report["pages"]:
        print(f"\n--- Page {page['page_number']} ---")
        print(f"Characters: {page['character_count']}")
        print(f"Tables detected: {page['table_count']}")

        # Only print tables belonging to this page.
        for table in page["tables"]:
            print(
                f"\nTable {table['table_number']}: "
                f"{table['row_count']} rows, "
                f"{table['column_count']} columns"
            )

            for row in table["preview_rows"]:
                print(row)



        

    








