
import re

from loader import load_pdf_file

# small helper that splits the extracted text into individual question-and-answer entries.
# To extract better chunks as per the sample pdf 

pdf_path = "/Users/nishiigupta/Documents/AI_Learnings_Projects/ai-knowledge-assistant/documents/neural_networks_sample.pdf"
text = load_pdf_file(pdf_path)

# Find question markers such as Q1:, Q2:, Q3:
matches = list(re.finditer(r"Q\d+:", text))

print(f"Questions found: {len(matches)}")

for index, match in enumerate(matches):
    start = match.start()

    if index + 1 < len(matches):
        end = matches[index + 1].start()
    else:
        end = len(text)

    entry = text[start:end].strip()

    print(f"\n===== ENTRY {index + 1} =====")
    print(entry)
