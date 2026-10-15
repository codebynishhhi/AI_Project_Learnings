import re 


import re


def split_qa_entries(text: str) -> list[str]:
    # Locate the start of each question.
    matches = list(re.finditer(r"(?m)^Q\d+:", text))

    entries = []

    for index, match in enumerate(matches):
        start = match.start()

        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            end = len(text)

        entry = text[start:end].strip()

        if entry:
            entries.append(entry)

    return entries


if __name__ == "__main__":
    from loader import load_pdf_file

    text = load_pdf_file("/Users/nishiigupta/Documents/AI_Learnings_Projects/ai-knowledge-assistant/documents/neural_networks_sample.pdf")

    entries = split_qa_entries(text)

    print(f"Q&A entries found: {len(entries)}")

    for index, entry in enumerate(entries, start=1):
        print(f"\n===== Q&A CHUNK {index} =====")
        print(entry)
        print(f"Characters: {len(entry)}")
