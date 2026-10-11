
import re


def split_into_paragraphs(text: str) -> list[str]:
    paragraphs = text.split("\n\n")

    return [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]


def split_into_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def split_large_paragraph(
    paragraph: str,
    max_characters: int
) -> list[str]:
    if max_characters <= 0:
        raise ValueError("max_characters must be positive")

    sentences = split_into_sentences(paragraph)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # Last resort: split a sentence that is too large.
        if len(sentence) > max_characters:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            for start in range(0, len(sentence), max_characters):
                chunks.append(
                    sentence[start:start + max_characters]
                )

        elif not current_chunk:
            current_chunk = sentence

        elif len(current_chunk) + 1 + len(sentence) <= max_characters:
            current_chunk += " " + sentence

        else:
            chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def get_overlap_sentences(
    sentences: list[str],
    overlap_characters: int
) -> list[str]:
    if overlap_characters <= 0:
        return []

    overlap = []
    current_length = 0

    for sentence in reversed(sentences):
        additional_length = len(sentence)

        if overlap:
            additional_length += 1  # Space separator

        if current_length + additional_length > overlap_characters:
            break

        overlap.insert(0, sentence)
        current_length += additional_length

    # Preserve at least the final sentence when possible.
    if not overlap and sentences:
        overlap = [sentences[-1]]

    return overlap


def create_chunks(
    paragraphs: list[str],
    max_characters: int = 100,
    overlap_characters: int = 20
) -> list[str]:
    if max_characters <= 0:
        raise ValueError("max_characters must be positive")

    if overlap_characters < 0:
        raise ValueError("overlap_characters cannot be negative")

    chunks = []
    current_sentences = []

    def save_current_chunk() -> None:
        if current_sentences:
            chunks.append(" ".join(current_sentences))

    for paragraph in paragraphs:
        sentences = split_into_sentences(paragraph)

        for sentence in sentences:
            # Handle sentences larger than the size limit.
            if len(sentence) > max_characters:
                save_current_chunk()
                current_sentences.clear()

                chunks.extend(
                    split_large_paragraph(
                        sentence,
                        max_characters
                    )
                )
                continue

            candidate = " ".join(current_sentences + [sentence])

            if len(candidate) <= max_characters:
                current_sentences.append(sentence)
                continue

            # Save the previous chunk before carrying overlap forward.
            save_current_chunk()

            overlap = get_overlap_sentences(
                current_sentences,
                overlap_characters
            )

            # Drop overlap sentences if necessary to fit the new sentence.
            while overlap and len(" ".join(overlap + [sentence])) > max_characters:
                overlap.pop(0)

            current_sentences = overlap + [sentence]

    save_current_chunk()

    return chunks


if __name__ == "__main__":
    text = """
    Artificial intelligence is transforming software engineering.

    Large language models can assist developers with coding and documentation.

    Production AI systems require evaluation, monitoring, and security controls.

    Reliable evaluation is important for production AI systems.
    """

    paragraphs = split_into_paragraphs(text)

    print("===== PARAGRAPHS =====")
    for index, paragraph in enumerate(paragraphs, start=1):
        print(f"\nParagraph {index}: {paragraph}")

    chunks = create_chunks(
        paragraphs,
        max_characters=150,
        overlap_characters=60
    )

    print("\n===== CHUNKS =====")
    for index, chunk in enumerate(chunks, start=1):
        print(f"\nChunk {index}:")
        print(chunk)
        print(f"Characters: {len(chunk)}")
