
def score_answer(
    fact_results: list[bool],
    answerable: bool,
    abstained: bool = False,
) -> float:
    """Score fact correctness or abstention."""

    # If answerable isn't a Boolean, stop and report invalid input. For example:The string "yes" is not the Boolean True, so the function raises an error.
    if not isinstance(answerable, bool):
        raise ValueError("Invalid input - answerable must be a boolean")

    if not isinstance(abstained, bool):
        raise ValueError("abstained must be a boolean")

    if any(type(result) is not bool for result in fact_results):
        raise ValueError("fact_results must contain only booleans")

    # Case 1: Handle questions the document cannot answer - The source does not contain enough information.
    if not answerable:
        if fact_results:
            raise ValueError("Unanswerable questions must not have fact judgments")

        return 1.0 if abstained else 0.0

    # Case 2: The source can answer the question. - Reject contradictory inputs
    if abstained:
        raise ValueError("An answerable question cannot be marked as abstained")

    # Make sure there are facts to evaluate
    if not fact_results:
        raise ValueError("Answerable questions require at least one fact judgment")

    correct_facts = sum(fact_results)
    total_facts = len(fact_results)

    if correct_facts == total_facts:
        return 1.0

    if correct_facts == 0:
        return 0.0

    return 0.5
