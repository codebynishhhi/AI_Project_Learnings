def build_tutor_prompt(user_input):
    return [
        {
            "role":"system",
            "content":(
                "You are an AI engineering tutor. "
                "Explain concepts using simple mental models. "
                "Use practical examples when useful. "
                "Keep explanations beginner-friendly."
            )
        },
        {
            "role":"user",
            "content":user_input
        }
    ]

def build_summarization_prompt(text):
    return [
        {
            "role": "system",
            "content": (
                "You are a concise text summarization assistant. "
                "Summarize the provided text while preserving "
                "important facts and meaning. "
                "Do not invent information."
            )
        },
        {
            "role": "user",
            "content": text
        }
    ]

def analyze_sentiment_prompt(user_input):
    return [
        {
            "role": "system",
            "content": (
                "You are an expert sentiment analysis agent. "
                "Analyze the sentiment of the provided text. "
                "The sentiment must be one of: positive, negative, or neutral. "
                "Provide a confidence score based on the classification. "
                "Provide a concise, fact-based explanation."
            )
        },
        {
            "role": "user",
            "content": user_input
        }
    ]

def key_points_giver_prompt(user_input: str) -> list[dict[str, str]]:
    """Generates a structured system and user prompt layout for key point extraction."""
    return [
        {   
           "role": "system",
           "content": (
               "You are an expert information extraction agent.\n\n"
               "CRITICAL TASK:\n"
               "Analyze the provided text and extract its most critical core concepts, actions, or takeaways. "
               "Condense the text into a clean, structured set of key points.\n\n"
               "RULES FOR EXTRACTION:\n"
               "- Extract only factual, high-value information present in the text.\n"
               "- Omit conversational filler, redundant examples, or introductory fluff.\n"
               "- Express each key point as a punchy, single-sentence fragment or short statement.\n"
               "- Ensure each point can stand alone with clear context.\n\n"
               "OUTPUT FORMAT:\n"
               "You must populate the requested JSON schema accurately. Do not wrap the output in markdown code blocks "
               "(e.g., do not use ```json). Do not add any conversational pleasantries or commentary."
           ), 
        },
        {
            "role": "user",
            "content": f"<source_text>\n{user_input}\n</source_text>"
        }
    ]

def action_item_generator_prompt(user_input: str) -> list[dict[str, str]]:
    """Generates a structured system and user prompt layout for action item extraction."""
    return [
        {
            "role": "system",
            "content": (
                "You are an expert project management and operations assistant.\n\n"
                "CRITICAL TASK:\n"
                "Analyze the provided text (such as meeting notes, transcripts, or emails) and extract all actionable tasks, "
                "next steps, and deliverables. Transform raw discussions into a clear execution plan.\n\n"
                "RULES FOR EXTRACTION:\n"
                "- Extract concrete, verifiable actions. Do not include vague, open-ended discussions.\n"
                "- Identify the assignee or owner for each task if explicitly stated; use 'Unassigned' if no clear owner is found.\n"
                "- Identify any mentioned deadlines, target dates, or timeframes; use 'None Specified' if missing.\n"
                "- Phrase each action item starting with an imperative verb (e.g., 'Update client', 'Finalize report', 'Review metrics').\n\n"
                "OUTPUT FORMAT:\n"
                "You must populate the requested JSON schema accurately. Do not wrap the output in markdown code blocks "
                "(e.g., do not use ```json). Do not add any conversational pleasantries or commentary."
            )
        },
        {
            "role": "user",
            "content": f"<meeting_or_text_data>\n{user_input}\n</meeting_or_text_data>"
        }
    ]

