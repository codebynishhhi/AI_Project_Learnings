import logging

from prompts import (
    build_summarization_prompt,
    analyze_sentiment_prompt,
    key_points_giver_prompt,
    action_item_generator_prompt,
)

from schemas import (
    SummaryResponse,
    SentimentResponse,
    KeyPointsResponse,
    ActionItemsResponse,
)

from llm import ask_llm_structured

# ================================== Logging ================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)

def summarize_text(text:str):
    messages = build_summarization_prompt(text)
    response = ask_llm_structured(
        messages=messages, 
        response_schema = SummaryResponse, 
        schema_name = "summarize_text_schema", 
        error_message="LLM request has failed to summarize the text!"
            )
    
    return response


def analyze_sentiment(user_input:str):
    messages = analyze_sentiment_prompt(user_input)
    response = ask_llm_structured(
        messages=messages, 
        response_schema=SentimentResponse, 
        schema_name="sentiment_analyzer_schema", 
        error_message="LLM request has failed to analyse the sometime please try again!"
        )

    return response

def key_points_response_giver(user_input:str):
    message = key_points_giver_prompt(user_input)
    response = ask_llm_structured(
        messages = message,
        response_schema=KeyPointsResponse, 
        schema_name="key_points_schema",
        error_message="LLM request has failed to analyse the key points!"
    )
    return response


def extract_action_items(user_input:str):
    message = action_item_generator_prompt(user_input)
    response = ask_llm_structured(
        messages=message,
        response_schema=ActionItemsResponse, 
        schema_name="extract_action_items_schema",
        error_message="LLM request has failed. Please try again to generate the action items. "
    )
    return response

# ================================== CLI ==================================
if __name__ == "__main__":

    print("\n==========================================")
    print("       AI TEXT INTELLIGENCE")
    print("==========================================")
    print("1. Summarize Text")
    print("2. Analyze Sentiment")
    print("3. Extract Key Points")
    print("4. Extract Action Items")
    print("5. Exit")
    print("==========================================")

    choice = input("Choose an option (1-5): ").strip()

    if choice == "5":
        logger.info("Application exited by user")
        print("Goodbye!")

    elif choice in {"1", "2", "3", "4"}:

        user_input = input("\nEnter the text:\n").strip()

        if not user_input:
            logger.warning("No user input provided")
            print("Please enter some text.")

        elif choice == "1":
            result = summarize_text(user_input)

            if result is not None:
                print("\nSummary:")
                print(result)

        elif choice == "2":
            result = analyze_sentiment(user_input)

            if result is not None:
                print("\nSentiment:")
                print(result)

        elif choice == "3":
            result = key_points_response_giver(user_input)

            if result is not None:
                print("\nKey Points:")
                print(result)

        elif choice == "4":
            result = extract_action_items(user_input)

            if result is not None:
                print("\nAction Items:")
                print(result)

    else:
        logger.warning("Invalid menu option selected: %s", choice)
        print("Invalid option. Please choose between 1 and 5.")



