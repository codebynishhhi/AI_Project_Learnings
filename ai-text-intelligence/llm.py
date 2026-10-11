import json
from groq import Groq
from config import GROQ_API_KEY, MODEL_NAME
import logging

logger = logging.getLogger(__name__)

client = Groq(api_key=GROQ_API_KEY)

def ask_llm_structured(messages, response_schema, schema_name, error_message):
    try:

        response = client.chat.completions.create(
            messages=messages,
            model=MODEL_NAME,
            response_format={
                "type" : "json_schema",
                "json_schema" :{
                    "name" : schema_name,
                    "schema": response_schema.model_json_schema(),
                    "strict" : True
                }
            }
        )
        raw_response = response.choices[0].message.content
        data = json.loads(raw_response)
        validated_data = response_schema.model_validate(data)

    except Exception as e:
        logger.error("%s: %s", error_message, e)
        return None

    return validated_data
