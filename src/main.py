import json
from typing import List, Optional
from pydantic import BaseModel, Field
from gemini_model import get_gemini_response


# --- Configuration (path)---
json_file_path = "src/data/llamaindex_embedding_data/transformed_oscar_data.json"
output_path = "src/data/test_data.json"
failed_validation_output_path = "src/data/failed_conversations.json"


# --- Functions ---
def load_json_data(file_path: str):
    """
    Loads data from the JSON file.
    """
    print("Loading data...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            return raw_data

    except FileNotFoundError:
        # Raise a proper Exception object, not a string
        raise Exception(f"Error: The file at {file_path} was not found.")
        
    except json.JSONDecodeError:
        # Raise a proper Exception object, not a string
        raise Exception(f"Error: The file at {file_path} is not a valid JSON file.")


class ConversationInfo(BaseModel):
    """Data model for extracted information from a conversation."""

    conversation_id: str = Field(..., description="The ID of the conversation.")
    products: List[str] = Field(..., description="List of all product names or product numbers mentioned in the conversation. If you are certain a metioned product is not a product of the company, e.g. 'a customers own mobile phone', 'customers own car', do not add it to this list")
    store_location: Optional[str] = Field(None, description="The specific store location or address mentioned, if any.")
    product_category: Optional[str] = Field(None, description="The general category of the product being discussed (e.g., 'storage', 'kitchen', 'lighting').")
    service_rendered: Optional[str] = Field(None, description="Any service that was provided or discussed (e.g., 'refund', 'delivery', 'assembly').")
    customer_satisfaction: str = Field(..., description="Overall customer satisfaction level. Must be one of: 'Positive', 'Negative', 'Neutral'.")
    case_or_order_number: Optional[str] = Field(None, description="Any mentioned order or case numbers.")
    


def prompt(transcript: str, metadata: dict) -> str:
    query = f"""
        You are an expert data analyst. Your task is to carefully read the following customer support conversation
        and extract specific pieces of information. 

        Conversation Transcript:
        ---
        {transcript}
        ---
        {metadata}
        ---

        Based on the conversation, provide the requested information in a valid JSON format,
        adhering to the following schema. Do not include any extra text or explanations outside of the JSON object.
        Never lie or make up information, always base your answers on the information from the documents.

        JSON Schema:
        {{
            "conversation_id": "conversation_id",
            "products": ["list of product names or numbers"] (If you are certain a metioned product is not a product of the company, e.g. 'a customers own mobile phone', 'customers own car', do not add it to this list).",
            "store_location": "store address or location",
            "product_category": "category of the product",
            "service_rendered": "service provided or discussed",
            "customer_satisfaction": "Positive, Negative, or Neutral",
            "case_or_order_number": "order or case number",
        }}
    """
    return query


def save_json_file(extracted_info: List[dict], output_path: str):
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(extracted_info, f, indent=4)
        print(f"Successfully saved extracted info to {output_path}")
    except IOError as e:
        print(f"Error writing to {output_path}: {e}")


def format_conversation(json_file_path: str) -> tuple[List[dict], List[dict]]:
    """Processes all conversations and extracts information from each.
    
    Returns:
        tuple: (successful_extractions, failed_extractions)
    """

    transformed_json_data = load_json_data(json_file_path)
    all_extracted_info = []
    failed_extractions = []

    for message in transformed_json_data:
        transcript = message.get("transcript")
        metadata = message.get("metadata")
        conversation_id = message.get("metadata", {}).get("conversation_id")

        try:
            response_text = get_gemini_response(prompt(transcript, metadata))
            # The response from the LLM should be a JSON string.
            # We clean it up and parse it.
            json_response = json.loads(response_text.strip().replace("```json", "").replace("```", "").strip())
            
            # Validate the data using the Pydantic model
            extracted_info = ConversationInfo(**json_response)
            all_extracted_info.append(extracted_info.dict())
            print(f"Successfully processed conversation: {extracted_info.conversation_id}")

        except (json.JSONDecodeError, TypeError) as e:
            error_info = {
                "conversation_id": conversation_id,
                "original_transcript": transcript,
                "original_metadata": metadata,
                "llm_response": response_text,
                "error_type": "JSON_DECODE_ERROR",
                "error_message": str(e)
            }
            failed_extractions.append(error_info)
            print(f"Could not parse LLM response for conversation {conversation_id}. Raw response was: {response_text}")
            continue
        except Exception as e:
            error_info = {
                "conversation_id": conversation_id,
                "original_transcript": transcript,
                "original_metadata": metadata,
                "llm_response": response_text if 'response_text' in locals() else "N/A",
                "error_type": "VALIDATION_ERROR",
                "error_message": str(e)
            }
            failed_extractions.append(error_info)
            print(f"An unexpected error occurred for conversation {i}: {e}")
            continue
    
    return all_extracted_info, failed_extractions


if __name__ == "__main__":
    # Process the conversations and extract information
    extracted_data, failed_data = format_conversation(json_file_path)
    
    # Save the extracted information to test.json
    if extracted_data:
        save_json_file(extracted_data, output_path)
        print(f"Extracted and saved information for {len(extracted_data)} conversations.")
    else:
        print("No data was extracted.")

    # Save failed conversations to failed_conversations.json
    if failed_data:
        save_json_file(failed_data, failed_validation_output_path)
        print(f"Saved {len(failed_data)} failed conversations to {failed_validation_output_path}")
    else:
        print("No failed conversations were found.")