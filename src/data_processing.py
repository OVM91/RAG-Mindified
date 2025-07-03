import json
from ordered_set import OrderedSet
from main import load_json_data

# --- Configuration (path)---
json_file_path = "src/data/raw_oscar_data.json"


# --- Functions ---
def parse_json_data(raw_json_data: str) -> json:
    """
    Processes raw conversation data by combining all the different messages of the same conversation_id (every msg also has its own metadata) 
    into one transcript and extracting the relevant metadata.
    
    Takes raw JSON conversation data and processes each conversation by:
    - Combining all messages into a single transcript string
    - Removing duplicate system messages to clean the data
    - Counting occurrences of specific wait messages for analytics
    - Extracting conversation metadata for further analysis
    
    Args:
        raw_json_data (str): Raw conversation data containing messages and metadata.
                            Expected to be a list of conversation dictionaries.
    
    Returns:
        json: List of processed conversations, each containing:
              - conversation_id: Unique identifier for the conversation
              - transcript: Combined text of all messages formatted as "user_type: message"
              - metadata: Dictionary with conversation details and duplicate message counts
    """
    processed_conversations = []
    for conv in raw_json_data:
        # Combine messages into a single transcript and remove system msg duplicates, sorry - this should be its own func for better readability
        transcript = ""
        seen_system_messages = OrderedSet()
        wait_message_duplicates_count = 0
        specific_wait_message = "We are currently busier than usual at the moment and are experiencing extended wait times."

        for msg in conv.get("messages", []):
            user_type = msg.get("user_type", "unknown").lower()
            text = msg.get("text_raw", "")

            # Counting duplicate system messages in regards to 'specific_wait_message', metadata
            if 'system' in user_type:
                if text in seen_system_messages:
                    if text == specific_wait_message:
                        wait_message_duplicates_count += 1
                    continue  # Skip duplicate system message
                seen_system_messages.add(text)

            if text:
                transcript += f"{user_type}: {text}\n"

        # Store the full transcript and top-level metadata
        if transcript:
            processed_conversations.append({
                "conversation_id": conv.get("conversation_id"),
                "transcript": transcript.strip(),
                "metadata": {
                    "conversation_id": conv.get("conversation_id"),
                    "country": conv.get("country"),
                    "channel": conv.get("channel"),
                    "start_time": conv.get("start_time"),
                    "end_time": conv.get("end_time"),
                    "published": conv.get("published"),
                    "translator": conv.get("translator"),
                    "wait_message_duplicates": wait_message_duplicates_count
                }
            })

    print(f"Successfully processed {len(processed_conversations)} conversations.")
    return processed_conversations
    

def main(file_path: str):
    """Main function to run the data processing, indexing and dump the json."""
    
    # --- Processing ---
    raw_json_data = load_json_data(json_file_path)
    conversations = parse_json_data(raw_json_data)
    
    if not conversations:
        print("No conversations to process. Exiting.")
        return
    
    # let's inspect the first processed conversation, making sure its OK
    if conversations:
        print("\n--- Test: first processed conversation ---")
        print(len(conversations))
        print(json.dumps(conversations[0], indent=2))

    output_json_path = "src/data/transformed_oscar_data.json"
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)
        

if __name__ == "__main__":
    main(json_file_path) 