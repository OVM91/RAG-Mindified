import json
from ordered_set import OrderedSet

# --- Configuration (path)---
json_file_path = "src/data/raw_oscar_data.json"


# --- Functions ---
def load_json_data(file_path: str):
    """Loads data from the JSON file."""

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


def parse_json_data(raw_json_data: str) -> json:

    processed_conversations = []
    for conv in raw_json_data:
        # Combine messages into a single transcript and remove system msg duplicates
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
        
    # In the next steps, we will add chunking and indexing here.
    print("\n--- Data Processing Complete ---")
    
    # let's just inspect the first processed conversation
    if conversations:
        print("\n--- Test: first processed conversation ---")
        print(len(conversations))
        print(json.dumps(conversations[0], indent=2))

    output_json_path = "src/data/transformed_oscar_data.json"
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)
        

if __name__ == "__main__":
    main(json_file_path) 