import json
#import chromadb
#from chromadb.utils import embedding_functions


def load_json_data(file_path: str):
    """
    Loads data from the JSON file and processes it into a list of conversations,
    each with its full transcript and associated metadata.
    """
    print("Loading data...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            return raw_data

    except FileNotFoundError:
        raise(f"Error: The file at {file_path} was not found.")
        
    except json.JSONDecodeError:
        raise(f"Error: The file at {file_path} is not a valid JSON file.")


def parse_json_data(raw_json_data: str) -> json:

    processed_conversations = []
    for conv in raw_json_data:
        # Combine messages into a single transcript
        transcript = ""
        for msg in conv.get("messages", []):
            user_type = msg.get("user_type", "unknown")
            text = msg.get("text_raw", "")
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
                    "translator": conv.get("translator")
                }
            })

    print(f"Successfully processed {len(processed_conversations)} conversations.")
    return processed_conversations

def main():
    """Main function to run the data processing and indexing."""
    
    # --- Configuration ---
    json_file_path = "src/data/oscar_data.json"
    
    
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
        print(json.dumps(conversations[0], indent=2))



if __name__ == "__main__":
    main() 