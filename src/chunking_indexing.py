import json
from data_processing import load_json_data

# --- Configuration (path) ---
json_file_path = "src/data/transformed_oscar_data.json"


# --- Functions ---
def chunking_data(json_file_path: str):
    
    json_data = load_json_data(json_file_path)

    final_structured_chunks = []

    for msg in json_data:
        chunks_list = []
        customer_response_count = 0

        full_transcript = msg.get("transcript")
        metadata = msg.get("metadata")

        chunked_transcript_list = full_transcript.split('\n')
        
        for chunk in chunked_transcript_list:
            chunks_list.append(chunk)

            if 'customer:' in chunk:
                customer_response_count += 1

            if customer_response_count == 4:
                # 1) Combine the collected lines into a single text block
                final_chunk_text = "\n".join(chunks_list)
                
                # 2) Structure the final chunk with separate text and metadata
                final_structured_chunks.append({
                    "chunked_text": final_chunk_text,
                    "metadata": metadata
                    })
                
                # 3) Reset for the next chunk
                chunks_list = [] 
                customer_response_count = 0
                
        if chunks_list:
            final_chunk_text = "\n".join(chunks_list)
            final_structured_chunks.append({
                "chunked_text": final_chunk_text,
                "metadata": metadata
            })
    
    if final_structured_chunks:
        print(len(final_structured_chunks))
        print(json.dumps(final_structured_chunks, indent=2))

chunking_data(json_file_path)


#def embedding_data()