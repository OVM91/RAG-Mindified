import os
import json
from dotenv import load_dotenv
import logging
from llm_models import gemini_2_5_flash_lite_preview_reponse

# --- Configuration ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

output_file = "src/data/synthetic_data.jsonl"


# --- Example (Based on provided data) ---

example_transcript = """customer: Customer connected from Live Chat
system: Please hold, we're connecting you to IKEA live chat
system: Your estimated wait time is more than 30 minutes
system: We are currently busier than usual at the moment and are experiencing extended wait times.
agent: Thank you for contacting IKEA Customer Support Center! My name is Bobba . How may I assist you?
customer: Hi Bobba, an item I bought arrived damaged. I am seeming a refund for the item.
agent: I am sorry to know that items in your order are damaged.I know this is not what you expected from the delivery
agent: I'd be glad to assist!
customer: Thank you!
agent: Welcome
agent: Before, I proceed could you please help me with your order number, phone number and which store you bought the items from ?
customer: yes one second
customer: phone number: 1243243512
customer: Order number: 471212321
customer: the store is the one in menlo, the broken item is the \tSOCKERBIT Storage box with Lid (15x30x11 ¾ \")
agent: Meanwhile I am checking. May i know how is weather at  San Francisco ?
customer: Fine, thank you. and your weather?
agent: Welcome . Here is summer going on
agent: How many quantity are damage ?
customer: Just 1
agent: I have successfully created your case, your case id is  71235234    . The refund process typically takes approximately 10-14 business days. Your refund must go through several departments prior to being issued. Please allow additional time to receive your refund. We apologize if there is a delay in receiving your refund.
customer: Thank you - please attach the following images to the ticket
agent: Welcome . Images are not required we have created case for the refund
customer: Thank you -- I would apprecaite them being attached anyways to support the speedy resolution of the ticket.
agent: Welcome
agent: I hope you are satisfied with the resolution provided, is there anything more I can assist you with?
customer: Not today, thank you for your help!
agent: After this conversation, you will receive a short survey pop-up regarding our chat interaction today. We would love to hear about your experience with us! Kindly take a moment to fill out the survey
agent: Welcome
agent: It was my pleasure chatting with you. Thank you for selecting IKEA. Please feel free to chat with us again. We are available from 08:00 AM to 12:00 AM EST. Have a lovely day ahead."""

example_json_output = {
    "conversation_id": "0",
    "products": ["SOCKERBIT Storage box with Lid"],
    "store_location": "Menlo",
    "product_category": "Storage",
    "service_rendered": "Refund for damaged item",
    "customer_satisfaction": "Positive",
    "case_or_order_number": "71235234 or 471212321"
}


# --- Prompt Engineering ---

def llm_prompt(transcript: str, json_output: dict) -> str:
    """Creates the prompt for the synthetic data generation of a LLM."""
    
    # Convert the JSON output to a formatted string
    json_string = json.dumps(json_output, indent=4)

    prompt = f"""You are an expert in creating synthetic data for training AI models.
    Your task is to generate a realistic customer service chat transcript and then extract key information from it into a structured JSON format.
    You will be given an example transcript and the corresponding JSON extraction.
    You must generate a NEW, entirely different transcript and its corresponding JSON extraction.
    The generated conversation should be plausible for a real customer service scenario (e.g., product questions, order issues, returns, delivery problems).
    The final output must be a single, valid JSON object with two keys: "transcript" and "structured_output".
    Do not repeat the example provided but it is important that you keep IKEA as the company.

    Here is an example of a transcript and its extracted JSON data:

    --- EXAMPLE TRANSCRIPT START ---
    {transcript}
    --- EXAMPLE TRANSCRIPT END ---

    --- EXAMPLE JSON OUTPUT START ---
    {json_string}
    --- EXAMPLE JSON OUTPUT END ---

    Now, please generate a new, completely different transcript and its corresponding JSON output.
    The JSON `structured_output` should strictly follow the schema of the example, but the values should be based on the new transcript you create.
    The `conversation_id` should be a unique identifier string.
    Please provide your response as a single JSON object with the keys "transcript" and "structured_output".
    """

    return prompt


def generate_synthetic_data() -> json:
    """Generates a single synthetic example of (transcript, structured_output) using LLM."""
    
    prompt = llm_prompt(example_transcript, example_json_output)
    
    try:
        llm_output = gemini_2_5_flash_lite_preview_reponse(prompt)
        
        cleaned_llm_output = llm_output.strip().replace("```json", "").replace("```", "").strip()
        json_llm_output = json.loads(cleaned_llm_output)
        
        # Basic validation
        if "transcript" in json_llm_output and "structured_output" in json_llm_output:
            logging.info("Successfully generated synthetic example")
            return json_llm_output
        else:
            logging.warning("Generated data is missing required keys ('transcript', 'structured_output').")
            return None
            
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON from the model's response: {e}")
        logging.error(f"Response was: {json_llm_output[:500]}...")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return None


def main(output_file: str):
    """Main function to generate a dataset of synthetic examples using LLM."""
    
    num_synthetic_data_generate = 3 
    
    logging.info(f"Starting synthetic data generation for {num_synthetic_data_generate} examples using LLM")
    
    successful_examples = []
    
    # Generate examples sequentially
    for i in range(num_synthetic_data_generate):
        logging.info(f"Generating example {i+1}/{num_synthetic_data_generate}...")
        example = generate_synthetic_data()

        if example:
            successful_examples.append(example)
        else:
            logging.warning(f"Failed to generate example {i+1}")
    
    logging.info(f"Successfully generated {len(successful_examples)} examples.")

    # Save the generated data to a JSONL file
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:

            for example in successful_examples:
                f.write(json.dumps(example) + "\n")

        logging.info(f"Data successfully saved to {output_file}")

    except IOError as e:
        logging.error(f"Failed to write to file {output_file}: {e}")

if __name__ == "__main__":
    main(output_file) 