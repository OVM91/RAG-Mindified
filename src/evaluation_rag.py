#import os 
#import nest_asyncio
#import asyncio
#import logging
#import sys
import pandas as pd
from typing import List
from llm_models import get_gemini_response
from main import load_json_data, save_json_file
import json


# Logging
"""nest_asyncio.apply()

# Set up the root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Set logger level to INFO

# Clear out any existing handlers
logger.handlers = []

# Set up the StreamHandler to output to sys.stdout (Colab's output)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)  # Set handler level to INFO

# Add the handler to the logger
logger.addHandler(handler)


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))"""



# --- Configuration (path)---
facit_data_json = "src/data/evaluation/facit.json"
new_data_json = "src/data/llm_output_data/test_data.json"
output_path = "src/data/evaluation/evaluation_results.json"


# --- Functions ---
def evaluate(test_conv: dict, facit_conv: dict) -> str:
    query = f"""
        You are an expert data analyst. Your task is to carefully read and compare the answers from the new data set (response from LLM)
        with the answers in the Answer sheet. The correct answers are found in the 'Answer sheet'.

        Correct Answer (Answer sheet):
        {facit_conv}
        ---
        Answer from the LLM:
        {test_conv}
        ---

        Keep score and give one point for each correct answer. Write 'CORRECT' or 'WRONG' after each key-value pair as in the template below.
        Sum up the points and type it in the field "total_score", MUST BE an INT (INTEGER).
        Use the following template as example, Return ONLY valid JSON in the following format (no extra text or comments):
       
        Template:
        {{
            "conversation_id": "0",
            "conversation_id_status": "CORRECT",
            "products": ["chair", "lamp"],
            "products_status": "WRONG", 
            "store_location": "Silicon Valley",
            "store_location_status": "WRONG",
            "product_category": "Home goods",
            "product_category_status": "CORRECT",
            "service_rendered": "Refunding",
            "service_rendered_status": "CORRECT",
            "customer_satisfaction": "Positive",
            "customer_satisfaction_status": "CORRECT",
            "case_or_order_number": "24156722",
            "case_or_order_number_status": "WRONG",
            "total_score": 4
        }}
    """
    return query

# Load the data
facit_data = load_json_data(facit_data_json)
new_data = load_json_data(new_data_json)

all_extracted_conv_score = []

for i, (facit_conv, test_conv) in enumerate(zip(facit_data, new_data)):
    llm_eval = evaluate(test_conv, facit_conv)
    gemini_output = get_gemini_response(llm_eval)
    print(f"Evaluated conversation {i}: {gemini_output}")

    cleaned_json_response = gemini_output.strip().replace("```json", "").replace("```", "").strip()
    
    try:
        # Parse the string into an actual JSON object
        json_response = json.loads(cleaned_json_response)
        all_extracted_conv_score.append(json_response)
        print(f"Successfully parsed JSON for conversation {i}")

    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON for conversation {i}: {e}")
        print(f"Raw response: {cleaned_json_response}")
        # Save the failed case as a string for debugging
        all_extracted_conv_score.append({
            "error": "JSON_PARSE_ERROR",
            "conversation_index": i,
            "raw_response": cleaned_json_response,
            "error_message": str(e)
        })

    # Save results for each iteration
    if all_extracted_conv_score:
        save_json_file(all_extracted_conv_score, output_path)
        print(f"Saved evaluation results")
    else:
        print("No evaluation results to save.")


def sum_conversation_scores(output_path: str):

    final_score = 0
    evaluation_data = load_json_data(output_path)

    for conversation in evaluation_data:
        conv_score = conversation.get('total_score')
        final_score += conv_score
    
    accuracy = final_score/84
    print(f'Final score: {final_score}/84\nAccuracy: {round(accuracy, 4)}')
        

sum_conversation_scores(output_path)


#evaluation_df = pd.DataFrame(data)
# Save as excel file
#with pd.ExcelWriter('eval report.xlsx', engine='xlsxwriter') as writer:
#    evaluation_df.to_excel(writer, sheet_name='Page_1', index=False)