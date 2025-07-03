from typing import List
from llm_models import get_gemini_response
from main import load_json_data, save_json_file
import json
import time

# --- Configuration (path)---
facit_data_json = "src/data/evaluation/facit.json"
new_data_json = "src/data/llm_output_data/test_data_gemini_1.5_flash.json"
output_path = "src/data/evaluation/test_data_gemini_1.5_flashv2.json"


# --- Functions ---
def evaluate(test_conv: dict, facit_conv: dict) -> str:
    query = f"""
        You are an expert data analyst. Your task is to carefully read and compare the answers from the new data set (response from LLM)
        with the answers in the Answer sheet. The correct answers are found in the 'Answer sheet'.
        **IMPORTANT** The llm response doesnt have to match 100 %, word for word with the Answer sheet, the semantic meaning is the most important.
        (e.g 'Order cancellation request, inquiry about order status/shipping, and discussion of refund process for delayed order.' and
        'Order cancellation inquiry, Order status inquiry, Refund process explanation' are very similar therefore both answers are correct.

        **Partial Matching for Slashed Answers:** If a field in the 'Answer sheet' contains multiple answers separated by a slash ('/'), 
        the LLM's response is considered CORRECT if it matches at least ONE of those answers.

        Correct Answer (Answer sheet/Grounded truth):
        {facit_conv}
        ---
        LLM response (output):
        {test_conv}
        ---

        Keep score and give one point for each correct answer. Write 'CORRECT' or 'WRONG' after each key-value pair as in the template below.
        Sum up the points and type it in the field "total_score", MUST BE an INT (INTEGER).
        Use the following template as example, Return ONLY valid JSON in the following format (no extra text or comments):
       
        Template:
        {{
            "conversation_id": "0",
            "conversation_id_status": "CORRECT",
            "products": ["Chair", "Lamp"],
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


def eval_main(facit_data_json: str, new_data_json: str) -> None:
    """
    Main evaluation function that compares LLM responses against facit/answer sheet.
    
    Loads reference (facit) and test data, evaluates each conversation pair using
    an LLM evaluator, and saves results incrementally. Handles JSON parsing errors
    gracefully by saving error information for debugging.
    
    Args:
        facit_data_json (str): Path to JSON file containing reference/ground truth data.
        new_data_json (str): Path to JSON file containing LLM-generated data.
    """
    facit_data = load_json_data(facit_data_json)
    new_data = load_json_data(new_data_json)

    all_extracted_conv_score = []

    # Comparing facit with the new data (new data is the output of the llm)
    for i, (facit_conv, test_conv) in enumerate(zip(facit_data, new_data)):
        llm_eval = evaluate(test_conv, facit_conv)
        print('Adding slight delay to manage the llm_quota (RPM)')
        time.sleep(2)

        llm_output = get_gemini_response(llm_eval)
        print(f"Evaluated conversation {i}: {llm_output}")

        try:
            # Cleaning the json output of the llm and parse it
            cleaned_json_response = llm_output.strip().replace("```json", "").replace("```", "").strip()
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

        # Save results for each iteration in case of unexpected error, so I can have a quick peek at what the data looks like
        if all_extracted_conv_score:
            save_json_file(all_extracted_conv_score, output_path)
            print(f"Saved evaluation results")
        else:
            print("No evaluation results to save.")


def sum_conversation_scores(output_path: str) -> None:
    """
    Calculates and prints the total score and accuracy from evaluation results.
    Loads evaluation data from a JSON file, sums up all conversation scores,
    and calculates the accuracy as a percentage of the maximum possible score (84).
    
    Args:
        output_path (str): Path to the JSON file containing evaluation results.
                          Each conversation should have a 'total_score' field.
    """
    final_score = 0
    evaluation_data = load_json_data(output_path)

    for conversation in evaluation_data:
        conv_score = conversation.get('total_score')
        final_score += conv_score
    
    max_possible_score = 84
    accuracy = final_score / max_possible_score
    print(f'Final score: {final_score}/84\nAccuracy: {round(accuracy, 4)}')
        

if __name__ == "__main__":
    eval_main(facit_data_json, new_data_json)
    sum_conversation_scores(output_path)

