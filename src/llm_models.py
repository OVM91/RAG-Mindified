import os
from dotenv import load_dotenv
import google.generativeai as genai


def get_gemini_response(prompt: str):
    """
    Sends a prompt to the Gemini model and returns the response.
    """
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')

    response = model.generate_content(prompt)
    return response.text


def gemini_1_5_flash_8b_reponse(prompt: str):
    """
    Sends a prompt to the Gemini model and returns the response.
    """
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash-8b')
    #model_name = model.model_name[7::] # Further down when I want to adjust for dynamic filenames

    response = model.generate_content(prompt)
    return response.text
