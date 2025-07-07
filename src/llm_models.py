import os
from dotenv import load_dotenv
from google import genai


def get_gemini_response(prompt: str) -> str:
    """
    Sends a prompt to the Gemini model and returns the response.
    """
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=prompt)
    
    if response.text is None:
        return "Error: No response text generated"
    
    return response.text


def gemini_1_5_flash_8b_reponse(prompt: str) -> str:
    """
    Sends a prompt to the Gemini model and returns the response.
    """
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
    model='gemini-1.5-flash-8b',
    contents=prompt)
    
    if response.text is None:
        return "Error: No response text generated"
    
    return response.text


def gemini_2_5_flash_lite_preview_reponse(prompt:str ) -> str:
    """
    Sends a prompt to the Gemini model and returns the response.
    """
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
    model='gemini-2.5-flash-lite-preview-06-17',
    contents=prompt)
    
    if response.text is None:
        return "Error: No response text generated"
    
    return response.text


