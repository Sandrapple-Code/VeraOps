from groq import Groq
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_groq_client(api_key: str) -> Groq:
    """
    Dynamically initializes and returns a Groq client using the provided API key.
    
    :param api_key: User-provided Groq API key.
    :return: An instance of the Groq client.
    :raises ValueError: If the API key is empty or invalid format.
    """
    if not api_key or not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("A valid, non-empty Groq API key must be provided.")
    return Groq(api_key=api_key.strip())

def generate_response(api_key: str, prompt: str) -> str:
    """
    Generates a response from the llama-3.3-70b-versatile model using the provided API key and prompt.
    
    :param api_key: User-provided Groq API key.
    :param prompt: Prompt text to send to the LLM.
    :return: The generated response text.
    :raises Exception: For API errors or network issues.
    """
    if not prompt or not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must be a non-empty string.")
        
    try:
        from services.config_service import get_setting
        client = get_groq_client(api_key)
        model = get_setting("model_selection", "llama-3.3-70b-versatile")
        temperature = get_setting("temperature", 0.1)
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt.strip()
                }
            ],
            model=model,
            temperature=temperature,
        )
        
        # Access response content safely
        if chat_completion.choices and len(chat_completion.choices) > 0:
            content = chat_completion.choices[0].message.content
            if content is not None:
                return content
            raise RuntimeError("Received null response content from Groq API.")
        else:
            raise RuntimeError("Received empty response choices from Groq API.")
            
    except Exception as e:
        logger.error(f"Failed to generate response from Groq API: {str(e)}")
        # Re-raise the exception to allow calling code to catch and display detailed errors
        raise e

if __name__ == "__main__":
    import os
    print("Testing Groq Client module...")
    
    # Retrieve the API key from the environment for testing if set
    api_key = os.environ.get("GROQ_API_KEY")
    
    if not api_key:
        print("\n[NOTE] Set the GROQ_API_KEY environment variable to test a live API call.")
        print("Example (PowerShell): $env:GROQ_API_KEY='your_api_key'")
        print("Example (Command Prompt): set GROQ_API_KEY=your_api_key")
        
        # Run local test with invalid API key to verify error handling
        print("\nTesting error handling with an invalid/mock API key:")
        try:
            generate_response("invalid_key_mock", "Hello")
        except Exception as e:
            print("\nSuccessfully caught expected API client exception:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
    else:
        print("\nGROQ_API_KEY found in environment. Testing live API call...")
        try:
            response = generate_response(api_key, "Say 'Hello, World!' in one sentence.")
            print("\nResponse from Llama 3 on Groq:")
            print(response)
        except Exception as e:
            print(f"\nAPI Call failed: {e}")
