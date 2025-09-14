# scorers/instruction_following.py

import os
from config import OPENAI_API_KEY, OPENAI_MODEL
import re
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Load environment variables from .env file
load_dotenv()

# Get API key from .env file
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

# Initialize the model
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

# Test the model


def rule_based_instruction_score(prompt, response):
    """Simple rule-based check for instruction following."""
    score = 1.0
    
    # Check for bullet points if requested
    if "bullet" in prompt.lower():
        bullet_points = re.findall(r'•|\d+\.|[-*]', response)
        if not bullet_points:
            score -= 0.5
    
    # Check for numerical answers if it's a QA task
    if "what is" in prompt.lower() and "?" in prompt:
        numbers = re.findall(r'\d+', response)
        if not numbers:
            score -= 0.3
    
    return max(0, score)  # Ensure score doesn't go below 0
def ai_judge_instruction_score(prompt, response):
    """Use Google's native API for better reliability."""
    try:
        judge_prompt = f"""
        Evaluate the Instruction and answer based on
        1) correctness=How well the agent follows the instructions.
        2)assumptions=Whether the agent makes unwarranted assumptions beyond the given information.
        3)readability=How logically organized, structured, and readable the response is.
        Instruction: {prompt}
        Response: {response}
        
        Score from 0.0 to 1.0,
        score = (correctness)*0.5+(assumption)*0.25+(readability)*0.25
        Respond only with the number.
        """
        
        # Use Google's native API
       
        response =  model.invoke(f"{judge_prompt}")
        print(response)    
        score_text = response.content.strip()
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", score_text)
        
        if numbers:
            score = float(numbers[0])
            return max(0.0, min(1.0, score))
        else:
            print(f"Could not parse score from: {score_text}")
            return 0.5
            
    except Exception as e:
        print(f"Google API error: {e}")
        return 0.5

def score_instruction_following(prompt, response):
    """Hybrid scoring for instruction following."""
    rule_score = rule_based_instruction_score(prompt, response)
    ai_score = ai_judge_instruction_score(prompt, response)
    
    # Weight the scores (you can adjust these weights)
    final_score = (rule_score * 0.3) + (ai_score * 0.7)
    return round(final_score, 2)