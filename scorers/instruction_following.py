# scorers/instruction_following.py

import os
from config import OPENAI_API_KEY, OPENAI_MODEL
import re
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model


load_dotenv()

# Get API key from .env file
api_key = os.environ.get("GOOGLE_API_KEY")


# Initialize the model
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

## considered formatting,numbers,response length,
def rule_based_instruction_score(prompt, response):
    """Enhanced rule-based check for instruction following."""
    score = 1.0

    prompt_lower = prompt.lower()
    response_lower = response.lower()


    if "bullet" in prompt_lower or "list" in prompt_lower:
        bullet_points = re.findall(r'•|\d+\.|[-*]', response)
        if not bullet_points:
            score -= 0.15


    if ("what is" in prompt_lower or "how many" in prompt_lower) and "?" in prompt_lower:
        numbers = re.findall(r'\d+(\.\d+)?', response) 
        if not numbers:
            score -= 0.15


    if any(keyword in prompt_lower for keyword in ["explain", "describe", "list", "define","summarize","conclude"]):
        if len(response.split()) < 10:
            score -= 0.35


    if re.search(r"\b(i don't know|no idea|n/a|cannot answer)\b", response_lower):
        score -= 0.35

    return round(score, 4)  

def ai_judge_instruction_score(prompt, response):
    """Use Google's native API for better reliability."""
    try:
        judge_prompt = f"""
        Evaluate the Instruction and answer based on
        1)instruction follow=How well the agent follows the instructions.
        2)assumptions=Whether the agent makes unwarranted assumptions beyond the given information.
        3)readability=How logically organized, structured, and readable the response is.
        Instruction: {prompt}
        Response: {response}
        
        Score from 0.0 to 1.0,
        score = (instruction follow)*0.5+(assumption)*0.25+(readability)*0.25
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