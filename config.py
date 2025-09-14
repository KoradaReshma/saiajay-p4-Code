# config.py
import os

# OpenAI API Key for AI Judge
OPENAI_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyBtqYea_2-ou6LpnQy28s9c4b4bBCQj8tE")
OPENAI_MODEL = "gemini-2.5-flash"  # Use a cheaper model for prototyping

# Hugging Face Models for NLI
NLI_MODEL_NAME = "facebook/bart-large-mnli"

# Embedding Model for Semantic Similarity
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Paths
SYNTHETIC_DATA_PATH = "agent_responses.json"
RESULTS_PATH = "results/evaluation_results.csv"
REPORT_PATH = "results/agent_report.html"
# config.py
# For zero-shot classification
