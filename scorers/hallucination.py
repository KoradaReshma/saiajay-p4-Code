
# scorers/hallucination.py
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import re

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load once
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
model.eval()

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt', quiet=True)

# Initialize models (lazy loading)
_nli_pipeline = None
_embedding_model = None


qa_dataset = {
    "Explain the concept of quantum entanglement and its implications for quantum computing in simple terms.": 
    "Quantum entanglement is a phenomenon where the quantum states of two or more particles become interconnected, such that the state of one particle cannot be described independently of the others, even when separated by large distances. In quantum computing, entanglement enables quantum parallelism, allowing computers to perform multiple calculations simultaneously, vastly improving performance on complex tasks like cryptography, optimization, and simulation of quantum systems.",

    "Describe how transformer architecture works in large language models and why it's effective for natural language processing.": 
    "The transformer architecture uses self-attention mechanisms to weigh the importance of each word in a sentence relative to others, regardless of their position. This enables models to capture long-range dependencies efficiently and allows parallel processing during training. Transformers have revolutionized NLP by enabling models like GPT and BERT to excel in tasks such as machine translation, text summarization, and question answering.",

    "What are the main ethical considerations in developing advanced AI systems, and how can we address them?": 
    "Key ethical considerations include fairness and bias mitigation, transparency of AI decisions, accountability for automated outcomes, user data privacy, safety against adversarial attacks, and addressing the socio-economic impact of job displacement. These can be addressed by implementing ethical AI principles, rigorous model testing, continuous monitoring, transparent reporting, and robust governance frameworks.",

    "Explain the difference between supervised, unsupervised, and reinforcement learning with practical examples.": 
    "Supervised learning trains models on labeled data to perform classification or regression, e.g., predicting spam emails. Unsupervised learning works on unlabeled data to find patterns, like customer segmentation through clustering. Reinforcement learning trains agents to make sequential decisions based on rewards and penalties, such as teaching a robot to navigate a maze.",

    "How does gradient descent optimization work in neural networks, and what are some common variants like Adam and RMSprop?": 
    "Gradient descent iteratively adjusts model parameters by calculating gradients of the loss function and moving in the opposite direction to minimize error. Momentum accelerates convergence by considering past gradients. RMSprop adapts learning rates based on recent gradient magnitudes. Adam combines momentum and RMSprop, dynamically adjusting learning rates per parameter for efficient and stable optimization.",

    "Describe the CAP theorem in distributed systems and its relevance to modern database design.": 
    "The CAP theorem states that a distributed system can achieve at most two of three guarantees: Consistency, Availability, and Partition Tolerance. Designers must prioritize based on application needs, e.g., banking systems favor consistency and partition tolerance, while social media platforms may favor availability and partition tolerance to handle network failures gracefully.",

    "What are the key challenges in achieving artificial general intelligence (AGI) compared to narrow AI?": 
    "AGI requires solving challenges such as common sense reasoning, transfer learning across domains, transparent decision-making, aligning with human values, and ensuring safe autonomous operation. Unlike narrow AI focused on specific tasks, AGI demands broad, flexible intelligence capable of adapting to diverse and unforeseen problems.",

    "Explain how attention mechanisms improve sequence-to-sequence models and their applications beyond NLP.": 
    "Attention mechanisms allow sequence models to focus selectively on important parts of the input when generating output, improving performance on tasks like machine translation by handling long-range dependencies. Beyond NLP, attention is used in image captioning, visual question answering, and protein structure prediction, enabling models to focus on relevant input features dynamically.",

    "Describe the concept of transfer learning and how it accelerates model training in deep learning applications.": 
    "Transfer learning involves taking a model pre-trained on a large dataset and fine-tuning it for a specific task. It allows models to reuse learned features, improving performance and reducing training time, especially when labeled data is limited. For example, using a pre-trained ResNet model for a specialized medical image classification task.",

    "What are the fundamental differences between symbolic AI and connectionist approaches, and how do they complement each other?": 
    "Symbolic AI uses explicit rules and logic for reasoning, excelling in interpretability and structured problem-solving. Connectionist approaches (neural networks) learn patterns from data, effectively handling unstructured data like images and audio. Combining them enables systems that can reason over structured knowledge while learning from large datasets, enhancing robustness and explainability."
}
def get_nli_pipeline():
    """Get or initialize the NLI pipeline."""
    global _nli_pipeline
    if _nli_pipeline is None:
        try:
            _nli_pipeline = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
                # device=-1  # Use CPU (-1), use 0 for GPU if available
            )
        except Exception as e:
            print(f"Failed to initialize NLI pipeline: {e}")
    return _nli_pipeline

def get_embedding_model():
    """Get or initialize the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"Failed to initialize embedding model: {e}")
    return _embedding_model



# For bart-large-mnli the logits order -> [contradiction, neutral, entailment]
def nli_probs(premise: str, hypothesis: str):
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits.squeeze(0)
        probs = F.softmax(logits, dim=-1).cpu().tolist()
    return {"contradiction": probs[0], "neutral": probs[1], "entailment": probs[2]}

def extract_claims(text):
    sents = [s.strip() for s in re.split(r'[.!?]\s+', text) if s.strip()]
    return sents

# -> shows high contradiction prob for this bad claim

def nli_hallucination_score(premise: str, response: str):
    # claims = extract_claims(response)
    # if not claims:
    #     return 0.0  # nothing to check

    contradiction_probs = []
  
    probs = nli_probs(premise, response)
    contradiction_probs.append(probs["contradiction"])

    # choose an aggregation: max is conservative (flag if any claim contradicts)
    max_contradiction = max(contradiction_probs)
    print(max_contradiction)
    return max_contradiction
        

def semantic_similarity_score(prompt, response):
    """Calculate semantic similarity score."""
    model = get_embedding_model()
    if model is None:
        return 0.5
    
    try:
        # Encode both texts
        prompt_embedding = model.encode(prompt, convert_to_tensor=True)
        response_embedding = model.encode(response, convert_to_tensor=True)
        
        # Calculate cosine similarity (0-1)
        similarity = util.pytorch_cos_sim(prompt_embedding, response_embedding).item()
        
        # Convert to dissimilarity score (0-1)
        return 1 - similarity
        
    except Exception as e:
        print(f"Semantic similarity error: {e}")
        return 0.5

def score_hallucination(prompt, response):
    """Main hallucination scoring function."""
    try:
        # Get both scores
        nli_score = nli_hallucination_score(qa_dataset[prompt], response)
        semantic_score = semantic_similarity_score(qa_dataset[prompt], response)
        
        # Combine scores (weighted average)
        final_score = (nli_score * 0.7) + (semantic_score * 0.3)
        return round(final_score, 2)
        
    except Exception as e:
        print(f"Hallucination scoring failed: {e}")
        return 0.5



