
# scorers/hallucination.py
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import numpy as np
import re
import gc

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load once
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli")
model.eval()

# Initialize models (lazy loading)
_embedding_model = None


qa_dataset = {
    "Explain the concept of quantum entanglement and its implications for quantum computing in simple terms.": 
    """ 
Linked Fate: Imagine two coins that are entangled. If you flip one and it lands heads, you instantly know the other coin must be tails, no matter how far apart they are. Quantum particles behave similarly: their quantum states are correlated, and measuring one instantly influences the state of the other. 
Shared System: Entangled particles cannot be described independently; their individual states only make sense as part of a single, unified quantum system. 
Not Faster-Than-Light Communication: While the correlation is instantaneous, entanglement doesn't allow for faster-than-light communication because the outcome of the measurement on the first particle is random. You can't control what state it collapses into, so you can't send a specific message. 
Implications for Quantum Computing:
Enhanced Computational Power: Entanglement allows quantum computers to perform calculations that are impossible for classical computers. By correlating the states of multiple qubits (quantum bits), a quantum computer can explore many possibilities simultaneously, leading to exponentially faster processing for certain problems. 
Complex Algorithms: Entanglement is a fundamental resource used in various quantum algorithms to create complex interactions between qubits. 
Quantum Error Correction: Quantum information is fragile and can be corrupted by noise, a process called decoherence. Entanglement enables quantum error correction, where information is encoded across multiple entangled qubits, allowing the system to detect and correct errors without destroying the encoded data. 
Quantum Teleportation: Entanglement is essential for quantum teleportation, a process that transfers a quantum state from one location to another, a feat not possible with classical technology. """,




   "Describe how transformer architecture works in large language models and why it's effective for natural language processing.": 
    """The Transformer architecture, a deep learning model introduced in the paper "Attention Is All You Need" by Vaswani et al. (2017), is fundamental to large language models (LLMs) due to its efficiency and ability to capture long-range dependencies in sequential data. 
How Transformer Architecture Works:
Encoder-Decoder Structure (Optional, but common): Transformers often employ an encoder-decoder architecture. The encoder processes the input sequence (e.g., a sentence), and the decoder generates the output sequence (e.g., a translation or a continuation of the text). Some LLMs, like GPT, are decoder-only.
Self-Attention Mechanism: This is the core innovation. Unlike recurrent neural networks (RNNs) that process words sequentially, self-attention allows the model to weigh the importance of all other words in the input sequence when processing each individual word. This is achieved by computing "query," "key," and "value" vectors for each word, enabling the model to identify relationships and dependencies between words regardless of their position. Multi-head attention further enhances this by performing multiple attention calculations in parallel, capturing diverse types of relationships.
Positional Encoding: Since self-attention lacks inherent sequential awareness, positional encodings are added to the input embeddings to provide information about the relative or absolute position of words in the sequence. This ensures the model understands the order of words.
Feed-Forward Networks: After the attention mechanism, each position in the sequence passes through an independent feed-forward neural network, which adds non-linearity and further transforms the representations.
Residual Connections and Layer Normalization: These techniques are used throughout the architecture to facilitate training deeper networks by allowing gradients to flow more easily and preventing vanishing/exploding gradients.

Parallel Processing: The self-attention mechanism allows Transformers to process entire sequences in parallel, unlike RNNs which process sequentially. This significantly speeds up training and inference, especially for long sequences.
Capturing Long-Range Dependencies: Self-attention effectively addresses the vanishing gradient problem faced by RNNs and LSTMs, enabling the model to establish connections between words that are far apart in a sentence, crucial for understanding complex linguistic structures and context.
Contextual Understanding: By attending to all other words in a sequence, Transformers can develop a rich contextual understanding of each word's meaning, leading to more accurate and nuanced interpretations and generations of text.
Scalability: The architecture's efficiency and ability to handle long sequences make it highly scalable for training on massive datasets, which is essential for developing powerful LLMs capable of diverse NLP tasks.""",




"What are the main ethical considerations in developing advanced AI systems, and how can we address them?": 
"""The Importance of Ethical AI Development
Ethical considerations in AI development are paramount due to the profound impact AI systems can have on individuals, communities, and societies as a whole. Unlike traditional software systems, AI possesses the ability to make autonomous decisions, often based on complex algorithms and vast amounts of data. Therefore, ensuring that AI operates in a manner consistent with ethical principles is crucial to prevent harm, promoting fairness, and uphold human rights.
User Experience: AI systems should be designed with the user in mind, providing intuitive interfaces and clear explanations for their actions. This enhances user trust and facilitates the effective use of AI technologies.
Ethical Use Cases
Beneficial Applications: AI should be developed for applications that benefit society, such as healthcare, education, and environmental sustainability. Developers should avoid creating AI systems for harmful purposes, such as autonomous weapons or invasive surveillance.
Social Impact: Assessing the social impact of AI technologies is crucial. Developers should consider how their AI systems affect different social groups and work towards minimizing negative consequences.
Regulatory Compliance
Legal Frameworks: AI development must comply with existing legal and regulatory frameworks. This includes data protection laws, non-discrimination laws, and industry-specific regulations. Staying informed about evolving legal requirements is essential for ethical AI development.
Standardization: Adopting industry standards and best practices for AI development promotes consistency and reliability. Standardization helps in setting benchmarks for ethical AI practices.
Continuous Monitoring and Improvement
Ongoing Evaluation: Ethical considerations in AI development are not one-time tasks. Continuous monitoring and evaluation of AI systems are necessary to identify and address ethical issues as they arise. This includes regular audits, user feedback, and performance assessments.
Adaptability: AI systems should be adaptable to changing ethical standards and societal expectations. Developers must be willing to update and improve their AI technologies to align with evolving ethical norms.
Principles of Ethical AI
To address these ethical challenges, various organizations and experts have proposed principles and guidelines for ethical AI development. Some key principles include:
Fairness: AI systems should be designed and deployed in a manner that promotes fairness and avoids discrimination against individuals or groups based on attributes such as race, gender, or socioeconomic status.
Transparency: AI systems should be transparent and explainable, enabling users to understand how decisions are made and allowing for scrutiny and accountability.
Privacy: AI developers should prioritize the protection of individuals' privacy rights and adhere to data protection regulations and best practices.
Accountability: Developers and organizations deploying AI systems should be accountable for the outcomes of their technologies, including mechanisms for addressing errors, biases, and unintended consequences.
Human-Centered Design: AI systems should be designed with the well-being and interests of humans in mind, ensuring that they enhance, rather than diminish, human autonomy, dignity, and welfare.""",



"Explain the difference between supervised, unsupervised, and reinforcement learning with practical examples.": 
   """Supervised Learning
Supervised learning is like learning with a teacher. The model is trained on a labeled dataset, meaning each input has a corresponding output. The key characteristics of supervised learning are:

Labeled Data: Training data has predefined labels.
Types of Problems: Used for classification task like spam detection and regression task like predicting house prices.
Algorithms: Linear Regression, Logistic Regression, SVM, Decision Trees, Neural Networks.
Unsupervised Learning
Unsupervised learning works with data that has no predefined labels. The model identifies patterns, clusters or associations independently. The key characteristics of unsupervised learning are:

Unlabeled Data: No predefined outputs.
Types of Problems: Used for Clusteringtask like customer segmentation and association task like market basket analysis.
Algorithms: K-Means, Hierarchical Clustering, PCA, Autoencoders.
Reinforcement Learning (RL)
Reinforcement learning involves an agent that interacts with an environment, learning through rewards and penalties to maximize long-term success. The key characteristics of reinforcement learning are:

Interaction-Based Learning: The agent learns by taking actions and receiving feedback.
No Labeled Data: Learns from trial and error.
Algorithms: Q-learning, SARSA, Deep Q-Networks (DQN).
Comparison Table: Supervised vs Unsupervised vs Reinforcement Learning
""",





"How does gradient descent optimization work in neural networks, and what are some common variants like Adam and RMSprop?": 
"""Gradient Descent is a fundamental optimization algorithm used in neural networks to minimize the loss function and find the optimal weights and biases. It works by iteratively adjusting these parameters in the direction opposite to the gradient of the loss function, which represents the direction of the steepest ascent.
How Gradient Descent Works:
Initialize Parameters:
Start with random initial values for the network's weights and biases.
Calculate Loss:
Compute the loss function's value based on the current parameters and the training data.
Compute Gradient:
Calculate the gradient of the loss function with respect to each parameter. The gradient indicates the direction and magnitude of the steepest increase in loss. 
Update Parameters:
Adjust the parameters by moving in the negative direction of the gradient, scaled by a learning rate. The update rule is:
Code
new_parameter = current_parameter - (learning_rate * gradient_of_loss)
Repeat: Steps 2-4 are repeated until the loss function converges to a minimum (or a local minimum in the case of non-convex functions).
Common Variants:
RMSprop (Root Mean Square Propagation):
This optimizer addresses the issue of varying gradient magnitudes across different parameters. It adapts the learning rate for each parameter by dividing the global learning rate by a moving average of the squared gradients. This helps to accelerate convergence for parameters with small gradients and slow down updates for parameters with large gradients, preventing oscillations.
Adam (Adaptive Moment Estimation):
Adam combines the benefits of RMSprop and Momentum. It calculates adaptive learning rates for each parameter based on estimates of both the first moment (mean) and the second moment (uncentered variance) of the gradients. This allows Adam to handle sparse gradients, accelerate convergence, and often achieve better performance than other optimizers in various deep learning tasks.""",



"Describe the CAP theorem in distributed systems and its relevance to modern database design.": 
"The CAP theorem states that a distributed system can achieve at most two of three guarantees: Consistency, Availability, and Partition Tolerance. Designers must prioritize based on application needs, e.g., banking systems favor consistency and partition tolerance, while social media platforms may favor availability and partition tolerance to handle network failures gracefully.",





"What are the key challenges in achieving artificial general intelligence (AGI) compared to narrow AI?": 
"""Key challenges in achieving AGI compared to narrow AI include replicating human-like cognition, common sense, and adaptability, which narrow AI lacks, requiring breakthroughs in understanding consciousness and abstract reasoning. Overcoming computational limitations, massive data requirements, and energy efficiency is crucial, as AGI needs to learn and operate flexibly across domains. Furthermore, ensuring AGI's alignment with human values, ethical considerations, and safety to prevent misuse and harmful outcomes poses significant technical and societal hurdles.  
Technical Challenges
Cognition and Reasoning:
Narrow AI specializes in specific tasks, but AGI must possess human-like understanding, common sense, and the ability to reason across diverse domains. 
Adaptability and Learning:
Humans can learn and apply new concepts flexibly, whereas AGI requires flexible, generalizable intelligence that current narrow AI systems, trained on vast datasets for specific tasks, cannot replicate effectively. 
Common Sense:
Machines lack the intuitive common sense humans use to understand context and make inferences, a fundamental capability needed for AGI to interact with the world meaningfully. 
Multi-Modal Sensory Perception:
AGI needs to integrate and understand sensory inputs (vision, hearing, touch) in real-time, a capability beyond the siloed sensory processing of narrow AI systems. 
Emotional and Social Intelligence:
Humans understand and respond to emotional cues and social dynamics, a complex aspect of intelligence that current AI systems, focused on logical patterns, have not developed. 
Computational and Data Challenges
Computational Power:
AGI development requires immense computational resources to process and learn from vast, diverse datasets, significantly exceeding the capabilities of current narrow AI systems. 
Data Requirements:
Beyond large datasets for specific tasks, AGI needs access to high-quality, diverse, and unstructured data to learn and generalize effectively, mimicking human experience. 
As AGI approaches and surpasses human-level intelligence, ensuring its safety and control becomes paramount to prevent potential risks and misuse. """,
 
 
 


"Explain how attention mechanisms improve sequence-to-sequence models and their applications beyond NLP.": 
"""Attention mechanisms significantly enhance sequence-to-sequence (Seq2Seq) models by addressing the limitations of fixed-size context vectors in traditional encoder-decoder architectures. Instead of relying on a single, compressed representation of the entire input sequence, attention allows the decoder to selectively focus on different parts of the input sequence during each step of generating the output. 
How Attention Improves Seq2Seq Models:
Handling Long Sequences:
Attention mitigates the "information bottleneck" of fixed-size context vectors, which struggle to retain information from long input sequences. By dynamically focusing on relevant input parts, attention allows the model to effectively process longer sequences without significant performance degradation.
Improved Contextual Understanding:
Attention enables the model to assign varying weights to different input elements, highlighting the most relevant information for generating the current output element. This leads to a richer and more accurate contextual understanding, particularly crucial in tasks like machine translation where word order and dependencies can differ across languages.
Enhanced Interpretability:
The attention weights provide a degree of interpretability by indicating which parts of the input sequence the model prioritizes when generating each output element. This allows for a better understanding of the model's decision-making process.
Addressing Alignment Issues:
In tasks like machine translation, attention helps align corresponding words or phrases between source and target languages, even when their positions or grammatical structures differ.
Applications Beyond NLP:
While attention mechanisms originated and are prominently used in Natural Language Processing (NLP), their benefits extend to various other domains:
Computer Vision:
Image Captioning: Attention allows models to focus on specific regions of an image when generating descriptive captions.
Visual Question Answering (VQA): Models can attend to relevant image regions and question words to answer visual queries.
Speech Recognition:
Attention can help models focus on specific parts of the audio input when transcribing speech, particularly in noisy environments or with varying speaker characteristics.
Time Series Forecasting:
Attention mechanisms enable models to identify and weigh relevant historical data points when making future predictions, improving accuracy in financial forecasting, weather prediction, and more.
.""",
 
    
       
"Describe the concept of transfer learning and how it accelerates model training in deep learning applications.": 
"""Transfer learning is a deep learning technique where a model pre-trained on a large, general dataset is reused as a foundation for a new, related task, accelerating training by providing a starting point with already learned features. Instead of training a model from scratch, which requires vast amounts of data and computational power, the pre-trained model's early layers (which learn generic features like edges or basic language structures) are kept, while the final layers are fine-tuned on a smaller, task-specific dataset. This process reduces training time, requires less data, and often leads to better performance and accuracy for the new task.  
How Transfer Learning Accelerates Training
Transfer learning speeds up model training in deep learning in several key ways:
Reduced Training Time:
Because the model already has a base level of knowledge, it doesn't need to start from random weights. It can begin its training on the new task with a significant head start, drastically cutting down the time required to reach high accuracy. 
Less Data Required:
Training a deep learning model from scratch requires a massive amount of data to learn complex features effectively. Transfer learning allows you to leverage the patterns learned from the large, pre-training dataset, so you need far less labeled data for your specific task to achieve good results. 
Better Initial Performance:
The pre-trained model provides a highly effective set of learned features, such as edge detectors in image recognition or grammatical patterns in natural language processing. By using these features as a starting point, the new model can often achieve higher performance and accuracy compared to a model trained from zero. 
Resource Efficiency:
By reducing the need for massive datasets and lengthy training periods, transfer learning significantly lowers the computational cost and resource expenditure (like GPU usage) associated with developing a deep learning model. 
How It Works
1. Start with a Pre-Trained Model:
Select a model that has been trained on a large, diverse dataset for a related task (e.g., an image recognition model trained on millions of general images). 
2. Adapt the Model:
Retain Generic Features: The initial layers of the pre-trained model have learned general features that are useful across many tasks. These layers are often "frozen," meaning their weights are kept as they are, or they are lightly fine-tuned. 
Replace/Retrain Task-Specific Layers: The later layers of the pre-trained model, which are specific to the original task, are replaced with new layers designed for your new task. These new layers, along with possibly the earlier ones, are then trained or "fine-tuned" on your smaller, task-specific dataset. 
3. Achieve Faster and Better Results:
The model quickly adapts to the new task because it doesn t have to relearn fundamental features from scratch, leading to a more efficient and accurate solution.""", 




"What are the fundamental differences between symbolic AI and connectionist approaches, and how do they complement each other?": 
"""Symbolic AI uses explicit, human-readable symbols and rules for reasoning, excelling in tasks requiring logic and explainability, while connectionist AI (neural networks) learns from vast data to recognize complex patterns, proving superior for tasks like image recognition and natural language processing. They complement each other by combining symbolic AI's structured reasoning with connectionism's adaptive pattern learning, leading to hybrid systems that can both understand explicit knowledge and learn from unstructured data, creating more robust and intelligent AI.  
Symbolic AI
Knowledge Representation: Uses symbols, rules, and logical statements to represent knowledge explicitly. 
Reasoning: Manipulates these symbols logically to derive conclusions. 
Strengths: Ideal for tasks requiring explicit knowledge, logical inference, and transparency, such as expert systems, theorem proving, and planning. 
Learning: Requires human experts to manually encode rules and knowledge. 
Connectionist AI
Knowledge Representation:
Distributed across a network of interconnected neurons, making it less interpretable. 
Reasoning:
Learns associations and patterns from data through parallel distributed processing. 
Strengths:
Excellent for pattern recognition, adapting to new data, and handling large datasets, as seen in image recognition, speech recognition, and some natural language processing tasks. 
Learning:
Automatically learns from examples and vast amounts of data. 
How They Complement Each Other
Hybrid Systems:
The integration of symbolic reasoning and connectionist learning creates hybrid systems that combine the strengths of both approaches. 
 """}

def extract_claims(text):
    sents = [s.strip() for s in re.split(r'[.!?]\s+', text) if s.strip()]
    return sents

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
probs=nli_probs("1+2=3","1+2=4")
print(probs)



def nli_hallucination_score(premise: str, response: str):

    contradiction_probs = []
    correctness_probs =[]
    sentences= extract_claims(response)
    # for claim in sentences:
    #      probs = nli_probs(premise, claim)
    #      contradiction_probs.append(probs["contradiction"])
    #      correctness_probs.append(probs["entailment"])
    #      print(probs)
        
    # choose an aggregation: max is conservative (flag if any claim contradicts)
    probs = nli_probs(premise, response)
    print(probs)
    # avg_contradiction = round(sum(correctness_probs)/len(correctness_probs),4)
    # print(max_contradiction)
    return probs["contradiction"],probs["entailment"]

# dummy_val=nli_probs(text1,text2)
# print(dummy_val)
def semantic_similarity_score(prompt, response):
    model = get_embedding_model()
    if model is None:
        return 0.5
    
    try:
        with torch.no_grad():
            embeddings = model.encode([prompt, response], convert_to_tensor=True)
            prompt_emb, response_emb = embeddings[0], embeddings[1]
            similarity = util.pytorch_cos_sim(prompt_emb, response_emb).item()

            # Free memory
            del embeddings, prompt_emb, response_emb
            gc.collect()
            torch.cuda.empty_cache()

            return 1 - similarity
        
    except Exception as e:
        print(f"Semantic similarity error: {e}")
        return 0.5

def score_hallucination(prompt, response):
    """Main hallucination scoring function."""
    try:
        # Get both scores
        nli_score,correctness_score = nli_hallucination_score(qa_dataset[prompt], response)
        semantic_score = semantic_similarity_score(qa_dataset[prompt], response)
        
        # Combine scores (weighted average)
        final_score = (correctness_score * 0.8) + (semantic_score * 0.2)
        print(final_score)
        return round(nli_score, 4),round(final_score,4)
        
    except Exception as e:
        print(f"Hallucination scoring failed: {e}")
        return 0.5,0.5



