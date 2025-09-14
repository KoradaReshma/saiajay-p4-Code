import os
import asyncio
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict, Any
import json
import time

# Load environment variables
load_dotenv()

# Define 10 complex questions
COMPLEX_QUESTIONS = [
    "Explain the concept of quantum entanglement and its implications for quantum computing in simple terms.",
    "Describe how transformer architecture works in large language models and why it's effective for natural language processing.",
    "What are the main ethical considerations in developing advanced AI systems, and how can we address them?",
    "Explain the difference between supervised, unsupervised, and reinforcement learning with practical examples.",
    "How does gradient descent optimization work in neural networks, and what are some common variants like Adam and RMSprop?",
    "Describe the CAP theorem in distributed systems and its relevance to modern database design.",
    "What are the key challenges in achieving artificial general intelligence (AGI) compared to narrow AI?",
    "Explain how attention mechanisms improve sequence-to-sequence models and their applications beyond NLP.",
    "Describe the concept of transfer learning and how it accelerates model training in deep learning applications.",
    "What are the fundamental differences between symbolic AI and connectionist approaches, and how do they complement each other?"
]

# Define 5 different agents using only free Gemini models
AGENTS_CONFIG = [
    {
        "agent_id": "agent_gemini_1.5_flash",
        "model_name": "gemini-1.5-flash",
        "temperature": 0.1,
        "system_message": "You are a precise and technical AI assistant. Provide detailed, accurate explanations."
    },
    {
        "agent_id": "agent_gemini_2.0_flash",
        "model_name": "gemini-2.0-flash",
        "temperature": 0.3,
        "system_message": "You are a helpful AI assistant. Provide clear and comprehensive answers."
    },
    {
        "agent_id": "agent_gemini_2.5_flash",
        "model_name": "gemini-2.5-flash",
        "temperature": 0.2,
        "system_message": "You are an expert AI assistant. Provide insightful and well-structured responses."
    },
    {
        "agent_id": "agent_gemini_2.5_flash_lite",
        "model_name": "gemini-2.5-flash-lite",
        "temperature": 0.4,
        "system_message": "You are a concise AI assistant. Provide direct and efficient answers."
    },
    {
         "agent_id": "agent_gemini_2.0_flash_lite",
        "model_name": "gemini-2.0-flash-lite",
        "temperature": 0.5,
        "system_message": "You are a knowledgeable AI assistant. Provide thorough and detailed explanations."
    }
]

class AgentRunner:
    def __init__(self, agent_config: Dict[str, Any]):
        self.agent_id = agent_config["agent_id"]
        self.model_name = agent_config["model_name"]
        self.temperature = agent_config["temperature"]
        self.system_message = agent_config["system_message"]
        
        # Initialize the Gemini model
        try:
            self.model = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                google_api_key=os.environ.get("GOOGLE_API_KEY")
            )
        except Exception as e:
            print(f"Error initializing {self.agent_id}: {e}")
            self.model = None
    
    def run_query(self, question: str) -> Dict[str, Any]:
        """Run a single query through the agent."""
        if not self.model:
            return {
                "agent_id": self.agent_id,
                "model_name": self.model_name,
                "question": question,
                "answer": "Model initialization failed",
                "error": "Model not initialized",
                "success": False
            }
        
        try:
            # Add system message context to the question
            full_prompt = f"{self.system_message}\n\nQuestion: {question}"
            
            # Run the query
            start_time = time.time()
            response = self.model.invoke(full_prompt)
            end_time = time.time()
            
            return {
                "agent_id": self.agent_id,
                "model_name": self.model_name,
                "question": question,
                "answer": str(response.content),
                "response_time": end_time - start_time,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            return {
                "agent_id": self.agent_id,
                "model_name": self.model_name,
                "question": question,
                "answer": f"Error: {str(e)}",
                "response_time": 0,
                "success": False,
                "error": str(e)
            }

async def run_agent_on_questions(agent_runner: AgentRunner, questions: List[str]) -> List[Dict[str, Any]]:
    """Run an agent on all questions asynchronously."""
    results = []
    
    for i, question in enumerate(questions, 1):
        print(f"Running {agent_runner.agent_id} on question {i}/{len(questions)}...")
        
        result = agent_runner.run_query(question)
        results.append(result)
        
        # Add small delay to avoid rate limiting
        await asyncio.sleep(1)
    
    return results

async def main():
    print("Starting agent evaluation with free Gemini models...")
    print(f"Number of agents: {len(AGENTS_CONFIG)}")
    print(f"Number of questions: {len(COMPLEX_QUESTIONS)}")
    print("-" * 50)
    
    # Check if API key is available
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        print("Please add your Gemini API key to the .env file.")
        return
    
    # Initialize all agents
    agents = []
    for config in AGENTS_CONFIG:
        agent = AgentRunner(config)
        if agent.model:  # Only add if model initialized successfully
            agents.append(agent)
            print(f"Initialized: {config['agent_id']}")
        else:
            print(f"Failed to initialize: {config['agent_id']}")
    
    if not agents:
        print("No agents initialized successfully. Exiting.")
        return
    
    # Run all agents on all questions
    all_results = []
    
    for agent in agents:
        print(f"\nRunning {agent.agent_id}...")
        agent_results = await run_agent_on_questions(agent, COMPLEX_QUESTIONS)
        all_results.extend(agent_results)
        
        # Save intermediate results after each agent
        df = pd.DataFrame(all_results)
        df.to_csv("agent_responses_intermediate.csv", index=False)
        print(f"Saved intermediate results for {agent.agent_id}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results to CSV
    results_df.to_csv("agent_responses.csv", index=False)
    print(f"\nSaved all results to agent_responses.csv")
    
    # Save results to JSON for better readability
    results_dict = results_df.to_dict('records')
    with open("agent_responses.json", "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    print("Saved results to agent_responses.json")
    
    # Generate summary statistics
    summary = results_df.groupby(['agent_id', 'model_name']).agg({
        'success': 'mean',
        'response_time': 'mean',
        'answer': 'count'
    }).round(3)
    
    summary.columns = ['success_rate', 'avg_response_time_seconds', 'total_questions']
    summary.to_csv("agent_performance_summary.csv")
    print("Saved performance summary to agent_performance_summary.csv")
    
    # Print summary
    print("\n" + "="*60)
    print("AGENT PERFORMANCE SUMMARY")
    print("="*60)
    print(summary)
    
    # Print some sample responses
    print("\n" + "="*60)
    print("SAMPLE RESPONSES")
    print("="*60)
    
    sample_questions = COMPLEX_QUESTIONS[:2]  # First 2 questions
    for question in sample_questions:
        print(f"\nQuestion: {question}")
        print("-" * 40)
        
        question_responses = results_df[results_df['question'] == question]
        for _, row in question_responses.iterrows():
            print(f"\n{row['agent_id']}:")
            answer_preview = row['answer'][:200] + "..." if len(row['answer']) > 200 else row['answer']
            print(f"  {answer_preview}")

if __name__ == "__main__":
    # Run the main async function
    asyncio.run(main())