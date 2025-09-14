# main.py
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from config import SYNTHETIC_DATA_PATH, RESULTS_PATH, REPORT_PATH
from scorers.instruction_following import score_instruction_following
from scorers.hallucination import score_hallucination

import json

def load_data(file_path):
    """Load data from a JSON array file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Ensure it's a list
        if isinstance(data, list):
            return data
        else:
            print(f"Warning: JSON file does not contain an array. Got: {type(data)}")
            return [data]  # Wrap single object in a list
    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return []
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return []
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []

def process_batch(data_batch):
    """Process a batch of data entries."""
    results = []
    
    for entry in tqdm(data_batch, desc="Processing responses"):
        prompt = entry['question']
        response = entry['answer']
        
        # Score each dimension
        instruction_score = score_instruction_following(prompt, response)
        hallucination_score = score_hallucination(prompt, response)
        
        # For prototype, we'll simplify other scores

        
        result = {
            'prompt_id': entry['question'],
            'agent_id': entry['agent_id'],
            'instruction_score': instruction_score,
            'hallucination_score': hallucination_score,
            
            'composite_score': round((instruction_score + (1 - hallucination_score) ) / 2, 2),
            'response': response[:100] + "..." if len(response) > 100 else response  # Preview
        }
        results.append(result)
    
    return results

def generate_report(results_df):
    """Generate a simple HTML report."""
    # Group by agent and calculate averages
    agent_report = results_df.groupby('agent_id').agg({
        'composite_score': 'mean',
        'hallucination_score': 'mean',
        'instruction_score': 'mean'
    }).round(2)
    
    # Create HTML report
    html_content = f"""
    <html>
    <head>
        <title>Agent Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>AI Agent Evaluation Report</h1>
        <p>Evaluated {len(results_df)} responses across {len(agent_report)} agents</p>
        
        <h2>Agent Performance Summary</h2>
        {agent_report.to_html()}
        
        <h2>Detailed Results</h2>
        {results_df.to_html(index=False)}
    </body>
    </html>
    """
    
    # Save report
    Path(REPORT_PATH).parent.mkdir(exist_ok=True)
    with open(REPORT_PATH, 'w') as f:
        f.write(html_content)
    
    print(f"Report generated: {REPORT_PATH}")

def main():
   
    print("Loading data...")
    data = load_data(SYNTHETIC_DATA_PATH)
    
    # Process all entries
    print("Evaluating responses...")
    results = process_batch(data)
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"Results saved to: {RESULTS_PATH}")
    
    # Generate report
    generate_report(results_df)
    
    # Show summary
    print("\nEvaluation Complete!")
    print(f"Agents evaluated: {results_df['agent_id'].nunique()}")
    print(f"Average Composite Score: {results_df['composite_score'].mean():.2f}")

if __name__ == "__main__":
    main()