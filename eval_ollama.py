import json
import requests
import logging
import time
from pathlib import Path

# Configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen:32b"
COMBINED_DATA_FILE = "combined_data.json"
OUTPUT_FILE = "evaluated_data.json"
MAX_RETRIES = 3
RETRY_DELAY = 2

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def call_ollama(prompt, model=MODEL_NAME):
    """Call Ollama API and get response"""
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(OLLAMA_API_URL, headers=headers, json=data)
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                logger.error("All attempts failed")
                raise

def create_evaluation_prompt(problem_id, ground_truth, elaborated_solution, ai_solution):
    """Create the evaluation prompt for Ollama"""
    return f"""You are an expert physics problem evaluator. Your task is to compare an AI-generated solution to a manual, ground-truth solution for a given physics problem.
Evaluate the AI-generated solution based on the following categories:
1.  **mathematical_accuracy**: (Score 1-5) How correct are the calculations, numerical answers, and units?
2.  **logical_consistency**: (Score 1-5) Does the solution follow a logical step-by-step progression?
3.  **completeness**: (Score 1-5) Does the AI-generated solution address all parts of the problem?
4.  **clarity_and_coherence**: (Score 1-5) Is the explanation clear, concise, easy to understand?
5.  **formulas_principles**: (Score 1-5) Are correct physical formulas/principles applied correctly?
6.  **assumptions_made**: (Score 1-5) Are AI assumptions explicit, justified, and reasonable?
7.  **overall_correctness**: (Score 0-10) How correct and sound is the AI's approach overall?

Problem ID: {problem_id}

Ground Truth Solution:
{ground_truth}

Elaborated Solution Steps:
{elaborated_solution}

AI-Generated Solution:
{ai_solution}

Provide your evaluation as a JSON object with the problem_id and scores for each category.
Response format:
{{
    "problem_id": "{problem_id}",
    "mathematical_accuracy": <1-5>,
    "logical_consistency": <1-5>,
    "completeness": <1-5>,
    "clarity_and_coherence": <1-5>,
    "formulas_principles": <1-5>,
    "assumptions_made": <1-5>,
    "overall_correctness": <0-10>
}}"""

def validate_evaluation(evaluation, problem_id):
    """Validate the evaluation JSON structure and scores"""
    try:
        if not isinstance(evaluation, dict):
            evaluation = json.loads(evaluation)
        
        required_fields = [
            "mathematical_accuracy", "logical_consistency", "completeness",
            "clarity_and_coherence", "formulas_principles", "assumptions_made",
            "overall_correctness"
        ]
        
        # Check all fields exist
        for field in required_fields:
            if field not in evaluation:
                logger.error(f"Missing field: {field}")
                return False
                
        # Validate score ranges
        for field in required_fields[:-1]:  # All except overall_correctness
            score = evaluation[field]
            if not isinstance(score, (int, float)) or score < 1 or score > 5:
                logger.error(f"Invalid score for {field}: {score}")
                return False
        
        # Validate overall_correctness
        overall = evaluation["overall_correctness"]
        if not isinstance(overall, (int, float)) or overall < 0 or overall > 10:
            logger.error(f"Invalid score for overall_correctness: {overall}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Validation error for {problem_id}: {str(e)}")
        return False

def main():
    try:
        # Load existing combined data
        with open(COMBINED_DATA_FILE, 'r', encoding='utf-8') as f:
            combined_data = json.load(f)
        
        evaluated_data = []
        
        for item in combined_data:
            problem_id = item.get('Problem_ID')
            ground_truth = item.get('ground_truth', '')
            elaborated_solution = item.get('elaborated_solution', '')
            openai_solution = item.get('openai_solution', '')
            
            if not all([problem_id, ground_truth, elaborated_solution, openai_solution]):
                logger.warning(f"Skipping incomplete item: {problem_id}")
                continue
                
            logger.info(f"Processing problem: {problem_id}")
            
            # Create prompt and get evaluation from Ollama
            prompt = create_evaluation_prompt(
                problem_id, 
                ground_truth, 
                elaborated_solution, 
                openai_solution
            )
            
            try:
                ollama_response = call_ollama(prompt)
                evaluation = json.loads(ollama_response)
                
                if validate_evaluation(evaluation, problem_id):
                    # Add Ollama evaluation to the item
                    item['ollama_evaluation'] = evaluation
                    evaluated_data.append(item)
                    logger.info(f"Successfully evaluated {problem_id}")
                else:
                    logger.warning(f"Invalid evaluation for {problem_id}")
                
            except Exception as e:
                logger.error(f"Error processing {problem_id}: {str(e)}")
                continue
            
            time.sleep(1)  # Rate limiting
        
        # Save the evaluated data
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(evaluated_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation complete. Processed {len(evaluated_data)} items.")
        
    except Exception as e:
        logger.error(f"Main process error: {str(e)}")

if __name__ == "__main__":
    main()