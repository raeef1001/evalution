import json
import requests
import logging
import time
from pathlib import Path
import re # Added for robust JSON extraction

# Configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:32b" # Ensure this model is suitable for JSON output
COMBINED_DATA_FILE = "combined_data.json"
OUTPUT_FILE = "evaluated_data.json" # Stores all successfully evaluated items
LOG_FILE = "evaluation_run.log"     # Log file name
MAX_ITERATIONS_PER_RUN = 10         # Max new items to process per script execution
MAX_API_RETRIES = 3                 # For Ollama API calls
API_RETRY_DELAY = 2                 # For Ollama API calls

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'), # Append mode for the log file
        logging.StreamHandler()                  # To also log to console
    ]
)
logger = logging.getLogger(__name__)

def call_ollama(prompt, model=MODEL_NAME):
    """Call Ollama API and get response string"""
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": { # Options for potentially better JSON output
            "temperature": 0.2 # Lower temperature for more deterministic JSON
        }
    }
    
    for attempt in range(MAX_API_RETRIES):
        try:
            logger.debug(f"Attempting to call Ollama API (Attempt {attempt + 1}/{MAX_API_RETRIES})")
            response = requests.post(OLLAMA_API_URL, headers=headers, json=data, timeout=120) # Increased timeout
            response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
            
            json_response = response.json()
            if "response" in json_response and isinstance(json_response["response"], str):
                return json_response["response"]
            else:
                logger.warning(f"Ollama API response format unexpected. Full response: {str(json_response)[:500]}...")
                # Fallback: try to find any string value if "response" key is missing/wrong type
                if isinstance(json_response, dict):
                    for key, value in json_response.items():
                        if isinstance(value, str):
                            logger.warning(f"Using fallback: returning content of key '{key}' as Ollama response.")
                            return value
                raise ValueError("Ollama API response does not contain a 'response' string field or suitable alternative.")

        except requests.exceptions.RequestException as e:
            logger.warning(f"Ollama API call attempt {attempt + 1} failed: {str(e)}")
            if attempt < MAX_API_RETRIES - 1:
                time.sleep(API_RETRY_DELAY * (attempt + 1)) # Basic exponential backoff
            else:
                logger.error(f"All {MAX_API_RETRIES} attempts to call Ollama API failed.")
                raise
        except Exception as e:
            logger.warning(f"An unexpected error occurred during Ollama API call attempt {attempt + 1}: {str(e)}")
            if attempt < MAX_API_RETRIES - 1:
                time.sleep(API_RETRY_DELAY * (attempt + 1))
            else:
                logger.error(f"All {MAX_API_RETRIES} attempts failed due to unexpected errors during API call.")
                raise
    return None # Should be unreachable if raise is hit

def extract_json_from_response(response_text: str, problem_id: str) -> dict | None:
    """Attempts to parse response_text as JSON, or extract JSON from it."""
    try:
        # Attempt direct parsing first
        evaluation = json.loads(response_text)
        logger.debug(f"Successfully parsed JSON directly for {problem_id}.")
        return evaluation
    except json.JSONDecodeError:
        logger.warning(f"Direct JSON parsing failed for {problem_id}. Attempting to extract JSON from text.")
        
        # Try to extract JSON from markdown code block (common for LLMs)
        # Handles ```json ... ``` or just ``` ... ```
        match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", response_text, re.IGNORECASE)
        if not match:
            # If not in a markdown block, try to find the first valid JSON object pattern
            # This regex looks for a balanced pair of braces.
            match = re.search(r"(\{[\s\S]*\})", response_text)

        if match:
            json_str = match.group(1)
            try:
                evaluation = json.loads(json_str)
                logger.info(f"Successfully extracted and parsed JSON for {problem_id} after initial parse failure.")
                return evaluation
            except json.JSONDecodeError as json_e_inner:
                logger.error(f"Failed to parse extracted JSON for {problem_id}: {json_e_inner}")
                logger.debug(f"Original response text for {problem_id} (snippet): {response_text[:500]}...")
                logger.debug(f"Extracted JSON string for {problem_id} (snippet): {json_str[:500]}...")
                return None
        else:
            logger.error(f"Could not find or extract JSON from Ollama response for {problem_id}.")
            logger.debug(f"Original response text for {problem_id} (snippet): {response_text[:500]}...")
            return None

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

Provide your evaluation STRICTLY as a JSON object with the problem_id and scores for each category.
Your entire response should be ONLY the JSON object, starting with {{{{ and ending with }}}}.
Example JSON format:
{{
    "problem_id": "{problem_id}",
    "mathematical_accuracy": <1-5>,
    "logical_consistency": <1-5>,
    "completeness": <1-5>,
    "clarity_and_coherence": <1-5>,
    "formulas_principles": <1-5>,
    "assumptions_made": <1-5>,
    "overall_correctness": <0-10>
}}""" # Note: Escaped curlies {{{{ and }}}} for f-string within the example.

def validate_evaluation(evaluation: dict, problem_id: str) -> bool:
    """Validate the evaluation JSON structure and scores. Expects a dictionary."""
    if not isinstance(evaluation, dict):
        logger.error(f"Validation input for {problem_id} is not a dictionary. Received type: {type(evaluation)}")
        return False
        
    required_fields_types = {
        "problem_id": str,
        "mathematical_accuracy": (int, float), "logical_consistency": (int, float),
        "completeness": (int, float), "clarity_and_coherence": (int, float),
        "formulas_principles": (int, float), "assumptions_made": (int, float),
        "overall_correctness": (int, float)
    }
    
    for field, expected_type in required_fields_types.items():
        if field not in evaluation:
            logger.error(f"Missing field '{field}' in evaluation for {problem_id}.")
            return False
        if not isinstance(evaluation[field], expected_type):
            logger.error(f"Invalid type for field '{field}' in evaluation for {problem_id}. Expected {expected_type}, got {type(evaluation[field])}.")
            return False

    if evaluation["problem_id"] != problem_id:
        logger.error(f"Mismatched problem_id in evaluation for {problem_id}. Expected '{problem_id}', got '{evaluation['problem_id']}'.")
        return False
            
    score_1_5_fields = [
        "mathematical_accuracy", "logical_consistency", "completeness",
        "clarity_and_coherence", "formulas_principles", "assumptions_made"
    ]
    for field in score_1_5_fields:
        score = evaluation[field]
        if not (isinstance(score, (int, float)) and 1 <= score <= 5):
            logger.error(f"Invalid score for '{field}' in evaluation for {problem_id}: {score}. Must be a number between 1 and 5.")
            return False
    
    overall_score = evaluation["overall_correctness"]
    if not (isinstance(overall_score, (int, float)) and 0 <= overall_score <= 10):
        logger.error(f"Invalid score for 'overall_correctness' in evaluation for {problem_id}: {overall_score}. Must be a number between 0 and 10.")
        return False
        
    return True

def main():
    output_path = Path(OUTPUT_FILE)
    evaluated_data = []
    processed_problem_ids = set()

    if output_path.exists() and output_path.stat().st_size > 0:
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            if isinstance(loaded_data, list):
                for item in loaded_data:
                    if isinstance(item, dict) and 'Problem_ID' in item and 'ollama_evaluation' in item:
                        evaluated_data.append(item) # Keep valid, previously evaluated items
                        processed_problem_ids.add(item['Problem_ID'])
                    else:
                        logger.warning(f"Malformed or incomplete item found in {OUTPUT_FILE}, skipping: {str(item)[:100]}...")
            else:
                logger.warning(f"{OUTPUT_FILE} does not contain a list. Starting with an empty evaluation set.")
            logger.info(f"Loaded {len(evaluated_data)} valid previously evaluated items from {OUTPUT_FILE}. Found {len(processed_problem_ids)} unique processed problem IDs.")
        except json.JSONDecodeError:
            logger.warning(f"Could not decode JSON from {OUTPUT_FILE}. Starting with an empty evaluation set.")
        except Exception as e:
            logger.error(f"Error loading {OUTPUT_FILE}: {e}. Starting with an empty evaluation set.")
    else:
        logger.info(f"{OUTPUT_FILE} does not exist or is empty. Starting with an empty evaluation set.")

    try:
        with open(COMBINED_DATA_FILE, 'r', encoding='utf-8') as f:
            combined_data_input = json.load(f)
        if not isinstance(combined_data_input, list):
            logger.error(f"{COMBINED_DATA_FILE} does not contain a list. Aborting.")
            return
    except FileNotFoundError:
        logger.error(f"Input file {COMBINED_DATA_FILE} not found. Aborting.")
        return
    except json.JSONDecodeError:
        logger.error(f"Could not decode JSON from {COMBINED_DATA_FILE}. Aborting.")
        return
    except Exception as e:
        logger.error(f"Unexpected error loading {COMBINED_DATA_FILE}: {e}. Aborting.", exc_info=True)
        return
        
    new_items_to_consider = []
    for item_from_combined in combined_data_input:
        if not isinstance(item_from_combined, dict):
            logger.warning(f"Item in {COMBINED_DATA_FILE} is not a dictionary, skipping: {str(item_from_combined)[:100]}...")
            continue
        pid = item_from_combined.get('Problem_ID')
        if not pid:
            original_id_info = item_from_combined.get('Original_ID', item_from_combined.get('id', 'Unknown Original_ID'))
            logger.warning(f"Item in {COMBINED_DATA_FILE} missing 'Problem_ID', skipping (Original ID if available: {original_id_info}).")
            continue
        if pid not in processed_problem_ids:
            new_items_to_consider.append(item_from_combined)

    if not new_items_to_consider:
        logger.info("No new items to process from {COMBINED_DATA_FILE}. All items are already in {OUTPUT_FILE} or {COMBINED_DATA_FILE} is empty/fully processed.")
        return
    logger.info(f"Found {len(new_items_to_consider)} new items to potentially process.")

    items_attempted_this_session = 0
    
    for item in new_items_to_consider:
        if items_attempted_this_session >= MAX_ITERATIONS_PER_RUN:
            logger.info(f"Reached iteration limit of {MAX_ITERATIONS_PER_RUN} for this run.")
            break

        problem_id = item.get('Problem_ID')
        ground_truth = item.get('ground_truth', '')
        elaborated_solution = item.get('elaborated_solution', '')
        openai_solution = item.get('openai_solution', '') # This is the AI solution to be evaluated
        
        if not all([problem_id, ground_truth, elaborated_solution, openai_solution]):
            logger.warning(f"Skipping item {problem_id} due to missing essential data fields (ground_truth, elaborated_solution, or openai_solution). It will be skipped in future runs too unless data is fixed.")
            # This item is considered "unactionable" for now. It won't count towards the iteration limit
            # unless we decide that looking at it is an "iteration".
            # For now, let's not count it, to prioritize processable items for the limit.
            continue
            
        logger.info(f"Processing item: {problem_id} (Attempt {items_attempted_this_session + 1}/{min(len(new_items_to_consider), MAX_ITERATIONS_PER_RUN)} for this session)")
        items_attempted_this_session += 1 # Count this as an attempt for the session limit

        prompt_text = create_evaluation_prompt(
            problem_id, ground_truth, elaborated_solution, openai_solution
        )
        
        try:
            ollama_response_text = call_ollama(prompt_text)
            if ollama_response_text is None: # call_ollama now returns None if all retries fail without raising
                logger.error(f"Failed to get response from Ollama for {problem_id} after all retries. Item will be re-attempted in a future run.")
                time.sleep(1) # Standard delay before next item
                continue

            evaluation = extract_json_from_response(ollama_response_text, problem_id)

            if evaluation and validate_evaluation(evaluation, problem_id):
                item['ollama_evaluation'] = evaluation
                evaluated_data.append(item) # Add successfully processed item to the main list
                processed_problem_ids.add(problem_id) # Mark as processed for this run's context, though file is source of truth
                logger.info(f"Successfully evaluated and stored result for {problem_id}.")
            elif evaluation: # JSON was extracted but failed validation
                logger.warning(f"Validation failed for evaluation of {problem_id}. Ollama response (parsed): {str(evaluation)[:300]}... This item will be re-attempted.")
            else: # JSON could not be extracted
                logger.warning(f"Could not extract valid JSON evaluation from Ollama for {problem_id}. This item will be re-attempted.")
            
        except Exception as e: # Catch-all for errors from call_ollama or other unexpected issues
            logger.error(f"An unexpected error occurred while processing {problem_id} (item will be re-attempted in future run): {str(e)}", exc_info=True)
            
        time.sleep(1.5)  # Slightly increased rate limiting per item processing attempt

    # Save the (potentially updated) evaluated_data list
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluated_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Session complete. Attempted to process {items_attempted_this_session} new items in this run.")
        final_processed_ids = {item.get('Problem_ID') for item in evaluated_data if item.get('Problem_ID')}
        logger.info(f"Total evaluated items now in {OUTPUT_FILE}: {len(evaluated_data)} (representing {len(final_processed_ids)} unique problem IDs).")
    except Exception as e:
        logger.error(f"Failed to save evaluated data to {OUTPUT_FILE}: {e}", exc_info=True)

if __name__ == "__main__":
    logger.info("Starting evaluation script run.")
    main()
    logger.info("Evaluation script run finished.")
