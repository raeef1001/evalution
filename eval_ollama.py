

import json
import requests
import logging
import time
from pathlib import Path
import re

# Configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:32b"  # Ensure this model is suitable for JSON output
COMBINED_DATA_FILE = "combined_data.json"
OUTPUT_FILE = "evaluated_data.json"  # Stores all successfully evaluated items
LOG_FILE = "evaluation_run.log"      # Log file name
MAX_ITERATIONS_PER_RUN = 10          # Now acts as SAVE_CHECKPOINT_INTERVAL: Save after this many new items attempted
MAX_API_RETRIES = 3                  # For Ollama API calls
API_RETRY_DELAY = 2                  # For Ollama API calls (base delay, increases with attempts)
API_TIMEOUT = 180                    # Timeout for Ollama API calls in seconds

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'),  # Append mode for the log file
        logging.StreamHandler()                   # To also log to console
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
        "options": {
            "temperature": 0.1  # Lower temperature for more deterministic JSON
        }
    }

    for attempt in range(MAX_API_RETRIES):
        try:
            logger.debug(f"Attempting to call Ollama API (Attempt {attempt + 1}/{MAX_API_RETRIES})")
            response = requests.post(OLLAMA_API_URL, headers=headers, json=data, timeout=API_TIMEOUT)
            response.raise_for_status()  # Raises HTTPError for bad responses (4XX or 5XX)

            json_response = response.json()
            if "response" in json_response and isinstance(json_response["response"], str):
                return json_response["response"]
            else:
                logger.warning(f"Ollama API response format unexpected. Full response: {str(json_response)[:500]}...")
                if isinstance(json_response, dict):
                    for key, value in json_response.items():
                        if isinstance(value, str):
                            logger.warning(f"Using fallback: returning content of key '{key}' as Ollama response.")
                            return value
                raise ValueError("Ollama API response does not contain a 'response' string field or suitable alternative.")

        except requests.exceptions.RequestException as e:
            logger.warning(f"Ollama API call attempt {attempt + 1} failed: {str(e)}")
            if attempt < MAX_API_RETRIES - 1:
                time.sleep(API_RETRY_DELAY * (attempt + 1))  # Exponential backoff
            else:
                logger.error(f"All {MAX_API_RETRIES} attempts to call Ollama API failed.")
                raise  # Re-raise after all retries fail
        except Exception as e: # Catch other unexpected errors during API call
            logger.warning(f"An unexpected error occurred during Ollama API call attempt {attempt + 1}: {str(e)}")
            if attempt < MAX_API_RETRIES - 1:
                time.sleep(API_RETRY_DELAY * (attempt + 1))
            else:
                logger.error(f"All {MAX_API_RETRIES} attempts failed due to unexpected errors during API call.")
                raise # Re-raise after all retries fail
    return None # Should be unreachable if raise is consistently used on failure

def extract_json_from_response(response_text: str, problem_id: str) -> dict | None:
    """Attempts to parse response_text as JSON, or extract JSON from it."""
    try:
        # Attempt direct parsing first
        evaluation = json.loads(response_text)
        logger.debug(f"Successfully parsed JSON directly for {problem_id}.")
        return evaluation
    except json.JSONDecodeError:
        logger.warning(f"Direct JSON parsing failed for {problem_id}. Attempting to extract JSON from text.")

        # Try to extract JSON from markdown code block
        match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", response_text, re.IGNORECASE)
        if not match:
            # If not in a markdown block, try to find the first valid JSON object pattern
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
    """Create the evaluation prompt for Ollama with detailed scoring guidelines."""
    return f"""You are an expert physics problem evaluator. Your task is to meticulously compare an AI-generated solution to a manual, ground-truth solution for a given physics problem.
Evaluate the AI-generated solution based on the following categories and scoring guidelines. Provide your evaluation STRICTLY as a JSON object.

Evaluation Categories and Scoring Guidelines:

1.  **mathematical_accuracy**: (Score 1-5) How correct are the calculations, numerical answers, and units?
    *   5: All calculations, numerical results, and units are perfectly correct and appropriately presented.
    *   4: Minor calculation error or an incorrect/missing unit, but the underlying mathematical method is sound.
    *   3: Several minor errors, or one significant calculation error that impacts the result. Units might be inconsistently handled.
    *   2: Major calculation errors or fundamental misunderstandings of mathematical operations. Incorrect application of formulas numerically.
    *   1: Almost all calculations are incorrect, non-sensical, or missing.

2.  **logical_consistency**: (Score 1-5) Does the solution follow a logical step-by-step progression? Is the reasoning sound?
    *   5: The solution flows perfectly. Each step logically follows from the previous one. The reasoning is impeccable.
    *   4: Mostly logical and well-reasoned. Perhaps one step is slightly unclear or its justification is weak, but it doesn't break the overall logic.
    *   3: Some logical gaps, inconsistencies, or steps that don't clearly follow, making the solution harder to follow or verify.
    *   2: Significant logical flaws. Steps are out of order, reasoning is poor or contradictory.
    *   1: The solution is illogical, incoherent, or internally contradictory.

3.  **completeness**: (Score 1-5) Does the AI-generated solution address all parts of the problem as presented?
    *   5: All parts of the problem (including sub-questions, if any) are fully addressed and answered.
    *   4: A minor aspect of the problem is overlooked, or one sub-question is not fully answered or is missing.
    *   3: A significant part of the problem is ignored or left unanswered.
    *   2: Only a small portion of the problem is addressed; major components are missing.
    *   1: The problem is largely unaddressed or the solution is off-topic.

4.  **clarity_and_coherence**: (Score 1-5) Is the explanation clear, concise, and easy to understand?
    *   5: The explanation is exceptionally clear, concise, well-structured, and very easy to understand. Excellent use of language and terminology.
    *   4: The explanation is clear and generally easy to understand, with minor areas for improvement in conciseness, structure, or flow.
    *   3: The explanation is generally understandable but may be verbose, unclear in parts, poorly organized, or contain jargon without adequate explanation.
    *   2: The explanation is difficult to understand due to ambiguity, poor writing, or convoluted structure.
    *   1: The explanation is incomprehensible, extremely poorly written, or nonsensical.

5.  **formulas_principles**: (Score 1-5) Are correct physical formulas and principles identified and applied correctly?
    *   5: All necessary physical formulas and principles are correctly identified, stated, and applied appropriately to the problem.
    *   4: Mostly correct formulas/principles. Perhaps a minor error in recalling a formula, or a slight misapplication of a correct principle.
    *   3: Some incorrect formulas/principles are used, or correct ones are applied incorrectly in a significant way.
    *   2: Major errors in formula/principle selection or application. Fundamental physics concepts are misunderstood.
    *   1: Completely inappropriate formulas/principles are used, or relevant physics is entirely ignored.

6.  **assumptions_made**: (Score 1-5) Are AI assumptions explicit, justified, and reasonable for the problem context?
    *   5: All necessary assumptions are explicitly stated, well-justified, and perfectly reasonable for the problem context.
    *   4: Most necessary assumptions are stated and reasonable; some minor ones might be implicit but obvious, or lack full justification but are acceptable.
    *   3: Some key assumptions are missing, not clearly stated, unjustified, or questionable in reasonableness.
    *   2: Major unreasonable assumptions are made, or critical assumptions are not stated, leading to an incorrect or flawed solution path.
    *   1: Assumptions are entirely inappropriate, absent when clearly needed, or lead to a trivialization/misrepresentation of the problem.

7.  **overall_correctness**: (Score 0-10) How correct and sound is the AI's approach and final answer(s) overall, considering all the above?
    *   10: Perfect solution in all aspects. Correct approach, execution, and answer.
    *   8-9: Excellent solution. Fundamentally correct with very minor, inconsequential flaws.
    *   6-7: Good solution. Generally correct approach and largely correct answer, but with some noticeable errors, omissions, or areas for improvement.
    *   4-5: Partially correct. The solution demonstrates some understanding but contains significant flaws in reasoning, calculation, or application of principles.
    *   2-3: Mostly incorrect. Shows fundamental misunderstandings of the problem or physics principles.
    *   0-1: Completely incorrect, irrelevant, or no meaningful attempt made.

Problem ID: {problem_id}

Ground Truth Solution:


{ground_truth}

Elaborated Solution Steps (Manual):
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END

{elaborated_solution}

AI-Generated Solution to Evaluate:
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END

{ai_solution}

Provide your evaluation STRICTLY as a JSON object with the problem_id and scores for each category listed above.
Your entire response should be ONLY the JSON object, starting with {{{{ and ending with }}}}.
Example JSON format:
{{
    "problem_id": "{problem_id}",
    "mathematical_accuracy": <score_1_to_5>,
    "logical_consistency": <score_1_to_5>,
    "completeness": <score_1_to_5>,
    "clarity_and_coherence": <score_1_to_5>,
    "formulas_principles": <score_1_to_5>,
    "assumptions_made": <score_1_to_5>,
    "overall_correctness": <score_0_to_10>
}}"""

def validate_evaluation(evaluation: dict, problem_id: str) -> bool:
    """Validate the evaluation JSON structure and scores."""
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
        if not isinstance(evaluation[field], expected_type): # Allow int or float for scores
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

def save_evaluated_data(data_to_save, file_path):
    """Saves the evaluated data list to a JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved {len(data_to_save)} items to {file_path}.")
    except Exception as e:
        logger.error(f"Failed to save data to {file_path}: {e}", exc_info=True)

def main():
    output_path = Path(OUTPUT_FILE)
    evaluated_data = []  # This list will hold all items, loaded and newly evaluated
    processed_problem_ids = set() # IDs of items already successfully evaluated and stored

    # Load previously evaluated data
    if output_path.exists() and output_path.stat().st_size > 0:
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            if isinstance(loaded_data, list):
                for item in loaded_data:
                    if isinstance(item, dict) and 'Problem_ID' in item and 'ollama_evaluation' in item:
                        # Validate the loaded evaluation before considering it processed
                        if validate_evaluation(item['ollama_evaluation'], item['Problem_ID']):
                            evaluated_data.append(item)
                            processed_problem_ids.add(item['Problem_ID'])
                        else:
                            logger.warning(f"Invalid evaluation structure in {OUTPUT_FILE} for Problem_ID '{item['Problem_ID']}'. It will be re-evaluated if encountered in input.")
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

    # Load input data
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

    # Filter for new items to process
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
        logger.info(f"No new items to process from {COMBINED_DATA_FILE}. All eligible items seem to be in {OUTPUT_FILE}.")
        return
    logger.info(f"Found {len(new_items_to_consider)} new items to potentially process.")

    items_attempted_this_run = 0 # Counts items attempted in this script execution

    for item_index, item in enumerate(new_items_to_consider):
        problem_id = item.get('Problem_ID') # Already checked for existence
        ground_truth = item.get('ground_truth', '')
        elaborated_solution = item.get('elaborated_solution', '')
        openai_solution = item.get('openai_solution', '') # This is the AI solution

        if not all([ground_truth, elaborated_solution, openai_solution]): # Problem_ID is known to be present
            logger.warning(f"Skipping item {problem_id} (new item {item_index + 1}/{len(new_items_to_consider)}) due to missing essential data fields (ground_truth, elaborated_solution, or openai_solution). It will be skipped in future runs too unless data is fixed.")
            continue # This item is not "attempted" for checkpointing purposes

        items_attempted_this_run += 1
        logger.info(f"Processing new item {item_index + 1}/{len(new_items_to_consider)}: {problem_id} (Attempt #{items_attempted_this_run} in this script run)")

        prompt_text = create_evaluation_prompt(
            problem_id, ground_truth, elaborated_solution, openai_solution
        )

        evaluation_successful_for_item = False
        try:
            ollama_response_text = call_ollama(prompt_text)
            if ollama_response_text is None: # call_ollama re-raises on total failure, this path might not be hit if it always raises
                logger.error(f"Failed to get response from Ollama for {problem_id} after all retries. Item will be re-attempted in a future run.")
            else:
                evaluation = extract_json_from_response(ollama_response_text, problem_id)
                if evaluation and validate_evaluation(evaluation, problem_id):
                    item['ollama_evaluation'] = evaluation
                    # Add to the main list that gets saved.
                    # To avoid duplicates if an item was partially processed but not saved:
                    # Remove old entry if it exists (e.g. if script crashed before saving last time but PID was added to processed_problem_ids prematurely)
                    # However, current logic is: load from file, only new PIDs are in new_items_to_consider.
                    # So, `item` is guaranteed not to be in `evaluated_data` with a *valid* `ollama_evaluation` yet.
                    evaluated_data.append(item)
                    processed_problem_ids.add(problem_id) # Mark as processed for *this run's* internal tracking too
                    logger.info(f"Successfully evaluated and stored result for {problem_id}.")
                    evaluation_successful_for_item = True
                elif evaluation:
                    logger.warning(f"Validation failed for evaluation of {problem_id}. Ollama response (parsed): {str(evaluation)[:300]}... This item will be re-attempted in a future run.")
                else:
                    logger.warning(f"Could not extract valid JSON evaluation from Ollama for {problem_id}. This item will be re-attempted in a future run.")

        except Exception as e: # Catch errors from call_ollama (if it raises) or other unexpected issues
            logger.error(f"An unexpected error occurred while processing {problem_id} (item will be re-attempted in future run): {str(e)}", exc_info=True)

        # Save progress periodically
        if items_attempted_this_run > 0 and items_attempted_this_run % MAX_ITERATIONS_PER_RUN == 0:
            logger.info(f"Checkpoint: Attempted {items_attempted_this_run} items in this run. Saving progress...")
            save_evaluated_data(evaluated_data, output_path)

        # Rate limiting
        if not evaluation_successful_for_item and items_attempted_this_run < len(new_items_to_consider): # If failed, maybe a slightly longer pause
            time.sleep(API_RETRY_DELAY)
        elif items_attempted_this_run < len(new_items_to_consider): # Standard delay between items
             time.sleep(1.5)


    # Final save at the end of processing all new items for this run
    logger.info(f"Finished processing all {len(new_items_to_consider)} new items considered in this run (attempted {items_attempted_this_run}). Performing final save.")
    save_evaluated_data(evaluated_data, output_path)

    final_unique_ids_in_output = {item.get('Problem_ID') for item in evaluated_data if item.get('Problem_ID')}
    logger.info(f"Total unique evaluated items now in {OUTPUT_FILE}: {len(final_unique_ids_in_output)} ({len(evaluated_data)} total entries, may differ if duplicates somehow occurred before this version).")

if __name__ == "__main__":
    logger.info("Starting evaluation script run.")
    main()
    logger.info("Evaluation script run finished.")
