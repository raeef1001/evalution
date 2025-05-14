import json
import requests
import os
import time
import logging # Import the logging module

# --- Configuration ---
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
# IMPORTANT: Set your OpenRouter API Key as an environment variable
# Example: export OPENROUTER_API_KEY="your_api_key_here"
OPENROUTER_API_KEY = "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"

# Choose a model available on OpenRouter.
OPENROUTER_MODEL = "deepseek/deepseek-chat-v3-0324:free" # Default, change as needed

MANUAL_DATA_FILE = "Test Set copy 3.jsonl"
AI_GENERATED_FILE = "openai2.jsonl"


OUTPUT_EVALUATION_FILE = "evaluations_openrouter_full.jsonl"
LOG_FILE_NAME = "evaluation_script.log" # Name for the log file

MAX_EVALUATIONS_TO_PERFORM = 1_000_000
SECONDS_BETWEEN_REQUESTS = 2
MAX_API_RETRIES_PER_CALL = 3
MAX_VALIDATION_RETRIES_PER_PROBLEM = 2
SAVE_INTERVAL = 10

EVALUATION_CATEGORIES = [
    "mathematical_accuracy", "logical_consistency", "completeness",
    "clarity_and_coherence", "formulas_principles", "assumptions_made",
    "overall_correctness"
]

# Global logger object
logger = logging.getLogger(__name__)

# --- Logging Setup ---
def setup_logging(log_file_name, console_level=logging.INFO, file_level=logging.DEBUG):
    """Configures logging to console and file."""
    logger.setLevel(min(console_level, file_level)) # Set logger to the lower of the two levels

    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    formatter_ch = logging.Formatter('%(levelname)s: %(message)s') # Simpler for console
    ch.setFormatter(formatter_ch)
    logger.addHandler(ch)

    # File Handler
    try:
        fh = logging.FileHandler(log_file_name, mode='w') # Overwrite log file each run
        fh.setLevel(file_level)
        formatter_fh = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s')
        fh.setFormatter(formatter_fh)
        logger.addHandler(fh)
    except IOError as e:
        logger.error(f"Could not open log file {log_file_name} for writing: {e}")
        # Fallback to console only if file logging fails
        logger.removeHandler(ch) # Remove previous console handler
        logger.addHandler(logging.StreamHandler()) # Add default console handler
        logger.error("Logging to file failed. Will log to console only.")


# --- Helper Functions ---
def load_jsonl(file_path):
    logger.info(f"Attempting to load JSONL file: {file_path}")
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed line {i+1} in {file_path}: {line.strip()} - Error: {e}")
        logger.info(f"Successfully loaded {len(data)} records from {file_path}")
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}. Please ensure it exists.")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading {file_path}: {e}")
        return None
    return data

def save_evaluations_batch(evaluations_batch, file_path):
    if not evaluations_batch:
        return
    logger.info(f"Attempting to append {len(evaluations_batch)} evaluations to {file_path}")
    try:
        with open(file_path, 'a', encoding='utf-8') as f:
            for eval_item in evaluations_batch:
                f.write(json.dumps(eval_item) + '\n')
        logger.info(f"Successfully appended {len(evaluations_batch)} evaluations to {file_path}")
    except IOError as e:
        logger.error(f"Error writing batch to {file_path}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during batch save to {file_path}: {e}")


def call_openrouter_evaluator(problem_id_expected, problem_text, manual_ground_truth, manual_elaborated_steps, manual_final_answers, ai_generated_content):
    if not OPENROUTER_API_KEY:
        logger.critical("OPENROUTER_API_KEY is not set. Cannot make API calls.")
        return None

    prompt = f"""You are an expert physics problem evaluator. Your task is to compare an AI-generated solution to a manual, ground-truth solution for a given physics problem.
Evaluate the AI-generated solution based on the following categories:
1.  **mathematical_accuracy**: (Score 1-5) How correct are the calculations, numerical answers, and units? (1: Very Poor, 5: Excellent)
2.  **logical_consistency**: (Score 1-5) Does the solution follow a logical step-by-step progression? Are there contradictions? (1: Very Poor, 5: Excellent)
3.  **completeness**: (Score 1-5) Does the AI-generated solution address all parts of the problem? Miss any key steps/components from manual? (1: Very Poor, 5: Excellent)
4.  **clarity_and_coherence**: (Score 1-5) Is the explanation clear, concise, easy to understand? Precise language? (1: Very Poor, 5: Excellent)
5.  **formulas_principles**: (Score 1-5) Are correct physical formulas/principles applied and stated correctly? (1: Very Poor, 5: Excellent)
6.  **assumptions_made**: (Score 1-5) Are AI assumptions explicit, justified, and reasonable? (1: Very Poor, 5: Excellent)
7.  **overall_correctness**: (Score 0-10) Considering all aspects, how correct and sound is the AI's approach and final answer(s) compared to the ground truth? (0: Completely Incorrect, 10: Perfectly Correct and Well-Explained)

Provide your evaluation as a JSON object. The JSON object MUST have a key "problem_id" with the EXACT Problem ID provided below, and then keys for each of the seven categories listed above with their respective scores.

EXACT Problem ID: 



Manual Solution (Ground Truth):


Manual Solution (Elaborated Solution Steps):


AI-Generated Solution:


Now, provide your evaluation ONLY as a single JSON object matching the specified format. Do not include any other text, explanations, or markdown formatting outside of the JSON object itself.
Your entire response MUST be a valid JSON. Example structure:
{{
  
  "mathematical_accuracy": <score_1_5>,
  "logical_consistency": <score_1_5>,
  "completeness": <score_1_5>,
  "clarity_and_coherence": <score_1_5>,
  "formulas_principles": <score_1_5>,
  "assumptions_made": <score_1_5>,
  "overall_correctness": <score_0_10>
}}
"""
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }

    raw_content = None # Initialize to ensure it's defined for logging in case of early exit
    json_block = None  # Initialize

    for attempt in range(MAX_API_RETRIES_PER_CALL):
        logger.debug(f"Attempting API call {attempt + 1}/{MAX_API_RETRIES_PER_CALL} for problem: {problem_id_expected}")
        try:
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=180)
            logger.debug(f"API response status code: {response.status_code} for problem: {problem_id_expected}")
            response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
            response_data = response.json()
            logger.debug(f"Raw API response data for {problem_id_expected} (attempt {attempt+1}): {str(response_data)[:500]}...") # Log partial raw response

            if not response_data.get("choices") or not response_data["choices"]:
                logger.warning(f"API Error (Attempt {attempt + 1}) for {problem_id_expected}: 'choices' array missing/empty. Response: {response_data}")
                if attempt < MAX_API_RETRIES_PER_CALL - 1: time.sleep((attempt + 1) * SECONDS_BETWEEN_REQUESTS * 2)
                continue

            message = response_data["choices"][0].get("message")
            if not message or "content" not in message:
                logger.warning(f"API Error (Attempt {attempt + 1}) for {problem_id_expected}: 'message' or 'content' key missing. Response: {response_data}")
                if attempt < MAX_API_RETRIES_PER_CALL - 1: time.sleep((attempt + 1) * SECONDS_BETWEEN_REQUESTS * 2)
                continue
            
            raw_content = message['content']
            if not raw_content or not raw_content.strip():
                logger.warning(f"API Error (Attempt {attempt + 1}) for {problem_id_expected}: LLM returned empty content string. Response: {response_data}")
                if attempt < MAX_API_RETRIES_PER_CALL - 1: time.sleep((attempt + 1) * SECONDS_BETWEEN_REQUESTS * 2)
                continue

            json_block = raw_content.strip()
            if json_block.startswith("```json"):
                json_block = json_block[len("```json"):].strip()
                if json_block.endswith("```"):
                    json_block = json_block[:-len("```")].strip()
            
            logger.debug(f"Attempting to parse JSON from LLM content for {problem_id_expected}: '{json_block[:200]}...'")
            evaluation_result = json.loads(json_block)
            logger.info(f"Successfully received and parsed LLM response for {problem_id_expected}.")
            return evaluation_result

        except requests.exceptions.Timeout:
            logger.warning(f"API Timeout (Attempt {attempt + 1}) for {problem_id_expected}.")
        except requests.exceptions.RequestException as e:
            logger.warning(f"API RequestException (Attempt {attempt + 1}) for {problem_id_expected}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.debug(f"    Response status: {e.response.status_code}. Content: {e.response.text[:500]}...")
        except json.JSONDecodeError as e:
            logger.warning(f"LLM Format Error (Attempt {attempt + 1}): Not valid JSON from model for {problem_id_expected}: {e}")
            logger.debug(f"    Raw LLM content received for {problem_id_expected}: '{json_block if json_block else raw_content if raw_content else 'N/A'}'")
        except Exception as e: # Catch any other unexpected error during API call/parsing
            logger.error(f"Unexpected error during API call/processing for {problem_id_expected} (Attempt {attempt+1}): {e}")

        if attempt < MAX_API_RETRIES_PER_CALL - 1:
            wait_time = (attempt + 1) * SECONDS_BETWEEN_REQUESTS * 2
            logger.info(f"Waiting for {wait_time}s before next API retry for {problem_id_expected}...")
            time.sleep(wait_time)
        else:
            logger.error(f"All {MAX_API_RETRIES_PER_CALL} API call attempts failed for problem: {problem_id_expected}.")
    return None


def validate_evaluation_structure(evaluation_json, expected_problem_id):
    logger.debug(f"Validating evaluation structure for expected_id: {expected_problem_id}. Received: {str(evaluation_json)[:300]}...")
    if not isinstance(evaluation_json, dict):
        logger.warning(f"Validation Error for {expected_problem_id}: Evaluation is not a dictionary. Got: {type(evaluation_json)}")
        return False
    
    returned_problem_id = evaluation_json.get("problem_id")
    if returned_problem_id != expected_problem_id:
        logger.warning(f"Validation Error for {expected_problem_id}: Mismatched 'problem_id'. Expected: '{expected_problem_id}', Got: '{returned_problem_id}'")
        return False

    for cat in EVALUATION_CATEGORIES:
        score = evaluation_json.get(cat)
        if score is None:
            logger.warning(f"Validation Error for {expected_problem_id}: Category '{cat}' is missing from evaluation.")
            return False
        
        is_valid_score = False
        if cat == "overall_correctness":
            if isinstance(score, (int, float)) and (0 <= score <= 10):
                is_valid_score = True
            else:
                logger.warning(f"Validation Error for {expected_problem_id}: Invalid score for '{cat}'. Expected 0-10, Got: {score} (type: {type(score).__name__})")
        else:
            if isinstance(score, (int, float)) and (1 <= score <= 5):
                is_valid_score = True
            else:
                logger.warning(f"Validation Error for {expected_problem_id}: Invalid score for '{cat}'. Expected 1-5, Got: {score} (type: {type(score).__name__})")
        
        if not is_valid_score:
            return False # Stop validation if any score is invalid
            
    logger.debug(f"Evaluation structure is valid for {expected_problem_id}.")
    return True

# --- Main Script ---
def main():
    # Setup logging first
    # Use logging.INFO for console for less noise, logging.DEBUG for file for all details
    setup_logging(LOG_FILE_NAME, console_level=logging.INFO, file_level=logging.DEBUG)

    logger.info("======================================================================")
    logger.info("Starting New Evaluation Script Run")
    logger.info("======================================================================")

    if not OPENROUTER_API_KEY:
        logger.critical("CRITICAL ERROR: OPENROUTER_API_KEY environment variable not set. Exiting.")
        return

    logger.info(f"--- Configuration ---")
    logger.info(f"OpenRouter Model: {OPENROUTER_MODEL}")
    logger.info(f"Manual Data File: {MANUAL_DATA_FILE}")
    logger.info(f"AI Generated File: {AI_GENERATED_FILE}")
    logger.info(f"Output Evaluations File: {OUTPUT_EVALUATION_FILE}")
    logger.info(f"Log File: {LOG_FILE_NAME}")
    logger.info(f"Max Evaluations to Perform: {MAX_EVALUATIONS_TO_PERFORM}")
    logger.info(f"Seconds Between Requests: {SECONDS_BETWEEN_REQUESTS}")
    logger.info(f"Max API Retries per Call: {MAX_API_RETRIES_PER_CALL}")
    logger.info(f"Max Validation Retries per Problem: {MAX_VALIDATION_RETRIES_PER_PROBLEM}")
    logger.info(f"Save Interval: {SAVE_INTERVAL}")
    logger.info(f"----------------------")

    try:
        with open(OUTPUT_EVALUATION_FILE, 'w', encoding='utf-8') as f:
            pass # Create/truncate the file
        logger.info(f"Initialized output file (truncated if existed): {OUTPUT_EVALUATION_FILE}")
    except IOError as e:
        logger.error(f"Error initializing output file {OUTPUT_EVALUATION_FILE}: {e}. Exiting.")
        return

    manual_data_list = load_jsonl(MANUAL_DATA_FILE)
    if not manual_data_list: 
        logger.error("Failed to load manual data. Exiting.")
        return
    manual_data_map = {item.get("Problem_ID"): item for item in manual_data_list if item.get("Problem_ID")}
    if not manual_data_map:
        logger.error("No 'Problem_ID' found in any loaded manual data items. Exiting.")
        return
    logger.info(f"Successfully loaded and mapped {len(manual_data_map)} manual solutions.")

    ai_solutions_list = load_jsonl(AI_GENERATED_FILE)
    if not ai_solutions_list: 
        logger.error("Failed to load AI-generated solutions. Exiting.")
        return
    logger.info(f"Successfully loaded {len(ai_solutions_list)} AI-generated solutions.")

    current_batch_evaluations = []
    total_successful_evaluations = 0
    problems_attempted_for_api_call = 0 # Counts problems for which an API call was made or attempted

    for idx, ai_solution_item in enumerate(ai_solutions_list):
        logger.debug(f"Processing AI solution item {idx+1}/{len(ai_solutions_list)}")
        if problems_attempted_for_api_call >= MAX_EVALUATIONS_TO_PERFORM:
            logger.info(f"Reached overall processing limit of {MAX_EVALUATIONS_TO_PERFORM} problems for API calls. Stopping.")
            break

        custom_id = ai_solution_item.get("custom_id")
        if not custom_id:
            logger.warning(f"AI solution item {idx+1} missing 'custom_id'. Skipping.")
            continue

        ai_content_str = None
        try:
            ai_content_str = ai_solution_item.get("response", {}).get("body", {}).get("choices", [{}])[0].get("message", {}).get("content")
        except (IndexError, AttributeError, TypeError) as e:
            logger.warning(f"Could not extract AI content for custom_id {custom_id} (item {idx+1}) due to structure issue: {e}. Skipping.")
            continue # Skip to next ai_solution_item
        
        if not ai_content_str or not ai_content_str.strip():
            logger.warning(f"AI content empty/not found for custom_id {custom_id} (item {idx+1}). Skipping.")
            continue

        if custom_id in manual_data_map:
            problems_attempted_for_api_call += 1
            manual_item = manual_data_map[custom_id]
            expected_problem_id = manual_item.get("Problem_ID")
            
            logger.info(f"--- Processing Matched Problem {problems_attempted_for_api_call}/{MAX_EVALUATIONS_TO_PERFORM} (Custom ID: {custom_id}) ---")
            logger.debug(f"Expected Problem ID for match: {expected_problem_id}")

            final_evaluation_for_problem = None
            for validation_attempt_num in range(MAX_VALIDATION_RETRIES_PER_PROBLEM + 1):
                if validation_attempt_num > 0:
                    logger.info(f"Retrying evaluation for {expected_problem_id} (Validation Attempt {validation_attempt_num}/{MAX_VALIDATION_RETRIES_PER_PROBLEM}).")
                    if SECONDS_BETWEEN_REQUESTS > 0: # Add a small delay before validation retry API call
                        time.sleep(max(1, SECONDS_BETWEEN_REQUESTS / 2)) 

                llm_json_output = call_openrouter_evaluator(
                    expected_problem_id,
                    manual_item.get("problem", "N/A_problem_text"),
                    manual_item.get("ground_truth", "N/A_ground_truth"),
                    manual_item.get("elaborated_solution_steps", "N/A_elaborated_steps"),
                    json.dumps(manual_item.get("final_answers_in_brief")) if isinstance(manual_item.get("final_answers_in_brief"), dict) else str(manual_item.get("final_answers_in_brief", "N/A_final_answers")),
                    ai_content_str
                )

                if llm_json_output is None:
                    logger.error(f"Failed to get any response from LLM for {expected_problem_id} after all API retries.")
                    # If API call itself failed, no point in further validation retries for this attempt.
                    # If this was the last validation_attempt_num, the loop will naturally exit.
                    if validation_attempt_num < MAX_VALIDATION_RETRIES_PER_PROBLEM:
                        logger.info(f"Proceeding to next validation attempt for {expected_problem_id} (if any).")
                        continue # Try another validation attempt which means another API call
                    else: # All validation retries exhausted, and last API call failed
                        logger.error(f"All validation retries also exhausted for {expected_problem_id} (last API call failed).")
                        break # Break from validation retry loop for this problem

                if validate_evaluation_structure(llm_json_output, expected_problem_id):
                    final_evaluation_for_problem = llm_json_output
                    logger.info(f"Successfully evaluated and validated Problem ID: {expected_problem_id}")
                    break # Successful validation, exit validation retry loop
                else:
                    logger.warning(f"Evaluation from LLM failed validation for {expected_problem_id} on validation attempt {validation_attempt_num+1}.")
                    if validation_attempt_num == MAX_VALIDATION_RETRIES_PER_PROBLEM:
                        logger.error(f"All {MAX_VALIDATION_RETRIES_PER_PROBLEM + 1} validation attempts failed for {expected_problem_id}.")
                        try:
                            with open("failed_llm_evaluations.jsonl", "a", encoding="utf-8") as err_f:
                                err_f.write(json.dumps({"expected_id": expected_problem_id, "llm_output": llm_json_output, "attempt": validation_attempt_num+1}) + "\n")
                            logger.info(f"Logged problematic LLM output for {expected_problem_id} to failed_llm_evaluations.jsonl")
                        except Exception as e_log:
                            logger.error(f"Could not log problematic LLM output: {e_log}")
                        break # Exhausted validation retries

            if final_evaluation_for_problem:
                current_batch_evaluations.append(final_evaluation_for_problem)
                total_successful_evaluations += 1
                logger.debug(f"Added evaluation for {expected_problem_id} to current batch. Batch size: {len(current_batch_evaluations)}")

                if len(current_batch_evaluations) >= SAVE_INTERVAL:
                    save_evaluations_batch(current_batch_evaluations, OUTPUT_EVALUATION_FILE)
                    current_batch_evaluations = [] # Reset batch
            
            # API Rate Limit Delay (only if we are not at the very end of processing limit)
            if SECONDS_BETWEEN_REQUESTS > 0 and problems_attempted_for_api_call < MAX_EVALUATIONS_TO_PERFORM :
                logger.info(f"Waiting for {SECONDS_BETWEEN_REQUESTS}s before processing next problem...")
                time.sleep(SECONDS_BETWEEN_REQUESTS)
        else:
            logger.info(f"AI solution custom_id '{custom_id}' (item {idx+1}) not found in manual data map. Skipping.")

    # Final save for any remaining evaluations in the batch
    if current_batch_evaluations:
        logger.info("Saving remaining evaluations in the final batch.")
        save_evaluations_batch(current_batch_evaluations, OUTPUT_EVALUATION_FILE)

    logger.info(f"\n--- Final Evaluation Summary ---")
    logger.info(f"Total AI solutions loaded: {len(ai_solutions_list) if ai_solutions_list else 0}")
    logger.info(f"Manual solutions successfully mapped: {len(manual_data_map)}")
    logger.info(f"Problems matched and for which API calls were attempted: {problems_attempted_for_api_call}")
    logger.info(f"Total successfully validated evaluations written to file: {total_successful_evaluations}.")
    logger.info(f"Output evaluations file: {OUTPUT_EVALUATION_FILE}")
    logger.info(f"Log file: {LOG_FILE_NAME}")
    logger.info(f"Problematic LLM outputs (if any) logged to: failed_llm_evaluations.jsonl")
    logger.info("--- Script Run Finished ---")

if __name__ == "__main__":
    main()