import json
import jsonlines
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def combine_data(solutions_file, data_file):
    # Dictionary to store solutions data
    solutions = {}
    
    # Read solutions file
    try:
        with open(solutions_file, 'r', encoding='utf-8') as f:
            solutions_data = json.load(f)
            for item in solutions_data:
                solutions[item['Problem_ID']] = item
        logger.info(f"Loaded {len(solutions)} items from solutions file")
    except Exception as e:
        logger.error(f"Error reading solutions file: {e}")
        return []
            
    # Process jsonlines data file and combine with solutions
    combined_data = []
    valid_lines = 0
    invalid_lines = 0
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():  # Skip empty lines
                    continue
                try:
                    item = json.loads(line)
                    if item.get('custom_id') in solutions:
                        solution = solutions[item['custom_id']]
                        combined = {
                            'Problem_ID': item['custom_id'],
                            'openai_solution': item.get('content', ''),
                            'ground_truth': solution.get('ground_truth', ''),
                            'elaborated_solution': solution.get('elaborated_solution_steps', '')
                        }
                        combined_data.append(combined)
                        valid_lines += 1
                except json.JSONDecodeError as e:
                    invalid_lines += 1
                    logger.warning(f"Invalid JSON at line {line_num}: {e}")
                    continue
                
        logger.info(f"Processed {valid_lines} valid lines and {invalid_lines} invalid lines")
                
    except Exception as e:
        logger.error(f"Error processing data file: {e}")
        return []
                
    return combined_data

def main():
    solutions_file = 'extracted_solutions_test_set.json'
    data_file = 'extracted_data_openai.jsonl'
    
    # Combine data from both files
    combined = combine_data(solutions_file, data_file)
    
    if combined:
        # Write combined data to output file
        try:
            with open('combined_data.json', 'w', encoding='utf-8') as f:
                json.dump(combined, f, indent=2, ensure_ascii=False)
            logger.info(f"Successfully wrote {len(combined)} items to combined_data.json")
        except Exception as e:
            logger.error(f"Error writing output file: {e}")
    else:
        logger.error("No data to write to output file")

if __name__ == '__main__':
    main()