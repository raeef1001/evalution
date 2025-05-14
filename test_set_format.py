import json

def extract_solution_data(input_file, output_file):
    try:
        # Read the input JSON file
        with open(input_file, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
        
        # Extract required fields from each problem
        extracted_data = []
        for problem in data:
            extracted_problem = {
                "Problem_ID": problem.get("Problem_ID", ""),
                "elaborated_solution_steps": problem.get("elaborated_solution_steps", ""),
                "ground_truth": problem.get("ground_truth", "")
            }
            extracted_data.append(extracted_problem)
        
        # Write to output JSON file
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(extracted_data, outfile, indent=4, ensure_ascii=False)
            
        print(f"Successfully extracted data to {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file - {e}")
    except Exception as e:
        print(f"Error: An unexpected error occurred - {e}")

# Use the function
input_file = 'Test Set.json'
output_file = 'extracted_solutions_test_set.json'
extract_solution_data(input_file, output_file)