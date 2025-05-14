import json

def extract_fields(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            if line.strip():  # Skip empty lines
                try:
                    # Parse the JSON line
                    data = json.loads(line)
                    
                    # Extract custom_id and content
                    custom_id = data.get('custom_id', '')
                    content = data.get('response', {}).get('body', {}).get('choices', [{}])[0].get('message', {}).get('content', '')
                    
                    # Create new dictionary with just these fields
                    new_data = {
                        'custom_id': custom_id,
                        'content': content
                    }
                    
                    # Write to output file
                    json.dump(new_data, outfile, ensure_ascii=False)
                    outfile.write('\n')
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON line: {e}")
                except Exception as e:
                    print(f"Error processing line: {e}")

# Use the function
input_file = 'openai2.jsonl'
output_file = 'extracted_data.jsonl'
extract_fields(input_file, output_file)