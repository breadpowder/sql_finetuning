# prepare_dataset.py
# This script downloads the 'gretelai/synthetic_text_to_sql' dataset (test split)
# from Hugging Face and prepares it by extracting relevant fields into a JSONL file.

import json
from datasets import load_dataset

# Define the output file name
OUTPUT_FILE = "prepared_test_data.jsonl"
DATASET_NAME = "gretelai/synthetic_text_to_sql"
DATASET_SPLIT = "test"

def load_and_prepare_data():
    """
    Loads the 'gretelai/synthetic_text_to_sql' test split, extracts relevant fields,
    and saves them to a JSONL file.
    """
    print(f"Attempting to load dataset: {DATASET_NAME}, split: {DATASET_SPLIT}")
    try:
        # Load the test split of the dataset
        dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False

    print(f"Processing dataset and writing to {OUTPUT_FILE}...")
    records_processed = 0
    try:
        with open(OUTPUT_FILE, 'w') as outfile:
            # Iterate through each record in the dataset
            for record in dataset:
                # Extract the required fields
                extracted_data = {
                    "id": record.get("id"),
                    "sql_prompt": record.get("sql_prompt"),
                    "sql_context": record.get("sql_context"),
                    "sql": record.get("sql")  # Ground truth SQL
                }
                
                # Convert the dictionary to a JSON string and write it as a new line
                outfile.write(json.dumps(extracted_data) + '\n')
                records_processed += 1
        
        print(f"Successfully processed {records_processed} records.")
        print(f"Data saved to {OUTPUT_FILE}")
        return True
        
    except IOError as e:
        print(f"Error writing to file {OUTPUT_FILE}: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")
        return False

if __name__ == "__main__":
    print("--- Starting Dataset Preparation ---")
    success = load_and_prepare_data()
    if success:
        print("--- Dataset Preparation Completed Successfully ---")
    else:
        print("--- Dataset Preparation Failed ---")
