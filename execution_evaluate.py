# execution_evaluate.py
# This script performs execution-based evaluation of SQL queries.
# It uses an in-memory SQLite database to execute generated and ground truth SQL queries,
# compares their results, and logs outcomes to Langfuse and a local JSONL file.

import os
import json
import sqlite3
import re
from langfuse import Langfuse
from langfuse.model import CreateTrace, CreateGeneration, CreateScore, CreateEvent

# --- Configuration ---
INPUT_FILE = "prepared_test_data.jsonl"
OUTPUT_RESULTS_FILE = "execution_results.jsonl" # Different from llm_evaluate output

MODEL_UNDER_TEST_NAME = os.getenv("MODEL_UNDER_TEST_NAME", "dummy-text-to-sql-model-execution")

# --- Langfuse Initialization ---
langfuse_client = None
LANGFUSE_ENABLED = False

def initialize_langfuse():
    """Initializes the Langfuse client."""
    global langfuse_client, LANGFUSE_ENABLED
    if langfuse_client is None:
        try:
            public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
            secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
            if not public_key or not secret_key:
                print("Warning: Langfuse keys not found. Logging disabled.")
                LANGFUSE_ENABLED = False
                return
            langfuse_client = Langfuse()
            langfuse_client.auth_check()
            LANGFUSE_ENABLED = True
            print("Langfuse initialized successfully and authentication verified.")
        except Exception as e:
            print(f"Error initializing Langfuse: {e}. Logging disabled.")
            langfuse_client = None
            LANGFUSE_ENABLED = False
    return langfuse_client

# --- Helper Functions ---
def load_prepared_data(filepath: str) -> list:
    """Loads data from a JSONL file."""
    data = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        print(f"Successfully loaded {len(data)} records from {filepath}")
    except FileNotFoundError:
        print(f"Error: Input file {filepath} not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON in {filepath}.")
    except Exception as e:
        print(f"An unexpected error occurred while loading {filepath}: {e}")
    return data

def save_results_to_jsonl(filepath: str, result_data: dict):
    """Appends a result dictionary to a JSONL file."""
    try:
        with open(filepath, 'a') as f:
            f.write(json.dumps(result_data) + '\n')
    except IOError as e:
        print(f"Error writing result to {filepath}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving results: {e}")

# --- Model-Under-Test (Placeholder - consistent with llm_evaluate.py) ---
def get_generated_sql_from_model(sql_prompt: str, sql_context: str, ground_truth_sql: str = None) -> str:
    """
    Placeholder for calling the user's fine-tuned Text-to-SQL model.
    For now, it returns a dummy SQL string or the ground truth SQL for testing.
    """
    # Option 1: Return a fixed dummy SQL for basic pipeline testing
    # return "SELECT COUNT(*) FROM users WHERE country = 'USA';"

    # Option 2: Return the ground truth SQL (useful for testing the evaluation part)
    if ground_truth_sql:
        # Introduce a common error for testing:
        # if "customers" in ground_truth_sql.lower() and "ORDER BY" not in ground_truth_sql.upper():
        #     return ground_truth_sql.replace(";", " ORDER BY name;") # Simulate model adding an ORDER BY
        return ground_truth_sql
    
    if "customers" in sql_prompt.lower():
        return f"SELECT name, email FROM customers WHERE {sql_prompt.split(' ')[-1]} = 'some_value';"
    else:
        return "SELECT id FROM some_table LIMIT 10;"

# --- Database Operations ---
def split_sql_statements(sql_script: str) -> list[str]:
    """
    Splits a potentially multi-statement SQL script into individual statements.
    Handles simple cases; might need refinement for complex SQL with embedded semicolons in strings.
    """
    if not sql_script:
        return []
    # Remove comments first (simple block and line comments)
    sql_script = re.sub(r"/\*.*?\*/", "", sql_script, flags=re.DOTALL)
    sql_script = re.sub(r"--.*?\n", "", sql_script)
    sql_script = re.sub(r"#.*?\n", "", sql_script) # MySQL style comments

    statements = sql_script.split(';')
    # Filter out empty strings that may result from splitting
    return [stmt.strip() for stmt in statements if stmt.strip()]


def setup_in_memory_db(sql_context: str) -> tuple[sqlite3.Connection | None, str | None]:
    """
    Creates an in-memory SQLite database and populates it using the sql_context.
    Returns the connection object and an error message if any.
    """
    conn = None
    try:
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        
        # Split sql_context into individual statements
        # The context might contain CREATE TABLE and INSERT INTO statements.
        statements = split_sql_statements(sql_context)
        if not statements:
            return conn, "SQL context was empty or contained no valid statements after splitting."

        for stmt in statements:
            try:
                cursor.execute(stmt)
            except sqlite3.Error as e:
                # This error is specific to a statement within the context
                error_msg = f"Error executing statement from sql_context: '{stmt[:100]}...'. Error: {e}"
                # We might want to close the connection if a critical setup step fails
                if conn: conn.close()
                return None, error_msg
        
        conn.commit() # Not strictly necessary for :memory: for data persistence, but good practice.
        return conn, None
    except sqlite3.Error as e:
        if conn: conn.close() # Ensure connection is closed on setup failure
        return None, f"Failed to set up in-memory database: {e}"
    except Exception as e: # Catch other unexpected errors during setup
        if conn: conn.close()
        return None, f"Unexpected error during DB setup: {e}"


def execute_sql_query(conn: sqlite3.Connection, query: str) -> tuple[list | None, str | None]:
    """
    Executes a given SQL query (SELECT) on the provided database connection.
    Returns a list of results or an error message.
    """
    if not query or not isinstance(query, str) or query.strip() == "":
        return None, "SQL query is empty or not a string."
        
    results = None
    error_msg = None
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        # For SELECT queries, fetch results. For DML/DDL, this would be empty.
        # We assume evaluation queries are SELECTs.
        results = cursor.fetchall() 
    except sqlite3.Error as e:
        error_msg = str(e)
    except Exception as e: # Catch other unexpected errors
        error_msg = f"Unexpected error during query execution: {str(e)}"
    return results, error_msg


# --- Result Comparison ---
def compare_execution_results(results1: list | None, results2: list | None) -> bool:
    """
    Compares two sets of SQL execution results.
    Handles potential differences in row/column ordering.
    Converts all data to strings for consistent comparison.
    """
    if results1 is None and results2 is None:
        return True # Both failed to execute or returned None consistently
    if results1 is None or results2 is None:
        return False # One failed, the other didn't (or one returned None, other data)

    if not isinstance(results1, list) or not isinstance(results2, list):
        # This case implies one of the results is an error string or unexpected type
        return str(results1) == str(results2) 

    if len(results1) != len(results2):
        return False

    if not results1: # Both are empty lists
        return True

    # Convert all elements to strings and sort rows to handle order invariance.
    # Each row is a tuple. Convert tuple elements to string, then sort the list of these string-tuples.
    try:
        # Ensure all elements within tuples are converted to strings
        str_results1 = sorted([tuple(map(str, row)) for row in results1])
        str_results2 = sorted([tuple(map(str, row)) for row in results2])
    except TypeError as e:
        # This can happen if a row is not iterable (e.g., if an error message sneaked in as a result)
        print(f"Type error during result conversion/sorting: {e}. Results1: {results1}, Results2: {results2}")
        return False # Treat as mismatch if conversion fails

    return str_results1 == str_results2


# --- Main Evaluation Loop ---
def main():
    initialize_langfuse()
    
    dataset = load_prepared_data(INPUT_FILE)
    if not dataset:
        print("Exiting due to inability to load dataset.")
        return

    if os.path.exists(OUTPUT_RESULTS_FILE):
        print(f"Warning: Output file {OUTPUT_RESULTS_FILE} already exists. Results will be appended.")

    total_items = len(dataset)
    for i, item in enumerate(dataset):
        item_id = item.get("id", f"item-{i+1}")
        print(f"\n--- Evaluating item {i+1} of {total_items} (ID: {item_id}) ---")
        
        sql_prompt = item.get("sql_prompt")
        sql_context = item.get("sql_context")
        ground_truth_sql_from_data = item.get("sql") # This is 'sql' from the dataset

        if not all([sql_prompt, sql_context, ground_truth_sql_from_data]):
            print(f"Skipping item {item_id} due to missing critical data (prompt, context, or ground truth SQL).")
            save_results_to_jsonl(OUTPUT_RESULTS_FILE, {
                "item_id": item_id, "error": "Missing critical data", "langfuse_trace_id": None
            })
            continue

        current_trace = None
        if LANGFUSE_ENABLED and langfuse_client:
            try:
                current_trace = langfuse_client.trace(
                    CreateTrace(
                        id=f"exec-{item_id}", # Prefix to distinguish from LLM eval traces
                        name=f"SQL Execution Eval - {item_id}",
                        user_id="execution_eval_script_user",
                        metadata={"item_id": item_id, "sql_prompt_snippet": sql_prompt[:100]},
                        tags=["text-to-sql", "execution-evaluation"]
                    )
                )
            except Exception as e:
                print(f"Langfuse: Error creating trace for item {item_id}. {e}")
        
        # 1. Database Setup
        db_conn, db_setup_error = setup_in_memory_db(sql_context)
        if db_setup_error or not db_conn:
            print(f"Database setup failed for item {item_id}: {db_setup_error}")
            if current_trace:
                current_trace.event(CreateEvent(name="db_setup_failure", metadata={"error": db_setup_error}))
            save_results_to_jsonl(OUTPUT_RESULTS_FILE, {
                "item_id": item_id, "sql_prompt": sql_prompt, "error_db_setup": db_setup_error, "langfuse_trace_id": current_trace.id if current_trace else None
            })
            if db_conn: db_conn.close() # Ensure closed if partially opened
            continue
        print("In-memory database setup complete.")

        # 2. Get Generated SQL from Model-Under-Test
        model_gen_output = None
        if current_trace:
            model_under_test_generation = current_trace.generation(
                CreateGeneration(
                    name="model-under-test-sql-generation-exec",
                    model=MODEL_UNDER_TEST_NAME,
                    input={"sql_prompt": sql_prompt, "sql_context_snippet": sql_context[:200]},
                )
            )
        
        generated_sql = get_generated_sql_from_model(sql_prompt, sql_context, ground_truth_sql_from_data)
        print(f"Generated SQL (from placeholder): {generated_sql}")

        if current_trace and model_under_test_generation:
            model_under_test_generation.end(output={"generated_sql": generated_sql})
            model_gen_output = {"generated_sql": generated_sql}


        # 3. Execute Generated SQL
        gen_sql_results, gen_sql_error = execute_sql_query(db_conn, generated_sql)
        gen_sql_exec_success = gen_sql_error is None
        
        print(f"Generated SQL Execution Success: {gen_sql_exec_success}")
        if gen_sql_error: print(f"  Error: {gen_sql_error}")
        # else: print(f"  Results: {gen_sql_results}") # Can be verbose

        if current_trace:
            current_trace.score(CreateScore(name="generated_sql_execution_success", value=(1 if gen_sql_exec_success else 0)))
            if gen_sql_error:
                current_trace.score(CreateScore(name="generated_sql_error_message", value=0, comment=str(gen_sql_error)[:1000])) # Score is 0, comment has error
                # Or use an event for the error message
                # current_trace.event(CreateEvent(name="generated_sql_execution_error", metadata={"error": str(gen_sql_error)[:1000]}))

        # 4. Execute Ground Truth SQL
        gt_sql_results, gt_sql_error = execute_sql_query(db_conn, ground_truth_sql_from_data)
        gt_sql_exec_success = gt_sql_error is None

        print(f"Ground Truth SQL Execution Success: {gt_sql_exec_success}")
        if gt_sql_error: print(f"  Error: {gt_sql_error}")
        # else: print(f"  Results: {gt_sql_results}") # Can be verbose

        if current_trace:
            current_trace.score(CreateScore(name="ground_truth_sql_execution_success", value=(1 if gt_sql_exec_success else 0)))
            if gt_sql_error:
                 current_trace.score(CreateScore(name="ground_truth_sql_error_message", value=0, comment=str(gt_sql_error)[:1000]))

        # 5. Compare Results
        results_are_match = False
        if not gen_sql_exec_success or not gt_sql_exec_success:
            print("Cannot compare results as one or both queries failed to execute.")
            results_are_match = False # If one failed, they don't match in terms of successful output
        else:
            results_are_match = compare_execution_results(gen_sql_results, gt_sql_results)
        
        print(f"Execution Results Match: {results_are_match}")
        if current_trace:
            current_trace.score(CreateScore(name="results_match", value=(1 if results_are_match else 0)))

        # 6. Database Teardown
        db_conn.close()
        print("In-memory database closed.")

        # 7. Store Detailed Results
        result_to_save = {
            "item_id": item_id,
            "sql_prompt": sql_prompt,
            "sql_context_snippet": sql_context[:200] + "...",
            "model_under_test_name": MODEL_UNDER_TEST_NAME,
            "generated_sql": generated_sql,
            "ground_truth_sql": ground_truth_sql_from_data,
            "generated_sql_execution_success": gen_sql_exec_success,
            "generated_sql_error": gen_sql_error,
            "generated_sql_output": json.dumps(gen_sql_results) if gen_sql_exec_success else None, # Serialize list of tuples
            "ground_truth_sql_execution_success": gt_sql_exec_success,
            "ground_truth_sql_error": gt_sql_error,
            "ground_truth_sql_output": json.dumps(gt_sql_results) if gt_sql_exec_success else None,
            "results_match": results_are_match,
            "langfuse_trace_id": current_trace.id if current_trace else None
        }
        save_results_to_jsonl(OUTPUT_RESULTS_FILE, result_to_save)

        if LANGFUSE_ENABLED and langfuse_client:
            langfuse_client.flush()

    print(f"\n--- Execution Evaluation Complete ---")
    print(f"Detailed results saved to: {OUTPUT_RESULTS_FILE}")
    if LANGFUSE_ENABLED:
        print(f"Langfuse logging was enabled. Check your Langfuse project for traces.")
    else:
        print(f"Langfuse logging was disabled.")

if __name__ == "__main__":
    main()
```
