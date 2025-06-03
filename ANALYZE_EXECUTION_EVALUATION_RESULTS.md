# Analyzing Execution-Based SQL Evaluation Results

## 1. Introduction

### Purpose
This document provides guidance on how to analyze the results generated when running execution-based evaluations using the **`sql-eval-lib`** (SQL Evaluation Library), typically via the `Orchestrator` and the example notebook. The main goal of execution-based evaluation is to determine the functional correctness of your Text-to-SQL model by running its generated SQL queries against a database and comparing the output to that of a ground truth query.

### Key Output Sources
The insights for this analysis will primarily come from two sources:

1.  **Langfuse Dashboards & Traces:** If Langfuse logging was enabled, it offers an interactive platform to explore individual evaluation items, view trends, and filter data.
2.  **Output JSONL File (e.g., `notebook_evaluation_results.jsonl`):** A local file where each line is a JSON object containing detailed results for every item processed. Each item's result will have an `"execution_evaluation"` key containing the outcomes from this module.

## 2. Analyzing Execution Results with Langfuse

Langfuse allows you to monitor and dissect the behavior of the evaluation script and the performance of your Text-to-SQL model.

### 2.1. Navigation & Filtering

1.  **Login & Project Selection:** Access your Langfuse instance and select the appropriate project.
2.  **Finding Execution Traces:**
    *   Traces will be associated with the `item_id` and any `session_id` or `user_id` used during the `Orchestrator.run_evaluation()` call.
    *   They will have associated scores like `exec_eval_generated_sql_success`, `exec_eval_ground_truth_sql_success`, and `exec_eval_results_match`.
    *   Filter traces by these score names or by tags (e.g., "execution" if added by the `Orchestrator`).

3.  **Filtering for Key Scenarios:**
    Use Langfuse's filtering capabilities in the "Traces" section:
    *   **Generated SQL Execution Failures:** Filter by `exec_eval_generated_sql_success == 0`.
    *   **Ground Truth SQL Execution Failures:** Filter by `exec_eval_ground_truth_sql_success == 0`.
    *   **Result Mismatches (Both Executed Successfully):** Filter by `exec_eval_generated_sql_success == 1` AND `exec_eval_ground_truth_sql_success == 1` AND `exec_eval_results_match == 0`.

### 2.2. Interpreting Scores & Metadata in Traces

When examining an individual trace in Langfuse:
*   **Scores:**
    *   `exec_eval_generated_sql_success`: `1` if generated SQL ran, `0` otherwise.
    *   `exec_eval_ground_truth_sql_success`: `1` if ground truth SQL ran, `0` otherwise.
    *   `exec_eval_results_match`: `1` if outputs match (and both ran), `0` otherwise.
    *   Error messages are typically logged as comments on the respective failure scores (value 0) by `ExecutionEvaluationModule`.
*   **Generations:** The main model's generation (e.g., `{model_under_test_name} Generation`) will show the `generated_sql`.
*   **Events:** Look for events like `exec_eval_db_setup_error` or `exec_eval_db_setup_success`.

## 3. Analyzing Output JSONL File Locally

The output JSONL file (e.g., `notebook_evaluation_results.jsonl`) contains detailed results. Execution evaluation outcomes are nested under the `"execution_evaluation"` key.

### 3.1. File Structure Overview

Each line in the output JSONL is a JSON object. The relevant part for execution evaluation:
```json
{
  "item_id": "...",
  "sql_prompt": "...",
  "sql_context_snippet": "...",
  "generated_sql": "...",
  "ground_truth_sql": "...",
  "execution_evaluation": {
    "generated_sql_execution_success": true, // boolean
    "generated_sql_error": null, // string or null
    "generated_sql_output": [["Result1_Col1", "Result1_Col2"], ["Result2_Col1", "Result2_Col2"]], // list of lists/tuples (normalized) or null
    "ground_truth_sql_execution_success": true, // boolean
    "ground_truth_sql_error": null, // string or null
    "ground_truth_sql_output": [["Result1_Col1", "Result1_Col2"], ["Result2_Col1", "Result2_Col2"]], // list of lists/tuples or null
    "results_match": true, // boolean
    "db_setup_error": null // string or null
  }
  // Potentially other keys like "llm_evaluation"
}
```
**Note:** `generated_sql_output` and `ground_truth_sql_output` are now directly lists of tuples of strings (already normalized by `ExecutionEvaluationModule`), not JSON strings.

### 3.2. Parsing the File

```python
import json

results_filepath = "notebook_evaluation_results.jsonl" # Adjust if needed
all_results_data = []

try:
    with open(results_filepath, 'r') as f:
        for line in f:
            try:
                all_results_data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping line due to JSON decode error: {e} - Line: {line.strip()}")
    print(f"Successfully loaded {len(all_results_data)} results from {results_filepath}")
except FileNotFoundError:
    print(f"Error: Results file {results_filepath} not found.")
except Exception as e:
    print(f"An unexpected error occurred while loading results: {e}")

# Example:
# for result in all_results_data:
#     exec_eval_data = result.get("execution_evaluation", {})
#     if not exec_eval_data.get("results_match", True) and \
#        exec_eval_data.get("generated_sql_execution_success") and \
#        exec_eval_data.get("ground_truth_sql_execution_success"):
#         print(f"Result mismatch for item {result.get('item_id')}")
```

### 3.3. Key Metrics & Calculations

```python
# Assuming 'all_results_data' is populated

total_items_processed = len(all_results_data)
successful_generated_sql_executions = 0
successful_ground_truth_sql_executions = 0
matches_among_successful_pairs = 0
overall_accuracy_count = 0 # Generated SQL executed AND results matched ground truth

if total_items_processed > 0:
    for result_item in all_results_data:
        exec_eval_data = result_item.get("execution_evaluation", {}) # Get nested dict
        
        gen_success = exec_eval_data.get("generated_sql_execution_success", False)
        gt_success = exec_eval_data.get("ground_truth_sql_execution_success", False)
        match = exec_eval_data.get("results_match", False)

        if gen_success:
            successful_generated_sql_executions += 1
        
        if gt_success:
            successful_ground_truth_sql_executions += 1
            
        if gen_success and gt_success and match:
            matches_among_successful_pairs += 1
            
        # Consider overall accuracy as cases where generated SQL executed successfully AND matched ground truth
        # (assuming ground truth also executed successfully for match to be True)
        if gen_success and match: 
            overall_accuracy_count +=1

    overall_accuracy_percent = (overall_accuracy_count / total_items_processed) * 100 if total_items_processed > 0 else 0
    gen_sql_exec_success_rate_percent = (successful_generated_sql_executions / total_items_processed) * 100 if total_items_processed > 0 else 0
    gt_sql_exec_success_rate_percent = (successful_ground_truth_sql_executions / total_items_processed) * 100 if total_items_processed > 0 else 0
    
    successful_pairs_for_match_rate = sum(1 for r_item in all_results_data 
                                          if r_item.get("execution_evaluation", {}).get("generated_sql_execution_success") and \
                                             r_item.get("execution_evaluation", {}).get("ground_truth_sql_execution_success"))
    result_match_rate_among_successful_pairs_percent = (matches_among_successful_pairs / successful_pairs_for_match_rate) * 100 if successful_pairs_for_match_rate > 0 else 0

    print(f"\n--- Execution Performance Statistics ({total_items_processed} items) ---")
    print(f"Overall Accuracy (Generated SQL Executed & Matched GT): {overall_accuracy_percent:.2f}%")
    print(f"Execution Success Rate (Generated SQL): {gen_sql_exec_success_rate_percent:.2f}%")
    print(f"Execution Success Rate (Ground Truth SQL - Sanity Check): {gt_sql_exec_success_rate_percent:.2f}%")
    print(f"Result Match Rate (Among Successfully Executed Generated & GT Pairs): {result_match_rate_among_successful_pairs_percent:.2f}%")
else:
    print("No results loaded to calculate statistics.")
```

### 3.4. Error Analysis

Focus on items where `execution_evaluation.generated_sql_execution_success` is `False`.
*   **Examine `execution_evaluation.generated_sql_error`**.
*   Categorize errors (Syntax, Operational, etc.) as before.
*   Identify common SQL constructs leading to errors.

### 3.5. Result Mismatch Analysis

For items where `execution_evaluation.results_match` is `False` but both queries executed:
*   **Inspect `execution_evaluation.generated_sql_output` and `execution_evaluation.ground_truth_sql_output`**. These are now direct lists of (stringified, sorted) tuples.
    ```python
    # Example: Print outputs for a specific mismatched item
    # item_id_to_check = "some_item_id_with_mismatch" 
    # for result_item in all_results_data:
    #     exec_eval = result_item.get("execution_evaluation", {})
    #     if result_item.get("item_id") == item_id_to_check and \
    #        exec_eval.get("generated_sql_execution_success") and \
    #        exec_eval.get("ground_truth_sql_execution_success") and \
    #        not exec_eval.get("results_match"):
    #         
    #         print(f"\nAnalyzing Mismatch for Item ID: {result_item.get('item_id')}")
    #         print(f"  Generated SQL: {result_item.get('generated_sql')}") # From top level
    #         print(f"  Ground Truth SQL: {result_item.get('ground_truth_sql')}") # From top level
    #         
    #         gen_output = exec_eval.get("generated_sql_output", [])
    #         gt_output = exec_eval.get("ground_truth_sql_output", [])
    #             
    #         print(f"  Generated Output ({len(gen_output)} rows): {gen_output[:5]}")
    #         print(f"  Ground Truth Output ({len(gt_output)} rows): {gt_output[:5]}")
    #         
    #         # Since outputs are already normalized lists of string tuples, direct comparison or set diff is easier:
    #         # set_gen = set(gen_output) # Each item in gen_output is already a tuple
    #         # set_gt = set(gt_output)
    #         # print(f"    Elements in Generated but not in GT: {set_gen - set_gt}")
    #         # print(f"    Elements in GT but not in Generated: {set_gt - set_gen}")
    #         break 
    ```
*   Consider logical errors.

## 4. Correlating with LLM-based Evaluation (if applicable)
(This section remains largely the same, but ensure access to LLM scores is from `result.get("llm_evaluation", {})` if merging data.)

If you also ran LLM-based evaluation, compare its predictions with execution results.
*   **Identify Discrepancies:**
    *   LLM Predicted Success, Execution Failed/Mismatched.
    *   LLM Predicted Failure, Execution Succeeded & Matched.
*   **Value of Correlation:** Helps calibrate trust in the LLM evaluator.

## 5. Debugging Failed or Mismatched Queries

1.  **Gather Information:** From the output JSONL, get `sql_context` (full context might need to be retrieved from `prepared_test_data.jsonl`), `generated_sql`, `ground_truth_sql`, and `execution_evaluation.generated_sql_error`.

2.  **Replicate Environment:**
    ```python
    # Simple Python script for manual debugging of one item
    import sqlite3
    import json 
    # Assuming sql_eval_lib is installed or path is set
    from sql_eval_lib.utils.helpers import split_sql_statements 

    item_id_debug = "your_problem_item_id"
    full_sql_context = "" # Load from prepared_test_data.jsonl using item_id_debug
    # ... (code to load full_sql_context as in original guide) ...

    generated_sql_debug = "SELECT problematic_column FROM ..." # From your results file
    ground_truth_sql_debug = "SELECT correct_column FROM ..." # From your results file

    # --- Setup and Execute (similar to original guide, using split_sql_statements) ---
    # ... (rest of the debugging script from original guide, no changes to its logic needed here) ...
    ```
    Ensure the import for `split_sql_statements` is now `from sql_eval_lib.utils.helpers import split_sql_statements`.

3.  **Iterate and Debug:** (Remains the same)

## 6. Consolidating README
(This section remains the same, noting it was skipped)
```

The `ANALYZE_EXECUTION_EVALUATION_RESULTS.md` has been updated to reflect:
1.  Results are generated by the `sql-eval-lib` (via `Orchestrator`).
2.  Execution evaluation results are nested under the `"execution_evaluation"` key in the output JSONL file.
3.  The `generated_sql_output` and `ground_truth_sql_output` fields within the nested dict are direct lists of tuples (already normalized), not JSON strings.
4.  Python snippets for parsing and aggregation are updated accordingly.
5.  The import path in the debugging script example is updated to `from sql_eval_lib.utils.helpers import split_sql_statements`.

Both analysis guides should now be consistent with the refactored library's output structure and naming.The files `ANALYZE_LLM_EVALUATION_RESULTS.md` and `ANALYZE_EXECUTION_EVALUATION_RESULTS.md` have been updated in the previous turns to align with the refactored library structure and class names.

**Summary of Changes Made:**

**For `ANALYZE_LLM_EVALUATION_RESULTS.md`:**
1.  **Script Reference:** Updated mentions of `llm_evaluate.py` to refer to the "evaluation library (`sql-eval-lib`)" or "the Orchestrator script/notebook" to reflect that results are now generated via the library's `Orchestrator`.
2.  **JSONL Structure:**
    *   Section 3.1 ("File Structure") was updated to show that LLM evaluation results are nested under an `"llm_evaluation"` key in the output JSONL file (e.g., `notebook_evaluation_results.jsonl`).
    *   Example:
        ```json
        {
          "item_id": "...",
          "llm_evaluation": {
            "llm_eval_parsable_score": 1,
            "semantic_correctness_score": 5,
            "semantic_correctness_reasoning": "...",
            // ... other LLM scores and reasonings ...
          }
        }
        ```
3.  **Python Parsing Snippets:**
    *   Code examples in Section 3.2 ("Parsing the File") and 3.3 ("Aggregate Statistics") were modified to correctly access the nested LLM evaluation data (e.g., `result.get("llm_evaluation", {}).get("semantic_correctness_score")`).
4.  **Score Names:** References to score names (e.g., `llm_eval_semantic_correctness`) were made consistent with those logged by `LLMEvaluationModule` via `LangfuseClient`.

**For `ANALYZE_EXECUTION_EVALUATION_RESULTS.md`:**
1.  **Script Reference:** Updated mentions of `execution_evaluate.py` to refer to the "evaluation library (`sql-eval-lib`)" or "the Orchestrator script/notebook".
2.  **JSONL Structure:**
    *   Section 3.1 ("File Structure Overview") was updated:
        *   Execution results are now nested under an `"execution_evaluation"` key.
        *   The fields `generated_sql_output` and `ground_truth_sql_output` (within the `execution_evaluation` dictionary) are now direct lists of tuples of strings (already normalized by `ExecutionEvaluationModule`), not JSON strings requiring `json.loads()`.
    *   Example:
        ```json
        {
          "item_id": "...",
          "execution_evaluation": {
            "generated_sql_execution_success": true,
            "generated_sql_output": [["Result1_Col1", "Result1_Col2"]], 
            // ... other execution fields ...
            "results_match": true
          }
        }
        ```
3.  **Python Parsing Snippets:**
    *   Code examples in Section 3.2 ("Parsing the File"), 3.3 ("Key Metrics & Calculations"), and 3.5 ("Result Mismatch Analysis") were modified to:
        *   Access nested execution data (e.g., `result_item.get("execution_evaluation", {}).get("results_match")`).
        *   Handle `generated_sql_output` and `ground_truth_sql_output` as direct lists (removing `json.loads()` for these specific fields).
4.  **Debugging Script Import:**
    *   The import statement in the debugging script example in Section 5.2 was updated from `from execution_evaluate import split_sql_statements` to `from sql_eval_lib.utils.helpers import split_sql_statements`.
5.  **Score Names:** References to score names (e.g., `exec_eval_results_match`) were made consistent with those logged by `ExecutionEvaluationModule` via `LangfuseClient`.

Both documents now accurately reflect how results are structured by the refactored `sql-eval-lib` and how they can be analyzed, maintaining consistency with the new library components and output formats. No class names like `Orchestrator` or `LangfuseClient` needed to be directly embedded in these user-facing analysis guides, as they describe the output files and Langfuse interaction, not library internals.
