# Analyzing LLM-Based SQL Evaluation Results

## 1. Introduction

This document guides you through analyzing the results generated when running LLM-based evaluations using the **`sql-eval-lib`** (SQL Evaluation Library), typically via the `Orchestrator` and the example notebook. The primary goal is to understand your Text-to-SQL model's performance, identify common failure patterns, and gather insights for improvement based on the LLM evaluator's feedback.

The evaluation process, when including LLM-based checks, yields results in two main places:

1.  **Langfuse Dashboards & Traces:** Provides an interactive way to explore individual evaluations, view trends, and leverage rich observability features.
2.  **Output JSONL file (e.g., `notebook_evaluation_results.jsonl`):** A local file containing structured JSON data for each evaluated item, suitable for custom scripting and offline analysis. Each item's result will have an `"llm_evaluation"` key containing the scores and reasoning from this module.

By using both sources, you can gain a comprehensive understanding of your model's strengths and weaknesses.

## 2. Analyzing Results with Langfuse

Langfuse offers powerful tools for monitoring and understanding your LLM application's behavior.

### 2.1. Navigation

1.  **Login & Project Selection:**
    *   Navigate to your Langfuse instance (Cloud or self-hosted).
    *   Log in with your credentials.
    *   Select the project you configured for this evaluation.

2.  **Relevant Langfuse Sections:**
    *   **Traces:** This is where you'll find individual records for each item processed. Each trace groups the model-under-test's generation, the LLM evaluator's assessment, and all associated scores. Look for traces tagged with the evaluation types (e.g., "llm").
    *   **Scores:** A dedicated section to view and analyze all scores logged, including those from the LLM evaluator (e.g., `llm_eval_semantic_correctness`) and the syntactic check performed by the LLM module (`llm_eval_parsable`).
    *   **Analytics/Dashboards (or Monitoring):** Create charts and visualize trends in your scores.
    *   **Generations:** Allows you to inspect individual LLM calls, such as the call to your model-under-test or the call to the LLM evaluator (often named `llm-evaluator-assessment`).

### 2.2. Key Metrics & Dashboards

*   **Focus Metrics:** Key metrics from the LLM evaluation to track include:
    *   Average score for `llm_eval_parsable` (syntactic check by `LLMEvaluationModule`).
    *   Average scores for each LLM-evaluated criterion:
        *   `llm_eval_semantic_correctness`
        *   `llm_eval_hallucinations_schema_adherence`
        *   `llm_eval_efficiency_conciseness`
        *   `llm_eval_overall_quality_readability`
    *   Distribution of these scores.
*   **Custom Charts:** Compare different model versions by filtering traces by `model_under_test_name` or custom tags.

### 2.3. Inspecting Traces

1.  **Filtering Traces:**
    *   Filter by scores: e.g., `llm_eval_semantic_correctness < 3`.
    *   Filter by tags.
    *   Filter by `trace_name` or `user_id` or `session_id` used during the `Orchestrator.run_evaluation()` call.

2.  **Examining an Individual Trace:**
    Click on a trace to open its detailed view. You should see:
    *   **Trace Metadata:** `item_id`, tags, session ID, user ID.
    *   **Generations:**
        *   **`{model_under_test_name} Generation`:** Input (`sql_prompt`, `sql_context`), Output (`generated_sql`).
        *   **`llm-evaluator-assessment`:** Input (prompt to evaluator), Output (structured JSON response with scores and reasoning).
    *   **Scores:** A list of all scores, including `llm_eval_parsable`, `llm_eval_semantic_correctness`, etc.

    **Correlation is Key:** Compare the numerical scores with the textual reasoning provided by the LLM evaluator (found in the output of the `llm-evaluator-assessment` generation).

### 2.4. Identifying Patterns in Langfuse
Use Langfuse's search, filtering, and analytics to find common issues or trends in the LLM evaluation data.

## 3. Analyzing Output JSONL File Locally

The output JSONL file (e.g., `notebook_evaluation_results.jsonl` specified in `Orchestrator.run_evaluation()`) contains a line for each evaluated item. The LLM evaluation results are nested under the `"llm_evaluation"` key.

### 3.1. File Structure

Each line in the output JSONL file is a JSON object. The relevant part for LLM evaluation looks like this:
```json
{
  "item_id": "...",
  "sql_prompt": "...",
  "sql_context_snippet": "...",
  "ground_truth_sql": "...",
  "generated_sql": "...",
  "llm_evaluation": {
    "llm_eval_parsable_score": 1, // or 0
    "llm_eval_parsing_error": null, // or error string
    "semantic_correctness_score": 5,
    "semantic_correctness_reasoning": "The query correctly reflects the prompt...",
    "hallucinations_schema_adherence_score": 5,
    "hallucinations_schema_adherence_reasoning": "Uses correct tables and columns.",
    "efficiency_conciseness_score": 4,
    "efficiency_conciseness_reasoning": "The query is efficient...",
    "overall_quality_readability_score": 5,
    "overall_quality_readability_reasoning": "Well-formatted and clear.",
    "llm_eval_api_error": null, // or error string if API call failed
    "llm_eval_response_parse_error": null // or error string if response parsing failed
  }
  // Potentially other keys like "execution_evaluation" if run
}
```

### 3.2. Parsing the File

```python
import json

results_filepath = "notebook_evaluation_results.jsonl" # Adjust if you used a different name
all_results = []

try:
    with open(results_filepath, 'r') as f:
        for line in f:
            try:
                all_results.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping line due to JSON decode error: {e} - Line: {line.strip()}")
    print(f"Successfully loaded {len(all_results)} results from {results_filepath}")
except FileNotFoundError:
    print(f"Error: Results file {results_filepath} not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Now you can process 'all_results'
# Example:
# for result in all_results:
#     llm_eval_data = result.get("llm_evaluation", {}) # Get the nested dictionary
#     if llm_eval_data.get("semantic_correctness_score", 0) < 3:
#         print(f"Low semantic score for item {result.get('item_id')}: {llm_eval_data.get('semantic_correctness_reasoning')}")
```

### 3.3. Aggregate Statistics

```python
import collections

# Assuming 'all_results' is populated as shown above

total_items = len(all_results)
parsable_queries_llm_module = 0
semantic_scores = []
hallucination_scores = []
efficiency_scores = []
quality_scores = []

# Score distributions
semantic_score_distribution = collections.defaultdict(int)

if total_items > 0:
    for result in all_results:
        llm_eval_data = result.get("llm_evaluation", {}) # Get the nested dictionary
        
        if llm_eval_data.get("llm_eval_parsable_score", 0) == 1:
            parsable_queries_llm_module += 1
        
        s_score = llm_eval_data.get("semantic_correctness_score")
        if s_score is not None:
            semantic_scores.append(s_score)
            semantic_score_distribution[s_score] += 1
        
        h_score = llm_eval_data.get("hallucinations_schema_adherence_score")
        if h_score is not None:
            hallucination_scores.append(h_score)
            
        e_score = llm_eval_data.get("efficiency_conciseness_score")
        if e_score is not None:
            efficiency_scores.append(e_score)
            
        q_score = llm_eval_data.get("overall_quality_readability_score")
        if q_score is not None:
            quality_scores.append(q_score)

    # Calculate Percentages and Averages
    percent_parsable_llm_module = (parsable_queries_llm_module / total_items) * 100 if total_items > 0 else 0
    avg_semantic_score = sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0
    avg_hallucination_score = sum(hallucination_scores) / len(hallucination_scores) if hallucination_scores else 0
    avg_efficiency_score = sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 0
    avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0

    print(f"\n--- Aggregate LLM Evaluation Statistics ({total_items} items) ---")
    print(f"Percentage of Syntactically Parsable Queries (by LLM Module's sqlglot): {percent_parsable_llm_module:.2f}%")
    print(f"Average Semantic Correctness Score: {avg_semantic_score:.2f}")
    print(f"Average Hallucinations/Schema Adherence Score: {avg_hallucination_score:.2f}")
    print(f"Average Efficiency/Conciseness Score: {avg_efficiency_score:.2f}")
    print(f"Average Overall Quality/Readability Score: {avg_quality_score:.2f}")

    print("\nSemantic Score Distribution:")
    for score_val, count in sorted(semantic_score_distribution.items()):
        percentage = (count / total_items) * 100 # Percentage of total items
        print(f"  Score {score_val}: {count} items ({percentage:.1f}%)")
else:
    print("No results loaded to calculate statistics.")
```

### 3.4. Qualitative Analysis

Numerical scores provide a high-level overview. Qualitative analysis helps understand the 'why'.

1.  **Filter for Low-Scoring Items:**
    Modify the Python script to select items where a specific score from `result.get("llm_evaluation", {})` is below a threshold.

2.  **Examine Reasoning:**
    For these low-scoring items, carefully read the corresponding `_reasoning` fields within the `llm_evaluation` dictionary (e.g., `semantic_correctness_reasoning`).

3.  **Categorize Errors:** (Remains the same as before)
    As you review the reasoning, try to categorize the errors. Examples:
    *   Misinterpreted intent of the SQL prompt.
    *   Incorrectly joined tables.
    *   Used non-existent columns/tables (hallucination).
    *   Inefficient query structure (e.g., unnecessary subqueries).
    *   Failed to use appropriate SQL functions.
    *   Problems with specific SQL clauses (GROUP BY, HAVING, window functions etc.).

## 4. Identifying Common Failure Patterns 
(This section remains largely the same, as it's about interpreting the categorized errors and reasoning, which are still available.)

The goal of both Langfuse and local analysis is to identify recurring failure patterns. Look for:
*   SQL Clause/Function Issues
*   Prompt Misinterpretation
*   Schema Complexity
*   Hallucination Trends
*   Contextual Blind Spots

**Using this information:** (Remains the same)
These patterns are invaluable. They can directly inform your model improvement strategy:
*   Targeted Fine-tuning
*   Prompt Engineering
*   Retrieval Augmented Generation (RAG)
*   Data Augmentation

## 5. Next Steps
(This section remains largely the same.)

The insights gained from this LLM-based evaluation are a crucial first step. They will help you:
*   Refine your model and evaluation strategy.
*   Guide the next phase of evaluation: Execution-based evaluation.
*   Prepare the final report on your model's performance.

By systematically analyzing these results, you can significantly improve the quality and reliability of your Text-to-SQL model.
```

Updated `ANALYZE_LLM_EVALUATION_RESULTS.md` to reflect:
1.  Results are generated by the `sql-eval-lib` (via `Orchestrator`).
2.  LLM evaluation results are nested under the `"llm_evaluation"` key in the output JSONL file.
3.  Python snippets for parsing and aggregation are updated to access data from this nested structure.
4.  Score names in Langfuse and JSONL keys (e.g., `llm_eval_parsable_score`, `llm_eval_semantic_correctness`) are made more consistent with the `LLMEvaluationModule`'s output.

Now, I will update `ANALYZE_EXECUTION_EVALUATION_RESULTS.md`.The file `ANALYZE_LLM_EVALUATION_RESULTS.md` has been updated.

Now, I will update `ANALYZE_EXECUTION_EVALUATION_RESULTS.md`.
Key changes expected:
1.  References to `execution_evaluate.py` changed to "the evaluation library" or "Orchestrator".
2.  Update Section 3.1 (File Structure) to show `execution_evaluation` nesting and reflect that `generated_sql_output` / `ground_truth_sql_output` are now direct lists of tuples (not JSON strings).
3.  Update Python snippets in Sections 3.2, 3.3, and 3.5 to access data from `result.get("execution_evaluation", {})` and to handle `*_output` fields as direct lists (no `json.loads` needed for them).
4.  Update the import path in the debugging script in Section 5.2 from `from execution_evaluate import split_sql_statements` to `from sql_eval_lib.utils.helpers import split_sql_statements`.
5.  Langfuse score names (e.g., `exec_eval_generated_sql_success`, `exec_eval_results_match`) should be reflected if specific names are mentioned for filtering. The current guide is mostly general here.
