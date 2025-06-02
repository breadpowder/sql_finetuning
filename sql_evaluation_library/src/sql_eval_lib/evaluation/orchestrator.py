# sql_evaluation_library/src/sql_eval_lib/evaluation/orchestrator.py
import json
import os # For os.path.exists in run_evaluation
from datetime import datetime # Use this for consistency

try:
    from ..models.interface import ModelInterface
    from ..langfuse.manager import LangfuseClient # Renamed and new path
    from .llm_evaluator import LLMEvaluationModule
    from .exec_evaluator import ExecutionEvaluationModule
except ImportError:
    # This allows the file to be potentially parsed in environments where relative imports might struggle initially
    # For actual runtime, these imports need to succeed.
    print("Orchestrator: Could not perform relative imports. Ensure library structure is correct.")
    ModelInterface = type("ModelInterface", (object,), {})
    LangfuseClient = type("LangfuseClient", (object,), {}) # Renamed
    LLMEvaluationModule = type("LLMEvaluationModule", (object,), {})
    ExecutionEvaluationModule = type("ExecutionEvaluationModule", (object,), {})


class Orchestrator: # Renamed from SQLEvaluator
    """
    Orchestrates the evaluation of Text-to-SQL models using various evaluation modules.
    """
    def __init__(self, model_under_test: ModelInterface, 
                 langfuse_client: LangfuseClient = None,  # Allow None, Renamed
                 llm_eval_config: dict = None, 
                 exec_eval_config: dict = None): # exec_eval_config is for future use or consistency
        """
        Initializes the Orchestrator.

        Args:
            model_under_test (ModelInterface): An instance of a class implementing ModelInterface.
            langfuse_client (LangfuseClient, optional): An instance of LangfuseClient. Defaults to None.
            llm_eval_config (dict, optional): Configuration for LLM evaluation.
                Expected keys: "api_key" (for evaluator LLM), "model_name" (for evaluator LLM).
            exec_eval_config (dict, optional): Configuration for execution evaluation.
                Currently not used by ExecutionEvaluationModule constructor but reserved.
        """
        if not isinstance(model_under_test, ModelInterface):
            raise TypeError("model_under_test must be an instance of ModelInterface.")

        self.model_under_test = model_under_test
        self.langfuse_client = langfuse_client # Renamed

        if llm_eval_config and isinstance(llm_eval_config, dict):
            self.llm_eval_module = LLMEvaluationModule(
                evaluator_llm_api_key=llm_eval_config.get("api_key"), # LLMEvalModule handles None key
                evaluator_llm_model=llm_eval_config.get("model_name", "gpt-3.5-turbo"), # Default in LLMEvalModule too
                langfuse_manager=self.langfuse_client # Pass LangfuseClient instance
            )
            print("Orchestrator: LLMEvaluationModule initialized.") # Updated class name in log
        else:
            self.llm_eval_module = None
            print("Orchestrator: LLMEvaluationModule not initialized (no llm_eval_config provided or invalid).") # Updated

        # ExecutionEvaluationModule's constructor currently only takes langfuse_manager (now langfuse_client)
        # exec_eval_config is present for future compatibility or if it needs config later
        if exec_eval_config is not None: # Check if config is provided, even if not used by module yet
            self.exec_eval_module = ExecutionEvaluationModule(langfuse_manager=self.langfuse_client) # Pass LangfuseClient
            print("Orchestrator: ExecutionEvaluationModule initialized (config provided).") # Updated
        else: # Default behavior: initialize it if no specific instruction not to.
            self.exec_eval_module = ExecutionEvaluationModule(langfuse_manager=self.langfuse_client) # Pass LangfuseClient
            print("Orchestrator: ExecutionEvaluationModule initialized (default).") # Updated
        
        # If exec_eval_config being None should disable it:
        # if exec_eval_config is None:
        #    self.exec_eval_module = None
        #    print("Orchestrator: ExecutionEvaluationModule not initialized (no exec_eval_config).")
        # else:
        #    self.exec_eval_module = ExecutionEvaluationModule(langfuse_manager=self.langfuse_client)
        #    print("Orchestrator: ExecutionEvaluationModule initialized.")



    def _load_dataset(self, dataset_path: str, item_ids_filter: list[str] = None) -> list[dict]:
        """Loads dataset from a JSONL file, optionally filtering by item_ids."""
        loaded_data = []
        try:
            with open(dataset_path, 'r') as f:
                for line in f:
                    try:
                        loaded_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping invalid JSON line in {dataset_path}: {line.strip()}")
            
            if item_ids_filter:
                # Ensure item_ids_filter contains strings for comparison, as item IDs from JSON are strings
                item_ids_filter_str = set(map(str, item_ids_filter)) 
                filtered_data = [item for item in loaded_data if str(item.get("id")) in item_ids_filter_str]
                print(f"Loaded {len(loaded_data)} items, filtered to {len(filtered_data)} items based on item_ids_filter.")
                return filtered_data
            else:
                print(f"Loaded {len(loaded_data)} items from {dataset_path}.")
                return loaded_data
        except FileNotFoundError:
            print(f"Error: Dataset file {dataset_path} not found.")
            return []
        except Exception as e:
            print(f"An unexpected error occurred while loading dataset {dataset_path}: {e}")
            return []


    def run_evaluation(self, dataset_path: str, evaluation_types: list[str], 
                       item_ids: list[str] = None, session_id: str = None, user_id: str = None,
                       output_file: str = None) -> list[dict]:
        """
        Runs the evaluation process for the specified dataset and evaluation types.

        Args:
            dataset_path (str): Path to the JSONL dataset file.
            evaluation_types (list[str]): List of evaluation types to perform (e.g., ["llm", "execution"]).
            item_ids (list[str], optional): A list of specific item IDs to evaluate. 
                                            If None, all items are evaluated.
            session_id (str, optional): Session ID for Langfuse tracing.
            user_id (str, optional): User ID for Langfuse tracing.
            output_file (str, optional): If provided, saves results of each item to this JSONL file.

        Returns:
            list[dict]: A list of dictionaries, where each dictionary contains the
                        evaluation results for a single item.
        """
        results_list = []
        dataset = self._load_dataset(dataset_path, item_ids_filter=item_ids)

        if not dataset:
            return results_list
        
        if output_file and os.path.exists(output_file):
            print(f"Warning: Output file {output_file} already exists. Results will be appended.")


        for item_index, item in enumerate(dataset):
            # Fallback for item_id using current timestamp if 'id' is missing
            item_id_str = str(item.get("id", f"generated-id-{datetime.utcnow().timestamp()}-{item_index}"))
            
            sql_prompt = item.get("sql_prompt")
            sql_context = item.get("sql_context")
            ground_truth_sql = item.get("sql")

            current_item_results = {
                "item_id": item_id_str,
                "sql_prompt": sql_prompt,
                # "sql_context": sql_context, # Context can be very large, consider snippet or exclude from item results
                "sql_context_snippet": sql_context[:200] + "..." if sql_context else None,
                "ground_truth_sql": ground_truth_sql
            }
            
            print(f"\nProcessing item {item_index + 1}/{len(dataset)} (ID: {item_id_str})...")

            trace = None
            if self.langfuse_client and self.langfuse_client.enabled: # Renamed
                # Construct tags for the trace. Include evaluation types being performed for this item.
                trace_tags = list(set(evaluation_types)) # Ensure unique tags
                trace = self.langfuse_client.get_trace( # Renamed
                    item_id=item_id_str, 
                    session_id=session_id, 
                    user_id=user_id, 
                    tags=trace_tags # Pass the requested evaluation types as tags
                )
            
            # Get Model Output
            model_metadata_for_call = {"item_id": item_id_str, "ground_truth_sql": ground_truth_sql} # For dummy model or other uses
            if trace: model_metadata_for_call["langfuse_trace_id"] = trace.id

            start_time_model = datetime.utcnow()
            generated_sql = self.model_under_test.get_sql(sql_prompt, sql_context, metadata=model_metadata_for_call)
            end_time_model = datetime.utcnow()
            
            current_item_results["generated_sql"] = generated_sql
            
            if trace and self.langfuse_client and self.langfuse_client.enabled: # Renamed
                self.langfuse_client.log_model_generation( # Renamed
                    trace, sql_prompt, sql_context, generated_sql, 
                    start_time_model, end_time_model, 
                    metadata={"item_id": item_id_str} # Basic metadata for the generation step
                )

            # LLM Evaluation
            if "llm" in evaluation_types and self.llm_eval_module:
                print(f"  Running LLM evaluation for item {item_id_str}...")
                llm_scores = self.llm_eval_module.evaluate_single_item(
                    trace, sql_prompt, sql_context, generated_sql, ground_truth_sql
                )
                current_item_results.update({"llm_evaluation": llm_scores}) # Nest results
            
            # Execution Evaluation
            if "execution" in evaluation_types and self.exec_eval_module:
                print(f"  Running Execution evaluation for item {item_id_str}...")
                exec_scores = self.exec_eval_module.evaluate_single_item(
                    trace, generated_sql, ground_truth_sql, sql_context
                )
                current_item_results.update({"execution_evaluation": exec_scores}) # Nest results
            
            results_list.append(current_item_results)

            if output_file:
                try:
                    with open(output_file, 'a') as f:
                        f.write(json.dumps(current_item_results) + '\n')
                except IOError as e:
                    print(f"Error writing item result to {output_file}: {e}")
            
            # Optional: Flush Langfuse per item if immediate visibility is needed
            # if trace and self.langfuse_client and self.langfuse_client.enabled: # Renamed
            #    self.langfuse_client.flush() # Renamed

        if self.langfuse_client and self.langfuse_client.enabled: # Renamed
            self.langfuse_client.flush() # Final flush after processing all items # Renamed
            print("Langfuse: All buffered events flushed.")
        
        print(f"\nEvaluation run complete. Processed {len(dataset)} items.")
        if output_file:
            print(f"Detailed results saved to: {output_file}")
            
        return results_list

```

This implementation of `SQLEvaluator`:
*   Imports necessary modules.
*   Constructor:
    *   Stores `model_under_test` and `langfuse_manager`.
    *   Initializes `llm_eval_module` if `llm_eval_config` is provided, passing API key, model name, and `langfuse_manager`.
    *   Initializes `exec_eval_module` (passing `langfuse_manager`). The `exec_eval_config` is checked for presence but not used by `ExecutionEvaluationModule`'s constructor yet, aligning with the design.
*   `_load_dataset`: Helper method to load and optionally filter data.
*   `run_evaluation`:
    *   Loads and filters data.
    *   Iterates through items:
        *   Gets `item_id`, `sql_prompt`, `sql_context`, `ground_truth_sql`.
        *   Initializes `current_item_results`.
        *   Gets/creates a Langfuse trace if `langfuse_manager` is enabled, passing `session_id`, `user_id`, and `evaluation_types` as tags.
        *   Calls `model_under_test.get_sql()`, records timing, and logs the generation to Langfuse.
        *   If "llm" in `evaluation_types` and `llm_eval_module` exists, calls `llm_eval_module.evaluate_single_item()` and updates results.
        *   If "execution" in `evaluation_types` and `exec_eval_module` exists, calls `exec_eval_module.evaluate_single_item()` and updates results.
        *   Appends `current_item_results` to `results_list`.
        *   Optionally saves each item's result to an output JSONL file.
    *   Flushes Langfuse at the end.
    *   Returns `results_list`.

I've also added an `output_file` parameter to `run_evaluation` for saving results incrementally, which is good practice for long runs. The `item_id` fallback now includes the item index for better uniqueness if multiple items lack an ID. The `exec_eval_config` handling in `__init__` is set up to initialize `exec_eval_module` by default, which is reasonable given its current lack of specific config needs, but can be adjusted if stricter control is desired.

Next, I will update `sql_evaluation_library/__init__.py`.The `SQLEvaluator` class has been implemented in `sql_evaluation_library/evaluator.py`.

Now, I will update `sql_evaluation_library/__init__.py` to export `SQLEvaluator`.
