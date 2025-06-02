# sql_evaluation_library/src/sql_eval_lib/evaluation/llm_evaluator.py
import os
import json
import time
from datetime import datetime
import openai # Direct import, assuming openai library is used
import sqlglot

# Attempt to import LangfuseClient for type hinting and structure
try:
    from ..langfuse.manager import LangfuseClient # Updated path and class name
except ImportError: # Fallback for environments where the file isn't discoverable yet by linters
    print("LLM_Evaluator: Could not import LangfuseClient from ..langfuse.manager, using a dummy type hint if needed.")
    LangfuseClient = type("LangfuseClient", (object,), {}) # Dummy type for hinting

class LLMEvaluationModule:
    """
    Evaluates generated SQL queries using an LLM evaluator.
    """
    def __init__(self, evaluator_llm_api_key: str = None, 
                 evaluator_llm_model: str = "gpt-3.5-turbo", 
                 langfuse_manager: LangfuseClient = None): # Updated type hint
        """
        Initializes the LLM Evaluation Module.

        Args:
            evaluator_llm_api_key (str, optional): API key for the evaluator LLM service.
                If None, uses OPENAI_API_KEY environment variable.
            evaluator_llm_model (str, optional): Name of the evaluator LLM model.
                Defaults to "gpt-3.5-turbo".
            langfuse_manager (LangfuseClient, optional): Instance for logging. # Updated type hint
                Defaults to None, in which case logging is skipped.
        """
        self.evaluator_llm_model = evaluator_llm_model
        
        resolved_api_key = evaluator_llm_api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            # Not raising an error immediately to allow for scenarios where the module might be
            # initialized but not used, or if the user intends to mock the API call.
            # A warning is printed instead. Actual usage will fail if key remains None.
            print("Warning: OpenAI API key not provided and not found in OPENAI_API_KEY environment variable for LLMEvaluationModule.")
            self.openai_client = None
        else:
            self.openai_client = openai.OpenAI(api_key=resolved_api_key)
            
        self.langfuse_manager = langfuse_manager
        self.sql_dialect_for_parsing = 'sqlite' # Default, can be made configurable
        print(f"LLMEvaluationModule initialized with model: {self.evaluator_llm_model}. Langfuse logging {'enabled' if self.langfuse_manager and self.langfuse_manager.enabled else 'disabled'}.")

    def _check_sql_syntax(self, sql_query: str) -> tuple[bool, str | None]:
        """Uses sqlglot to parse the SQL query and check for syntactic validity."""
        if not sql_query or not isinstance(sql_query, str) or sql_query.strip() == "":
            return False, "SQL query is empty or not a string."
        try:
            sqlglot.parse_one(sql_query, read=self.sql_dialect_for_parsing)
            return True, None
        except sqlglot.errors.ParseError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Unexpected sqlglot error: {str(e)}"

    def _construct_evaluator_prompt(self, sql_prompt: str, sql_context: str, generated_sql: str, ground_truth_sql: str) -> tuple[str, str]:
        """Constructs system and user prompts for the LLM evaluator."""
        system_prompt = f"""You are an expert SQL evaluator. Your task is to evaluate a generated SQL query based on several criteria.
Provide a score from 1 (worst) to 5 (best) for each criterion, along with detailed textual reasoning for each score.
Output your response strictly as a single JSON object. Do not include any text outside of this JSON object.

The JSON object should have the following structure, with scores as integers and reasoning as strings:
{{
    "semantic_correctness": {{ "score": <integer>, "reasoning": "<text>" }},
    "hallucinations_schema_adherence": {{ "score": <integer>, "reasoning": "<text>" }},
    "efficiency_conciseness": {{ "score": <integer>, "reasoning": "<text>" }},
    "overall_quality_readability": {{ "score": <integer>, "reasoning": "<text>" }}
}}
"""
        user_prompt = f"""Please evaluate the **Generated SQL Query** based on the provided context.

**1. Natural Language Question (SQL Prompt):**
{sql_prompt}

**2. Database Schema (SQL Context):**
```sql
{sql_context}
```

**3. Generated SQL Query (to be evaluated):**
```sql
{generated_sql}
```

**4. Ground Truth SQL Query (for reference, DO NOT SCORE THIS ONE, use it to inform your score of the Generated SQL):**
```sql
{ground_truth_sql}
```

Evaluate the **Generated SQL Query** against the criteria and provide your response in the specified JSON format.
Focus on:
- **Semantic Correctness:** Does the generated SQL accurately represent the intent of the Natural Language Question? Does it fetch the correct data as per the question and schema? Compare its logic with the Ground Truth SQL.
- **Hallucinations / Schema Adherence:** Does the generated SQL only use tables and columns defined in the Database Schema (SQL Context)? Are there any fabricated table or column names?
- **Efficiency and Conciseness:** Is the generated SQL efficient? Are there any redundant operations or overly complex structures compared to the Ground Truth SQL or optimal SQL practices? Is it concise?
- **Overall Quality & Readability:** Is the generated SQL well-formatted and easy to understand? Does it follow common SQL coding conventions? Considering all aspects, what is its overall quality?
"""
        return system_prompt, user_prompt

    def _call_evaluator_llm(self, system_prompt: str, user_prompt: str) -> tuple[str | None, str | None]:
        """Calls the configured OpenAI LLM to get the evaluation."""
        if not self.openai_client:
            return None, "OpenAI client not initialized (likely missing API key)."

        max_retries = 2 # Reduced for library context
        retry_delay = 3
        for attempt in range(max_retries):
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.evaluator_llm_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1, # Low temperature for consistent JSON output
                    response_format={"type": "json_object"} # Request JSON mode
                )
                response_content = response.choices[0].message.content
                return response_content, None
            except openai.RateLimitError as e:
                error_msg = f"OpenAI RateLimitError: {e}. Attempt {attempt + 1}/{max_retries}."
                print(error_msg)
                if attempt < max_retries - 1: time.sleep(retry_delay)
            except openai.APIError as e:
                error_msg = f"OpenAI APIError: {e}. Attempt {attempt + 1}/{max_retries}."
                print(error_msg)
                if attempt < max_retries - 1: time.sleep(retry_delay)
            except Exception as e:
                error_msg = f"Unexpected error calling OpenAI: {e}"
                print(error_msg)
                return None, error_msg # Don't retry on unexpected errors

        return None, error_msg # Return last error if all retries fail

    def _parse_llm_response(self, response_content: str | None) -> tuple[dict, str | None]:
        """Parses the JSON response from the LLM, extracts scores and reasonings."""
        if not response_content:
            return {}, "LLM response was empty."

        try:
            parsed_json = json.loads(response_content)
            # Basic validation of structure
            criteria = ["semantic_correctness", "hallucinations_schema_adherence", "efficiency_conciseness", "overall_quality_readability"]
            extracted_results = {}
            valid_structure = True
            for crit in criteria:
                if isinstance(parsed_json.get(crit), dict) and \
                   "score" in parsed_json[crit] and "reasoning" in parsed_json[crit]:
                    extracted_results[f"{crit}_score"] = int(parsed_json[crit]["score"])
                    extracted_results[f"{crit}_reasoning"] = str(parsed_json[crit]["reasoning"])
                else:
                    valid_structure = False
                    extracted_results[f"{crit}_score"] = 0 # Default on structure error
                    extracted_results[f"{crit}_reasoning"] = "Invalid structure in LLM response for this criterion."
            
            if not valid_structure:
                 return extracted_results, "LLM response JSON did not match expected structure for one or more criteria."
            return extracted_results, None
        except json.JSONDecodeError as e:
            return {}, f"Failed to decode LLM response as JSON: {e}. Response: {response_content[:200]}..."
        except (TypeError, ValueError) as e: # For int conversion or other parsing issues
            return {}, f"Error parsing values from LLM JSON: {e}. Response: {response_content[:200]}..."


    def evaluate_single_item(self, trace, sql_prompt: str, sql_context: str, generated_sql: str, ground_truth_sql: str) -> dict:
        """
        Evaluates a single generated SQL query item using an LLM.
        """
        results = {}
        start_time_overall = datetime.utcnow()

        # 1. Syntactic Check
        is_parsable, parsing_error = self._check_sql_syntax(generated_sql)
        results["llm_eval_parsable_score"] = 1 if is_parsable else 0
        results["llm_eval_parsing_error"] = parsing_error
        
        if self.langfuse_manager and self.langfuse_manager.enabled and trace:
            self.langfuse_manager.log_score(
                trace, 
                name="llm_eval_parsable", 
                value=results["llm_eval_parsable_score"], 
                comment=parsing_error or "OK"
            )

        # Proceed to LLM evaluation only if parsable, or based on a flag (currently always proceeds)
        # if not is_parsable:
        #     # Populate default scores if not parsable and skipping LLM eval for unparsable queries
        #     return results # Or add default failure scores for LLM part

        # 2. Construct Evaluator LLM Prompt
        system_prompt, user_prompt = self._construct_evaluator_prompt(
            sql_prompt, sql_context, generated_sql, ground_truth_sql
        )
        prompt_for_logging = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}"


        # 3. Call Evaluator LLM
        start_time_llm_call = datetime.utcnow()
        llm_response_content, llm_call_error = self._call_evaluator_llm(system_prompt, user_prompt)
        end_time_llm_call = datetime.utcnow()

        # 4. Parse Evaluator LLM Response
        parsed_llm_scores, parsing_llm_response_error = self._parse_llm_response(llm_response_content)
        results.update(parsed_llm_scores)
        
        if llm_call_error: # Error during API call
            results["llm_eval_api_error"] = llm_call_error
            # Log all criteria as failed if API call itself failed
            default_fail_reason = f"LLM API call failed: {llm_call_error}"
            criteria_keys = ["semantic_correctness", "hallucinations_schema_adherence", "efficiency_conciseness", "overall_quality_readability"]
            for key_base in criteria_keys:
                results.setdefault(f"{key_base}_score", 0)
                results.setdefault(f"{key_base}_reasoning", default_fail_reason)

        elif parsing_llm_response_error: # Error parsing the response from LLM
            results["llm_eval_response_parse_error"] = parsing_llm_response_error
            # Log all criteria as failed if response parsing failed
            default_fail_reason = f"LLM response parsing failed: {parsing_llm_response_error}"
            criteria_keys = ["semantic_correctness", "hallucinations_schema_adherence", "efficiency_conciseness", "overall_quality_readability"]
            for key_base in criteria_keys:
                results.setdefault(f"{key_base}_score", 0)
                results.setdefault(f"{key_base}_reasoning", default_fail_reason)
        
        # Log evaluator generation to Langfuse
        if self.langfuse_manager and self.langfuse_manager.enabled and trace:
            self.langfuse_manager.log_evaluator_generation(
                trace,
                evaluator_model_name=self.evaluator_llm_model,
                prompt_to_evaluator=prompt_for_logging, # Could be just user_prompt or combined
                evaluation_result={"raw_response": llm_response_content, "parsed_scores": parsed_llm_scores, "api_error": llm_call_error, "parsing_error": parsing_llm_response_error},
                start_time=start_time_llm_call,
                end_time=end_time_llm_call,
                metadata={"model": self.evaluator_llm_model}
            )
            # Log individual scores extracted from LLM
            for key, value in parsed_llm_scores.items():
                if key.endswith("_score"):
                    score_name_for_langfuse = f"llm_eval_{key.replace('_score', '')}"
                    self.langfuse_manager.log_score(trace, name=score_name_for_langfuse, value=value)
                # Reasonings are part of the logged generation output (evaluation_result.raw_response or parsed_scores)

        # Add overall duration? (Optional)
        # results["llm_eval_duration_seconds"] = (datetime.utcnow() - start_time_overall).total_seconds()
        return results

```
