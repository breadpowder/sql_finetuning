Improvement:
1. generation.py. Remove __main__.py to test functionality, create test_<model_name> after `/tests` folders.
2. Code refactor evaluation/ for extensibility.
- Missing the abstracted class for llm_evaluator and exec_evaluator. Sql project is a one concrete implementation and we need to provide common abstract interface and concrete implementation for SQL project
    -- for LLM evaluator, self._construct_evaluator_prompt can be abstraced and can extend for each concrete implementation, i.e. sql task.
    -- for exec_evaluator, each concrete task must execute single item, get the outcome, and decide the evaluation result. 

3. Now prompts are mixed with agentic flow. Please extract prompts in seperate python files prompts.py for modularity. Provide a few examples if the task is complicated.

Plan:
1. Using conda env sql_fine_tuning to run the test. 
2. For testing database, set up docker images and container for sqlite, trino, mysql for test run.
3. for Model Adapter and model implementation. Add gemini model adapter and default to gemini-2.5-flash
4. I don't see how ollama self-hosted model can be plugged in, please implement it.