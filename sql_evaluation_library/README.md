# SQL Evaluation Library (`sql-eval-lib`)

## Overview

This Python library provides a modular framework for evaluating Text-to-SQL models. It's designed to help you assess the performance of your models by running them against a dataset and evaluating the generated SQL queries through various methodologies.

**Key Features:**

*   **LLM-based Evaluation:** Leverages a powerful Large Language Model (e.g., GPT-3.5 Turbo, GPT-4) to analyze generated SQL for semantic correctness, schema adherence, efficiency, and overall quality, providing both scores and textual reasoning.
*   **Execution-based Evaluation:** Executes generated SQL and ground truth SQL against an in-memory SQLite database to compare their results for functional correctness.
*   **Langfuse Integration:** Seamlessly integrates with a locally hosted Langfuse instance (via Docker) for comprehensive logging and observability of traces, generations, scores, and events throughout the evaluation process.
*   **Modular Design:** Built with clear interfaces and distinct components for model interaction, evaluation methods, and logging, allowing for extensibility.
*   **Easy to Use:** Designed for straightforward use, especially within Jupyter notebooks for interactive evaluation and analysis.

## Package Structure

The library is structured as an installable Python package named `sql_eval_lib`. The source code resides in the `src/sql_eval_lib/` directory. When installed, you can import its components using `from sql_eval_lib import ...`.

## Prerequisites

1.  **Python:** Version >=3.8 (as specified in `pyproject.toml`).
2.  **`uv` (Recommended):** For Python package management and virtual environments. Installation instructions for `uv` can be found at [astral.sh/uv](https://astral.sh/uv). (`pip` can also be used).
3.  **Docker & Docker Compose:** Required for running a local Langfuse instance. [Docker Desktop](https://www.docker.com/products/docker-desktop/) is a common way to get these.
4.  **Prepared Dataset:** A dataset in JSONL format is required. The `scripts/prepare_dataset.py` script can be used to generate `prepared_test_data.jsonl` from the `gretelai/synthetic_text_to_sql` dataset.

## Setup & Installation

Follow these steps to set up the environment and install the library:

### 1. Prepare Dataset

Before running evaluations, you need a dataset. If you're using the `gretelai/synthetic_text_to_sql` dataset, run the provided script from the root of this project:
```bash
python scripts/prepare_dataset.py
```
This will create `prepared_test_data.jsonl` in the project root, containing the necessary fields (`id`, `sql_prompt`, `sql_context`, `sql`).

### 2. Setup Langfuse Locally

This library integrates with Langfuse for detailed logging. You'll need to run a local Langfuse instance using Docker:
*   Follow the instructions in **`LANGFUSE_SETUP_INSTRUCTIONS.MD`** (located in the project root) to clone the Langfuse repository, configure, and start the Langfuse Docker containers.
*   Ensure you set the following environment variables in your terminal/shell (or a `.env` file loaded by your environment) to match your Langfuse Docker setup:
    ```bash
    export LANGFUSE_HOST="http://localhost:3000"
    export LANGFUSE_PUBLIC_KEY="your_chosen_public_key_for_docker" # e.g., pk-lf-...
    export LANGFUSE_SECRET_KEY="your_chosen_secret_key_for_docker"
    ```

### 3. Install the Library (`sql-eval-lib`)

It's recommended to use a virtual environment.

**Using `uv` (Recommended):**
```bash
# Create and activate a virtual environment (optional but good practice)
uv venv
source .venv/bin/activate # On Linux/macOS
# .venv\\Scripts\\activate # On Windows

# Install the library in editable mode from the project root
# (where pyproject.toml is located)
uv pip install -e .
```
This command installs `sql-eval-lib` along with its core dependencies specified in `pyproject.toml` (e.g., `langfuse`, `openai`, `sqlglot`).

*(Conceptual) If the library were published to PyPI, you would install it using:*
```bash
# uv pip install sql-eval-lib
```

### 4. Set API Keys for LLMs

If you plan to use LLM-based evaluation or an OpenAI model as your model-under-test, set the necessary API key(s):
```bash
export OPENAI_API_KEY="your_openai_api_key_here"
# If your evaluator LLM is different and needs a separate key:
# export EVALUATOR_LLM_API_KEY="your_evaluator_llm_key_here"
```
Also, set the model names you intend to use if they differ from defaults:
```bash
export MODEL_UNDER_TEST_NAME="MyTextToSQLModel-v1"
export EVALUATOR_LLM_MODEL="gpt-4-turbo-preview" # For LLM-based evaluation
```

## Basic Usage Example

Here's a Python snippet demonstrating how to use the library:

```python
import os
import json
from sql_eval_lib import Orchestrator, DummyModelAdapter, LangfuseClient, OpenAIModelAdapter # Key imports

# Ensure environment variables are set as described in Setup steps.
# Particularly: LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
# And OPENAI_API_KEY if using OpenAIModelAdapter or LLM-based evaluation.

# 1. Initialize LangfuseClient
# LangfuseClient will attempt to read credentials from environment variables if not passed directly.
# Ensure MODEL_UNDER_TEST_NAME is set in your environment or pass it here.
model_name_for_langfuse = os.getenv("MODEL_UNDER_TEST_NAME", "MyDefaultSQLModel")
langfuse_client = None
try:
    langfuse_client = LangfuseClient(model_under_test_name=model_name_for_langfuse)
    print("LangfuseClient initialized successfully.")
except (ValueError, ConnectionError) as e:
    print(f"Langfuse initialization failed: {e}. Logging to Langfuse will be disabled.")
    # Proceed without Langfuse logging if desired, or handle error.

# 2. Initialize a model adapter
# Using DummyModelAdapter for this basic example.
# Replace with your actual model adapter (e.g., OpenAIModelAdapter or a custom one).
model_to_evaluate = DummyModelAdapter(fixed_query="SELECT COUNT(*) FROM users;")
# Example with OpenAIModelAdapter (if you want to test it as a model-under-test):
# openai_api_key_for_model = os.getenv("OPENAI_API_KEY")
# if openai_api_key_for_model:
#     model_to_evaluate = OpenAIModelAdapter(api_key=openai_api_key_for_model, model_name="gpt-3.5-turbo")
# else:
#     print("OPENAI_API_KEY not set, using DummyModelAdapter.")


# 3. Configure LLM evaluator (if doing LLM-based evaluation)
# Uses OPENAI_API_KEY by default if EVALUATOR_LLM_API_KEY is not set.
eval_llm_api_key = os.getenv("EVALUATOR_LLM_API_KEY", os.getenv("OPENAI_API_KEY"))
eval_llm_model_name = os.getenv("EVALUATOR_LLM_MODEL", "gpt-3.5-turbo")

llm_eval_module_config = None
if eval_llm_api_key:
    llm_eval_module_config = {
        "api_key": eval_llm_api_key,
        "model_name": eval_llm_model_name
    }
    print(f"LLM Evaluation will use: {eval_llm_model_name}")
else:
    print("Evaluator LLM API key not found. LLM-based evaluation will be skipped if requested.")

# 4. Initialize Orchestrator
# Pass langfuse_client (which might be None if initialization failed)
orchestrator = Orchestrator(
    model_under_test=model_to_evaluate,
    langfuse_client=langfuse_client,
    llm_eval_config=llm_eval_module_config, # Pass None if not doing LLM eval or if config is missing
    exec_eval_config={} # Empty dict enables execution eval by default if requested
)

# 5. Run evaluation
# Ensure 'prepared_test_data.jsonl' exists (e.g., in project root or specify full path)
# This example assumes it's in the same directory or a known path.
# For a project structure where this README is in 'sql_evaluation_library/',
# and 'prepared_test_data.jsonl' is at the true project root, the path might be '../prepared_test_data.jsonl'.
# For simplicity, assuming it's accessible directly here.
dataset_file_path = "prepared_test_data.jsonl" 
# A more robust path assuming this README is at project_root/sql_evaluation_library/README.md:
# dataset_file_path = "../prepared_test_data.jsonl" 
# However, if the library is installed, the user provides the path from their script's location.

if not os.path.exists(dataset_file_path):
    print(f"ERROR: Dataset file not found at '{dataset_file_path}'. Generate it using 'scripts/prepare_dataset.py'.")
    results = []
else:
    results = orchestrator.run_evaluation(
        dataset_path=dataset_file_path,
        evaluation_types=["llm", "execution"], # Choose one or both
        # item_ids=["1", "2"] # Optional: to test with a subset of items
        output_file="notebook_run_results.jsonl" # Saves results to this file
    )

if results:
    print(f"\\nProcessed {len(results)} items. First item's result:")
    print(json.dumps(results[0], indent=2))
else:
    print("\\nNo results processed.")

if langfuse_client and langfuse_client.enabled:
    print("\\nCheck Langfuse at http://localhost:3000 for detailed traces!")
else:
    print("\\nLangfuse logging was disabled for this run.")

```

## Detailed Examples

For a more comprehensive, step-by-step guide on using the library, including detailed setup and analysis, please refer to the Jupyter Notebook:
*   **`examples/run_evaluation_notebook.ipynb`**

This notebook provides interactive examples of initializing components, running evaluations, and inspecting results.

## Contributing / Development

If you're contributing to the development of this library:
*   Use `uv` to manage dependencies and the development environment.
*   Install development dependencies (e.g., `pytest`, `ruff`) if defined in `pyproject.toml` under `[project.optional-dependencies.dev]`:
    ```bash
    uv pip install -e ".[dev]"
    ```
*   Follow coding standards and run tests before submitting changes. (Testing infrastructure to be detailed separately).
```

This new README content is comprehensive and covers all the points requested in the subtask. It assumes the `README.md` is located at `sql_evaluation_library/README.md`, which is the root of the project as defined in the `pyproject.toml`. The paths in the example code (like for `dataset_path`) are written with this assumption, but users would adjust them based on where they run their scripts if the library is installed.The `sql_evaluation_library/README.md` file was successfully created with the comprehensive content outlined in the previous response.

**Verification against requirements:**

1.  **Located `README.md`:** The file was created at `sql_evaluation_library/README.md`.
2.  **Rewritten/Updated Content:**
    *   **Project Title:** "SQL Evaluation Library (`sql-eval-lib`)" - Implemented.
    *   **Overview:**
        *   Purpose (evaluating Text-to-SQL models) - Implemented.
        *   Key features (LLM-based, execution-based, Langfuse, modular design) - Implemented.
    *   **Package Structure:**
        *   Mention of `src/sql_eval_lib` layout and `sql_eval_lib` as importable package name - Implemented.
    *   **Prerequisites:**
        *   Python (>=3.8), `uv`, Docker, prepared dataset - Implemented.
    *   **Setup & Installation:**
        *   **1. Prepare Dataset:** Instruction to run `python scripts/prepare_dataset.py` - Implemented.
        *   **2. Setup Langfuse:** Reference to `LANGFUSE_SETUP_INSTRUCTIONS.MD` and environment variables - Implemented.
        *   **3. Install the Library:** Using `uv pip install -e .` and conceptual published install - Implemented. Mention of dependencies being installed - Implemented.
        *   **4. Set API Keys:** Reminder about `OPENAI_API_KEY` and other model-related environment variables - Implemented.
    *   **Basic Usage Example (Python code block):**
        *   The provided Python code block was included, demonstrating:
            *   Imports (`Orchestrator`, `DummyModelAdapter`, `LangfuseClient`).
            *   LangfuseClient initialization (with error handling).
            *   Model adapter initialization (`DummyModelAdapter` example).
            *   LLM evaluator configuration (conditional on API key).
            *   `Orchestrator` initialization.
            *   Calling `orchestrator.run_evaluation` with `dataset_path` and `evaluation_types`.
            *   Printing results and a reminder to check Langfuse.
    *   **Detailed Examples:**
        *   Pointed to `examples/run_evaluation_notebook.ipynb` - Implemented.
    *   **Contributing/Development (Optional):**
        *   Brief mention of using `uv` for dev dependencies - Implemented.

The created `sql_evaluation_library/README.md` accurately reflects the project's structure, setup, and usage as per the subtask requirements.
