# Langfuse Setup for Python Evaluation Library (Local Docker Method)

This document provides instructions on how to set up Langfuse locally using Docker. This local instance will serve as the backend for the Python evaluation library used in this project, allowing you to log and visualize evaluation traces, generations, and scores.

## 1. Langfuse Python SDK Installation

To interact with Langfuse from your Python scripts, you need to install the Langfuse Python SDK:

```bash
pip install langfuse
```
This command will download and install the latest version of the Langfuse SDK and its dependencies.

## 2. Setting up Langfuse Locally via Docker

These instructions guide you to run Langfuse on your local machine using Docker Compose. This is suitable for development and testing purposes.

### 2.1. Prerequisites
*   **Docker and Docker Compose:** Ensure you have Docker and Docker Compose installed. The easiest way is often via [Docker Desktop](https://www.docker.com/products/docker-desktop/).
*   **Git:** Required to clone the Langfuse repository.

### 2.2. Clone Langfuse Repository
The official Langfuse repository contains the necessary `docker-compose.yml` file.
```bash
git clone https://github.com/langfuse/langfuse.git
cd langfuse
```

### 2.3. Configure Langfuse API Keys for Docker
Before starting the Langfuse services, you need to define your API keys. Langfuse's `docker-compose.yml` typically sources these from an `.env` file in the same directory (the `langfuse` directory you just cloned into).

1.  **Create or Check `.env` file:**
    Inside the `langfuse` directory, look for a `docker-compose.yml` file. This file will specify how environment variables (including API keys) are loaded for the Langfuse services. Often, it references an `.env` file. If an `.env` file (e.g., `.env.example` that you can copy to `.env`) is not present or if the `docker-compose.yml` directly lists environment variables, you might need to edit `docker-compose.yml` directly for the `LANGFUSE_SECRET_KEY` and `LANGFUSE_PUBLIC_KEY` variables under the `langfuse-server` or similar service.

2.  **Set Your API Keys for Docker:**
    You need to **choose and set** the `LANGFUSE_SECRET_KEY` and `LANGFUSE_PUBLIC_KEY` that your local Langfuse instance will use. Add or modify these lines in the `.env` file (or directly in `docker-compose.yml` if applicable):

    ```env
    # .env file inside the 'langfuse' directory (used by docker-compose)
    LANGFUSE_SECRET_KEY=your_chosen_strong_secret_key_for_docker
    LANGFUSE_PUBLIC_KEY=your_chosen_public_key_for_docker_pk_prefix 
    # Example: LANGFUSE_PUBLIC_KEY=pk-lf-1234567890abcdef1234567890abcdef
    
    # Other variables like DATABASE_URL, REDIS_URL, etc., are usually pre-configured
    # in the provided docker-compose.yml or .env.example.
    # Ensure POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB are also set as per Langfuse docs.
    # Default values are often provided in .env.example or the compose file.
    ```
    **Important:** These are the keys your local Langfuse Docker instance will use. You will use these *same keys* in your host environment (see Section 3) for the Python SDK to authenticate with this local Langfuse instance.

### 2.4. Start Langfuse Services
Navigate to the directory where you cloned the Langfuse repository (and where `docker-compose.yml` is located) and run:
```bash
docker compose up -d
```
The `-d` flag runs the containers in detached mode (in the background). You can view logs using `docker compose logs -f`.
Wait for a few minutes for the services to start. The `langfuse-web` container logs should eventually indicate it's ready (e.g., "Ready" or similar message).

### 2.5. Access Langfuse UI
Once started, the Langfuse UI should be accessible in your browser at:
`http://localhost:3000` (or the port configured in your `docker-compose.yml` if different).

You can now create a project in the UI. However, the API keys for SDK interaction are those you defined in the `.env` file (or `docker-compose.yml`) for the Docker services.

## 3. Configuring Your Python Environment for Local Langfuse

Your Python evaluation scripts need to be configured to send data to this local Langfuse instance. This is done by setting environment variables in your host machine's terminal (the environment where you run your Python scripts).

1.  **`LANGFUSE_PUBLIC_KEY`**:
    Set this to the **same public key** you configured for the Docker container (e.g., in the `.env` file within the `langfuse` cloned directory).
    ```bash
    export LANGFUSE_PUBLIC_KEY="your_chosen_public_key_for_docker_pk_prefix" 
    ```

2.  **`LANGFUSE_SECRET_KEY`**:
    Set this to the **same secret key** you configured for the Docker container.
    ```bash
    export LANGFUSE_SECRET_KEY="your_chosen_strong_secret_key_for_docker"
    ```

3.  **`LANGFUSE_HOST`**:
    This tells the Python SDK where to send the data. For the local Docker setup, this is:
    ```bash
    export LANGFUSE_HOST="http://localhost:3000"
    ```

Make these environment variables persistent by adding them to your shell's configuration file (e.g., `~/.bashrc`, `~/.zshrc`).

## 4. API Keys for Evaluator LLMs (e.g., OpenAI)

The setup of API keys for external services used by your evaluation logic (like an OpenAI model for LLM-based evaluation) remains unchanged. These are independent of Langfuse's own API keys.

### OpenAI API Key
If your evaluation scripts use OpenAI models:
```bash
export OPENAI_API_KEY="your_actual_openai_api_key_here"
```

### Other Model/Service API Keys
If your model-under-test or other evaluation components require different API keys:
```bash
export MY_OTHER_MODEL_API_KEY="your_other_model_key_here"
```

## 5. Langfuse Python Client Initialization

The Langfuse Python client, when initialized, will automatically pick up the `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST` environment variables you set in your Python environment (Section 3).

Here's the relevant initialization snippet (as used in the evaluation scripts):

```python
# From your Python evaluation script (e.g., llm_evaluate.py or execution_evaluate.py)
import os
from langfuse import Langfuse

langfuse_client = None
LANGFUSE_ENABLED = False

def initialize_langfuse():
    global langfuse_client, LANGFUSE_ENABLED
    if langfuse_client is None: # Initialize only once
        try:
            # These environment variables must be set in the environment
            # where this Python script is executed.
            public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
            secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
            host = os.environ.get("LANGFUSE_HOST") # Crucial for local setup

            if not all([public_key, secret_key, host]):
                print("Warning: Langfuse environment variables (PUBLIC_KEY, SECRET_KEY, HOST) not fully set. Langfuse logging will be disabled.")
                LANGFUSE_ENABLED = False
                return

            langfuse_client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host 
                # Alternatively, if all env vars are reliably set:
                # langfuse_client = Langfuse() 
            )
            langfuse_client.auth_check() # Verifies connection and authentication
            LANGFUSE_ENABLED = True
            print(f"Langfuse initialized successfully to host: {host}")

        except Exception as e:
            print(f"Error initializing Langfuse or during auth check. Langfuse logging will be disabled.")
            print(f"Details: {e}")
            langfuse_client = None
            LANGFUSE_ENABLED = False
    return langfuse_client

# Example: Call at the start of your script
# initialize_langfuse()
# if LANGFUSE_ENABLED:
#     print("Ready to log to local Langfuse instance.")
# else:
#     print("Langfuse logging is disabled.")
```
The conceptual logging examples provided previously (e.g., in `evaluate_text_to_sql_item`) remain valid for interacting with this locally hosted Langfuse instance.

## 6. Shutting Down Local Langfuse
To stop your local Langfuse services, navigate to the `langfuse` directory (where you cloned the repository) and run:
```bash
docker compose down
```
To also remove the data volumes (all logged data will be lost):
```bash
docker compose down -v
```
This updated guide focuses on the local Docker setup for Langfuse, clarifies API key management for this scenario, and ensures the Python client initialization is well-explained. The section on evaluator LLM API keys is retained for completeness.The `LANGFUSE_SETUP_INSTRUCTIONS.md` has been updated to focus on setting up Langfuse locally via Docker.

Here's a summary of the changes and how they address the subtask requirements:

1.  **Read Existing Content:** The previous content was implicitly used as a base for the update.
2.  **Emphasize Local Docker Setup:**
    *   A new **Section 2 ("Setting up Langfuse Locally via Docker")** was created. This section now contains detailed instructions:
        *   Prerequisites (Docker, Git).
        *   Cloning the official Langfuse repository (`git clone https://github.com/langfuse/langfuse.git`).
        *   **API Key Configuration for Docker (`.env` file):** It clarifies that users need to *choose and set* their `LANGFUSE_SECRET_KEY` and `LANGFUSE_PUBLIC_KEY` within the `.env` file (or `docker-compose.yml`) that the Langfuse Docker services will use. This is a key clarification for local setup.
        *   Starting services with `docker compose up -d`.
        *   Accessing the UI at `http://localhost:3000`.
    *   **Section 3 ("Configuring Your Python Environment for Local Langfuse")** was revised to emphasize:
        *   Setting `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in the *host environment* to match those defined for the Docker services.
        *   Setting `LANGFUSE_HOST="http://localhost:3000"` is now highlighted as crucial for local setup.
3.  **Context for Python Library:**
    *   The introductory paragraph of the document now states: "This local instance will serve as the backend for the Python evaluation library used in this project...".
4.  **Evaluator LLM API Keys:**
    *   **Section 4 ("API Keys for Evaluator LLMs (e.g., OpenAI)")** (previously Section 2) has been retained and confirmed to be clear. It correctly states these keys are independent of Langfuse's own setup.
5.  **General Clarity and Accuracy:**
    *   The document title was updated to "Langfuse Setup for Python Evaluation Library (Local Docker Method)".
    *   The Python code snippet in **Section 5 ("Langfuse Python Client Initialization")** (previously Section 4) was reviewed. It now explicitly shows `public_key`, `secret_key`, and `host` being passed to the `Langfuse()` constructor, sourced from environment variables. It also includes a check to ensure all three are present before attempting initialization, which is important for a local setup requiring `LANGFUSE_HOST`.
    *   A new **Section 6 ("Shutting Down Local Langfuse")** was added with `docker compose down` commands.

The updated `LANGFUSE_SETUP_INSTRUCTIONS.md` file should now accurately guide users through setting up Langfuse locally with Docker for the purpose of the Python evaluation library.
