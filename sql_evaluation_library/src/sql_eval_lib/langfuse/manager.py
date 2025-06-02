# sql_evaluation_library/src/sql_eval_lib/langfuse/manager.py
import os
import langfuse
from langfuse.model import TraceClient # For type hinting
from datetime import datetime

class LangfuseClient: # Renamed from LangfuseManager
    """
    Manages interactions with the Langfuse service, including client initialization,
    trace creation, and logging of generations, scores, and events.
    """
    def __init__(self, model_under_test_name: str, 
                 langfuse_host: str = None, 
                 langfuse_public_key: str = None, 
                 langfuse_secret_key: str = None):
        """
        Initializes the LangfuseClient and the Langfuse client.

        Args:
            model_under_test_name (str): Name of the model being evaluated, for tagging/metadata.
            langfuse_host (str, optional): The Langfuse server host.
                Defaults to os.getenv("LANGFUSE_HOST") or "http://localhost:3000".
            langfuse_public_key (str, optional): The Langfuse public key.
                Defaults to os.getenv("LANGFUSE_PUBLIC_KEY").
            langfuse_secret_key (str, optional): The Langfuse secret key.
                Defaults to os.getenv("LANGFUSE_SECRET_KEY").

        Raises:
            ValueError: If public_key or secret_key is not provided and not found in environment.
            ConnectionError: If Langfuse authentication fails.
        """
        public_key = langfuse_public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = langfuse_secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        host = langfuse_host or os.getenv("LANGFUSE_HOST") or "http://localhost:3000" # Default for local Docker

        if not public_key:
            raise ValueError("Langfuse public key not provided and not found in LANGFUSE_PUBLIC_KEY environment variable.")
        if not secret_key:
            raise ValueError("Langfuse secret key not provided and not found in LANGFUSE_SECRET_KEY environment variable.")

        self.model_under_test_name = model_under_test_name
        self.host = host # Store host for reference if needed
        self.enabled = False # Will be set to True on successful auth

        try:
            print(f"Attempting to initialize Langfuse client with host: {host}...")
            self.langfuse = langfuse.Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
                # debug=True # Optional: for more verbose output from the SDK
            )
            if not self.langfuse.auth_check():
                # This specific check might not raise an error itself but return False.
                # The SDK might raise errors during initialization for blatant issues.
                raise ConnectionError("Langfuse authentication failed (auth_check returned False). Check credentials and host.")
            
            self.enabled = True
            print(f"LangfuseClient initialized successfully for host: {host}, model: {self.model_under_test_name}.") # Updated class name in log

        except Exception as e:
            # Catch any exception during Langfuse client init or auth_check
            # This includes potential network issues, SDK errors, etc.
            print(f"Error initializing Langfuse client or during auth_check: {e}")
            # Raising a specific error type helps upstream error handling.
            # Re-raise or raise a custom error if the caller should handle this failure.
            raise ConnectionError(f"Langfuse client initialization failed: {e}")


    def get_trace(self, item_id: str, session_id: str = None, user_id: str = None, tags: list = None) -> TraceClient | None:
        """
        Creates and returns a Langfuse trace object.

        Args:
            item_id (str): A unique identifier for the item being evaluated, used in trace ID and name.
            session_id (str, optional): Session ID for grouping traces.
            user_id (str, optional): User ID associated with the trace.
            tags (list, optional): Additional tags for the trace.

        Returns:
            langfuse.model.TraceClient or None: The created trace object, or None if Langfuse is disabled or trace creation fails.
        """
        if not self.enabled:
            print("LangfuseClient: Logging disabled, cannot get trace.") # Updated class name in log
            return None
        
        default_tags = ["text-to-sql-evaluation", self.model_under_test_name]
        if tags:
            combined_tags = default_tags + [t for t in tags if t not in default_tags]
        else:
            combined_tags = default_tags
            
        try:
            # Using item_id directly as trace id, as per design.
            # The name includes more context.
            trace = self.langfuse.trace(
                id=item_id, 
                name=f"SQL Eval Item {item_id}",
                session_id=session_id,
                user_id=user_id,
                tags=combined_tags
            )
            return trace
        except Exception as e:
            print(f"Langfuse: Error creating trace for item_id '{item_id}'. Error: {e}")
            return None

    def log_model_generation(self, trace: TraceClient, sql_prompt: str, sql_context: str, 
                             generated_sql: str, start_time: datetime, end_time: datetime, 
                             metadata: dict = None) -> None:
        """
        Logs the generation details of the model-under-test.
        """
        if not self.enabled or not trace:
            return

        try:
            trace.generation(
                name=f"{self.model_under_test_name} Generation",
                model=self.model_under_test_name,
                input={"prompt": sql_prompt, "context": sql_context}, 
                output=generated_sql, 
                start_time=start_time,
                end_time=end_time,
                metadata=metadata
            )
        except Exception as e:
            print(f"Langfuse: Error logging model generation for trace_id '{trace.id}'. Error: {e}")


    def log_evaluator_generation(self, trace: TraceClient, prompt_to_evaluator: any, 
                                 evaluation_result: any, start_time: datetime, end_time: datetime, 
                                 model_name: str, metadata: dict = None) -> None:
        """
        Logs the generation details of an LLM evaluator.
        """
        if not self.enabled or not trace:
            return
        
        try:
            trace.generation(
                name=f"{model_name} Evaluation", 
                model=model_name, 
                input=prompt_to_evaluator, 
                output=evaluation_result,  
                start_time=start_time,
                end_time=end_time,
                metadata=metadata
            )
        except Exception as e:
            print(f"Langfuse: Error logging evaluator generation for trace_id '{trace.id}'. Error: {e}")

    def log_score(self, trace: TraceClient, name: str, value: any, comment: str = None) -> None:
        """
        Logs a score to the given trace.
        """
        if not self.enabled or not trace:
            return
            
        try:
            trace.score(name=name, value=value, comment=comment)
        except Exception as e:
            print(f"Langfuse: Error logging score '{name}' for trace_id '{trace.id}'. Error: {e}")
    
    def log_event(self, trace: TraceClient, name: str, metadata: dict = None, **kwargs) -> None:
        """
        Logs a generic event to the trace.
        """
        if not self.enabled or not trace:
            return
        try:
            trace.event(name=name, metadata=metadata, **kwargs)
        except Exception as e:
            print(f"Langfuse: Error logging event '{name}' for trace_id '{trace.id}'. Error: {e}")


    def flush(self) -> None:
        """
        Flushes any buffered Langfuse events to the server.
        """
        if not self.enabled:
            return
            
        try:
            self.langfuse.flush()
        except Exception as e:
            print(f"Langfuse: Error during flush. Error: {e}")
```

The `LangfuseManager` code is now correctly placed in the file.
The `get_trace` method uses `id=item_id` as specified in the design.
Added `log_event` method as it was in the placeholder and is generally useful.

Next, I will update `sql_evaluation_library/__init__.py`.
