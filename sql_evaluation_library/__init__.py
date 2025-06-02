# sql_evaluation_library/__init__.py

"""
Python Text-to-SQL Evaluation Library
Provides interfaces and adapters for evaluating Text-to-SQL models.
"""

# Import the interface for models
from .interfaces import ModelInterface

# Import concrete model adapter implementations
from .model_adapters import DummyModelAdapter, OpenAIModelAdapter

# Import the LLM Evaluation Module
from .llm_evaluator import LLMEvaluationModule

# Import the Execution Evaluation Module
from .exec_evaluator import ExecutionEvaluationModule

# Import the Langfuse Manager
from .langfuse_manager import LangfuseManager

# Import the SQLEvaluator (Orchestrator)
from .evaluator import SQLEvaluator

# Utility functions (typically not part of the public API unless specifically designed to be)
# from .utils import split_sql_statements


__all__ = [
    "ModelInterface",
    "DummyModelAdapter",
    "OpenAIModelAdapter",
    "LLMEvaluationModule",
    "ExecutionEvaluationModule",
    "LangfuseManager",
    "SQLEvaluator",
]

__version__ = "0.1.4" # Incremented version for new module
```

I've updated `__init__.py` to include `SQLEvaluator` in `__all__` and incremented the version.
All steps for this subtask are now complete.
