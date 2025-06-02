# sql_evaluation_library/src/sql_eval_lib/__init__.py
from .evaluation.orchestrator import Orchestrator
from .models.interface import ModelInterface
from .models.adapters import DummyModelAdapter, OpenAIModelAdapter
from .langfuse.manager import LangfuseClient
# Potentially LLMEvaluationModule and ExecutionEvaluationModule if they are to be used directly
from .evaluation.llm_evaluator import LLMEvaluationModule
from .evaluation.exec_evaluator import ExecutionEvaluationModule

__version__ = "0.1.0" # Matches pyproject.toml

__all__ = [
    "Orchestrator",
    "ModelInterface",
    "DummyModelAdapter",
    "OpenAIModelAdapter",
    "LangfuseClient",
    "LLMEvaluationModule",
    "ExecutionEvaluationModule",
]
