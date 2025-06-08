# sql_evaluation_library/src/sql_eval_lib/evaluation/__init__.py
"""
The evaluation module for the SQL-Eval Library.

This module provides a flexible framework for evaluating model-generated
SQL queries using various metrics and strategies.

Key components:
- `BaseEvaluator`: Abstract base class for all evaluators.
- `LLMEvaluator`: Abstract base class for LLM-based evaluators.
- `ExecutionEvaluator`: Abstract base class for execution-based evaluators.
- `SQLLLMEvaluator`: Concrete evaluator for SQL tasks using an LLM.
- `SQLExecutionEvaluator`: Concrete evaluator for SQL execution tasks.
- `EvaluationSuite`: A container for running a collection of metrics.
- `EvaluationMetric`: The abstract base class for all metrics.
"""

from .base import (
    EvaluationStatus,
    EvaluationContext,
    MetricResult,
    SuiteResult,
    EvaluationResult,
    EvaluationMetric,
    EvaluationSuite,
    BaseEvaluator,
    LLMEvaluator,
    ExecutionEvaluator,
)

from .sql.sql_llm_evaluator import SQLLLMEvaluator, SQLEvaluationContext
from .sql.sql_exec_evaluator import SQLExecutionEvaluator

# Public API for this module
__all__ = [
    # Base Framework
    "EvaluationStatus",
    "EvaluationContext",
    "MetricResult",
    "SuiteResult",
    "EvaluationResult",
    "EvaluationMetric",
    "EvaluationSuite",
    "BaseEvaluator",
    "LLMEvaluator",
    "ExecutionEvaluator",
    # SQL-Specific Implementations
    "SQLEvaluationContext",
    "SQLLLMEvaluator",
    "SQLExecutionEvaluator",
]
