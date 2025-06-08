# sql_evaluation_library/src/sql_eval_lib/__init__.py

"""
SQL Evaluation Library

A comprehensive library for evaluating SQL query generation using multiple evaluation methods.
"""

# Import the refactored evaluation framework components
from .evaluation import (
    # Base Classes
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
    # Concrete SQL Classes
    SQLEvaluationContext,
    SQLLLMEvaluator,
    SQLExecutionEvaluator,
)

# Import metrics
from .evaluation.metrics import (
    SemanticSimilarityMetric,
    ExecutionAccuracyMetric,
    SyntaxValidityMetric,
    SQLSyntaxMetric,
    SQLExecutionAccuracyMetric,
)

# Configuration and utilities
from .config import EvaluationConfig, LangfuseConfig, get_config

__version__ = "0.1.0" # Matches pyproject.toml

__all__ = [
    # Evaluation Framework
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
    "SQLEvaluationContext",
    "SQLLLMEvaluator",
    "SQLExecutionEvaluator",

    # Metrics
    "SemanticSimilarityMetric",
    "ExecutionAccuracyMetric",
    "SyntaxValidityMetric",
    "SQLSyntaxMetric",
    "SQLExecutionAccuracyMetric",

    # Configuration
    "EvaluationConfig",
    "LangfuseConfig", 
    "get_config",
]
