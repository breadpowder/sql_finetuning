"""
Evaluation metrics module.

This module provides composable metrics that can be used with the evaluation
framework. Metrics are separated into domain-agnostic base metrics and
domain-specific metrics (e.g., for SQL).
"""

from .base_metrics import (
    SemanticSimilarityMetric,
    ExecutionAccuracyMetric,
    SyntaxValidityMetric,
)

from .sql_metrics import (
    SQLSyntaxMetric,
    SQLExecutionAccuracyMetric,
    SQLSchemaAdherenceMetric,
    SQLSemanticCorrectnessMetric,
)


__all__ = [
    # Base (Domain-Agnostic) Metrics
    "SemanticSimilarityMetric",
    "ExecutionAccuracyMetric", 
    "SyntaxValidityMetric",
    
    # SQL-Specific Metrics
    "SQLSyntaxMetric",
    "SQLExecutionAccuracyMetric",
    "SQLSchemaAdherenceMetric",
    "SQLSemanticCorrectnessMetric",
] 