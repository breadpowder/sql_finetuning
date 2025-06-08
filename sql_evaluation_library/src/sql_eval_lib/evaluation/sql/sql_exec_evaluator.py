"""
Concrete implementation of an execution-based evaluator for the SQL domain.
"""

from typing import Dict, Any

from ..base import ExecutionEvaluator, EvaluationSuite
from .sql_llm_evaluator import SQLEvaluationContext # Re-use the same context

class SQLExecutionEvaluator(ExecutionEvaluator):
    """
    A concrete execution evaluator for SQL tasks.
    It implements the abstract `_create_evaluation_context` method
    to produce a SQL-specific context for execution metrics.
    """

    def __init__(self, suite: EvaluationSuite, **kwargs):
        """
        Initializes the SQL Execution Evaluator.
        
        Args:
            suite: An evaluation suite containing SQL execution metrics
                   (e.g., SQLExecutionAccuracyMetric).
            **kwargs: Additional arguments for the base evaluator.
        """
        # Set a default name if not provided
        kwargs.setdefault("name", "sql_execution_evaluator")
        super().__init__(suite=suite, **kwargs)

    async def _create_evaluation_context(
        self, prompt: str, response: str, context: Dict[str, Any]
    ) -> SQLEvaluationContext:
        """
        Creates a SQL-specific evaluation context from the provided inputs.
        
        This method fulfills the contract required by the BaseEvaluator's
        template method, preparing the data needed by SQL execution metrics.

        Args:
            prompt: The natural language question.
            response: The generated SQL query to be executed and evaluated.
            context: A dictionary containing additional data, expected to have
                     'ground_truth' (the reference SQL) and 
                     'sql_context' (the DB schema and data).

        Returns:
            An SQLEvaluationContext instance populated with all necessary data.
        """
        return SQLEvaluationContext(
            prompt=prompt,
            response=response,
            ground_truth=context.get("ground_truth"),
            sql_context=context.get("sql_context", ""),
            metadata=context
        ) 