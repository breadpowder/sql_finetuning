"""
Concrete implementation of an LLM-based evaluator for the SQL domain.
"""

from typing import Dict, Any
from dataclasses import dataclass, field

from ..base import LLMEvaluator, EvaluationContext, EvaluationSuite

@dataclass
class SQLEvaluationContext(EvaluationContext):
    """
    A domain-specific context for SQL evaluation.
    It provides typed access to SQL-related data.
    """
    sql_context: str = ""
    
    def __post_init__(self):
        """
        Retrieves SQL context from metadata if not provided directly.
        This ensures backward compatibility with the generic context.
        """
        super().__post_init__()
        if not self.sql_context and self.metadata:
            self.sql_context = self.metadata.get("sql_context", "")

class SQLLLMEvaluator(LLMEvaluator):
    """
    A concrete LLM evaluator for SQL tasks.
    It implements the abstract `_create_evaluation_context` method
    to produce a SQL-specific context.
    """

    def __init__(self, suite: EvaluationSuite, llm_client: Any, **kwargs):
        """
        Initializes the SQL LLM Evaluator.
        
        Args:
            suite: An evaluation suite containing SQL-specific metrics
                   (e.g., SQLSyntaxMetric, SQLSemanticCorrectnessMetric).
            llm_client: Client for LLM API calls.
            **kwargs: Additional arguments for the base evaluator.
        """
        # Set a default name if not provided
        kwargs.setdefault("name", "sql_llm_evaluator")
        super().__init__(suite=suite, llm_client=llm_client, **kwargs)

    async def _create_evaluation_context(
        self, prompt: str, response: str, context: Dict[str, Any]
    ) -> SQLEvaluationContext:
        """
        Creates a SQL-specific evaluation context from the provided inputs.
        
        This method is the core of the concrete implementation, fulfilling the
        contract required by the BaseEvaluator's template method.

        Args:
            prompt: The natural language question.
            response: The generated SQL query to be evaluated.
            context: A dictionary containing additional data, expected to have
                     'ground_truth' and 'sql_context' (the DB schema).

        Returns:
            An SQLEvaluationContext instance populated with all necessary data
            for the SQL metrics in the evaluation suite.
        """
        return SQLEvaluationContext(
            prompt=prompt,
            response=response,
            ground_truth=context.get("ground_truth"),
            sql_context=context.get("sql_context", ""),
            metadata=context  # Pass the full context dict for other uses
        ) 