"""
Tests for the SQL-specific evaluation metrics.
"""
import pytest

from sql_eval_lib.evaluation.metrics.sql_metrics import SQLSyntaxMetric
from sql_eval_lib.evaluation.base import EvaluationContext

@pytest.mark.asyncio
@pytest.mark.parametrize("sql_query, expected_score, expect_error", [
    # Valid cases
    ("SELECT * FROM my_table", 1.0, False),
    ("SELECT a, b FROM c WHERE d > 5", 1.0, False),
    
    # Invalid cases
    ("SELECT FROM my_table WHERE", 0.0, True), # Guaranteed parse failure
    ("this is not sql", 0.0, True),
    ("", 0.0, True),
    ("  ", 0.0, True),
])
async def test_sql_syntax_metric(sql_query, expected_score, expect_error):
    """
    Tests the SQLSyntaxMetric with various valid and invalid SQL queries.
    """
    # Arrange
    metric = SQLSyntaxMetric()
    context = EvaluationContext(prompt="test", response=sql_query)
    
    # Act
    result = await metric.compute(context)
    
    # Assert
    assert result.value == expected_score
    assert result.name == "sql_syntax_validity"
    if expect_error:
        assert result.error is not None
        print(f"Validated error: {result.error}")
    else:
        assert result.error is None 