"""
Tests for the abstract evaluation framework, including the BaseEvaluator
and the metric suite.
"""
from typing import Dict, Any

import pytest
from pytest_mock import MockerFixture

from sql_eval_lib.evaluation.base import (
    BaseEvaluator, EvaluationContext, EvaluationSuite, MetricResult,
    EvaluationStatus, SuiteResult
)

# A simple, concrete implementation of BaseEvaluator for testing purposes.
class MockEvaluator(BaseEvaluator):
    """A mock concrete evaluator to test the abstract base class."""
    async def _create_evaluation_context(
        self, prompt: str, response: str, context: Dict[str, Any]
    ) -> EvaluationContext:
        """Creates a basic evaluation context for testing."""
        return EvaluationContext(
            prompt=prompt, response=response, ground_truth=context.get("gt"), metadata=context
        )

@pytest.mark.asyncio
async def test_base_evaluator_template_method_success(mocker: MockerFixture):
    """
    Verify the BaseEvaluator's `evaluate` template method correctly
    orchestrates the evaluation process on a successful run.
    """
    # Arrange
    mock_metric_result = MetricResult(name="test_metric", value=1.0)
    mock_suite_result = SuiteResult(
        context_id="test_id", metric_results=[mock_metric_result], overall_score=1.0
    )
    
    # Mock the EvaluationSuite to control its behavior
    mock_suite = mocker.MagicMock(spec=EvaluationSuite)
    # The mock's evaluate method must be an async mock
    mock_suite.evaluate = mocker.AsyncMock(return_value=mock_suite_result)

    evaluator = MockEvaluator(name="test_eval", suite=mock_suite)

    # Act
    result = await evaluator.evaluate("test prompt", "test response", {"gt": "test gt"})

    # Assert
    # 1. Check that the suite's evaluate method was called once
    mock_suite.evaluate.assert_called_once()
    
    # 2. Check that the context was created and passed to the suite
    call_args = mock_suite.evaluate.call_args[0]
    passed_context: EvaluationContext = call_args[0]
    assert isinstance(passed_context, EvaluationContext)
    assert passed_context.prompt == "test prompt"
    
    # 3. Check that the final EvaluationResult is correctly formed
    assert result.status == EvaluationStatus.SUCCESS
    assert result.evaluator_name == "test_eval"
    assert result.suite_result == mock_suite_result
    assert result.error is None
    assert result.metadata["evaluator_type"] == "MockEvaluator"

@pytest.mark.asyncio
async def test_base_evaluator_template_method_failure(mocker: MockerFixture):
    """
    Verify the BaseEvaluator's `evaluate` template method correctly
    handles exceptions during the process.
    """
    # Arrange
    # Mock the _create_evaluation_context to raise an exception
    mocker.patch.object(
        MockEvaluator, 
        '_create_evaluation_context',
        side_effect=ValueError("Failed to create context")
    )

    mock_suite = mocker.MagicMock(spec=EvaluationSuite)
    evaluator = MockEvaluator(name="test_eval_fail", suite=mock_suite)
    
    # Act
    result = await evaluator.evaluate("p", "r", {})
    
    # Assert
    # 1. The suite should not have been called
    mock_suite.evaluate.assert_not_called()
    
    # 2. The result should reflect the failure
    assert result.status == EvaluationStatus.FAILED
    assert result.evaluator_name == "test_eval_fail"
    assert result.suite_result is None
    assert "Failed to create context" in result.error 