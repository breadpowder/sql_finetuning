"""
Core evaluation framework with composition-based metrics and factory pattern.

This module provides the foundational classes for a flexible evaluation system
that uses composition for metrics and factory methods for creating evaluators.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class EvaluationStatus(Enum):
    """Status of an evaluation operation."""
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class EvaluationContext:
    """
    Context object containing all data needed for evaluation.
    
    This is a generic context that can be used for any domain.
    Domain-specific fields should be added via the metadata dictionary.
    """
    prompt: str
    response: str
    ground_truth: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Post-initialization to ensure metadata is not None."""
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MetricResult:
    """Result of a single metric computation."""
    name: str
    value: float
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Whether the metric computation was successful."""
        return self.error is None


@dataclass
class SuiteResult:
    """Result of evaluating a suite of metrics."""
    context_id: str
    metric_results: List[MetricResult]
    overall_score: Optional[float] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        """Whether all metrics were computed successfully."""
        return all(result.success for result in self.metric_results)
    
    @property
    def failed_metrics(self) -> List[MetricResult]:
        """List of metrics that failed to compute."""
        return [result for result in self.metric_results if not result.success]


@dataclass
class EvaluationResult:
    """Result of a complete evaluation using a specific evaluator."""
    evaluator_name: str
    status: EvaluationStatus
    suite_result: Optional[SuiteResult] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        """Whether the evaluation was successful."""
        return self.status == EvaluationStatus.SUCCESS


# ============================================================================
# Core Metric Framework
# ============================================================================

class EvaluationMetric(ABC):
    """
    Abstract base class for evaluation metrics.
    
    Metrics are composable units that compute specific aspects of evaluation.
    They should be domain-agnostic and focus on a single measurement.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the metric.
        
        Args:
            name: Unique name for this metric
            config: Optional configuration dictionary
        """
        self.name = name
        self.config = config or {}
    
    @abstractmethod
    async def compute(self, context: EvaluationContext) -> MetricResult:
        """
        Compute the metric for the given context.
        
        Args:
            context: Evaluation context containing prompt, response, etc.
            
        Returns:
            MetricResult with the computed value and details
        """
        pass
    
    def validate_context(self, context: EvaluationContext) -> tuple[bool, Optional[str]]:
        """
        Validate that the context contains required fields for this metric.
        
        Args:
            context: Evaluation context to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Default implementation checks basic fields
        if not context.prompt:
            return False, "Missing required field: prompt"
        if not context.response:
            return False, "Missing required field: response"
        return True, None


class EvaluationSuite:
    """
    A suite of metrics that can be evaluated together.
    
    This class uses composition to combine multiple metrics and provides
    aggregation strategies for computing overall scores.
    """
    
    def __init__(self, 
                 metrics: List[EvaluationMetric],
                 aggregation_strategy: str = "mean",
                 weights: Optional[Dict[str, float]] = None):
        """
        Initialize the evaluation suite.
        
        Args:
            metrics: List of metrics to include in the suite
            aggregation_strategy: How to aggregate metric scores ("mean", "weighted", "max", "min")
            weights: Optional weights for metrics (used with "weighted" strategy)
        """
        self.metrics = metrics
        self.aggregation_strategy = aggregation_strategy
        self.weights = weights or {}
        
        # Validate weights if provided
        if self.aggregation_strategy == "weighted" and self.weights:
            metric_names = {metric.name for metric in self.metrics}
            weight_names = set(self.weights.keys())
            if not weight_names.issubset(metric_names):
                raise ValueError(f"Weights contain unknown metrics: {weight_names - metric_names}")
    
    async def evaluate(self, context: EvaluationContext) -> SuiteResult:
        """
        Evaluate all metrics in the suite for the given context.
        
        Args:
            context: Evaluation context
            
        Returns:
            SuiteResult with all metric results and aggregated score
        """
        start_time = datetime.now()
        metric_results = []
        
        # Evaluate each metric
        for metric in self.metrics:
            try:
                # Validate context for this metric
                is_valid, error_msg = metric.validate_context(context)
                if not is_valid:
                    metric_results.append(MetricResult(
                        name=metric.name,
                        value=0.0,
                        error=f"Context validation failed: {error_msg}"
                    ))
                    continue
                
                # Compute metric
                result = await metric.compute(context)
                metric_results.append(result)
                
            except Exception as e:
                metric_results.append(MetricResult(
                    name=metric.name,
                    value=0.0,
                    error=f"Metric computation failed: {str(e)}"
                ))
        
        # Calculate execution time
        end_time = datetime.now()
        execution_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Aggregate scores
        overall_score = self._aggregate_scores(metric_results)
        
        # Generate context ID
        context_id = f"eval_{hash(context.prompt + context.response) % 1000000}"
        
        return SuiteResult(
            context_id=context_id,
            metric_results=metric_results,
            overall_score=overall_score,
            execution_time_ms=execution_time_ms,
            metadata={
                "aggregation_strategy": self.aggregation_strategy,
                "total_metrics": len(self.metrics),
                "successful_metrics": len([r for r in metric_results if r.success])
            }
        )
    
    def _aggregate_scores(self, metric_results: List[MetricResult]) -> Optional[float]:
        """Aggregate metric scores based on the configured strategy."""
        successful_results = [r for r in metric_results if r.success]
        
        if not successful_results:
            return None
        
        values = [r.value for r in successful_results]
        
        if self.aggregation_strategy == "mean":
            return sum(values) / len(values)
        elif self.aggregation_strategy == "weighted":
            total_weight = 0.0
            weighted_sum = 0.0
            for result in successful_results:
                weight = self.weights.get(result.name, 1.0)
                weighted_sum += result.value * weight
                total_weight += weight
            return weighted_sum / total_weight if total_weight > 0 else 0.0
        elif self.aggregation_strategy == "max":
            return max(values)
        elif self.aggregation_strategy == "min":
            return min(values)
        else:
            return None


# ============================================================================
# Refactored Abstract Evaluator Framework (Template Method Pattern)
# ============================================================================

class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluators using the Template Method Pattern.
    It defines the overall evaluation algorithm and allows subclasses to provide
    implementations for specific steps.
    """
    def __init__(self, name: str, suite: EvaluationSuite, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the evaluator.
        
        Args:
            name: Name of the evaluator
            suite: Evaluation suite to use for evaluation
            config: Optional configuration
        """
        self.name = name
        self.suite = suite
        self.config = config or {}

    @abstractmethod
    async def _create_evaluation_context(
        self, prompt: str, response: str, context: Dict[str, Any]
    ) -> EvaluationContext:
        """
        Create a domain-specific evaluation context.
        Subclasses MUST implement this to tailor the context object.
        """
        pass

    async def evaluate(self, prompt: str, response: str, context: Dict[str, Any]) -> EvaluationResult:
        """
        Template method that orchestrates the evaluation process.
        This method should not be overridden by subclasses.
        """
        try:
            eval_context = await self._create_evaluation_context(prompt, response, context)
            suite_result = await self.suite.evaluate(eval_context)
            
            return EvaluationResult(
                evaluator_name=self.name,
                status=EvaluationStatus.SUCCESS if suite_result.success else EvaluationStatus.FAILED,
                suite_result=suite_result,
                metadata=self._get_evaluator_metadata()
            )
        except Exception as e:
            return EvaluationResult(
                evaluator_name=self.name,
                status=EvaluationStatus.FAILED,
                error=str(e),
                metadata=self._get_evaluator_metadata()
            )

    def _get_evaluator_metadata(self) -> Dict[str, Any]:
        """
        Returns metadata about the evaluator. Can be extended by subclasses.
        """
        return {"evaluator_type": self.__class__.__name__}


class LLMEvaluator(BaseEvaluator):
    """
    Abstract base for evaluators that use an LLM.
    
    This class can be extended by concrete evaluators that require an LLM client.
    """
    def __init__(self, suite: EvaluationSuite, llm_client: Any, **kwargs):
        """
        Initialize the LLM evaluator.
        
        Args:
            suite: Evaluation suite with LLM-based metrics
            llm_client: Client for LLM API calls
            **kwargs: Additional arguments for BaseEvaluator
        """
        super().__init__(suite=suite, **kwargs)
        self.llm_client = llm_client

    def _get_evaluator_metadata(self) -> Dict[str, Any]:
        """Adds LLM client info to the metadata."""
        metadata = super()._get_evaluator_metadata()
        if self.llm_client:
            metadata["llm_client"] = self.llm_client.__class__.__name__
        return metadata


class ExecutionEvaluator(BaseEvaluator):
    """
    Abstract base for evaluators that execute code or queries.
    
    This class can be extended by concrete evaluators that perform execution.
    It may have its own abstract methods in the future for environment setup.
    """
    def __init__(self, suite: EvaluationSuite, **kwargs):
        """
        Initialize the execution evaluator.
        
        Args:
            suite: Evaluation suite with execution-based metrics
            **kwargs: Additional arguments for BaseEvaluator
        """
        super().__init__(suite=suite, **kwargs)

# The EvaluatorFactory is removed as it's tightly coupled with the old design.
# A new factory can be created later if the new pattern requires it. 