"""
Base metric implementations that are domain-agnostic.

These metrics provide fundamental evaluation capabilities that can be
used across different domains (e.g., SQL, code generation, text generation).
They should not contain any logic specific to a single domain.
"""

from abc import abstractmethod
from typing import Dict, Any, Optional

from ..base import EvaluationMetric, EvaluationContext, MetricResult


class SemanticSimilarityMetric(EvaluationMetric):
    """
    Measures semantic similarity between generated response and ground truth.
    
    This is a domain-agnostic metric that can work with any text-based evaluation.
    """
    
    def __init__(self, name: str = "semantic_similarity", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the semantic similarity metric.
        
        Args:
            name: Name of the metric
            config: Configuration with options like:
                - similarity_method: "cosine", "jaccard", "levenshtein" (default: "jaccard")
                - threshold: Minimum similarity for success (default: 0.7)
        """
        super().__init__(name, config)
        self.similarity_method = self.config.get("similarity_method", "jaccard")
        self.threshold = self.config.get("threshold", 0.7)
    
    async def compute(self, context: EvaluationContext) -> MetricResult:
        """Compute semantic similarity between response and ground truth."""
        try:
            if not context.ground_truth:
                return MetricResult(
                    name=self.name,
                    value=0.0,
                    error="No ground truth provided for similarity comparison"
                )
            
            # Compute similarity based on method
            if self.similarity_method == "jaccard":
                similarity = self._jaccard_similarity(context.response, context.ground_truth)
            elif self.similarity_method == "levenshtein":
                similarity = self._levenshtein_similarity(context.response, context.ground_truth)
            elif self.similarity_method == "cosine":
                similarity = self._cosine_similarity(context.response, context.ground_truth)
            else:
                raise ValueError(f"Unknown similarity method: {self.similarity_method}")
            
            return MetricResult(
                name=self.name,
                value=similarity,
                details={
                    "method": self.similarity_method,
                    "threshold": self.threshold,
                    "meets_threshold": similarity >= self.threshold,
                    "response_length": len(context.response),
                    "ground_truth_length": len(context.ground_truth)
                }
            )
            
        except Exception as e:
            return MetricResult(
                name=self.name,
                value=0.0,
                error=f"Failed to compute semantic similarity: {str(e)}"
            )
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity between two texts."""
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _levenshtein_similarity(self, text1: str, text2: str) -> float:
        """Compute normalized Levenshtein similarity."""
        def levenshtein_distance(s1: str, s2: str) -> int:
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        distance = levenshtein_distance(text1.lower(), text2.lower())
        max_len = max(len(text1), len(text2))
        return 1.0 - (distance / max_len) if max_len > 0 else 1.0
    
    def _cosine_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts (simple word-based)."""
        words1 = text1.lower().split()
        words2 = text2.lower().split()
        
        # Create word frequency vectors
        all_words = set(words1 + words2)
        vec1 = [words1.count(word) for word in all_words]
        vec2 = [words2.count(word) for word in all_words]
        
        # Compute cosine similarity
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)


class ExecutionAccuracyMetric(EvaluationMetric):
    """
    Measures accuracy by comparing execution results.
    
    This is a domain-agnostic metric. It relies on the pre-computed
    execution results for both the response and the ground truth to be present
    in the `metadata` of the EvaluationContext. It does not perform any
    execution itself.
    """
    
    def __init__(self, name: str = "execution_accuracy", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        # The key names in the context metadata where the results are stored.
        self.response_result_key = self.config.get("response_result_key", "response_exec_result")
        self.ground_truth_result_key = self.config.get("ground_truth_result_key", "ground_truth_exec_result")

    async def compute(self, context: EvaluationContext) -> MetricResult:
        """
        Computes execution accuracy based on pre-computed results in the context.
        """
        try:
            response_result = context.metadata.get(self.response_result_key)
            ground_truth_result = context.metadata.get(self.ground_truth_result_key)

            if response_result is None or ground_truth_result is None:
                return MetricResult(
                    name=self.name,
                    value=0.0,
                    error=f"Execution results not found in context metadata under keys "
                          f"'{self.response_result_key}' or '{self.ground_truth_result_key}'."
                )

            # The results are expected to be normalized (e.g., sorted list of tuples)
            # by the domain-specific logic that ran the execution.
            is_match = (response_result == ground_truth_result)
            
            return MetricResult(
                name=self.name,
                value=1.0 if is_match else 0.0,
                details={
                    "match": is_match
                }
            )
            
        except Exception as e:
            return MetricResult(
                name=self.name,
                value=0.0,
                error=f"Failed to compute execution accuracy: {str(e)}"
            )


class SyntaxValidityMetric(EvaluationMetric):
    """
    Measures syntax validity based on a pre-computed check.
    
    This is a domain-agnostic metric. It relies on the result of a
    domain-specific syntax check to be present in the `metadata` of the
    EvaluationContext. It does not perform any parsing itself.
    """

    def __init__(self, name: str = "syntax_validity", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.syntax_valid_key = self.config.get("syntax_valid_key", "is_syntactically_valid")
        self.syntax_error_key = self.config.get("syntax_error_key", "syntax_error_details")
    
    async def compute(self, context: EvaluationContext) -> MetricResult:
        """
        Computes syntax validity based on a pre-computed flag in the context.
        """
        try:
            is_valid = context.metadata.get(self.syntax_valid_key)

            if is_valid is None:
                return MetricResult(
                    name=self.name,
                    value=0.0,
                    error=f"Syntax validity flag not found in context metadata under key '{self.syntax_valid_key}'."
                )

            error_details = context.metadata.get(self.syntax_error_key)
            
            return MetricResult(
                name=self.name,
                value=1.0 if is_valid else 0.0,
                error=str(error_details) if not is_valid and error_details else None,
                details={
                    "is_valid": bool(is_valid)
                }
            )
            
        except Exception as e:
            return MetricResult(
                name=self.name,
                value=0.0,
                error=f"Failed to check syntax validity from context: {str(e)}"
            )