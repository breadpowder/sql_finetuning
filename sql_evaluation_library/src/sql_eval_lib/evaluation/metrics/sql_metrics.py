"""
SQL-specific evaluation metrics.

These metrics provide evaluation capabilities specific to the SQL domain,
such as syntax checking, execution comparison, and schema adherence.
"""

import sqlite3
from typing import Dict, Any, Optional, List, Tuple

import sqlglot

from ..base import EvaluationMetric, EvaluationContext, MetricResult
from ...utils.helpers import split_sql_statements

class SQLSyntaxMetric(EvaluationMetric):
    """Checks if the generated SQL is syntactically valid using sqlglot."""

    def __init__(self, name: str = "sql_syntax_validity", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.dialect = self.config.get("dialect", "sqlite")

    async def compute(self, context: EvaluationContext) -> MetricResult:
        """
        Parses the SQL query from the context to check for valid syntax.
        """
        sql_query = context.response
        if not sql_query or not isinstance(sql_query, str) or not sql_query.strip():
            return MetricResult(name=self.name, value=0.0, error="SQL query is empty or invalid.")

        try:
            sqlglot.parse_one(sql_query, read=self.dialect)
            return MetricResult(name=self.name, value=1.0, details={"dialect": self.dialect})
        except sqlglot.errors.ParseError as e:
            return MetricResult(name=self.name, value=0.0, error=str(e), details={"dialect": self.dialect})
        except Exception as e:
            return MetricResult(name=self.name, value=0.0, error=f"An unexpected error occurred during parsing: {e}")


class SQLExecutionAccuracyMetric(EvaluationMetric):
    """
    Executes both the generated and ground truth SQL queries against a database
    and compares if their normalized outputs match.
    """
    def _normalize_results(self, raw_results: List[Tuple]) -> List[Tuple[str, ...]]:
        """
        Normalizes raw database results for comparison by converting all values
        to strings and sorting the rows.
        """
        if not raw_results:
            return []
        stringified_rows = [tuple(map(str, row)) for row in raw_results]
        return sorted(stringified_rows)

    def _execute_sql(self, cursor: sqlite3.Cursor, sql: str) -> Tuple[Optional[List[Tuple[str, ...]]], Optional[str]]:
        """Executes a single SQL query and returns its normalized result or an error."""
        if not sql or not sql.strip():
            return None, "SQL query is empty."
        try:
            cursor.execute(sql)
            raw_output = cursor.fetchall()
            return self._normalize_results(raw_output), None
        except sqlite3.Error as e:
            return None, str(e)

    async def compute(self, context: EvaluationContext) -> MetricResult:
        """
        Sets up a database, executes queries, and compares results.
        """
        sql_setup_script = context.metadata.get("sql_context")
        if not sql_setup_script:
            return MetricResult(name=self.name, value=0.0, error="No 'sql_context' found in metadata to set up the database.")

        conn = None
        try:
            conn = sqlite3.connect(':memory:')
            cursor = conn.cursor()
            
            # Setup database schema and data
            db_setup_statements = split_sql_statements(sql_setup_script)
            for statement in db_setup_statements:
                cursor.execute(statement)
            conn.commit()

            # Execute generated and ground truth queries
            gen_output, gen_error = self._execute_sql(cursor, context.response)
            gt_output, gt_error = self._execute_sql(cursor, context.ground_truth)

            match = gen_output is not None and gt_output is not None and gen_output == gt_output
            
            return MetricResult(
                name=self.name,
                value=1.0 if match else 0.0,
                details={
                    "generated_sql_executed_successfully": gen_error is None,
                    "ground_truth_sql_executed_successfully": gt_error is None,
                    "generated_sql_error": gen_error,
                    "ground_truth_error": gt_error,
                    "outputs_match": match
                }
            )

        except sqlite3.Error as e:
            return MetricResult(name=self.name, value=0.0, error=f"Database setup failed: {e}")
        except Exception as e:
            return MetricResult(name=self.name, value=0.0, error=f"An unexpected error occurred: {e}")
        finally:
            if conn:
                conn.close()


class SQLSchemaAdherenceMetric(EvaluationMetric):
    """
    Checks if the generated SQL query only uses tables and columns
    from the provided database schema. (Placeholder)
    """
    async def compute(self, context: EvaluationContext) -> MetricResult:
        # Future implementation will:
        # 1. Parse schema from context.metadata['sql_context']
        # 2. Parse generated SQL (context.response) with sqlglot
        # 3. Extract all table and column identifiers from the query's AST
        # 4. Compare query identifiers against schema identifiers
        # 5. Return score based on adherence.
        return MetricResult(name=self.name, value=1.0, details={"status": "placeholder"})


class SQLSemanticCorrectnessMetric(EvaluationMetric):
    """
    Uses an LLM to evaluate the semantic correctness of the SQL query
    against the natural language prompt. (Placeholder)
    """
    def __init__(self, name: str = "sql_semantic_correctness", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        # This metric would require an llm_client, passed in via config
        self.llm_client = self.config.get("llm_client")

    async def compute(self, context: EvaluationContext) -> MetricResult:
        if not self.llm_client:
            return MetricResult(name=self.name, value=0.0, error="LLM client not provided in metric config.")
        
        # Future implementation will encapsulate the logic from the old
        # `llm_evaluator`:
        # 1. Construct a detailed prompt for an evaluator LLM.
        # 2. Call the LLM API.
        # 3. Parse the JSON response to get a score and reasoning.
        return MetricResult(name=self.name, value=1.0, details={"status": "placeholder"})


class SQLSemanticMetric(EvaluationMetric):
    """
    Evaluates SQL semantic correctness and logical structure.
    
    This metric analyzes the logical structure of SQL queries and checks
    for semantic issues like missing joins, incorrect aggregations, etc.
    """
    
    def __init__(self, name: str = "sql_semantic", config: Optional[Dict[str, Any]] = None):
        """Initialize the SQL semantic metric."""
        super().__init__(name, config)
        self.check_joins = self.config.get("check_joins", True)
        self.check_aggregations = self.config.get("check_aggregations", True)
        self.check_subqueries = self.config.get("check_subqueries", True)
    
    async def compute(self, context: EvaluationContext) -> MetricResult:
        """Compute SQL semantic correctness score."""
        try:
            if not sqlglot:
                return MetricResult(
                    name=self.name,
                    value=0.0,
                    error="sqlglot package not available for SQL semantic analysis"
                )
            
            # Get SQL query
            sql_query = context.response
            if hasattr(context, 'generated_sql') and getattr(context, 'generated_sql'):
                sql_query = getattr(context, 'generated_sql')
            
            # Parse SQL
            try:
                parsed = sqlglot.parse_one(sql_query, read=self.config.get("dialect", "sqlite"))
                if not parsed:
                    return MetricResult(
                        name=self.name,
                        value=0.0,
                        error="Failed to parse SQL for semantic analysis"
                    )
            except Exception as e:
                return MetricResult(
                    name=self.name,
                    value=0.0,
                    error=f"SQL parsing failed: {str(e)}"
                )
            
            # Perform semantic analysis
            semantic_score, issues = self._analyze_semantics(parsed, sql_query)
            
            return MetricResult(
                name=self.name,
                value=semantic_score,
                details={
                    "semantic_score": semantic_score,
                    "issues_found": issues,
                    "checks_performed": {
                        "joins": self.check_joins,
                        "aggregations": self.check_aggregations,
                        "subqueries": self.check_subqueries
                    }
                }
            )
            
        except Exception as e:
            return MetricResult(
                name=self.name,
                value=0.0,
                error=f"Failed to analyze SQL semantics: {str(e)}"
            )
    
    def _analyze_semantics(self, parsed_sql, original_query: str) -> tuple[float, List[str]]:
        """Analyze SQL semantics and return score and issues."""
        issues = []
        score_deductions = 0.0
        
        # Check for common semantic issues
        query_upper = original_query.upper()
        
        # Check for potential Cartesian products
        if self.check_joins and "FROM" in query_upper and "," in query_upper:
            if "JOIN" not in query_upper and "WHERE" not in query_upper:
                issues.append("Potential Cartesian product detected")
                score_deductions += 0.3
        
        # Check for GROUP BY without aggregate functions
        if self.check_aggregations and "GROUP BY" in query_upper:
            has_aggregates = any(func in query_upper for func in ["COUNT", "SUM", "AVG", "MAX", "MIN"])
            if not has_aggregates:
                issues.append("GROUP BY without aggregate functions")
                score_deductions += 0.2
        
        # Check for HAVING without GROUP BY
        if "HAVING" in query_upper and "GROUP BY" not in query_upper:
            issues.append("HAVING clause without GROUP BY")
            score_deductions += 0.2
        
        # Calculate final score
        final_score = max(0.0, 1.0 - score_deductions)
        
        return final_score, issues


def create_sql_metrics(config: Dict[str, Any]) -> List[EvaluationMetric]:
    """
    Factory function to create a standard set of SQL-specific metrics.
    
    Args:
        config: Configuration dictionary with metric settings
        
    Returns:
        List of configured SQL metrics
    """
    metrics = []
    
    # Default SQL metrics to include
    default_metrics = [
        ("syntax", SQLSyntaxMetric),
        ("semantic", SQLSemanticMetric),
        ("performance", SQLPerformanceMetric)
    ]
    
    # Get enabled metrics from config
    enabled_metrics = config.get("enabled_metrics", [m[0] for m in default_metrics])
    
    for metric_name, metric_class in default_metrics:
        if metric_name in enabled_metrics:
            metric_config = config.get(metric_name, {})
            # Inherit global SQL config
            metric_config.update({
                "dialect": config.get("dialect", "sqlite"),
                "strict_mode": config.get("strict_mode", False)
            })
            
            metrics.append(metric_class(
                name=f"sql_{metric_name}",
                config=metric_config
            ))
    
    return metrics 