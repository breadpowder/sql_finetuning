"""
Database connection validation and testing utilities.

This module provides comprehensive validation for database connections,
including connection string validation, schema verification, performance testing,
and error handling and recovery mechanisms.
"""

import time
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
from pathlib import Path

try:
    import sqlalchemy
    from sqlalchemy import create_engine, text, MetaData, inspect
    from sqlalchemy.engine import Engine
    from sqlalchemy.exc import SQLAlchemyError
except ImportError:
    print("Warning: sqlalchemy package not installed. Database validation will not work.")
    sqlalchemy = None


class ValidationStatus(Enum):
    """Status of validation checks."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"


class DatabaseType(Enum):
    """Supported database types."""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    TRINO = "trino"
    CUSTOM = "custom"


@dataclass
class DatabaseConfig:
    """Configuration for a database connection."""
    name: str
    db_type: DatabaseType
    connection_string: str
    test_query: str = "SELECT 1"
    timeout: int = 30
    pool_size: int = 5
    max_overflow: int = 10
    expected_schema: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of a database validation check."""
    database_name: str
    test_name: str
    status: ValidationStatus
    message: str
    duration_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for database operations."""
    database_name: str
    operation: str
    timestamp: datetime
    duration_ms: float
    success: bool
    rows_affected: Optional[int] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "database_name": self.database_name,
            "operation": self.operation,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "success": self.success,
            "rows_affected": self.rows_affected,
            "error_message": self.error_message
        }


class DatabaseValidator:
    """
    Validates database connections and performs comprehensive testing.
    
    This class provides validation of database connections, schema verification,
    performance testing, and error handling validation.
    """
    
    def __init__(self, databases: List[DatabaseConfig]):
        """
        Initialize the validator.
        
        Args:
            databases: List of database configurations to validate
        """
        self.databases = {db.name: db for db in databases}
        self.results: List[ValidationResult] = []
        self.performance_metrics: List[PerformanceMetrics] = []
        self._engines: Dict[str, Engine] = {}
    
    def validate_all(self) -> Dict[str, Any]:
        """
        Run validation tests on all databases.
        
        Returns:
            Dictionary with validation summary and detailed results
        """
        self.results.clear()
        
        for name, db_config in self.databases.items():
            print(f"\nValidating database: {name} ({db_config.db_type.value})")
            
            # Test 1: Connection string validation
            self._validate_connection_string(db_config)
            
            # Test 2: Basic connectivity
            self._validate_connectivity(db_config)
            
            # Test 3: Schema verification
            if db_config.expected_schema:
                self._validate_schema(db_config)
            
            # Test 4: Performance testing
            self._validate_performance(db_config)
            
            # Test 5: Error handling
            self._validate_error_handling(db_config)
        
        # Clean up connections
        self._cleanup_connections()
        
        return self._generate_summary()
    
    def validate_database(self, database_name: str) -> List[ValidationResult]:
        """
        Validate a specific database.
        
        Args:
            database_name: Name of the database to validate
            
        Returns:
            List of validation results for the database
        """
        if database_name not in self.databases:
            return [ValidationResult(
                database_name=database_name,
                test_name="database_exists",
                status=ValidationStatus.FAIL,
                message=f"Database '{database_name}' not found"
            )]
        
        db_config = self.databases[database_name]
        start_results_count = len(self.results)
        
        # Run all validation tests
        self._validate_connection_string(db_config)
        self._validate_connectivity(db_config)
        
        if db_config.expected_schema:
            self._validate_schema(db_config)
        
        self._validate_performance(db_config)
        self._validate_error_handling(db_config)
        
        # Return only the results for this database
        return self.results[start_results_count:]
    
    def _validate_connection_string(self, db_config: DatabaseConfig) -> None:
        """Validate database connection string format."""
        start_time = time.time()
        
        try:
            if sqlalchemy is None:
                self._add_result(db_config, "connection_string", ValidationStatus.FAIL,
                               "SQLAlchemy package not available")
                return
            
            # Basic format validation
            if not db_config.connection_string:
                self._add_result(db_config, "connection_string", ValidationStatus.FAIL,
                               "Connection string is empty")
                return
            
            # Check for required components based on database type
            conn_str = db_config.connection_string.lower()
            
            if db_config.db_type == DatabaseType.SQLITE:
                if not (conn_str.startswith("sqlite://") or conn_str.startswith("sqlite:////")):
                    self._add_result(db_config, "connection_string", ValidationStatus.FAIL,
                                   "SQLite connection string must start with sqlite:// or sqlite:///")
                    return
            
            elif db_config.db_type == DatabaseType.POSTGRESQL:
                if not (conn_str.startswith("postgresql://") or conn_str.startswith("postgres://")):
                    self._add_result(db_config, "connection_string", ValidationStatus.FAIL,
                                   "PostgreSQL connection string must start with postgresql:// or postgres://")
                    return
            
            elif db_config.db_type == DatabaseType.MYSQL:
                if not conn_str.startswith("mysql://"):
                    self._add_result(db_config, "connection_string", ValidationStatus.FAIL,
                                   "MySQL connection string must start with mysql://")
                    return
            
            elif db_config.db_type == DatabaseType.TRINO:
                if not conn_str.startswith("trino://"):
                    self._add_result(db_config, "connection_string", ValidationStatus.FAIL,
                                   "Trino connection string must start with trino://")
                    return
            
            # Try to parse the connection string
            try:
                engine = sqlalchemy.create_engine(
                    db_config.connection_string,
                    pool_size=1,
                    max_overflow=0,
                    pool_pre_ping=True
                )
                duration_ms = (time.time() - start_time) * 1000
                
                self._add_result(db_config, "connection_string", ValidationStatus.PASS,
                               "Connection string format is valid",
                               duration_ms=duration_ms,
                               details={"dialect": str(engine.dialect.name)})
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                self._add_result(db_config, "connection_string", ValidationStatus.FAIL,
                               f"Invalid connection string format: {str(e)}",
                               duration_ms=duration_ms)
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._add_result(db_config, "connection_string", ValidationStatus.FAIL,
                           f"Connection string validation error: {str(e)}",
                           duration_ms=duration_ms)
    
    def _validate_connectivity(self, db_config: DatabaseConfig) -> None:
        """Validate basic database connectivity."""
        start_time = time.time()
        
        try:
            if sqlalchemy is None:
                self._add_result(db_config, "connectivity", ValidationStatus.FAIL,
                               "SQLAlchemy package not available")
                return
            
            # Create engine with connection pooling
            engine = sqlalchemy.create_engine(
                db_config.connection_string,
                pool_size=db_config.pool_size,
                max_overflow=db_config.max_overflow,
                pool_pre_ping=True,
                connect_args={"connect_timeout": db_config.timeout}
            )
            
            # Test connection
            with engine.connect() as connection:
                result = connection.execute(text(db_config.test_query))
                test_result = result.fetchone()
                
            duration_ms = (time.time() - start_time) * 1000
            
            self._add_result(db_config, "connectivity", ValidationStatus.PASS,
                           "Successfully connected to database",
                           duration_ms=duration_ms,
                           details={
                               "test_query": db_config.test_query,
                               "test_result": str(test_result),
                               "dialect": str(engine.dialect.name)
                           })
            
            # Store engine for reuse in other tests
            self._engines[db_config.name] = engine
            
        except sqlalchemy.exc.TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            self._add_result(db_config, "connectivity", ValidationStatus.FAIL,
                           f"Connection timeout after {db_config.timeout} seconds",
                           duration_ms=duration_ms)
        except sqlalchemy.exc.OperationalError as e:
            duration_ms = (time.time() - start_time) * 1000
            self._add_result(db_config, "connectivity", ValidationStatus.FAIL,
                           f"Database connection failed: {str(e)}",
                           duration_ms=duration_ms)
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._add_result(db_config, "connectivity", ValidationStatus.FAIL,
                           f"Unexpected connectivity error: {str(e)}",
                           duration_ms=duration_ms)
    
    def _validate_schema(self, db_config: DatabaseConfig) -> None:
        """Validate database schema against expected structure."""
        start_time = time.time()
        
        try:
            engine = self._engines.get(db_config.name)
            if engine is None:
                self._add_result(db_config, "schema_validation", ValidationStatus.SKIP,
                               "No connection available for schema validation")
                return
            
            # Get database metadata
            metadata = MetaData()
            inspector = inspect(engine)
            
            # Get table names
            table_names = inspector.get_table_names()
            
            # Validate expected tables exist
            expected_tables = db_config.expected_schema.get("tables", [])
            missing_tables = []
            extra_tables = []
            
            for table_name in expected_tables:
                if table_name not in table_names:
                    missing_tables.append(table_name)
            
            # Check for unexpected tables (optional warning)
            if db_config.expected_schema.get("strict", False):
                for table_name in table_names:
                    if table_name not in expected_tables:
                        extra_tables.append(table_name)
            
            duration_ms = (time.time() - start_time) * 1000
            
            if missing_tables:
                self._add_result(db_config, "schema_validation", ValidationStatus.FAIL,
                               f"Missing expected tables: {', '.join(missing_tables)}",
                               duration_ms=duration_ms,
                               details={
                                   "missing_tables": missing_tables,
                                   "found_tables": table_names,
                                   "extra_tables": extra_tables
                               })
            elif extra_tables:
                self._add_result(db_config, "schema_validation", ValidationStatus.WARNING,
                               f"Found unexpected tables: {', '.join(extra_tables)}",
                               duration_ms=duration_ms,
                               details={
                                   "extra_tables": extra_tables,
                                   "found_tables": table_names
                               })
            else:
                self._add_result(db_config, "schema_validation", ValidationStatus.PASS,
                               "Schema validation successful",
                               duration_ms=duration_ms,
                               details={
                                   "found_tables": table_names,
                                   "validated_tables": expected_tables
                               })
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._add_result(db_config, "schema_validation", ValidationStatus.FAIL,
                           f"Schema validation error: {str(e)}",
                           duration_ms=duration_ms)
    
    def _validate_performance(self, db_config: DatabaseConfig) -> None:
        """Run performance tests on the database."""
        start_time = time.time()
        
        try:
            engine = self._engines.get(db_config.name)
            if engine is None:
                self._add_result(db_config, "performance", ValidationStatus.SKIP,
                               "No connection available for performance testing")
                return
            
            # Test 1: Simple query performance
            query_start = time.time()
            with engine.connect() as connection:
                result = connection.execute(text(db_config.test_query))
                result.fetchall()
            query_duration = (time.time() - query_start) * 1000
            
            # Record performance metric
            metric = PerformanceMetrics(
                database_name=db_config.name,
                operation="simple_query",
                timestamp=datetime.utcnow(),
                duration_ms=query_duration,
                success=True
            )
            self.performance_metrics.append(metric)
            
            # Test 2: Connection pool performance
            pool_start = time.time()
            connections = []
            try:
                # Create multiple connections to test pool
                for i in range(min(3, db_config.pool_size)):
                    conn = engine.connect()
                    connections.append(conn)
                    conn.execute(text(db_config.test_query))
                
                pool_duration = (time.time() - pool_start) * 1000
                
                # Record pool performance metric
                pool_metric = PerformanceMetrics(
                    database_name=db_config.name,
                    operation="connection_pool",
                    timestamp=datetime.utcnow(),
                    duration_ms=pool_duration,
                    success=True,
                    rows_affected=len(connections)
                )
                self.performance_metrics.append(pool_metric)
                
            finally:
                # Clean up connections
                for conn in connections:
                    conn.close()
            
            total_duration = (time.time() - start_time) * 1000
            
            # Evaluate performance
            if query_duration > 5000:  # 5 seconds
                status = ValidationStatus.WARNING
                message = f"Query performance is slow: {query_duration:.1f}ms"
            elif query_duration > 1000:  # 1 second
                status = ValidationStatus.WARNING
                message = f"Query performance is acceptable: {query_duration:.1f}ms"
            else:
                status = ValidationStatus.PASS
                message = f"Query performance is good: {query_duration:.1f}ms"
            
            self._add_result(db_config, "performance", status, message,
                           duration_ms=total_duration,
                           details={
                               "query_duration_ms": query_duration,
                               "pool_duration_ms": pool_duration,
                               "pool_connections_tested": len(connections)
                           })
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            # Record failed performance metric
            error_metric = PerformanceMetrics(
                database_name=db_config.name,
                operation="performance_test",
                timestamp=datetime.utcnow(),
                duration_ms=duration_ms,
                success=False,
                error_message=str(e)
            )
            self.performance_metrics.append(error_metric)
            
            self._add_result(db_config, "performance", ValidationStatus.FAIL,
                           f"Performance test failed: {str(e)}",
                           duration_ms=duration_ms)
    
    def _validate_error_handling(self, db_config: DatabaseConfig) -> None:
        """Test error handling and recovery mechanisms."""
        start_time = time.time()
        
        try:
            engine = self._engines.get(db_config.name)
            if engine is None:
                self._add_result(db_config, "error_handling", ValidationStatus.SKIP,
                               "No connection available for error handling testing")
                return
            
            error_tests_passed = 0
            total_error_tests = 3
            
            # Test 1: Invalid query handling
            try:
                with engine.connect() as connection:
                    connection.execute(text("SELECT * FROM non_existent_table_12345"))
            except Exception:
                error_tests_passed += 1  # Expected to fail
            
            # Test 2: Connection recovery after error
            try:
                with engine.connect() as connection:
                    connection.execute(text(db_config.test_query))
                error_tests_passed += 1  # Should work after error
            except Exception:
                pass  # Connection didn't recover
            
            # Test 3: Transaction rollback handling
            try:
                with engine.connect() as connection:
                    with connection.begin() as transaction:
                        connection.execute(text(db_config.test_query))
                        # Simulate error and rollback
                        transaction.rollback()
                error_tests_passed += 1  # Should handle rollback
            except Exception:
                pass  # Rollback didn't work
            
            duration_ms = (time.time() - start_time) * 1000
            
            if error_tests_passed == total_error_tests:
                self._add_result(db_config, "error_handling", ValidationStatus.PASS,
                               "Error handling tests passed",
                               duration_ms=duration_ms,
                               details={"tests_passed": f"{error_tests_passed}/{total_error_tests}"})
            elif error_tests_passed >= 2:
                self._add_result(db_config, "error_handling", ValidationStatus.WARNING,
                               f"Some error handling tests failed: {error_tests_passed}/{total_error_tests}",
                               duration_ms=duration_ms,
                               details={"tests_passed": f"{error_tests_passed}/{total_error_tests}"})
            else:
                self._add_result(db_config, "error_handling", ValidationStatus.FAIL,
                               f"Error handling tests mostly failed: {error_tests_passed}/{total_error_tests}",
                               duration_ms=duration_ms,
                               details={"tests_passed": f"{error_tests_passed}/{total_error_tests}"})
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._add_result(db_config, "error_handling", ValidationStatus.FAIL,
                           f"Error handling test failed: {str(e)}",
                           duration_ms=duration_ms)
    
    def _add_result(self, db_config: DatabaseConfig, test_name: str, status: ValidationStatus,
                   message: str, duration_ms: Optional[float] = None,
                   details: Optional[Dict[str, Any]] = None) -> None:
        """Add a validation result."""
        result = ValidationResult(
            database_name=db_config.name,
            test_name=test_name,
            status=status,
            message=message,
            duration_ms=duration_ms,
            details=details
        )
        self.results.append(result)
    
    def _cleanup_connections(self) -> None:
        """Clean up database connections."""
        for engine in self._engines.values():
            try:
                engine.dispose()
            except Exception as e:
                print(f"Error disposing engine: {e}")
        self._engines.clear()
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate validation summary."""
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r.status == ValidationStatus.PASS])
        failed_tests = len([r for r in self.results if r.status == ValidationStatus.FAIL])
        warning_tests = len([r for r in self.results if r.status == ValidationStatus.WARNING])
        skipped_tests = len([r for r in self.results if r.status == ValidationStatus.SKIP])
        
        overall_status = "PASS"
        if failed_tests > 0:
            overall_status = "FAIL"
        elif warning_tests > 0:
            overall_status = "WARNING"
        
        # Calculate database-specific summaries
        database_summaries = {}
        for database_name in self.databases.keys():
            db_results = [r for r in self.results if r.database_name == database_name]
            db_passed = len([r for r in db_results if r.status == ValidationStatus.PASS])
            db_failed = len([r for r in db_results if r.status == ValidationStatus.FAIL])
            
            db_status = "PASS" if db_failed == 0 else "FAIL"
            database_summaries[database_name] = {
                "status": db_status,
                "total_tests": len(db_results),
                "passed": db_passed,
                "failed": db_failed
            }
        
        total_duration = sum(r.duration_ms for r in self.results if r.duration_ms is not None)
        
        return {
            "overall_status": overall_status,
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "warnings": warning_tests,
                "skipped": skipped_tests,
                "total_duration_ms": total_duration
            },
            "database_summaries": database_summaries,
            "results": [
                {
                    "database_name": r.database_name,
                    "test_name": r.test_name,
                    "status": r.status.value,
                    "message": r.message,
                    "duration_ms": r.duration_ms,
                    "details": r.details
                }
                for r in self.results
            ],
            "performance_metrics": [m.to_dict() for m in self.performance_metrics],
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_results(self) -> List[ValidationResult]:
        """Get all validation results."""
        return self.results.copy()
    
    def get_performance_metrics(self) -> List[PerformanceMetrics]:
        """Get all performance metrics."""
        return self.performance_metrics.copy()


def create_database_configs_from_dict(configs: Dict[str, Any]) -> List[DatabaseConfig]:
    """
    Create database configurations from dictionary.
    
    Args:
        configs: Dictionary with database configurations
        
    Returns:
        List of database configurations
    """
    database_configs = []
    
    for name, config in configs.items():
        db_type_str = config.get("type", "custom")
        db_type = DatabaseType(db_type_str)
        
        db_config = DatabaseConfig(
            name=name,
            db_type=db_type,
            connection_string=config["connection_string"],
            test_query=config.get("test_query", "SELECT 1"),
            timeout=config.get("timeout", 30),
            pool_size=config.get("pool_size", 5),
            max_overflow=config.get("max_overflow", 10),
            expected_schema=config.get("expected_schema"),
            metadata=config.get("metadata", {})
        )
        
        database_configs.append(db_config)
    
    return database_configs


def validate_databases(databases: List[DatabaseConfig]) -> Dict[str, Any]:
    """
    Validate a list of database configurations.
    
    Args:
        databases: List of database configurations to validate
        
    Returns:
        Validation results dictionary
    """
    validator = DatabaseValidator(databases)
    return validator.validate_all()


def print_validation_report(validation_results: Dict[str, Any]) -> None:
    """
    Print a formatted validation report.
    
    Args:
        validation_results: Results from validate_databases()
    """
    print("\n" + "="*70)
    print("DATABASE VALIDATION REPORT")
    print("="*70)
    
    summary = validation_results["summary"]
    print(f"Overall Status: {validation_results['overall_status']}")
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Warnings: {summary['warnings']}")
    print(f"Skipped: {summary['skipped']}")
    print(f"Total Duration: {summary['total_duration_ms']:.1f}ms")
    
    print("\n" + "-"*70)
    print("DATABASE SUMMARIES")
    print("-"*70)
    
    for db_name, summary in validation_results.get("database_summaries", {}).items():
        status_symbol = "✅" if summary["status"] == "PASS" else "❌"
        print(f"{status_symbol} {db_name}: {summary['passed']}/{summary['total_tests']} tests passed")
    
    print("\n" + "-"*70)
    print("DETAILED RESULTS")
    print("-"*70)
    
    current_database = None
    for result in validation_results["results"]:
        if result["database_name"] != current_database:
            current_database = result["database_name"]
            print(f"\n🗄️  {current_database}")
        
        status_symbol = {
            "pass": "  ✅",
            "fail": "  ❌",
            "warning": "  ⚠️",
            "skip": "  ⏭️"
        }.get(result["status"], "  ?")
        
        duration = f" ({result['duration_ms']:.1f}ms)" if result["duration_ms"] else ""
        print(f"{status_symbol} {result['test_name']}: {result['message']}{duration}")
    
    # Performance metrics summary
    metrics = validation_results.get("performance_metrics", [])
    if metrics:
        print("\n" + "-"*70)
        print("PERFORMANCE METRICS")
        print("-"*70)
        
        for metric in metrics[-5:]:  # Show last 5 metrics
            status_symbol = "✅" if metric["success"] else "❌"
            print(f"{status_symbol} {metric['database_name']} - {metric['operation']}: "
                  f"{metric['duration_ms']:.1f}ms")
    
    print("="*70) 