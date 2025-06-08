"""
Database Backend Plugin Architecture for SQL Evaluation Library

This module provides a pluggable database backend system that supports
multiple database types including SQLite, Trino, PostgreSQL, MySQL, and custom backends.

Architecture:
- Abstract DatabaseBackend interface defines the contract
- Concrete implementations for each database type
- Factory pattern for backend instantiation
- Registry for automatic discovery
- Type-safe configuration with Pydantic
"""

# Legacy backend system
from .base import DatabaseBackend, DatabaseConfig, QueryResult
from .factory import DatabaseBackendFactory
from .registry import DatabaseBackendRegistry
from .sqlite_backend import SQLiteBackend
from .trino_backend import TrinoBackend
from .postgres_backend import PostgreSQLBackend  
from .mysql_backend import MySQLBackend

# Phase 2.3: Database Connection Validation - New components
from .validation import (
    DatabaseValidator,
    DatabaseConfig as ValidationDatabaseConfig,
    DatabaseType,
    ValidationResult,
    ValidationStatus,
    PerformanceMetrics,
    create_database_configs_from_dict,
    validate_databases,
    print_validation_report
)

# Registry instance for automatic discovery
registry = DatabaseBackendRegistry()

# Register built-in backends
registry.register('sqlite', SQLiteBackend)
registry.register('trino', TrinoBackend)
registry.register('postgres', PostgreSQLBackend)
registry.register('mysql', MySQLBackend)

# Factory instance using the registry
factory = DatabaseBackendFactory(registry)

__all__ = [
    # Legacy backend system
    'DatabaseBackend',
    'DatabaseConfig', 
    'QueryResult',
    'DatabaseBackendFactory',
    'DatabaseBackendRegistry',
    'SQLiteBackend',
    'TrinoBackend',
    'PostgreSQLBackend',
    'MySQLBackend',
    'registry',
    'factory',
    
    # Phase 2.3: Database Validation
    'DatabaseValidator',
    'ValidationDatabaseConfig',
    'DatabaseType',
    'ValidationResult',
    'ValidationStatus',
    'PerformanceMetrics',
    'create_database_configs_from_dict',
    'validate_databases',
    'print_validation_report',
] 