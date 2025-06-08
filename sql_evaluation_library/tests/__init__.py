"""
Test suite for SQL Evaluation Library.

This package contains comprehensive tests for all components of the
SQL evaluation library, including unit tests, integration tests,
and end-to-end tests.

Test Structure:
- test_models_generation.py: Tests for SQL generation models
- test_evaluation_framework.py: Tests for evaluation strategies and framework
- conftest.py: Shared fixtures and configuration

All tests follow pytest conventions and include proper typing annotations
and comprehensive docstrings.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # Import test-specific types when type checking

__all__ = [
    # Re-export common test utilities if needed
] 