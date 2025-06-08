# sql_evaluation_library/tests/conftest.py
"""
Shared fixtures and configuration for pytest.

This file contains shared fixtures and configuration that can be used
across all test modules in the test suite.
"""

from typing import TYPE_CHECKING, Generator
import pytest
import subprocess
import os
from pathlib import Path

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


@pytest.fixture(scope="session")
def project_root() -> str:
    """
    Returns the project root directory.
    
    Returns:
        str: Absolute path to the project root directory
    """
    # Assuming conftest.py is in sql_evaluation_library/tests/
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def mock_config() -> Generator[dict, None, None]:
    """
    Provides a mock configuration for testing.
    
    Yields:
        dict: Mock configuration dictionary with common test values
    """
    config = {
        "model": {
            "api_key": "fake-api-key",
            "adapter_type": "openai",
            "model_name": "gpt-3.5-turbo",
            "temperature": 0.1,
            "max_tokens": 1000
        },
        "database": {
            "connection_string": "sqlite:///:memory:",
            "type": "sqlite"
        }
    }
    yield config


@pytest.fixture
def sample_evaluation_context() -> dict:
    """
    Provides a sample evaluation context for testing.
    
    Returns:
        dict: Sample evaluation context with typical SQL evaluation data
    """
    return {
        "item_id": "test_1",
        "prompt": "Count all users in the database",
        "context": "CREATE TABLE users (id INTEGER PRIMARY KEY, name VARCHAR(100), age INTEGER);",
        "generated_output": "SELECT COUNT(*) FROM users;",
        "reference_output": "SELECT COUNT(id) FROM users;",
        "sql_prompt": "Count all users in the database",
        "sql_context": "CREATE TABLE users (id INTEGER PRIMARY KEY, name VARCHAR(100), age INTEGER);",
        "generated_sql": "SELECT COUNT(*) FROM users;",
        "ground_truth_sql": "SELECT COUNT(id) FROM users;"
    }


@pytest.fixture(scope="session", autouse=True)
def ensure_langfuse_submodule():
    """
    Ensure langfuse submodule is properly initialized before tests run.
    This fixture runs automatically for the entire test session.
    """
    langfuse_path = Path(__file__).parent.parent / "langfuse"
    
    if not langfuse_path.exists() or not (langfuse_path / ".git").exists():
        print("🔄 Initializing langfuse submodule...")
        try:
            # Initialize and update submodules
            subprocess.run(
                ["git", "submodule", "update", "--init", "--recursive"],
                cwd=Path(__file__).parent.parent,
                check=True,
                capture_output=True,
                text=True
            )
            print("✅ Langfuse submodule initialized successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to initialize langfuse submodule: {e}")
            print(f"Error output: {e.stderr}")
            # Don't fail tests if submodule init fails
            pass
    else:
        print("✅ Langfuse submodule already initialized") 