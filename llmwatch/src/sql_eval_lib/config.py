"""
Configuration Management System for SQL Evaluation Library

Implements hierarchical configuration with:
- Environment variables
- Configuration files (YAML, JSON, TOML)
- Programmatic configuration
- Type-safe validation with Pydantic
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Type
from enum import Enum
from dataclasses import dataclass

from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
import json
import yaml
from dotenv import load_dotenv


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class ConfigSource(Enum):
    DEFAULT = "default"
    FILE = "file"
    ENVIRONMENT = "environment"
    PROGRAMMATIC = "programmatic"


class BaseConfig(BaseModel):
    """Base class for all configuration sections"""
    
    class Config:
        extra = "forbid"  # Prevent typos
        validate_assignment = True


class DatabaseConfig(BaseConfig):
    """Database backend configuration"""
    backend_type: str = Field(default="sqlite", description="Database backend type")
    connection_params: Dict[str, Any] = Field(default_factory=dict)
    timeout: int = Field(default=30, ge=1, le=300, description="Connection timeout in seconds")
    pool_size: int = Field(default=1, ge=1, le=20, description="Connection pool size")
    
    @validator('backend_type')
    def validate_backend_type(cls, v):
        allowed_types = ['sqlite', 'trino', 'postgres', 'mysql', 'custom']
        if v not in allowed_types:
            raise ValueError(f'backend_type must be one of {allowed_types}')
        return v


class LangfuseConfig(BaseConfig):
    """Langfuse tracing configuration"""
    enabled: bool = Field(default=True, description="Enable Langfuse tracing")
    host: str = Field(default="http://localhost:3000", description="Langfuse server host")
    public_key: Optional[str] = Field(default=None, description="Langfuse public key")
    secret_key: Optional[str] = Field(default=None, description="Langfuse secret key")
    debug: bool = Field(default=False, description="Enable debug mode")
    flush_interval: int = Field(default=1, ge=1, description="Flush interval in seconds")


class ModelConfig(BaseConfig):
    """Model adapter configuration"""
    adapter_type: str = Field(default="openai", description="Model adapter type")
    api_key: Optional[str] = Field(default=None, description="API key for the model")
    model_name: str = Field(default="gpt-3.5-turbo", description="Model name")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Model temperature")
    max_tokens: Optional[int] = Field(default=None, ge=1, description="Maximum tokens")
    timeout: int = Field(default=60, ge=1, description="Request timeout in seconds")


class EvaluationConfig(BaseConfig):
    """Evaluation pipeline configuration"""
    default_strategies: List[str] = Field(default=["llm_based"], description="Default evaluation strategies")
    parallel_execution: bool = Field(default=False, description="Enable parallel strategy execution")
    aggregation_method: str = Field(default="weighted_average", description="Result aggregation method")
    timeout: int = Field(default=300, ge=1, description="Evaluation timeout in seconds")


class LoggingConfig(BaseConfig):
    """Logging configuration"""
    level: LogLevel = Field(default=LogLevel.INFO, description="Log level")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_path: Optional[str] = Field(default=None, description="Log file path")


class SQLEvalConfig(BaseSettings):
    """Main configuration class for SQL Evaluation Library"""
    
    # Configuration sections
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    langfuse: LangfuseConfig = Field(default_factory=LangfuseConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Global settings
    debug: bool = Field(default=False, description="Enable debug mode")
    config_file: Optional[str] = Field(default=None, description="Configuration file path")
    
    class Config:
        env_prefix = "SQL_EVAL_"
        env_nested_delimiter = "__"
        case_sensitive = False
        
    @validator('config_file')
    def validate_config_file(cls, v):
        if v and not os.path.exists(v):
            raise ValueError(f'Configuration file not found: {v}')
        return v


class ConfigManager:
    """Manages hierarchical configuration loading and validation"""
    
    def __init__(self):
        self._config_cache = {}
        self._validators = {}
        self._loaded_env = False
    
    def load_config(self, 
                   config_class: Type[BaseSettings] = SQLEvalConfig,
                   config_file: Optional[str] = None,
                   env_file: Optional[str] = None,
                   **overrides) -> BaseSettings:
        """Load configuration with hierarchical precedence"""
        
        # Load environment variables if not already loaded
        if not self._loaded_env:
            self._load_environment(env_file)
            self._loaded_env = True
        
        # Start with any existing configuration data
        config_data = {}
        
        # Load from configuration file if provided
        if config_file:
            file_data = self._load_config_file(config_file)
            config_data.update(file_data)
        
        # Apply programmatic overrides
        config_data.update(overrides)
        
        # Create and validate configuration
        # Pydantic will automatically handle environment variables
        config = config_class(**config_data)
        
        # Run additional validation
        self._validate_config(config)
        
        return config
    
    def _load_environment(self, env_file: Optional[str] = None) -> None:
        """Load environment variables from .env file"""
        if env_file and os.path.exists(env_file):
            load_dotenv(env_file)
        else:
            # Try to find .env in common locations
            possible_paths = [
                Path.cwd() / ".env",
                Path(__file__).parent / ".env",
                Path(__file__).parent.parent.parent / ".env"
            ]
            
            for path in possible_paths:
                if path.exists():
                    load_dotenv(path)
                    break
    
    def _load_config_file(self, file_path: str) -> Dict[str, Any]:
        """Load configuration from file (YAML, JSON, or TOML)"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        try:
            with open(file_path, 'r') as f:
                if suffix in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                elif suffix == '.json':
                    return json.load(f) or {}
                elif suffix == '.toml':
                    try:
                        import tomli
                        return tomli.load(f.buffer) or {}
                    except ImportError:
                        raise ImportError("tomli package required for TOML support")
                else:
                    raise ValueError(f"Unsupported config file format: {suffix}")
        except Exception as e:
            raise ValueError(f"Error loading config file {file_path}: {e}")
    
    def _validate_config(self, config: BaseSettings) -> None:
        """Run additional validation beyond Pydantic"""
        # Custom validation logic can be added here
        if hasattr(config, 'langfuse') and config.langfuse.enabled:
            if not config.langfuse.public_key or not config.langfuse.secret_key:
                raise ValueError("Langfuse public_key and secret_key are required when enabled")


# Global configuration manager instance
config_manager = ConfigManager()


def get_config(**overrides) -> SQLEvalConfig:
    """Get configuration with optional overrides"""
    return config_manager.load_config(**overrides)


def create_env_template(output_path: str = ".env.template") -> None:
    """Create environment template file"""
    template_content = """# SQL Evaluation Library Environment Configuration
# Copy this file to .env and fill in your values

# ================================
# Langfuse Configuration
# ================================
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_DEBUG=false

# ================================
# Model Configuration
# ================================
# OpenAI
OPENAI_API_KEY=

# Anthropic Claude
ANTHROPIC_API_KEY=

# Google AI
GOOGLE_AI_API_KEY=

# Ollama (Self-hosted)
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama2

# ================================
# Database Configuration
# ================================
# SQLite (default)
DATABASE_TYPE=sqlite
DATABASE_CONNECTION_STRING=:memory:

# Trino
# DATABASE_TYPE=trino
# TRINO_HOST=localhost
# TRINO_PORT=8080
# TRINO_CATALOG=memory
# TRINO_SCHEMA=default
# TRINO_USER=admin

# ================================
# Evaluation Configuration
# ================================
DEFAULT_EVALUATION_STRATEGIES=llm_based,execution_based
EVALUATION_TIMEOUT=300
PARALLEL_EVALUATION=false

# ================================
# Logging Configuration
# ================================
LOG_LEVEL=INFO
LOG_FILE_PATH=

# ================================
# Performance Configuration
# ================================
CONNECTION_POOL_SIZE=5
REQUEST_TIMEOUT=60
MAX_RETRIES=3
"""
    
    with open(output_path, 'w') as f:
        f.write(template_content)
    
    print(f"Environment template created at {output_path}")


if __name__ == "__main__":
    # Demo usage
    config = get_config()
    print("Configuration loaded successfully!")
    print(f"Database backend: {config.database.backend_type}")
    print(f"Langfuse enabled: {config.langfuse.enabled}")
    print(f"Model adapter: {config.model.adapter_type}") 