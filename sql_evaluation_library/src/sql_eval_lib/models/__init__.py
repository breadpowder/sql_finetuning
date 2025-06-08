# sql_evaluation_library/src/sql_eval_lib/models/__init__.py

# Legacy components
from .generation import get_generated_sql_from_model, GraphState, build_sql_generation_graph
from .adapters import DummyModelAdapter, OpenAIModelAdapter

# Phase 2.2: Model Endpoint Validation - New components
from .validation import (
    ModelProvider,
    ModelEndpoint,
    ModelValidator,
    ValidationResult,
    ValidationStatus,
    ModelResponse,
    create_model_endpoints_from_config,
    validate_model_endpoints,
    validate_from_env_config,
    print_validation_report
)

from .health_check import (
    ModelHealthMonitor,
    HealthStatus,
    FailoverStrategy,
    PerformanceMetrics,
    EndpointHealth,
    FailoverConfig,
    create_health_monitor,
    simple_health_callback,
    print_health_summary
)

__all__ = [
    # Legacy components
    "get_generated_sql_from_model",
    "GraphState",
    "build_sql_generation_graph",
    "DummyModelAdapter",
    "OpenAIModelAdapter",
    
    # Validation
    "ModelProvider",
    "ModelEndpoint",
    "ModelValidator",
    "ValidationResult",
    "ValidationStatus",
    "ModelResponse",
    "create_model_endpoints_from_config",
    "validate_model_endpoints",
    "validate_from_env_config",
    "print_validation_report",
    
    # Health Check & Monitoring
    "ModelHealthMonitor",
    "HealthStatus",
    "FailoverStrategy",
    "PerformanceMetrics",
    "EndpointHealth",
    "FailoverConfig",
    "create_health_monitor",
    "simple_health_callback",
    "print_health_summary",
]

# Placeholder for models module
