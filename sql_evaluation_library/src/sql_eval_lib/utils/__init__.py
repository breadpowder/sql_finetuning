# sql_evaluation_library/src/sql_eval_lib/utils/__init__.py

# Legacy utilities
from .helpers import split_sql_statements

# Phase 2.4: Comprehensive Error Handling - New components
from .error_handling import (
    # Core classes
    CircuitBreaker,
    RetryHandler,
    StatePersistence,
    GracefulDegradationManager,
    ErrorRecoveryOrchestrator,
    
    # Configuration classes
    RetryConfig,
    CircuitBreakerConfig,
    ErrorContext,
    RecoveryState,
    
    # Enums
    ErrorSeverity,
    CircuitState,
    RetryStrategy,
    
    # Exceptions
    CircuitBreakerOpenError,
    
    # Decorators
    resilient,
    circuit_breaker,
    retry_on_failure,
    
    # Global utilities
    get_global_orchestrator,
    configure_global_error_handling,
)

__all__ = [
    # Legacy utilities
    "split_sql_statements",
    
    # Phase 2.4: Error Handling
    # Core classes
    "CircuitBreaker",
    "RetryHandler", 
    "StatePersistence",
    "GracefulDegradationManager",
    "ErrorRecoveryOrchestrator",
    
    # Configuration classes
    "RetryConfig",
    "CircuitBreakerConfig",
    "ErrorContext",
    "RecoveryState",
    
    # Enums
    "ErrorSeverity",
    "CircuitState",
    "RetryStrategy",
    
    # Exceptions
    "CircuitBreakerOpenError",
    
    # Decorators
    "resilient",
    "circuit_breaker", 
    "retry_on_failure",
    
    # Global utilities
    "get_global_orchestrator",
    "configure_global_error_handling",
]
