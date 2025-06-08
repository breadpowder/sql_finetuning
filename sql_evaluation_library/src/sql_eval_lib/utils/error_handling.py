"""
Comprehensive error handling and recovery mechanisms.

This module provides robust error handling patterns including retry logic
with exponential backoff, circuit breaker pattern for external services,
graceful degradation strategies, and state persistence for recovery.
"""

import time
import random
import json
import threading
from typing import Dict, Any, Optional, List, Callable, TypeVar, Generic, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
from functools import wraps
import pickle
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


class RetryStrategy(Enum):
    """Retry strategies."""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    JITTER = "jitter"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    retry_on_exceptions: List[type] = field(default_factory=lambda: [Exception])
    stop_on_exceptions: List[type] = field(default_factory=list)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    timeout_seconds: float = 60.0
    success_threshold: int = 2
    monitor_window_seconds: float = 300.0


@dataclass
class ErrorContext:
    """Context information for error handling."""
    operation_name: str
    timestamp: datetime
    attempt_number: int
    error: Exception
    severity: ErrorSeverity
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "operation_name": self.operation_name,
            "timestamp": self.timestamp.isoformat(),
            "attempt_number": self.attempt_number,
            "error_type": type(self.error).__name__,
            "error_message": str(self.error),
            "severity": self.severity.value,
            "metadata": self.metadata
        }


@dataclass
class RecoveryState:
    """State information for recovery persistence."""
    operation_id: str
    operation_name: str
    current_attempt: int
    last_error: Optional[str]
    next_retry_time: Optional[datetime]
    context_data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "operation_id": self.operation_id,
            "operation_name": self.operation_name,
            "current_attempt": self.current_attempt,
            "last_error": self.last_error,
            "next_retry_time": self.next_retry_time.isoformat() if self.next_retry_time else None,
            "context_data": self.context_data,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecoveryState":
        """Create from dictionary."""
        return cls(
            operation_id=data["operation_id"],
            operation_name=data["operation_name"],
            current_attempt=data["current_attempt"],
            last_error=data.get("last_error"),
            next_retry_time=datetime.fromisoformat(data["next_retry_time"]) if data.get("next_retry_time") else None,
            context_data=data.get("context_data", {}),
            created_at=datetime.fromisoformat(data["created_at"])
        )


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for external services.
    
    Prevents cascading failures by failing fast when a service is
    consistently failing and allowing recovery testing.
    """
    
    def __init__(self, 
                 name: str,
                 config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker.
        
        Args:
            name: Name of the service/operation
            config: Optional configuration
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state_changed_time = datetime.utcnow()
        self._lock = threading.Lock()
        
        # Callbacks for state changes
        self._state_change_callbacks: List[Callable[[CircuitState, CircuitState], None]] = []
    
    def add_state_change_callback(self, callback: Callable[[CircuitState, CircuitState], None]) -> None:
        """Add callback for state changes."""
        self._state_change_callbacks.append(callback)
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
            Any exception from the function
        """
        with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._change_state(CircuitState.HALF_OPEN)
                else:
                    raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")
            
            if self.state == CircuitState.HALF_OPEN:
                if self.success_count >= self.config.success_threshold:
                    self._change_state(CircuitState.CLOSED)
        
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = datetime.utcnow() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.config.timeout_seconds
    
    def _record_success(self) -> None:
        """Record successful operation."""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
            else:
                self.failure_count = 0
                self.success_count = 0
    
    def _record_failure(self) -> None:
        """Record failed operation."""
        with self._lock:
            self.failure_count += 1
            self.success_count = 0
            self.last_failure_time = datetime.utcnow()
            
            if self.state == CircuitState.CLOSED and self.failure_count >= self.config.failure_threshold:
                self._change_state(CircuitState.OPEN)
            elif self.state == CircuitState.HALF_OPEN:
                self._change_state(CircuitState.OPEN)
    
    def _change_state(self, new_state: CircuitState) -> None:
        """Change circuit breaker state."""
        old_state = self.state
        self.state = new_state
        self.state_changed_time = datetime.utcnow()
        
        logger.info(f"Circuit breaker {self.name} changed from {old_state.value} to {new_state.value}")
        
        # Notify callbacks
        for callback in self._state_change_callbacks:
            try:
                callback(old_state, new_state)
            except Exception as e:
                logger.error(f"Error in circuit breaker callback: {e}")
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information."""
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
                "state_changed_time": self.state_changed_time.isoformat(),
                "time_since_state_change": (datetime.utcnow() - self.state_changed_time).total_seconds()
            }
    
    def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        with self._lock:
            old_state = self.state
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            self.state_changed_time = datetime.utcnow()
            
            logger.info(f"Circuit breaker {self.name} manually reset from {old_state.value} to CLOSED")


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class RetryHandler:
    """
    Implements various retry strategies with exponential backoff.
    
    Provides configurable retry logic with different backoff strategies,
    jitter, and exception filtering.
    """
    
    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Initialize retry handler.
        
        Args:
            config: Optional retry configuration
        """
        self.config = config or RetryConfig()
    
    def retry(self, 
             func: Callable[..., T],
             *args,
             operation_name: Optional[str] = None,
             **kwargs) -> T:
        """
        Execute function with retry logic.
        
        Args:
            func: Function to execute
            *args: Function arguments
            operation_name: Optional operation name for logging
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries fail
        """
        operation_name = operation_name or func.__name__
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                result = func(*args, **kwargs)
                if attempt > 1:
                    logger.info(f"Operation {operation_name} succeeded on attempt {attempt}")
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if we should stop retrying
                if any(isinstance(e, exc_type) for exc_type in self.config.stop_on_exceptions):
                    logger.info(f"Stopping retries for {operation_name} due to stop exception: {type(e).__name__}")
                    raise e
                
                # Check if we should retry
                if not any(isinstance(e, exc_type) for exc_type in self.config.retry_on_exceptions):
                    logger.info(f"Not retrying {operation_name} for exception type: {type(e).__name__}")
                    raise e
                
                if attempt < self.config.max_attempts:
                    delay = self._calculate_delay(attempt)
                    logger.warning(f"Operation {operation_name} failed on attempt {attempt}/{self.config.max_attempts}. "
                                 f"Retrying in {delay:.2f}s. Error: {str(e)}")
                    time.sleep(delay)
                else:
                    logger.error(f"Operation {operation_name} failed after {self.config.max_attempts} attempts")
        
        # All retries exhausted
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError(f"Operation {operation_name} failed without exception")
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        if self.config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.config.base_delay
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay * attempt
        elif self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay * (self.config.backoff_multiplier ** (attempt - 1))
        else:  # JITTER
            base_delay = self.config.base_delay * (self.config.backoff_multiplier ** (attempt - 1))
            delay = base_delay * (0.5 + random.random() * 0.5)  # 50-100% of base delay
        
        # Apply jitter if enabled
        if self.config.jitter and self.config.strategy != RetryStrategy.JITTER:
            jitter_factor = 0.1  # ±10% jitter
            jitter = delay * jitter_factor * (random.random() * 2 - 1)
            delay += jitter
        
        # Clamp to max delay
        return min(delay, self.config.max_delay)


class StatePersistence:
    """
    Provides state persistence for recovery operations.
    
    Allows operations to persist their state and recover from failures
    by resuming from the last known good state.
    """
    
    def __init__(self, state_file: Optional[Path] = None):
        """
        Initialize state persistence.
        
        Args:
            state_file: Optional path to state file
        """
        self.state_file = state_file or Path("recovery_state.json")
        self._states: Dict[str, RecoveryState] = {}
        self._lock = threading.Lock()
        
        # Load existing state
        self._load_state()
    
    def save_state(self, operation_id: str, state: RecoveryState) -> None:
        """
        Save operation state.
        
        Args:
            operation_id: Unique operation identifier
            state: Recovery state to save
        """
        with self._lock:
            self._states[operation_id] = state
            self._persist_state()
    
    def get_state(self, operation_id: str) -> Optional[RecoveryState]:
        """
        Get operation state.
        
        Args:
            operation_id: Operation identifier
            
        Returns:
            Recovery state or None if not found
        """
        with self._lock:
            return self._states.get(operation_id)
    
    def remove_state(self, operation_id: str) -> bool:
        """
        Remove operation state.
        
        Args:
            operation_id: Operation identifier
            
        Returns:
            True if state was removed, False if not found
        """
        with self._lock:
            if operation_id in self._states:
                del self._states[operation_id]
                self._persist_state()
                return True
            return False
    
    def list_states(self) -> List[RecoveryState]:
        """Get list of all recovery states."""
        with self._lock:
            return list(self._states.values())
    
    def cleanup_old_states(self, max_age_hours: int = 24) -> int:
        """
        Clean up old recovery states.
        
        Args:
            max_age_hours: Maximum age in hours
            
        Returns:
            Number of states removed
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        removed_count = 0
        
        with self._lock:
            old_states = [
                op_id for op_id, state in self._states.items()
                if state.created_at < cutoff_time
            ]
            
            for op_id in old_states:
                del self._states[op_id]
                removed_count += 1
            
            if removed_count > 0:
                self._persist_state()
        
        logger.info(f"Cleaned up {removed_count} old recovery states")
        return removed_count
    
    def _load_state(self) -> None:
        """Load state from file."""
        if not self.state_file.exists():
            return
        
        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)
            
            self._states = {
                op_id: RecoveryState.from_dict(state_data)
                for op_id, state_data in data.items()
            }
            
            logger.info(f"Loaded {len(self._states)} recovery states from {self.state_file}")
            
        except Exception as e:
            logger.error(f"Failed to load recovery state: {e}")
            self._states = {}
    
    def _persist_state(self) -> None:
        """Persist state to file."""
        try:
            data = {
                op_id: state.to_dict()
                for op_id, state in self._states.items()
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to persist recovery state: {e}")


class GracefulDegradationManager:
    """
    Manages graceful degradation strategies.
    
    Provides fallback mechanisms when primary services fail,
    allowing the system to continue operating with reduced functionality.
    """
    
    def __init__(self):
        """Initialize graceful degradation manager."""
        self._fallback_strategies: Dict[str, List[Callable]] = {}
        self._service_health: Dict[str, bool] = {}
        self._lock = threading.Lock()
    
    def register_fallback(self, 
                         service_name: str,
                         fallback_func: Callable,
                         priority: int = 0) -> None:
        """
        Register a fallback strategy for a service.
        
        Args:
            service_name: Name of the service
            fallback_func: Fallback function to execute
            priority: Priority (lower = higher priority)
        """
        with self._lock:
            if service_name not in self._fallback_strategies:
                self._fallback_strategies[service_name] = []
            
            self._fallback_strategies[service_name].append((priority, fallback_func))
            self._fallback_strategies[service_name].sort(key=lambda x: x[0])
    
    def execute_with_fallback(self,
                             service_name: str,
                             primary_func: Callable[..., T],
                             *args,
                             **kwargs) -> T:
        """
        Execute function with fallback strategies.
        
        Args:
            service_name: Name of the service
            primary_func: Primary function to try
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Result from primary or fallback function
            
        Raises:
            Exception if all strategies fail
        """
        # Try primary function
        try:
            result = primary_func(*args, **kwargs)
            self._mark_service_healthy(service_name)
            return result
        except Exception as primary_error:
            self._mark_service_unhealthy(service_name)
            logger.warning(f"Primary service {service_name} failed: {primary_error}")
        
        # Try fallback strategies
        with self._lock:
            strategies = self._fallback_strategies.get(service_name, [])
        
        for priority, fallback_func in strategies:
            try:
                logger.info(f"Trying fallback strategy for {service_name} (priority {priority})")
                result = fallback_func(*args, **kwargs)
                logger.info(f"Fallback strategy succeeded for {service_name}")
                return result
            except Exception as fallback_error:
                logger.warning(f"Fallback strategy failed for {service_name}: {fallback_error}")
                continue
        
        # All strategies failed
        raise RuntimeError(f"All strategies failed for service {service_name}")
    
    def _mark_service_healthy(self, service_name: str) -> None:
        """Mark service as healthy."""
        with self._lock:
            was_unhealthy = not self._service_health.get(service_name, True)
            self._service_health[service_name] = True
            
            if was_unhealthy:
                logger.info(f"Service {service_name} recovered")
    
    def _mark_service_unhealthy(self, service_name: str) -> None:
        """Mark service as unhealthy."""
        with self._lock:
            self._service_health[service_name] = False
    
    def get_service_health(self) -> Dict[str, bool]:
        """Get health status of all services."""
        with self._lock:
            return self._service_health.copy()


class ErrorRecoveryOrchestrator:
    """
    Orchestrates comprehensive error handling and recovery.
    
    Combines retry logic, circuit breakers, state persistence,
    and graceful degradation into a unified system.
    """
    
    def __init__(self,
                 retry_config: Optional[RetryConfig] = None,
                 circuit_config: Optional[CircuitBreakerConfig] = None,
                 state_file: Optional[Path] = None):
        """
        Initialize error recovery orchestrator.
        
        Args:
            retry_config: Optional retry configuration
            circuit_config: Optional circuit breaker configuration
            state_file: Optional state persistence file
        """
        self.retry_handler = RetryHandler(retry_config)
        self.state_persistence = StatePersistence(state_file)
        self.degradation_manager = GracefulDegradationManager()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()
    
    def get_or_create_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get or create circuit breaker for service."""
        with self._lock:
            if name not in self.circuit_breakers:
                self.circuit_breakers[name] = CircuitBreaker(name, self.circuit_config)
            return self.circuit_breakers[name]
    
    def execute_resilient_operation(self,
                                   operation_name: str,
                                   func: Callable[..., T],
                                   *args,
                                   service_name: Optional[str] = None,
                                   operation_id: Optional[str] = None,
                                   save_state: bool = False,
                                   **kwargs) -> T:
        """
        Execute operation with comprehensive error handling.
        
        Args:
            operation_name: Name of the operation
            func: Function to execute
            *args: Function arguments
            service_name: Optional service name for circuit breaking
            operation_id: Optional operation ID for state persistence
            save_state: Whether to save state for recovery
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        # Check for existing state
        if operation_id and save_state:
            existing_state = self.state_persistence.get_state(operation_id)
            if existing_state:
                logger.info(f"Found existing state for operation {operation_id}, attempt {existing_state.current_attempt}")
        
        def wrapped_operation():
            if service_name:
                # Use circuit breaker if service name provided
                circuit_breaker = self.get_or_create_circuit_breaker(service_name)
                return circuit_breaker.call(func, *args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        try:
            # Execute with retry logic
            result = self.retry_handler.retry(wrapped_operation, operation_name=operation_name)
            
            # Clean up state on success
            if operation_id and save_state:
                self.state_persistence.remove_state(operation_id)
            
            return result
            
        except Exception as e:
            # Save state for recovery if requested
            if operation_id and save_state:
                recovery_state = RecoveryState(
                    operation_id=operation_id,
                    operation_name=operation_name,
                    current_attempt=self.retry_handler.config.max_attempts,
                    last_error=str(e),
                    context_data=kwargs
                )
                self.state_persistence.save_state(operation_id, recovery_state)
            
            logger.error(f"Resilient operation {operation_name} failed after all recovery attempts: {e}")
            raise
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health information."""
        circuit_breaker_states = {
            name: cb.get_state_info()
            for name, cb in self.circuit_breakers.items()
        }
        
        service_health = self.degradation_manager.get_service_health()
        recovery_states = self.state_persistence.list_states()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "circuit_breakers": circuit_breaker_states,
            "service_health": service_health,
            "active_recovery_operations": len(recovery_states),
            "recovery_states": [state.to_dict() for state in recovery_states[-5:]]  # Last 5
        }


# Decorators for convenient usage

def resilient(retry_config: Optional[RetryConfig] = None,
             service_name: Optional[str] = None,
             operation_name: Optional[str] = None):
    """
    Decorator for resilient operation execution.
    
    Args:
        retry_config: Optional retry configuration
        service_name: Optional service name for circuit breaking
        operation_name: Optional operation name (defaults to function name)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        orchestrator = ErrorRecoveryOrchestrator(retry_config=retry_config)
        
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            op_name = operation_name or func.__name__
            return orchestrator.execute_resilient_operation(
                operation_name=op_name,
                func=func,
                *args,
                service_name=service_name,
                **kwargs
            )
        
        return wrapper
    return decorator


def circuit_breaker(name: Optional[str] = None,
                   config: Optional[CircuitBreakerConfig] = None):
    """
    Decorator for circuit breaker protection.
    
    Args:
        name: Circuit breaker name (defaults to function name)
        config: Optional circuit breaker configuration
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cb_name = name or func.__name__
        circuit_breaker = CircuitBreaker(cb_name, config)
        
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return circuit_breaker.call(func, *args, **kwargs)
        
        return wrapper
    return decorator


def retry_on_failure(config: Optional[RetryConfig] = None):
    """
    Decorator for retry logic.
    
    Args:
        config: Optional retry configuration
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        retry_handler = RetryHandler(config)
        
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return retry_handler.retry(func, *args, **kwargs)
        
        return wrapper
    return decorator


# Global orchestrator instance
_global_orchestrator: Optional[ErrorRecoveryOrchestrator] = None


def get_global_orchestrator() -> ErrorRecoveryOrchestrator:
    """Get or create global error recovery orchestrator."""
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = ErrorRecoveryOrchestrator()
    return _global_orchestrator


def configure_global_error_handling(retry_config: Optional[RetryConfig] = None,
                                   circuit_config: Optional[CircuitBreakerConfig] = None,
                                   state_file: Optional[Path] = None) -> None:
    """
    Configure global error handling settings.
    
    Args:
        retry_config: Optional retry configuration
        circuit_config: Optional circuit breaker configuration
        state_file: Optional state persistence file
    """
    global _global_orchestrator
    _global_orchestrator = ErrorRecoveryOrchestrator(retry_config, circuit_config, state_file) 