"""
Model endpoint health monitoring and automatic failover system.

This module provides continuous health monitoring of model endpoints,
automatic failover capabilities, performance metrics collection,
and cost optimization strategies.
"""

import time
import threading
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
from pathlib import Path
import statistics

from .validation import ModelEndpoint, ModelValidator, ValidationStatus, ModelProvider


class HealthStatus(Enum):
    """Health status of model endpoints."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class FailoverStrategy(Enum):
    """Failover strategies for model endpoints."""
    ROUND_ROBIN = "round_robin"
    PRIORITY = "priority"
    LEAST_LATENCY = "least_latency"
    LEAST_COST = "least_cost"
    LOAD_BALANCED = "load_balanced"


@dataclass
class PerformanceMetrics:
    """Performance metrics for a model endpoint."""
    endpoint_name: str
    timestamp: datetime
    response_time_ms: float
    success: bool
    tokens_used: Optional[int] = None
    cost_estimate: Optional[float] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "endpoint_name": self.endpoint_name,
            "timestamp": self.timestamp.isoformat(),
            "response_time_ms": self.response_time_ms,
            "success": self.success,
            "tokens_used": self.tokens_used,
            "cost_estimate": self.cost_estimate,
            "error_message": self.error_message
        }


@dataclass
class EndpointHealth:
    """Health information for a model endpoint."""
    endpoint: ModelEndpoint
    status: HealthStatus
    last_check: datetime
    consecutive_failures: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    average_response_time: float = 0.0
    availability_percentage: float = 100.0
    total_cost: float = 0.0
    priority: int = 1
    is_enabled: bool = True
    
    def update_metrics(self, metric: PerformanceMetrics) -> None:
        """Update health metrics with new performance data."""
        self.total_requests += 1
        self.last_check = datetime.utcnow()
        
        if metric.success:
            self.successful_requests += 1
            self.consecutive_failures = 0
            self.status = HealthStatus.HEALTHY
        else:
            self.consecutive_failures += 1
            if self.consecutive_failures >= 3:
                self.status = HealthStatus.UNHEALTHY
            elif self.consecutive_failures >= 1:
                self.status = HealthStatus.DEGRADED
        
        # Update availability percentage
        self.availability_percentage = (self.successful_requests / self.total_requests) * 100
        
        # Update average response time (moving average)
        if self.average_response_time == 0:
            self.average_response_time = metric.response_time_ms
        else:
            self.average_response_time = (self.average_response_time * 0.8) + (metric.response_time_ms * 0.2)
        
        # Update cost tracking
        if metric.cost_estimate:
            self.total_cost += metric.cost_estimate


@dataclass
class FailoverConfig:
    """Configuration for failover behavior."""
    strategy: FailoverStrategy = FailoverStrategy.PRIORITY
    max_retries: int = 3
    retry_delay: float = 1.0
    health_check_interval: int = 60
    failure_threshold: int = 3
    recovery_threshold: int = 2
    enable_cost_optimization: bool = True
    max_cost_per_request: Optional[float] = None


class ModelHealthMonitor:
    """
    Monitors health of model endpoints and provides failover capabilities.
    
    This class continuously monitors model endpoints, tracks performance metrics,
    and provides automatic failover when endpoints become unhealthy.
    """
    
    def __init__(self, 
                 endpoints: List[ModelEndpoint],
                 config: Optional[FailoverConfig] = None):
        """
        Initialize the health monitor.
        
        Args:
            endpoints: List of model endpoints to monitor
            config: Optional failover configuration
        """
        self.config = config or FailoverConfig()
        self.endpoint_health: Dict[str, EndpointHealth] = {}
        self.metrics_history: List[PerformanceMetrics] = []
        self.is_monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Initialize endpoint health
        for i, endpoint in enumerate(endpoints):
            self.endpoint_health[endpoint.name] = EndpointHealth(
                endpoint=endpoint,
                status=HealthStatus.UNKNOWN,
                last_check=datetime.utcnow(),
                priority=i + 1  # Default priority based on order
            )
        
        # Callbacks for health status changes
        self._health_change_callbacks: List[Callable[[str, HealthStatus], None]] = []
    
    def add_health_change_callback(self, callback: Callable[[str, HealthStatus], None]) -> None:
        """
        Add a callback for health status changes.
        
        Args:
            callback: Function that takes endpoint name and new status
        """
        self._health_change_callbacks.append(callback)
    
    def start_monitoring(self) -> bool:
        """
        Start continuous health monitoring.
        
        Returns:
            True if monitoring started successfully, False otherwise
        """
        if self.is_monitoring:
            print("Health monitoring is already running")
            return False
        
        self.is_monitoring = True
        self._stop_event.clear()
        
        try:
            self._monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                name="ModelHealthMonitor",
                daemon=True
            )
            self._monitor_thread.start()
            print("Model health monitoring started")
            return True
        except Exception as e:
            self.is_monitoring = False
            print(f"Failed to start health monitoring: {e}")
            return False
    
    def stop_monitoring(self) -> bool:
        """
        Stop continuous health monitoring.
        
        Returns:
            True if monitoring stopped successfully, False otherwise
        """
        if not self.is_monitoring:
            return True
        
        self._stop_event.set()
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=10)
        
        self.is_monitoring = False
        print("Model health monitoring stopped")
        return True
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        print(f"Starting health monitoring loop (interval: {self.config.health_check_interval}s)")
        
        while not self._stop_event.is_set():
            try:
                # Perform health checks on all endpoints
                for endpoint_name, health in self.endpoint_health.items():
                    if health.is_enabled:
                        self._perform_health_check(health)
                
                # Clean up old metrics
                self._cleanup_old_metrics()
                
                # Wait for next check
                self._stop_event.wait(self.config.health_check_interval)
                
            except Exception as e:
                print(f"Error in health monitoring loop: {e}")
                time.sleep(5)  # Brief pause before retrying
    
    def _perform_health_check(self, health: EndpointHealth) -> None:
        """Perform health check on a single endpoint."""
        start_time = time.time()
        
        try:
            # Use validator to check endpoint health
            validator = ModelValidator([health.endpoint])
            results = validator.validate_endpoint(health.endpoint.name)
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Determine if check was successful
            success = all(r.status == ValidationStatus.PASS for r in results)
            error_message = None
            
            if not success:
                error_messages = [r.message for r in results if r.status == ValidationStatus.FAIL]
                error_message = "; ".join(error_messages) if error_messages else "Health check failed"
            
            # Create performance metric
            metric = PerformanceMetrics(
                endpoint_name=health.endpoint.name,
                timestamp=datetime.utcnow(),
                response_time_ms=duration_ms,
                success=success,
                error_message=error_message
            )
            
            # Update health and track metric
            old_status = health.status
            health.update_metrics(metric)
            self.metrics_history.append(metric)
            
            # Notify if status changed
            if old_status != health.status:
                self._notify_health_change(health.endpoint.name, health.status)
            
            print(f"Health check {health.endpoint.name}: {'✅' if success else '❌'} ({duration_ms:.1f}ms)")
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            metric = PerformanceMetrics(
                endpoint_name=health.endpoint.name,
                timestamp=datetime.utcnow(),
                response_time_ms=duration_ms,
                success=False,
                error_message=str(e)
            )
            
            old_status = health.status
            health.update_metrics(metric)
            self.metrics_history.append(metric)
            
            if old_status != health.status:
                self._notify_health_change(health.endpoint.name, health.status)
            
            print(f"Health check {health.endpoint.name}: ❌ Error: {e}")
    
    def _notify_health_change(self, endpoint_name: str, new_status: HealthStatus) -> None:
        """Notify callbacks of health status change."""
        for callback in self._health_change_callbacks:
            try:
                callback(endpoint_name, new_status)
            except Exception as e:
                print(f"Error in health change callback: {e}")
    
    def _cleanup_old_metrics(self) -> None:
        """Remove old metrics to prevent memory growth."""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        self.metrics_history = [m for m in self.metrics_history if m.timestamp > cutoff_time]
    
    def get_healthy_endpoints(self) -> List[str]:
        """Get list of healthy endpoint names."""
        return [
            name for name, health in self.endpoint_health.items()
            if health.status == HealthStatus.HEALTHY and health.is_enabled
        ]
    
    def get_best_endpoint(self, strategy: Optional[FailoverStrategy] = None) -> Optional[str]:
        """
        Get the best endpoint based on the specified strategy.
        
        Args:
            strategy: Optional strategy override
            
        Returns:
            Name of the best endpoint or None if none available
        """
        strategy = strategy or self.config.strategy
        healthy_endpoints = self.get_healthy_endpoints()
        
        if not healthy_endpoints:
            # Fall back to degraded endpoints if no healthy ones
            degraded_endpoints = [
                name for name, health in self.endpoint_health.items()
                if health.status == HealthStatus.DEGRADED and health.is_enabled
            ]
            if degraded_endpoints:
                healthy_endpoints = degraded_endpoints
            else:
                return None
        
        if strategy == FailoverStrategy.PRIORITY:
            # Return highest priority (lowest number) healthy endpoint
            return min(healthy_endpoints, 
                      key=lambda name: self.endpoint_health[name].priority)
        
        elif strategy == FailoverStrategy.LEAST_LATENCY:
            # Return endpoint with lowest average response time
            return min(healthy_endpoints,
                      key=lambda name: self.endpoint_health[name].average_response_time)
        
        elif strategy == FailoverStrategy.LEAST_COST:
            # Return endpoint with lowest total cost
            return min(healthy_endpoints,
                      key=lambda name: self.endpoint_health[name].total_cost)
        
        elif strategy == FailoverStrategy.LOAD_BALANCED:
            # Return endpoint with fewest total requests
            return min(healthy_endpoints,
                      key=lambda name: self.endpoint_health[name].total_requests)
        
        else:  # ROUND_ROBIN
            # Simple round-robin based on total requests
            return min(healthy_endpoints,
                      key=lambda name: self.endpoint_health[name].total_requests)
    
    def execute_with_failover(self, 
                            operation: Callable[[ModelEndpoint], Any],
                            strategy: Optional[FailoverStrategy] = None) -> Tuple[Any, str]:
        """
        Execute an operation with automatic failover.
        
        Args:
            operation: Function that takes a ModelEndpoint and returns a result
            strategy: Optional strategy override
            
        Returns:
            Tuple of (result, endpoint_name_used)
            
        Raises:
            RuntimeError: If all endpoints fail
        """
        strategy = strategy or self.config.strategy
        
        for attempt in range(self.config.max_retries):
            endpoint_name = self.get_best_endpoint(strategy)
            
            if endpoint_name is None:
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                    continue
                else:
                    raise RuntimeError("No healthy endpoints available")
            
            endpoint = self.endpoint_health[endpoint_name].endpoint
            start_time = time.time()
            
            try:
                result = operation(endpoint)
                duration_ms = (time.time() - start_time) * 1000
                
                # Record successful metric
                metric = PerformanceMetrics(
                    endpoint_name=endpoint_name,
                    timestamp=datetime.utcnow(),
                    response_time_ms=duration_ms,
                    success=True
                )
                
                self.endpoint_health[endpoint_name].update_metrics(metric)
                self.metrics_history.append(metric)
                
                return result, endpoint_name
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                
                # Record failed metric
                metric = PerformanceMetrics(
                    endpoint_name=endpoint_name,
                    timestamp=datetime.utcnow(),
                    response_time_ms=duration_ms,
                    success=False,
                    error_message=str(e)
                )
                
                old_status = self.endpoint_health[endpoint_name].status
                self.endpoint_health[endpoint_name].update_metrics(metric)
                self.metrics_history.append(metric)
                
                # Notify if status changed
                if old_status != self.endpoint_health[endpoint_name].status:
                    self._notify_health_change(endpoint_name, self.endpoint_health[endpoint_name].status)
                
                print(f"Operation failed on {endpoint_name}: {e}")
                
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    raise RuntimeError(f"All endpoints failed. Last error: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary for all endpoints.
        
        Returns:
            Dictionary with performance metrics and summaries
        """
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "monitoring_active": self.is_monitoring,
            "total_endpoints": len(self.endpoint_health),
            "endpoints": {}
        }
        
        # Calculate overall stats
        healthy_count = len([h for h in self.endpoint_health.values() if h.status == HealthStatus.HEALTHY])
        degraded_count = len([h for h in self.endpoint_health.values() if h.status == HealthStatus.DEGRADED])
        unhealthy_count = len([h for h in self.endpoint_health.values() if h.status == HealthStatus.UNHEALTHY])
        
        summary["overall_health"] = {
            "healthy": healthy_count,
            "degraded": degraded_count,
            "unhealthy": unhealthy_count,
            "total": len(self.endpoint_health)
        }
        
        # Per-endpoint details
        for name, health in self.endpoint_health.items():
            summary["endpoints"][name] = {
                "provider": health.endpoint.provider.value,
                "model_name": health.endpoint.model_name,
                "status": health.status.value,
                "last_check": health.last_check.isoformat(),
                "consecutive_failures": health.consecutive_failures,
                "total_requests": health.total_requests,
                "successful_requests": health.successful_requests,
                "availability_percentage": round(health.availability_percentage, 2),
                "average_response_time_ms": round(health.average_response_time, 2),
                "total_cost": health.total_cost,
                "priority": health.priority,
                "is_enabled": health.is_enabled
            }
        
        # Recent metrics (last hour)
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > one_hour_ago]
        
        if recent_metrics:
            response_times = [m.response_time_ms for m in recent_metrics if m.success]
            if response_times:
                summary["recent_performance"] = {
                    "total_requests": len(recent_metrics),
                    "successful_requests": len([m for m in recent_metrics if m.success]),
                    "average_response_time_ms": round(statistics.mean(response_times), 2),
                    "median_response_time_ms": round(statistics.median(response_times), 2),
                    "min_response_time_ms": round(min(response_times), 2),
                    "max_response_time_ms": round(max(response_times), 2)
                }
        
        return summary
    
    def export_metrics(self, file_path: Path) -> bool:
        """
        Export metrics and health data to a file.
        
        Args:
            file_path: Path to export file
            
        Returns:
            True if export was successful, False otherwise
        """
        try:
            export_data = {
                "export_timestamp": datetime.utcnow().isoformat(),
                "config": {
                    "strategy": self.config.strategy.value,
                    "max_retries": self.config.max_retries,
                    "health_check_interval": self.config.health_check_interval,
                    "failure_threshold": self.config.failure_threshold
                },
                "performance_summary": self.get_performance_summary(),
                "metrics_history": [m.to_dict() for m in self.metrics_history]
            }
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"Metrics exported to {file_path}")
            return True
            
        except Exception as e:
            print(f"Failed to export metrics: {e}")
            return False
    
    def set_endpoint_priority(self, endpoint_name: str, priority: int) -> bool:
        """
        Set priority for an endpoint.
        
        Args:
            endpoint_name: Name of the endpoint
            priority: Priority value (lower = higher priority)
            
        Returns:
            True if priority was set, False if endpoint not found
        """
        if endpoint_name in self.endpoint_health:
            self.endpoint_health[endpoint_name].priority = priority
            return True
        return False
    
    def enable_endpoint(self, endpoint_name: str) -> bool:
        """Enable an endpoint for use."""
        if endpoint_name in self.endpoint_health:
            self.endpoint_health[endpoint_name].is_enabled = True
            return True
        return False
    
    def disable_endpoint(self, endpoint_name: str) -> bool:
        """Disable an endpoint from use."""
        if endpoint_name in self.endpoint_health:
            self.endpoint_health[endpoint_name].is_enabled = False
            return True
        return False


def create_health_monitor(endpoints: List[ModelEndpoint],
                         config: Optional[FailoverConfig] = None) -> ModelHealthMonitor:
    """
    Create a model health monitor.
    
    Args:
        endpoints: List of model endpoints to monitor
        config: Optional failover configuration
        
    Returns:
        Model health monitor instance
    """
    return ModelHealthMonitor(endpoints, config)


def simple_health_callback(endpoint_name: str, status: HealthStatus) -> None:
    """
    Simple callback for health status changes.
    
    Args:
        endpoint_name: Name of the endpoint
        status: New health status
    """
    status_symbols = {
        HealthStatus.HEALTHY: "✅",
        HealthStatus.DEGRADED: "⚠️",
        HealthStatus.UNHEALTHY: "❌",
        HealthStatus.UNKNOWN: "❓"
    }
    
    symbol = status_symbols.get(status, "❓")
    print(f"{symbol} Endpoint {endpoint_name} status changed to: {status.value.upper()}")


def print_health_summary(monitor: ModelHealthMonitor) -> None:
    """
    Print a formatted health summary.
    
    Args:
        monitor: Model health monitor instance
    """
    summary = monitor.get_performance_summary()
    
    print("\n" + "="*70)
    print("MODEL ENDPOINT HEALTH SUMMARY")
    print("="*70)
    
    overall = summary["overall_health"]
    print(f"Total Endpoints: {overall['total']}")
    print(f"Healthy: {overall['healthy']} | Degraded: {overall['degraded']} | Unhealthy: {overall['unhealthy']}")
    print(f"Monitoring Active: {'Yes' if summary['monitoring_active'] else 'No'}")
    
    print("\n" + "-"*70)
    print("ENDPOINT STATUS")
    print("-"*70)
    
    for name, endpoint_data in summary["endpoints"].items():
        status_symbol = {
            "healthy": "✅",
            "degraded": "⚠️", 
            "unhealthy": "❌",
            "unknown": "❓"
        }.get(endpoint_data["status"], "❓")
        
        availability = endpoint_data["availability_percentage"]
        response_time = endpoint_data["average_response_time_ms"]
        priority = endpoint_data["priority"]
        
        print(f"{status_symbol} {name} ({endpoint_data['provider']})")
        print(f"    Availability: {availability}% | Avg Response: {response_time}ms | Priority: {priority}")
        print(f"    Requests: {endpoint_data['successful_requests']}/{endpoint_data['total_requests']}")
    
    if "recent_performance" in summary:
        recent = summary["recent_performance"]
        print("\n" + "-"*70)
        print("RECENT PERFORMANCE (Last Hour)")
        print("-"*70)
        print(f"Total Requests: {recent['total_requests']}")
        print(f"Success Rate: {recent['successful_requests']}/{recent['total_requests']}")
        print(f"Response Time: Avg {recent['average_response_time_ms']}ms | "
              f"Median {recent['median_response_time_ms']}ms")
        print(f"Range: {recent['min_response_time_ms']}ms - {recent['max_response_time_ms']}ms")
    
    print("="*70) 