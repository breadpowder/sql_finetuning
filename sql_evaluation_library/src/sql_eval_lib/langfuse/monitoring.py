"""
Langfuse health monitoring and alerting system.

This module provides continuous monitoring of Langfuse instances,
including health checking, metrics collection, and alerting capabilities.
"""

import time
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path

from .validation import LangfuseValidator, ValidationConfig, ValidationStatus


class MonitoringStatus(Enum):
    """Status of monitoring system."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MonitoringConfig:
    """Configuration for Langfuse monitoring."""
    check_interval: int = 60  # seconds
    health_check_timeout: int = 30  # seconds
    max_consecutive_failures: int = 3
    metrics_retention_hours: int = 24
    alerts_enabled: bool = True
    log_file: Optional[Path] = None
    
    def __post_init__(self):
        """Set default log file if not provided."""
        if self.log_file is None:
            self.log_file = Path("langfuse_monitoring.log")


@dataclass
class HealthMetric:
    """A single health check metric."""
    timestamp: datetime
    is_healthy: bool
    response_time_ms: float
    status_code: Optional[int] = None
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """A monitoring alert."""
    timestamp: datetime
    level: AlertLevel
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class LangfuseMonitor:
    """
    Continuous health monitor for Langfuse instances.
    
    This class provides real-time monitoring of Langfuse instances with
    health checking, metrics collection, and alerting capabilities.
    """
    
    def __init__(self, 
                 validation_config: ValidationConfig,
                 monitoring_config: Optional[MonitoringConfig] = None):
        """
        Initialize the monitor.
        
        Args:
            validation_config: Configuration for Langfuse validation
            monitoring_config: Configuration for monitoring behavior
        """
        self.validation_config = validation_config
        self.monitoring_config = monitoring_config or MonitoringConfig()
        
        self.status = MonitoringStatus.STOPPED
        self.metrics: List[HealthMetric] = []
        self.alerts: List[Alert] = []
        
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._consecutive_failures = 0
        self._last_healthy_time: Optional[datetime] = None
        
        # Alert callbacks
        self._alert_callbacks: List[Callable[[Alert], None]] = []
    
    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """
        Add a callback function to be called when alerts are triggered.
        
        Args:
            callback: Function that takes an Alert object as parameter
        """
        self._alert_callbacks.append(callback)
    
    def start_monitoring(self) -> bool:
        """
        Start continuous monitoring.
        
        Returns:
            True if monitoring started successfully, False otherwise.
        """
        if self.status in [MonitoringStatus.RUNNING, MonitoringStatus.STARTING]:
            print("Monitoring is already running or starting")
            return False
        
        self.status = MonitoringStatus.STARTING
        self._stop_event.clear()
        
        try:
            self._monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                name="LangfuseMonitor",
                daemon=True
            )
            self._monitor_thread.start()
            
            # Wait a bit to ensure thread started successfully
            time.sleep(0.5)
            
            if self.status == MonitoringStatus.RUNNING:
                self._log_info("Monitoring started successfully")
                return True
            else:
                self._log_error("Failed to start monitoring")
                return False
                
        except Exception as e:
            self.status = MonitoringStatus.ERROR
            self._log_error(f"Error starting monitoring: {e}")
            return False
    
    def stop_monitoring(self) -> bool:
        """
        Stop continuous monitoring.
        
        Returns:
            True if monitoring stopped successfully, False otherwise.
        """
        if self.status == MonitoringStatus.STOPPED:
            return True
        
        self.status = MonitoringStatus.STOPPING
        self._stop_event.set()
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=10)
            
            if self._monitor_thread.is_alive():
                self._log_error("Monitor thread did not stop gracefully")
                return False
        
        self.status = MonitoringStatus.STOPPED
        self._log_info("Monitoring stopped")
        return True
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        self.status = MonitoringStatus.RUNNING
        self._log_info(f"Starting monitoring loop (interval: {self.monitoring_config.check_interval}s)")
        
        while not self._stop_event.is_set():
            try:
                # Perform health check
                metric = self._perform_health_check()
                self.metrics.append(metric)
                
                # Process the metric
                self._process_health_metric(metric)
                
                # Clean up old metrics
                self._cleanup_old_metrics()
                
                # Wait for next check
                self._stop_event.wait(self.monitoring_config.check_interval)
                
            except Exception as e:
                self._log_error(f"Error in monitoring loop: {e}")
                self.status = MonitoringStatus.ERROR
                break
    
    def _perform_health_check(self) -> HealthMetric:
        """Perform a single health check."""
        start_time = time.time()
        timestamp = datetime.utcnow()
        
        try:
            validator = LangfuseValidator(self.validation_config)
            
            # Perform basic connectivity and health endpoint checks
            validator._validate_basic_connectivity()
            validator._validate_health_endpoint()
            
            results = validator.get_results()
            duration_ms = (time.time() - start_time) * 1000
            
            # Determine overall health
            is_healthy = all(r.status == ValidationStatus.PASS for r in results)
            
            # Extract details
            details = {}
            status_code = None
            error_message = None
            
            for result in results:
                if result.name == "basic_connectivity" and result.details:
                    status_code = result.details.get("status_code")
                
                if result.status == ValidationStatus.FAIL:
                    if error_message is None:
                        error_message = result.message
                    else:
                        error_message += f"; {result.message}"
                
                details[result.name] = {
                    "status": result.status.value,
                    "message": result.message,
                    "duration_ms": result.duration_ms
                }
            
            return HealthMetric(
                timestamp=timestamp,
                is_healthy=is_healthy,
                response_time_ms=duration_ms,
                status_code=status_code,
                error_message=error_message,
                details=details
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthMetric(
                timestamp=timestamp,
                is_healthy=False,
                response_time_ms=duration_ms,
                error_message=str(e)
            )
    
    def _process_health_metric(self, metric: HealthMetric) -> None:
        """Process a health metric and trigger alerts if necessary."""
        if metric.is_healthy:
            # Reset consecutive failures
            if self._consecutive_failures > 0:
                self._create_alert(
                    AlertLevel.INFO,
                    "Langfuse instance is healthy again",
                    {"consecutive_failures_reset": self._consecutive_failures}
                )
            
            self._consecutive_failures = 0
            self._last_healthy_time = metric.timestamp
            
        else:
            # Increment consecutive failures
            self._consecutive_failures += 1
            
            # Trigger alerts based on failure count
            if self._consecutive_failures == 1:
                self._create_alert(
                    AlertLevel.WARNING,
                    "Langfuse health check failed",
                    {
                        "error_message": metric.error_message,
                        "response_time_ms": metric.response_time_ms
                    }
                )
            elif self._consecutive_failures == self.monitoring_config.max_consecutive_failures:
                self._create_alert(
                    AlertLevel.ERROR,
                    f"Langfuse has failed {self._consecutive_failures} consecutive health checks",
                    {
                        "consecutive_failures": self._consecutive_failures,
                        "last_healthy": self._last_healthy_time.isoformat() if self._last_healthy_time else None
                    }
                )
            elif self._consecutive_failures > self.monitoring_config.max_consecutive_failures:
                # Critical alert for extended outages
                minutes_down = (metric.timestamp - self._last_healthy_time).total_seconds() / 60 if self._last_healthy_time else None
                self._create_alert(
                    AlertLevel.CRITICAL,
                    f"Langfuse has been down for {self._consecutive_failures} checks",
                    {
                        "consecutive_failures": self._consecutive_failures,
                        "minutes_down": minutes_down,
                        "last_healthy": self._last_healthy_time.isoformat() if self._last_healthy_time else None
                    }
                )
        
        # Log the metric
        health_status = "HEALTHY" if metric.is_healthy else "UNHEALTHY"
        self._log_info(f"Health check: {health_status} ({metric.response_time_ms:.1f}ms)")
    
    def _create_alert(self, level: AlertLevel, message: str, details: Dict[str, Any]) -> None:
        """Create and process an alert."""
        alert = Alert(
            timestamp=datetime.utcnow(),
            level=level,
            message=message,
            details=details
        )
        
        self.alerts.append(alert)
        
        # Call alert callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self._log_error(f"Error in alert callback: {e}")
        
        # Log the alert
        self._log_alert(alert)
    
    def _cleanup_old_metrics(self) -> None:
        """Remove old metrics based on retention policy."""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.monitoring_config.metrics_retention_hours)
        self.metrics = [m for m in self.metrics if m.timestamp > cutoff_time]
    
    def get_current_status(self) -> Dict[str, Any]:
        """
        Get current monitoring status and recent metrics.
        
        Returns:
            Dictionary with current status information.
        """
        recent_metrics = self.metrics[-10:] if self.metrics else []
        recent_alerts = [a for a in self.alerts if not a.resolved][-5:]
        
        health_summary = self._get_health_summary()
        
        return {
            "monitoring_status": self.status.value,
            "current_health": {
                "is_healthy": recent_metrics[-1].is_healthy if recent_metrics else False,
                "consecutive_failures": self._consecutive_failures,
                "last_check": recent_metrics[-1].timestamp.isoformat() if recent_metrics else None,
                "last_healthy": self._last_healthy_time.isoformat() if self._last_healthy_time else None
            },
            "health_summary": health_summary,
            "recent_metrics": [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "is_healthy": m.is_healthy,
                    "response_time_ms": m.response_time_ms,
                    "error_message": m.error_message
                }
                for m in recent_metrics
            ],
            "active_alerts": [
                {
                    "timestamp": a.timestamp.isoformat(),
                    "level": a.level.value,
                    "message": a.message,
                    "details": a.details
                }
                for a in recent_alerts
            ],
            "config": {
                "check_interval": self.monitoring_config.check_interval,
                "max_consecutive_failures": self.monitoring_config.max_consecutive_failures,
                "metrics_retention_hours": self.monitoring_config.metrics_retention_hours
            }
        }
    
    def _get_health_summary(self) -> Dict[str, Any]:
        """Generate health summary statistics."""
        if not self.metrics:
            return {"total_checks": 0}
        
        # Last hour metrics
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_metrics = [m for m in self.metrics if m.timestamp > one_hour_ago]
        
        if not recent_metrics:
            recent_metrics = self.metrics[-12:]  # Last 12 checks as fallback
        
        total_checks = len(recent_metrics)
        healthy_checks = len([m for m in recent_metrics if m.is_healthy])
        avg_response_time = sum(m.response_time_ms for m in recent_metrics) / total_checks if total_checks > 0 else 0
        
        return {
            "total_checks": total_checks,
            "healthy_checks": healthy_checks,
            "health_percentage": (healthy_checks / total_checks * 100) if total_checks > 0 else 0,
            "average_response_time_ms": avg_response_time,
            "time_period_hours": 1 if len(self.metrics) > 12 else "last_12_checks"
        }
    
    def export_metrics(self, file_path: Path) -> bool:
        """
        Export metrics to a JSON file.
        
        Args:
            file_path: Path to export file
            
        Returns:
            True if export was successful, False otherwise.
        """
        try:
            export_data = {
                "export_timestamp": datetime.utcnow().isoformat(),
                "monitoring_status": self.status.value,
                "config": {
                    "validation_config": {
                        "host": self.validation_config.host,
                        "port": self.validation_config.port,
                        "base_url": self.validation_config.base_url
                    },
                    "monitoring_config": {
                        "check_interval": self.monitoring_config.check_interval,
                        "max_consecutive_failures": self.monitoring_config.max_consecutive_failures,
                        "metrics_retention_hours": self.monitoring_config.metrics_retention_hours
                    }
                },
                "metrics": [
                    {
                        "timestamp": m.timestamp.isoformat(),
                        "is_healthy": m.is_healthy,
                        "response_time_ms": m.response_time_ms,
                        "status_code": m.status_code,
                        "error_message": m.error_message,
                        "details": m.details
                    }
                    for m in self.metrics
                ],
                "alerts": [
                    {
                        "timestamp": a.timestamp.isoformat(),
                        "level": a.level.value,
                        "message": a.message,
                        "details": a.details,
                        "resolved": a.resolved,
                        "resolved_at": a.resolved_at.isoformat() if a.resolved_at else None
                    }
                    for a in self.alerts
                ]
            }
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self._log_info(f"Metrics exported to {file_path}")
            return True
            
        except Exception as e:
            self._log_error(f"Failed to export metrics: {e}")
            return False
    
    def _log_info(self, message: str) -> None:
        """Log an info message."""
        self._log_message("INFO", message)
    
    def _log_error(self, message: str) -> None:
        """Log an error message."""
        self._log_message("ERROR", message)
    
    def _log_alert(self, alert: Alert) -> None:
        """Log an alert."""
        self._log_message(f"ALERT-{alert.level.value.upper()}", alert.message)
    
    def _log_message(self, level: str, message: str) -> None:
        """Log a message to file and console."""
        timestamp = datetime.utcnow().isoformat()
        log_line = f"[{timestamp}] {level}: {message}"
        
        # Print to console
        print(log_line)
        
        # Write to log file if configured
        if self.monitoring_config.log_file:
            try:
                with open(self.monitoring_config.log_file, 'a') as f:
                    f.write(log_line + '\n')
            except Exception as e:
                print(f"Failed to write to log file: {e}")


def create_monitor(
    host: str = "localhost",
    port: int = 3000,
    public_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    check_interval: int = 60,
    max_consecutive_failures: int = 3
) -> LangfuseMonitor:
    """
    Create a Langfuse monitor with the given configuration.
    
    Args:
        host: Langfuse host
        port: Langfuse port
        public_key: Optional API public key
        secret_key: Optional API secret key
        check_interval: Health check interval in seconds
        max_consecutive_failures: Maximum consecutive failures before alert
        
    Returns:
        Configured LangfuseMonitor instance
    """
    validation_config = ValidationConfig(
        host=host,
        port=port,
        public_key=public_key,
        secret_key=secret_key
    )
    
    monitoring_config = MonitoringConfig(
        check_interval=check_interval,
        max_consecutive_failures=max_consecutive_failures
    )
    
    return LangfuseMonitor(validation_config, monitoring_config)


def simple_alert_callback(alert: Alert) -> None:
    """
    Simple alert callback that prints alerts to console.
    
    Args:
        alert: The alert that was triggered
    """
    level_symbols = {
        AlertLevel.INFO: "ℹ️",
        AlertLevel.WARNING: "⚠️",
        AlertLevel.ERROR: "❌",
        AlertLevel.CRITICAL: "🚨"
    }
    
    symbol = level_symbols.get(alert.level, "⚠️")
    print(f"\n{symbol} ALERT [{alert.level.value.upper()}]: {alert.message}")
    
    if alert.details:
        for key, value in alert.details.items():
            print(f"  {key}: {value}")
    print()


def start_monitoring(
    host: str = "localhost",
    port: int = 3000,
    public_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    check_interval: int = 60,
    alert_callback: Optional[Callable[[Alert], None]] = None
) -> LangfuseMonitor:
    """
    Start monitoring a Langfuse instance.
    
    Args:
        host: Langfuse host
        port: Langfuse port
        public_key: Optional API public key
        secret_key: Optional API secret key
        check_interval: Health check interval in seconds
        alert_callback: Optional callback for alerts
        
    Returns:
        Started monitor instance
    """
    monitor = create_monitor(host, port, public_key, secret_key, check_interval)
    
    # Add alert callback
    if alert_callback:
        monitor.add_alert_callback(alert_callback)
    else:
        monitor.add_alert_callback(simple_alert_callback)
    
    # Start monitoring
    if monitor.start_monitoring():
        print(f"Monitoring started for {host}:{port}")
        print(f"Check interval: {check_interval} seconds")
        return monitor
    else:
        raise RuntimeError("Failed to start monitoring") 