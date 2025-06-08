# llmwatch/src/sql_eval_lib/langfuse/__init__.py
from .manager import LangfuseClient

# Phase 2.1: Langfuse Enhancement - New components
from .deployment import (
    LangfuseDeployment,
    DeploymentConfig,
    DeploymentStatus,
    deploy_langfuse,
    stop_langfuse,
    get_langfuse_status
)

from .validation import (
    LangfuseValidator,
    ValidationConfig,
    ValidationResult,
    ValidationStatus,
    validate_langfuse_connection,
    validate_langfuse_from_env,
    quick_health_check,
    print_validation_report
)

from .monitoring import (
    LangfuseMonitor,
    MonitoringConfig,
    MonitoringStatus,
    HealthMetric,
    Alert,
    AlertLevel,
    create_monitor,
    start_monitoring,
    simple_alert_callback
)

from .project_manager import (
    LangfuseProjectManager,
    ProjectConfig,
    ApiKeyPair,
    create_project_manager,
    setup_default_project,
    print_project_summary
)

__all__ = [
    # Legacy components
    "LangfuseClient",
    
    # Deployment
    "LangfuseDeployment",
    "DeploymentConfig", 
    "DeploymentStatus",
    "deploy_langfuse",
    "stop_langfuse",
    "get_langfuse_status",
    
    # Validation
    "LangfuseValidator",
    "ValidationConfig",
    "ValidationResult",
    "ValidationStatus",
    "validate_langfuse_connection",
    "validate_langfuse_from_env",
    "quick_health_check",
    "print_validation_report",
    
    # Monitoring
    "LangfuseMonitor",
    "MonitoringConfig",
    "MonitoringStatus",
    "HealthMetric",
    "Alert",
    "AlertLevel",
    "create_monitor",
    "start_monitoring",
    "simple_alert_callback",
    
    # Project Management
    "LangfuseProjectManager",
    "ProjectConfig",
    "ApiKeyPair",
    "create_project_manager",
    "setup_default_project",
    "print_project_summary",
]
