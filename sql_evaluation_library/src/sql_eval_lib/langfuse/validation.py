"""
Langfuse connection validation and testing utilities.

This module provides comprehensive validation for Langfuse connections,
including API key validation, project access verification, and health checking.
"""

import os
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

try:
    import requests
except ImportError:
    print("Warning: requests package not installed. HTTP validation will not work.")
    requests = None

try:
    from langfuse import Langfuse
except ImportError:
    print("Warning: langfuse package not installed. Full validation will not work.")
    Langfuse = None


class ValidationStatus(Enum):
    """Status of validation checks."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    name: str
    status: ValidationStatus
    message: str
    details: Optional[Dict[str, Any]] = None
    duration_ms: Optional[float] = None


@dataclass
class ValidationConfig:
    """Configuration for Langfuse validation."""
    host: str = "localhost"
    port: int = 3000
    public_key: Optional[str] = None
    secret_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 30
    verify_ssl: bool = True
    
    def __post_init__(self):
        """Set derived configuration values."""
        if self.base_url is None:
            protocol = "https" if self.verify_ssl else "http"
            self.base_url = f"{protocol}://{self.host}:{self.port}"


class LangfuseValidator:
    """
    Validates Langfuse connectivity and configuration.
    
    This class provides comprehensive validation of Langfuse instances,
    including basic connectivity, API authentication, and functionality tests.
    """
    
    def __init__(self, config: ValidationConfig):
        """
        Initialize the validator.
        
        Args:
            config: Validation configuration
        """
        self.config = config
        self.results: List[ValidationResult] = []
    
    def validate_all(self) -> Dict[str, Any]:
        """
        Run all validation checks.
        
        Returns:
            Dictionary with validation summary and detailed results.
        """
        self.results.clear()
        
        # Basic connectivity tests
        self._validate_basic_connectivity()
        self._validate_health_endpoint()
        
        # API validation (if keys are provided)
        if self.config.public_key and self.config.secret_key:
            self._validate_api_authentication()
            self._validate_api_functionality()
        else:
            self.results.append(ValidationResult(
                name="api_authentication",
                status=ValidationStatus.SKIP,
                message="No API keys provided - skipping API validation"
            ))
            self.results.append(ValidationResult(
                name="api_functionality",
                status=ValidationStatus.SKIP,
                message="No API keys provided - skipping functionality tests"
            ))
        
        return self._generate_summary()
    
    def _validate_basic_connectivity(self) -> None:
        """Validate basic HTTP connectivity to Langfuse."""
        start_time = time.time()
        
        try:
            if requests is None:
                self.results.append(ValidationResult(
                    name="basic_connectivity",
                    status=ValidationStatus.FAIL,
                    message="requests package not available",
                    duration_ms=(time.time() - start_time) * 1000
                ))
                return
            
            response = requests.get(
                self.config.base_url,
                timeout=self.config.timeout,
                verify=self.config.verify_ssl
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code in [200, 301, 302]:
                self.results.append(ValidationResult(
                    name="basic_connectivity",
                    status=ValidationStatus.PASS,
                    message=f"Successfully connected to {self.config.base_url}",
                    details={"status_code": response.status_code, "response_time_ms": duration_ms},
                    duration_ms=duration_ms
                ))
            else:
                self.results.append(ValidationResult(
                    name="basic_connectivity",
                    status=ValidationStatus.FAIL,
                    message=f"HTTP {response.status_code} received from {self.config.base_url}",
                    details={"status_code": response.status_code, "response_time_ms": duration_ms},
                    duration_ms=duration_ms
                ))
                
        except requests.exceptions.ConnectTimeout:
            duration_ms = (time.time() - start_time) * 1000
            self.results.append(ValidationResult(
                name="basic_connectivity",
                status=ValidationStatus.FAIL,
                message=f"Connection timeout to {self.config.base_url}",
                details={"timeout_seconds": self.config.timeout},
                duration_ms=duration_ms
            ))
        except requests.exceptions.ConnectionError as e:
            duration_ms = (time.time() - start_time) * 1000
            self.results.append(ValidationResult(
                name="basic_connectivity",
                status=ValidationStatus.FAIL,
                message=f"Connection error: {str(e)}",
                duration_ms=duration_ms
            ))
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.results.append(ValidationResult(
                name="basic_connectivity",
                status=ValidationStatus.FAIL,
                message=f"Unexpected error: {str(e)}",
                duration_ms=duration_ms
            ))
    
    def _validate_health_endpoint(self) -> None:
        """Validate Langfuse health endpoint."""
        start_time = time.time()
        
        try:
            if requests is None:
                self.results.append(ValidationResult(
                    name="health_endpoint",
                    status=ValidationStatus.SKIP,
                    message="requests package not available",
                    duration_ms=(time.time() - start_time) * 1000
                ))
                return
            
            health_url = f"{self.config.base_url}/api/health"
            response = requests.get(
                health_url,
                timeout=self.config.timeout,
                verify=self.config.verify_ssl
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                try:
                    health_data = response.json()
                    self.results.append(ValidationResult(
                        name="health_endpoint",
                        status=ValidationStatus.PASS,
                        message="Health endpoint is working",
                        details={"health_data": health_data, "response_time_ms": duration_ms},
                        duration_ms=duration_ms
                    ))
                except Exception:
                    self.results.append(ValidationResult(
                        name="health_endpoint",
                        status=ValidationStatus.WARNING,
                        message="Health endpoint responded but returned invalid JSON",
                        details={"status_code": response.status_code, "response_time_ms": duration_ms},
                        duration_ms=duration_ms
                    ))
            else:
                self.results.append(ValidationResult(
                    name="health_endpoint",
                    status=ValidationStatus.FAIL,
                    message=f"Health endpoint returned HTTP {response.status_code}",
                    details={"status_code": response.status_code, "response_time_ms": duration_ms},
                    duration_ms=duration_ms
                ))
                
        except requests.exceptions.Timeout:
            duration_ms = (time.time() - start_time) * 1000
            self.results.append(ValidationResult(
                name="health_endpoint",
                status=ValidationStatus.FAIL,
                message=f"Health endpoint timeout after {self.config.timeout}s",
                duration_ms=duration_ms
            ))
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.results.append(ValidationResult(
                name="health_endpoint",
                status=ValidationStatus.FAIL,
                message=f"Health endpoint error: {str(e)}",
                duration_ms=duration_ms
            ))
    
    def _validate_api_authentication(self) -> None:
        """Validate API key authentication."""
        start_time = time.time()
        
        try:
            if Langfuse is None:
                self.results.append(ValidationResult(
                    name="api_authentication",
                    status=ValidationStatus.FAIL,
                    message="langfuse package not available",
                    duration_ms=(time.time() - start_time) * 1000
                ))
                return
            
            # Create Langfuse client
            client = Langfuse(
                public_key=self.config.public_key,
                secret_key=self.config.secret_key,
                host=self.config.base_url
            )
            
            # Try to create a simple trace to test authentication
            trace = client.trace(name="validation_test")
            
            # Flush to ensure the request is sent
            client.flush()
            
            duration_ms = (time.time() - start_time) * 1000
            
            self.results.append(ValidationResult(
                name="api_authentication",
                status=ValidationStatus.PASS,
                message="API authentication successful",
                details={"trace_id": trace.id, "response_time_ms": duration_ms},
                duration_ms=duration_ms
            ))
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = str(e).lower()
            
            if "401" in error_msg or "unauthorized" in error_msg:
                status = ValidationStatus.FAIL
                message = "API authentication failed - invalid credentials"
            elif "403" in error_msg or "forbidden" in error_msg:
                status = ValidationStatus.FAIL
                message = "API authentication failed - access forbidden"
            elif "timeout" in error_msg:
                status = ValidationStatus.FAIL
                message = "API authentication timed out"
            else:
                status = ValidationStatus.FAIL
                message = f"API authentication error: {str(e)}"
            
            self.results.append(ValidationResult(
                name="api_authentication",
                status=status,
                message=message,
                duration_ms=duration_ms
            ))
    
    def _validate_api_functionality(self) -> None:
        """Validate core API functionality."""
        start_time = time.time()
        
        try:
            if Langfuse is None:
                self.results.append(ValidationResult(
                    name="api_functionality",
                    status=ValidationStatus.SKIP,
                    message="langfuse package not available",
                    duration_ms=(time.time() - start_time) * 1000
                ))
                return
            
            client = Langfuse(
                public_key=self.config.public_key,
                secret_key=self.config.secret_key,
                host=self.config.base_url
            )
            
            # Test creating a trace with generation and score
            trace_name = f"validation_test_{int(time.time())}"
            trace = client.trace(name=trace_name)
            
            # Add a generation
            generation = trace.generation(
                name="test_generation",
                model="test-model",
                input="test input",
                output="test output"
            )
            
            # Add a score
            trace.score(
                name="test_score",
                value=0.95,
                comment="Validation test score"
            )
            
            # Flush to ensure requests are sent
            client.flush()
            
            duration_ms = (time.time() - start_time) * 1000
            
            self.results.append(ValidationResult(
                name="api_functionality",
                status=ValidationStatus.PASS,
                message="API functionality test successful",
                details={
                    "trace_id": trace.id,
                    "generation_id": generation.id,
                    "response_time_ms": duration_ms
                },
                duration_ms=duration_ms
            ))
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.results.append(ValidationResult(
                name="api_functionality",
                status=ValidationStatus.FAIL,
                message=f"API functionality test failed: {str(e)}",
                duration_ms=duration_ms
            ))
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate validation summary."""
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r.status == ValidationStatus.PASS])
        failed_tests = len([r for r in self.results if r.status == ValidationStatus.FAIL])
        warning_tests = len([r for r in self.results if r.status == ValidationStatus.WARNING])
        skipped_tests = len([r for r in self.results if r.status == ValidationStatus.SKIP])
        
        overall_status = "PASS"
        if failed_tests > 0:
            overall_status = "FAIL"
        elif warning_tests > 0:
            overall_status = "WARNING"
        
        total_duration = sum(r.duration_ms for r in self.results if r.duration_ms is not None)
        
        return {
            "overall_status": overall_status,
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "warnings": warning_tests,
                "skipped": skipped_tests,
                "total_duration_ms": total_duration
            },
            "config": {
                "host": self.config.host,
                "port": self.config.port,
                "base_url": self.config.base_url,
                "has_credentials": bool(self.config.public_key and self.config.secret_key)
            },
            "results": [
                {
                    "name": r.name,
                    "status": r.status.value,
                    "message": r.message,
                    "details": r.details,
                    "duration_ms": r.duration_ms
                }
                for r in self.results
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_results(self) -> List[ValidationResult]:
        """Get all validation results."""
        return self.results.copy()


def validate_langfuse_connection(
    host: str = "localhost",
    port: int = 3000,
    public_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Validate Langfuse connection with the given parameters.
    
    Args:
        host: Langfuse host
        port: Langfuse port
        public_key: Optional API public key
        secret_key: Optional API secret key
        base_url: Optional complete base URL (overrides host/port)
        timeout: Request timeout in seconds
        
    Returns:
        Validation results dictionary
    """
    config = ValidationConfig(
        host=host,
        port=port,
        public_key=public_key,
        secret_key=secret_key,
        base_url=base_url,
        timeout=timeout
    )
    
    validator = LangfuseValidator(config)
    return validator.validate_all()


def validate_langfuse_from_env() -> Dict[str, Any]:
    """
    Validate Langfuse connection using environment variables.
    
    Expected environment variables:
    - LANGFUSE_PUBLIC_KEY: API public key
    - LANGFUSE_SECRET_KEY: API secret key
    - LANGFUSE_HOST: Host (default: localhost)
    - LANGFUSE_PORT: Port (default: 3000)
    - LANGFUSE_BASE_URL: Complete base URL (optional)
    
    Returns:
        Validation results dictionary
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "localhost")
    port = int(os.getenv("LANGFUSE_PORT", "3000"))
    base_url = os.getenv("LANGFUSE_BASE_URL")
    
    return validate_langfuse_connection(
        host=host,
        port=port,
        public_key=public_key,
        secret_key=secret_key,
        base_url=base_url
    )


def quick_health_check(base_url: str) -> Tuple[bool, str]:
    """
    Quick health check for Langfuse instance.
    
    Args:
        base_url: Base URL of Langfuse instance
        
    Returns:
        Tuple of (is_healthy, status_message)
    """
    try:
        if requests is None:
            return False, "requests package not available"
        
        response = requests.get(f"{base_url}/api/health", timeout=10)
        if response.status_code == 200:
            return True, "Healthy"
        else:
            return False, f"HTTP {response.status_code}"
    except requests.exceptions.ConnectTimeout:
        return False, "Connection timeout"
    except requests.exceptions.ConnectionError:
        return False, "Connection error"
    except Exception as e:
        return False, f"Error: {str(e)}"


def print_validation_report(validation_results: Dict[str, Any]) -> None:
    """
    Print a formatted validation report.
    
    Args:
        validation_results: Results from validate_langfuse_connection()
    """
    print("\n" + "="*60)
    print("LANGFUSE VALIDATION REPORT")
    print("="*60)
    
    summary = validation_results["summary"]
    print(f"Overall Status: {validation_results['overall_status']}")
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Warnings: {summary['warnings']}")
    print(f"Skipped: {summary['skipped']}")
    print(f"Total Duration: {summary['total_duration_ms']:.1f}ms")
    
    print("\n" + "-"*60)
    print("DETAILED RESULTS")
    print("-"*60)
    
    for result in validation_results["results"]:
        status_symbol = {
            "pass": "✅",
            "fail": "❌",
            "warning": "⚠️",
            "skip": "⏭️"
        }.get(result["status"], "?")
        
        duration = f" ({result['duration_ms']:.1f}ms)" if result["duration_ms"] else ""
        print(f"{status_symbol} {result['name']}: {result['message']}{duration}")
        
        if result.get("details"):
            for key, value in result["details"].items():
                if key != "response_time_ms":  # Already shown in duration
                    print(f"   {key}: {value}")
    
    print("="*60) 