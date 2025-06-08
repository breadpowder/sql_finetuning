"""
Multi-provider model endpoint validation and testing utilities.

This module provides comprehensive validation for various LLM providers
including OpenAI, Gemini, Ollama, Claude, and other model endpoints.
"""

import os
import time
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

# Optional imports for different providers
try:
    import openai
except ImportError:
    print("Warning: openai package not installed. OpenAI validation will not work.")
    openai = None

try:
    import google.generativeai as genai
except ImportError:
    print("Warning: google-generativeai package not installed. Gemini validation will not work.")
    genai = None

try:
    import requests
except ImportError:
    print("Warning: requests package not installed. HTTP validation will not work.")
    requests = None

try:
    import anthropic
except ImportError:
    print("Warning: anthropic package not installed. Claude validation will not work.")
    anthropic = None


class ModelProvider(Enum):
    """Supported model providers."""
    OPENAI = "openai"
    GEMINI = "gemini"
    OLLAMA = "ollama"
    CLAUDE = "claude"
    CUSTOM = "custom"


class ValidationStatus(Enum):
    """Status of validation checks."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"


@dataclass
class ModelEndpoint:
    """Configuration for a model endpoint."""
    provider: ModelProvider
    name: str
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    timeout: int = 30
    max_tokens: int = 1000
    temperature: float = 0.1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set provider-specific defaults."""
        if self.provider == ModelProvider.OPENAI and self.base_url is None:
            self.base_url = "https://api.openai.com/v1"
        elif self.provider == ModelProvider.GEMINI and self.base_url is None:
            self.base_url = "https://generativelanguage.googleapis.com"
        elif self.provider == ModelProvider.CLAUDE and self.base_url is None:
            self.base_url = "https://api.anthropic.com"
        elif self.provider == ModelProvider.OLLAMA and self.base_url is None:
            self.base_url = "http://localhost:11434"


@dataclass
class ValidationResult:
    """Result of a model validation check."""
    endpoint_name: str
    provider: ModelProvider
    test_name: str
    status: ValidationStatus
    message: str
    duration_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class ModelResponse:
    """Response from a model endpoint."""
    content: str
    provider: ModelProvider
    model_name: str
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class ModelValidator:
    """
    Validates model endpoints across multiple providers.
    
    This class provides comprehensive validation of model endpoints,
    including connectivity, authentication, and functionality tests.
    """
    
    def __init__(self, endpoints: List[ModelEndpoint]):
        """
        Initialize the validator.
        
        Args:
            endpoints: List of model endpoints to validate
        """
        self.endpoints = {ep.name: ep for ep in endpoints}
        self.results: List[ValidationResult] = []
    
    def validate_all(self) -> Dict[str, Any]:
        """
        Run validation tests on all endpoints.
        
        Returns:
            Dictionary with validation summary and detailed results
        """
        self.results.clear()
        
        for name, endpoint in self.endpoints.items():
            print(f"\nValidating endpoint: {name} ({endpoint.provider.value})")
            
            # Run provider-specific validations
            if endpoint.provider == ModelProvider.OPENAI:
                self._validate_openai(endpoint)
            elif endpoint.provider == ModelProvider.GEMINI:
                self._validate_gemini(endpoint)
            elif endpoint.provider == ModelProvider.OLLAMA:
                self._validate_ollama(endpoint)
            elif endpoint.provider == ModelProvider.CLAUDE:
                self._validate_claude(endpoint)
            else:
                self._validate_custom(endpoint)
        
        return self._generate_summary()
    
    def validate_endpoint(self, endpoint_name: str) -> List[ValidationResult]:
        """
        Validate a specific endpoint.
        
        Args:
            endpoint_name: Name of the endpoint to validate
            
        Returns:
            List of validation results for the endpoint
        """
        if endpoint_name not in self.endpoints:
            return [ValidationResult(
                endpoint_name=endpoint_name,
                provider=ModelProvider.CUSTOM,
                test_name="endpoint_exists",
                status=ValidationStatus.FAIL,
                message=f"Endpoint '{endpoint_name}' not found"
            )]
        
        endpoint = self.endpoints[endpoint_name]
        start_results_count = len(self.results)
        
        # Run validation based on provider
        if endpoint.provider == ModelProvider.OPENAI:
            self._validate_openai(endpoint)
        elif endpoint.provider == ModelProvider.GEMINI:
            self._validate_gemini(endpoint)
        elif endpoint.provider == ModelProvider.OLLAMA:
            self._validate_ollama(endpoint)
        elif endpoint.provider == ModelProvider.CLAUDE:
            self._validate_claude(endpoint)
        else:
            self._validate_custom(endpoint)
        
        # Return only the results for this endpoint
        return self.results[start_results_count:]
    
    def _validate_openai(self, endpoint: ModelEndpoint) -> None:
        """Validate OpenAI endpoint."""
        if openai is None:
            self._add_result(endpoint, "package_availability", ValidationStatus.FAIL,
                           "OpenAI package not available")
            return
        
        # Test 1: API Key validation
        if not endpoint.api_key:
            self._add_result(endpoint, "api_key_check", ValidationStatus.FAIL,
                           "No API key provided")
            return
        
        # Test 2: Basic connectivity
        self._test_openai_connectivity(endpoint)
        
        # Test 3: Model availability
        self._test_openai_model_availability(endpoint)
        
        # Test 4: Simple generation
        self._test_openai_generation(endpoint)
    
    def _test_openai_connectivity(self, endpoint: ModelEndpoint) -> None:
        """Test OpenAI API connectivity."""
        start_time = time.time()
        
        try:
            client = openai.OpenAI(
                api_key=endpoint.api_key,
                base_url=endpoint.base_url,
                timeout=endpoint.timeout
            )
            
            # Try to list models
            response = client.models.list()
            duration_ms = (time.time() - start_time) * 1000
            
            self._add_result(endpoint, "connectivity", ValidationStatus.PASS,
                           "Successfully connected to OpenAI API",
                           duration_ms=duration_ms,
                           details={"models_count": len(response.data)})
            
        except openai.AuthenticationError:
            duration_ms = (time.time() - start_time) * 1000
            self._add_result(endpoint, "connectivity", ValidationStatus.FAIL,
                           "Authentication failed - invalid API key",
                           duration_ms=duration_ms)
        except openai.APIConnectionError as e:
            duration_ms = (time.time() - start_time) * 1000
            self._add_result(endpoint, "connectivity", ValidationStatus.FAIL,
                           f"Connection error: {str(e)}",
                           duration_ms=duration_ms)
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._add_result(endpoint, "connectivity", ValidationStatus.FAIL,
                           f"Unexpected error: {str(e)}",
                           duration_ms=duration_ms)
    
    def _test_openai_model_availability(self, endpoint: ModelEndpoint) -> None:
        """Test if the specified OpenAI model is available."""
        start_time = time.time()
        
        try:
            client = openai.OpenAI(
                api_key=endpoint.api_key,
                base_url=endpoint.base_url,
                timeout=endpoint.timeout
            )
            
            # Try to retrieve the specific model
            model = client.models.retrieve(endpoint.model_name)
            duration_ms = (time.time() - start_time) * 1000
            
            self._add_result(endpoint, "model_availability", ValidationStatus.PASS,
                           f"Model '{endpoint.model_name}' is available",
                           duration_ms=duration_ms,
                           details={"model_id": model.id, "owned_by": model.owned_by})
            
        except openai.NotFoundError:
            duration_ms = (time.time() - start_time) * 1000
            self._add_result(endpoint, "model_availability", ValidationStatus.FAIL,
                           f"Model '{endpoint.model_name}' not found",
                           duration_ms=duration_ms)
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._add_result(endpoint, "model_availability", ValidationStatus.FAIL,
                           f"Error checking model availability: {str(e)}",
                           duration_ms=duration_ms)
    
    def _test_openai_generation(self, endpoint: ModelEndpoint) -> None:
        """Test OpenAI text generation."""
        start_time = time.time()
        
        try:
            client = openai.OpenAI(
                api_key=endpoint.api_key,
                base_url=endpoint.base_url,
                timeout=endpoint.timeout
            )
            
            response = client.chat.completions.create(
                model=endpoint.model_name,
                messages=[{"role": "user", "content": "Say 'Hello, this is a test!'"}],
                max_tokens=50,
                temperature=0.1
            )
            
            duration_ms = (time.time() - start_time) * 1000
            content = response.choices[0].message.content
            
            if content and "test" in content.lower():
                self._add_result(endpoint, "generation", ValidationStatus.PASS,
                               "Text generation successful",
                               duration_ms=duration_ms,
                               details={
                                   "generated_text": content,
                                   "usage": response.usage.__dict__ if response.usage else None
                               })
            else:
                self._add_result(endpoint, "generation", ValidationStatus.WARNING,
                               "Generation completed but response unexpected",
                               duration_ms=duration_ms,
                               details={"generated_text": content})
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._add_result(endpoint, "generation", ValidationStatus.FAIL,
                           f"Generation failed: {str(e)}",
                           duration_ms=duration_ms)
    
    def _validate_gemini(self, endpoint: ModelEndpoint) -> None:
        """Validate Gemini endpoint."""
        if genai is None:
            self._add_result(endpoint, "package_availability", ValidationStatus.FAIL,
                           "Google GenerativeAI package not available")
            return
        
        # Test 1: API Key validation
        if not endpoint.api_key:
            self._add_result(endpoint, "api_key_check", ValidationStatus.FAIL,
                           "No API key provided")
            return
        
        # Test 2: Configuration and connectivity
        self._test_gemini_connectivity(endpoint)
        
        # Test 3: Model availability
        self._test_gemini_model_availability(endpoint)
        
        # Test 4: Simple generation
        self._test_gemini_generation(endpoint)
    
    def _test_gemini_connectivity(self, endpoint: ModelEndpoint) -> None:
        """Test Gemini API connectivity."""
        start_time = time.time()
        
        try:
            genai.configure(api_key=endpoint.api_key)
            
            # Try to list models
            models = list(genai.list_models())
            duration_ms = (time.time() - start_time) * 1000
            
            self._add_result(endpoint, "connectivity", ValidationStatus.PASS,
                           "Successfully connected to Gemini API",
                           duration_ms=duration_ms,
                           details={"models_count": len(models)})
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            if "401" in str(e) or "unauthorized" in str(e).lower():
                self._add_result(endpoint, "connectivity", ValidationStatus.FAIL,
                               "Authentication failed - invalid API key",
                               duration_ms=duration_ms)
            else:
                self._add_result(endpoint, "connectivity", ValidationStatus.FAIL,
                               f"Connection error: {str(e)}",
                               duration_ms=duration_ms)
    
    def _test_gemini_model_availability(self, endpoint: ModelEndpoint) -> None:
        """Test if the specified Gemini model is available."""
        start_time = time.time()
        
        try:
            genai.configure(api_key=endpoint.api_key)
            
            # Check if model exists in available models
            available_models = [model.name for model in genai.list_models()]
            model_found = any(endpoint.model_name in model for model in available_models)
            
            duration_ms = (time.time() - start_time) * 1000
            
            if model_found:
                self._add_result(endpoint, "model_availability", ValidationStatus.PASS,
                               f"Model '{endpoint.model_name}' is available",
                               duration_ms=duration_ms,
                               details={"available_models": available_models[:5]})  # Show first 5
            else:
                self._add_result(endpoint, "model_availability", ValidationStatus.FAIL,
                               f"Model '{endpoint.model_name}' not found",
                               duration_ms=duration_ms,
                               details={"available_models": available_models[:5]})
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._add_result(endpoint, "model_availability", ValidationStatus.FAIL,
                           f"Error checking model availability: {str(e)}",
                           duration_ms=duration_ms)
    
    def _test_gemini_generation(self, endpoint: ModelEndpoint) -> None:
        """Test Gemini text generation."""
        start_time = time.time()
        
        try:
            genai.configure(api_key=endpoint.api_key)
            model = genai.GenerativeModel(endpoint.model_name)
            
            response = model.generate_content(
                "Say 'Hello, this is a test!'",
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=50,
                    temperature=endpoint.temperature
                )
            )
            
            duration_ms = (time.time() - start_time) * 1000
            content = response.text
            
            if content and "test" in content.lower():
                self._add_result(endpoint, "generation", ValidationStatus.PASS,
                               "Text generation successful",
                               duration_ms=duration_ms,
                               details={"generated_text": content})
            else:
                self._add_result(endpoint, "generation", ValidationStatus.WARNING,
                               "Generation completed but response unexpected",
                               duration_ms=duration_ms,
                               details={"generated_text": content})
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._add_result(endpoint, "generation", ValidationStatus.FAIL,
                           f"Generation failed: {str(e)}",
                           duration_ms=duration_ms)
    
    def _validate_ollama(self, endpoint: ModelEndpoint) -> None:
        """Validate Ollama endpoint."""
        if requests is None:
            self._add_result(endpoint, "package_availability", ValidationStatus.FAIL,
                           "Requests package not available")
            return
        
        # Test 1: Basic connectivity
        self._test_ollama_connectivity(endpoint)
        
        # Test 2: Model availability
        self._test_ollama_model_availability(endpoint)
        
        # Test 3: Simple generation
        self._test_ollama_generation(endpoint)
    
    def _test_ollama_connectivity(self, endpoint: ModelEndpoint) -> None:
        """Test Ollama server connectivity."""
        start_time = time.time()
        
        try:
            response = requests.get(f"{endpoint.base_url}/api/tags", timeout=endpoint.timeout)
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                models_data = response.json()
                self._add_result(endpoint, "connectivity", ValidationStatus.PASS,
                               "Successfully connected to Ollama API",
                               duration_ms=duration_ms,
                               details={"models_count": len(models_data.get("models", []))})
            else:
                self._add_result(endpoint, "connectivity", ValidationStatus.FAIL,
                               f"HTTP {response.status_code} from Ollama API",
                               duration_ms=duration_ms)
            
        except requests.exceptions.ConnectTimeout:
            duration_ms = (time.time() - start_time) * 1000
            self._add_result(endpoint, "connectivity", ValidationStatus.FAIL,
                           "Connection timeout to Ollama server",
                           duration_ms=duration_ms)
        except requests.exceptions.ConnectionError:
            duration_ms = (time.time() - start_time) * 1000
            self._add_result(endpoint, "connectivity", ValidationStatus.FAIL,
                           "Connection error - Ollama server may not be running",
                           duration_ms=duration_ms)
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._add_result(endpoint, "connectivity", ValidationStatus.FAIL,
                           f"Unexpected error: {str(e)}",
                           duration_ms=duration_ms)
    
    def _test_ollama_model_availability(self, endpoint: ModelEndpoint) -> None:
        """Test if the specified Ollama model is available."""
        start_time = time.time()
        
        try:
            response = requests.get(f"{endpoint.base_url}/api/tags", timeout=endpoint.timeout)
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                models_data = response.json()
                available_models = [model["name"] for model in models_data.get("models", [])]
                
                if endpoint.model_name in available_models:
                    self._add_result(endpoint, "model_availability", ValidationStatus.PASS,
                                   f"Model '{endpoint.model_name}' is available",
                                   duration_ms=duration_ms,
                                   details={"available_models": available_models})
                else:
                    self._add_result(endpoint, "model_availability", ValidationStatus.FAIL,
                                   f"Model '{endpoint.model_name}' not found",
                                   duration_ms=duration_ms,
                                   details={"available_models": available_models})
            else:
                self._add_result(endpoint, "model_availability", ValidationStatus.FAIL,
                               f"HTTP {response.status_code} when checking models",
                               duration_ms=duration_ms)
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._add_result(endpoint, "model_availability", ValidationStatus.FAIL,
                           f"Error checking model availability: {str(e)}",
                           duration_ms=duration_ms)
    
    def _test_ollama_generation(self, endpoint: ModelEndpoint) -> None:
        """Test Ollama text generation."""
        start_time = time.time()
        
        try:
            payload = {
                "model": endpoint.model_name,
                "prompt": "Say 'Hello, this is a test!'",
                "stream": False,
                "options": {
                    "temperature": endpoint.temperature,
                    "num_predict": 50
                }
            }
            
            response = requests.post(
                f"{endpoint.base_url}/api/generate",
                json=payload,
                timeout=endpoint.timeout
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("response", "")
                
                if content and "test" in content.lower():
                    self._add_result(endpoint, "generation", ValidationStatus.PASS,
                                   "Text generation successful",
                                   duration_ms=duration_ms,
                                   details={"generated_text": content})
                else:
                    self._add_result(endpoint, "generation", ValidationStatus.WARNING,
                                   "Generation completed but response unexpected",
                                   duration_ms=duration_ms,
                                   details={"generated_text": content})
            else:
                self._add_result(endpoint, "generation", ValidationStatus.FAIL,
                               f"HTTP {response.status_code} during generation",
                               duration_ms=duration_ms)
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._add_result(endpoint, "generation", ValidationStatus.FAIL,
                           f"Generation failed: {str(e)}",
                           duration_ms=duration_ms)
    
    def _validate_claude(self, endpoint: ModelEndpoint) -> None:
        """Validate Claude endpoint."""
        if anthropic is None:
            self._add_result(endpoint, "package_availability", ValidationStatus.FAIL,
                           "Anthropic package not available")
            return
        
        # Test 1: API Key validation
        if not endpoint.api_key:
            self._add_result(endpoint, "api_key_check", ValidationStatus.FAIL,
                           "No API key provided")
            return
        
        # Test 2: Basic connectivity and generation
        self._test_claude_generation(endpoint)
    
    def _test_claude_generation(self, endpoint: ModelEndpoint) -> None:
        """Test Claude text generation (includes connectivity test)."""
        start_time = time.time()
        
        try:
            client = anthropic.Anthropic(api_key=endpoint.api_key)
            
            response = client.messages.create(
                model=endpoint.model_name,
                max_tokens=50,
                temperature=endpoint.temperature,
                messages=[{"role": "user", "content": "Say 'Hello, this is a test!'"}]
            )
            
            duration_ms = (time.time() - start_time) * 1000
            content = response.content[0].text if response.content else ""
            
            if content and "test" in content.lower():
                self._add_result(endpoint, "generation", ValidationStatus.PASS,
                               "Text generation successful",
                               duration_ms=duration_ms,
                               details={
                                   "generated_text": content,
                                   "usage": response.usage.__dict__ if hasattr(response, "usage") else None
                               })
            else:
                self._add_result(endpoint, "generation", ValidationStatus.WARNING,
                               "Generation completed but response unexpected",
                               duration_ms=duration_ms,
                               details={"generated_text": content})
            
        except anthropic.AuthenticationError:
            duration_ms = (time.time() - start_time) * 1000
            self._add_result(endpoint, "generation", ValidationStatus.FAIL,
                           "Authentication failed - invalid API key",
                           duration_ms=duration_ms)
        except anthropic.APIError as e:
            duration_ms = (time.time() - start_time) * 1000
            self._add_result(endpoint, "generation", ValidationStatus.FAIL,
                           f"API error: {str(e)}",
                           duration_ms=duration_ms)
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._add_result(endpoint, "generation", ValidationStatus.FAIL,
                           f"Generation failed: {str(e)}",
                           duration_ms=duration_ms)
    
    def _validate_custom(self, endpoint: ModelEndpoint) -> None:
        """Validate custom endpoint."""
        if requests is None:
            self._add_result(endpoint, "package_availability", ValidationStatus.FAIL,
                           "Requests package not available for custom endpoint")
            return
        
        # Basic HTTP connectivity test
        self._test_custom_connectivity(endpoint)
    
    def _test_custom_connectivity(self, endpoint: ModelEndpoint) -> None:
        """Test custom endpoint connectivity."""
        start_time = time.time()
        
        try:
            headers = {}
            if endpoint.api_key:
                headers["Authorization"] = f"Bearer {endpoint.api_key}"
            
            response = requests.get(
                endpoint.base_url,
                headers=headers,
                timeout=endpoint.timeout
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code in [200, 404]:  # 404 is OK for base URL
                self._add_result(endpoint, "connectivity", ValidationStatus.PASS,
                               f"Successfully connected to custom endpoint",
                               duration_ms=duration_ms,
                               details={"status_code": response.status_code})
            else:
                self._add_result(endpoint, "connectivity", ValidationStatus.WARNING,
                               f"HTTP {response.status_code} from custom endpoint",
                               duration_ms=duration_ms)
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._add_result(endpoint, "connectivity", ValidationStatus.FAIL,
                           f"Connection failed: {str(e)}",
                           duration_ms=duration_ms)
    
    def _add_result(self, endpoint: ModelEndpoint, test_name: str, status: ValidationStatus,
                   message: str, duration_ms: Optional[float] = None,
                   details: Optional[Dict[str, Any]] = None) -> None:
        """Add a validation result."""
        result = ValidationResult(
            endpoint_name=endpoint.name,
            provider=endpoint.provider,
            test_name=test_name,
            status=status,
            message=message,
            duration_ms=duration_ms,
            details=details
        )
        self.results.append(result)
    
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
        
        # Calculate endpoint-specific summaries
        endpoint_summaries = {}
        for endpoint_name in self.endpoints.keys():
            endpoint_results = [r for r in self.results if r.endpoint_name == endpoint_name]
            endpoint_passed = len([r for r in endpoint_results if r.status == ValidationStatus.PASS])
            endpoint_failed = len([r for r in endpoint_results if r.status == ValidationStatus.FAIL])
            
            endpoint_status = "PASS" if endpoint_failed == 0 else "FAIL"
            endpoint_summaries[endpoint_name] = {
                "status": endpoint_status,
                "total_tests": len(endpoint_results),
                "passed": endpoint_passed,
                "failed": endpoint_failed
            }
        
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
            "endpoint_summaries": endpoint_summaries,
            "results": [
                {
                    "endpoint_name": r.endpoint_name,
                    "provider": r.provider.value,
                    "test_name": r.test_name,
                    "status": r.status.value,
                    "message": r.message,
                    "duration_ms": r.duration_ms,
                    "details": r.details
                }
                for r in self.results
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_results(self) -> List[ValidationResult]:
        """Get all validation results."""
        return self.results.copy()


def create_model_endpoints_from_config(config: Dict[str, Any]) -> List[ModelEndpoint]:
    """
    Create model endpoints from configuration dictionary.
    
    Args:
        config: Configuration dictionary with endpoint definitions
        
    Returns:
        List of model endpoints
    """
    endpoints = []
    
    for name, endpoint_config in config.items():
        provider_str = endpoint_config.get("provider", "custom")
        provider = ModelProvider(provider_str)
        
        endpoint = ModelEndpoint(
            provider=provider,
            name=name,
            model_name=endpoint_config["model_name"],
            api_key=endpoint_config.get("api_key"),
            base_url=endpoint_config.get("base_url"),
            api_version=endpoint_config.get("api_version"),
            timeout=endpoint_config.get("timeout", 30),
            max_tokens=endpoint_config.get("max_tokens", 1000),
            temperature=endpoint_config.get("temperature", 0.1),
            metadata=endpoint_config.get("metadata", {})
        )
        
        endpoints.append(endpoint)
    
    return endpoints


def validate_model_endpoints(endpoints: List[ModelEndpoint]) -> Dict[str, Any]:
    """
    Validate a list of model endpoints.
    
    Args:
        endpoints: List of model endpoints to validate
        
    Returns:
        Validation results dictionary
    """
    validator = ModelValidator(endpoints)
    return validator.validate_all()


def validate_from_env_config() -> Dict[str, Any]:
    """
    Validate model endpoints using environment variables.
    
    Expected environment variables:
    - OPENAI_API_KEY: OpenAI API key
    - GEMINI_API_KEY: Google Gemini API key
    - ANTHROPIC_API_KEY: Anthropic Claude API key
    - OLLAMA_BASE_URL: Ollama server URL (default: http://localhost:11434)
    
    Returns:
        Validation results dictionary
    """
    endpoints = []
    
    # OpenAI
    if os.getenv("OPENAI_API_KEY"):
        endpoints.append(ModelEndpoint(
            provider=ModelProvider.OPENAI,
            name="openai_gpt35",
            model_name="gpt-3.5-turbo",
            api_key=os.getenv("OPENAI_API_KEY")
        ))
    
    # Gemini
    if os.getenv("GEMINI_API_KEY"):
        endpoints.append(ModelEndpoint(
            provider=ModelProvider.GEMINI,
            name="gemini_pro",
            model_name="gemini-pro",
            api_key=os.getenv("GEMINI_API_KEY")
        ))
    
    # Claude
    if os.getenv("ANTHROPIC_API_KEY"):
        endpoints.append(ModelEndpoint(
            provider=ModelProvider.CLAUDE,
            name="claude_3",
            model_name="claude-3-sonnet-20240229",
            api_key=os.getenv("ANTHROPIC_API_KEY")
        ))
    
    # Ollama
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    endpoints.append(ModelEndpoint(
        provider=ModelProvider.OLLAMA,
        name="ollama_llama2",
        model_name="llama2",
        base_url=ollama_url
    ))
    
    if not endpoints:
        return {
            "overall_status": "SKIP",
            "summary": {"total_tests": 0, "message": "No API keys found in environment"},
            "results": [],
            "timestamp": datetime.utcnow().isoformat()
        }
    
    return validate_model_endpoints(endpoints)


def print_validation_report(validation_results: Dict[str, Any]) -> None:
    """
    Print a formatted validation report.
    
    Args:
        validation_results: Results from validate_model_endpoints()
    """
    print("\n" + "="*70)
    print("MODEL ENDPOINT VALIDATION REPORT")
    print("="*70)
    
    summary = validation_results["summary"]
    print(f"Overall Status: {validation_results['overall_status']}")
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Warnings: {summary['warnings']}")
    print(f"Skipped: {summary['skipped']}")
    print(f"Total Duration: {summary['total_duration_ms']:.1f}ms")
    
    print("\n" + "-"*70)
    print("ENDPOINT SUMMARIES")
    print("-"*70)
    
    for endpoint_name, summary in validation_results.get("endpoint_summaries", {}).items():
        status_symbol = "✅" if summary["status"] == "PASS" else "❌"
        print(f"{status_symbol} {endpoint_name}: {summary['passed']}/{summary['total_tests']} tests passed")
    
    print("\n" + "-"*70)
    print("DETAILED RESULTS")
    print("-"*70)
    
    current_endpoint = None
    for result in validation_results["results"]:
        if result["endpoint_name"] != current_endpoint:
            current_endpoint = result["endpoint_name"]
            print(f"\n📡 {current_endpoint} ({result['provider']})")
        
        status_symbol = {
            "pass": "  ✅",
            "fail": "  ❌",
            "warning": "  ⚠️",
            "skip": "  ⏭️"
        }.get(result["status"], "  ?")
        
        duration = f" ({result['duration_ms']:.1f}ms)" if result["duration_ms"] else ""
        print(f"{status_symbol} {result['test_name']}: {result['message']}{duration}")
    
    print("="*70) 