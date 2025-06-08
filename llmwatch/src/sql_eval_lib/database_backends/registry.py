"""
Registry for database backend discovery and management
"""

from typing import Dict, Type, List, Optional
import logging
from .base import DatabaseBackend, DatabaseType, BackendNotFoundError

logger = logging.getLogger(__name__)


class DatabaseBackendRegistry:
    """
    Registry for database backend plugins
    
    Manages registration and discovery of database backend implementations.
    Supports both built-in backends and custom user-defined backends.
    """
    
    def __init__(self):
        self._backends: Dict[str, Type[DatabaseBackend]] = {}
        self._metadata: Dict[str, Dict] = {}
    
    def register(self, 
                name: str, 
                backend_class: Type[DatabaseBackend],
                description: Optional[str] = None,
                version: Optional[str] = None,
                author: Optional[str] = None) -> None:
        """
        Register a database backend
        
        Args:
            name: Unique name for the backend (e.g., 'sqlite', 'trino')
            backend_class: Backend implementation class
            description: Human-readable description
            version: Backend version
            author: Backend author/maintainer
            
        Raises:
            ValueError: If backend name already exists or class is invalid
        """
        if not name or not isinstance(name, str):
            raise ValueError("Backend name must be a non-empty string")
            
        if name in self._backends:
            logger.warning(f"Overriding existing backend '{name}'")
            
        if not issubclass(backend_class, DatabaseBackend):
            raise ValueError(f"Backend class must inherit from DatabaseBackend, got {backend_class}")
        
        self._backends[name] = backend_class
        self._metadata[name] = {
            'description': description or f"{backend_class.__name__} database backend",
            'version': version or '1.0.0',
            'author': author or 'Unknown',
            'class_name': backend_class.__name__,
            'module': backend_class.__module__
        }
        
        logger.info(f"Registered database backend: {name} ({backend_class.__name__})")
    
    def unregister(self, name: str) -> None:
        """
        Unregister a database backend
        
        Args:
            name: Backend name to remove
            
        Raises:
            BackendNotFoundError: If backend is not registered
        """
        if name not in self._backends:
            raise BackendNotFoundError(f"Backend '{name}' not found in registry")
            
        del self._backends[name]
        del self._metadata[name]
        logger.info(f"Unregistered database backend: {name}")
    
    def get_backend(self, name: str) -> Type[DatabaseBackend]:
        """
        Get a registered backend class
        
        Args:
            name: Backend name
            
        Returns:
            Backend class
            
        Raises:
            BackendNotFoundError: If backend is not registered
        """
        if name not in self._backends:
            available = list(self._backends.keys())
            raise BackendNotFoundError(
                f"Backend '{name}' not found. Available backends: {available}"
            )
            
        return self._backends[name]
    
    def has_backend(self, name: str) -> bool:
        """
        Check if a backend is registered
        
        Args:
            name: Backend name
            
        Returns:
            True if backend exists, False otherwise
        """
        return name in self._backends
    
    def list_backends(self) -> List[str]:
        """
        Get list of all registered backend names
        
        Returns:
            List of backend names
        """
        return list(self._backends.keys())
    
    def get_backend_info(self, name: str) -> Dict:
        """
        Get metadata about a registered backend
        
        Args:
            name: Backend name
            
        Returns:
            Backend metadata dictionary
            
        Raises:
            BackendNotFoundError: If backend is not registered
        """
        if name not in self._metadata:
            raise BackendNotFoundError(f"Backend '{name}' not found in registry")
            
        return self._metadata[name].copy()
    
    def get_all_backend_info(self) -> Dict[str, Dict]:
        """
        Get metadata for all registered backends
        
        Returns:
            Dictionary mapping backend names to metadata
        """
        return {name: info.copy() for name, info in self._metadata.items()}
    
    def discover_backends(self, package_name: str = "sql_eval_lib.database_backends") -> None:
        """
        Automatically discover and register backends from a package
        
        Args:
            package_name: Package to scan for backend implementations
        """
        try:
            import importlib
            import pkgutil
            
            package = importlib.import_module(package_name)
            
            for _, module_name, _ in pkgutil.iter_modules(package.__path__, package_name + "."):
                try:
                    module = importlib.import_module(module_name)
                    
                    # Look for classes that inherit from DatabaseBackend
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        
                        if (isinstance(attr, type) and 
                            issubclass(attr, DatabaseBackend) and 
                            attr != DatabaseBackend):
                            
                            # Auto-register with module name
                            backend_name = module_name.split('.')[-1].replace('_backend', '')
                            if not self.has_backend(backend_name):
                                self.register(
                                    backend_name,
                                    attr,
                                    description=f"Auto-discovered {attr.__name__}",
                                    version="auto"
                                )
                                
                except Exception as e:
                    logger.warning(f"Failed to discover backends in {module_name}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to discover backends from {package_name}: {e}")
    
    def validate_registry(self) -> Dict[str, List[str]]:
        """
        Validate all registered backends
        
        Returns:
            Dictionary with 'valid' and 'invalid' lists
        """
        valid = []
        invalid = []
        
        for name, backend_class in self._backends.items():
            try:
                # Basic validation checks
                if not issubclass(backend_class, DatabaseBackend):
                    invalid.append(f"{name}: Not a DatabaseBackend subclass")
                    continue
                    
                # Check required methods exist
                required_methods = [
                    'connect', 'disconnect', 'execute_query', 
                    'setup_database', 'validate_connection', 'get_connection_info'
                ]
                
                for method in required_methods:
                    if not hasattr(backend_class, method):
                        invalid.append(f"{name}: Missing required method '{method}'")
                        break
                else:
                    valid.append(name)
                    
            except Exception as e:
                invalid.append(f"{name}: Validation error - {e}")
        
        return {'valid': valid, 'invalid': invalid}
    
    def clear(self) -> None:
        """Clear all registered backends"""
        self._backends.clear()
        self._metadata.clear()
        logger.info("Cleared all registered backends")
    
    def __len__(self) -> int:
        """Get number of registered backends"""
        return len(self._backends)
    
    def __contains__(self, name: str) -> bool:
        """Check if backend is registered (supports 'in' operator)"""
        return self.has_backend(name)
    
    def __iter__(self):
        """Iterate over backend names"""
        return iter(self._backends.keys())
    
    def __repr__(self) -> str:
        """String representation"""
        return f"DatabaseBackendRegistry(backends={len(self._backends)})" 