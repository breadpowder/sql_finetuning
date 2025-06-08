"""
Factory for creating database backend instances
"""

from typing import Dict, Any, Optional, Type
import logging
from .base import DatabaseBackend, DatabaseConfig, DatabaseType, BackendNotFoundError
from .registry import DatabaseBackendRegistry

logger = logging.getLogger(__name__)


class DatabaseBackendFactory:
    """
    Factory for creating database backend instances
    
    Provides a centralized way to create database backends with proper configuration
    and validation. Supports both built-in and custom backends through the registry.
    """
    
    def __init__(self, registry: DatabaseBackendRegistry):
        self.registry = registry
    
    async def create_backend(self, 
                           backend_type: str,
                           connection_params: Optional[Dict[str, Any]] = None,
                           **config_kwargs) -> DatabaseBackend:
        """
        Create a database backend instance
        
        Args:
            backend_type: Type of backend to create (e.g., 'sqlite', 'trino')
            connection_params: Database-specific connection parameters
            **config_kwargs: Additional configuration parameters
            
        Returns:
            Configured database backend instance
            
        Raises:
            BackendNotFoundError: If backend type is not registered
            ValueError: If configuration is invalid
        """
        if not self.registry.has_backend(backend_type):
            available = self.registry.list_backends()
            raise BackendNotFoundError(
                f"Backend '{backend_type}' not found. Available backends: {available}"
            )
        
        # Get backend class from registry
        backend_class = self.registry.get_backend(backend_type)
        
        # Create configuration
        config = self._create_config(backend_type, connection_params or {}, **config_kwargs)
        
        # Validate configuration
        self._validate_config(config)
        
        # Create backend instance
        try:
            backend = backend_class(config)
            logger.info(f"Created {backend_type} backend instance")
            return backend
        except Exception as e:
            logger.error(f"Failed to create {backend_type} backend: {e}")
            raise ValueError(f"Failed to create {backend_type} backend: {e}")
    
    def _create_config(self, 
                      backend_type: str,
                      connection_params: Dict[str, Any],
                      **config_kwargs) -> DatabaseConfig:
        """
        Create a DatabaseConfig instance with validation
        
        Args:
            backend_type: Backend type string
            connection_params: Connection parameters
            **config_kwargs: Additional config parameters
            
        Returns:
            Validated DatabaseConfig instance
        """
        # Map string to enum
        try:
            db_type = DatabaseType(backend_type)
        except ValueError:
            # Allow custom backend types
            db_type = DatabaseType.CUSTOM
        
        # Merge config parameters
        config_data = {
            'backend_type': db_type,
            'connection_params': connection_params,
            **config_kwargs
        }
        
        return DatabaseConfig(**config_data)
    
    def _validate_config(self, config: DatabaseConfig) -> None:
        """
        Validate configuration for the specific backend type
        
        Args:
            config: Configuration to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        # The DatabaseConfig model handles most validation via Pydantic
        # Additional backend-specific validation can be added here
        
        backend_type = config.backend_type
        params = config.connection_params
        
        if backend_type == DatabaseType.SQLITE:
            # Validate SQLite-specific parameters
            if 'database' in params and 'path' in params:
                raise ValueError("SQLite config cannot have both 'database' and 'path' parameters")
        
        elif backend_type == DatabaseType.TRINO:
            # Validate Trino-specific parameters
            if 'port' in params:
                port = params['port']
                if not isinstance(port, int) or port <= 0 or port > 65535:
                    raise ValueError(f"Trino port must be between 1-65535, got {port}")
        
        elif backend_type == DatabaseType.POSTGRES:
            # Validate PostgreSQL-specific parameters
            if 'port' in params:
                port = params['port']
                if not isinstance(port, int) or port <= 0 or port > 65535:
                    raise ValueError(f"PostgreSQL port must be between 1-65535, got {port}")
        
        elif backend_type == DatabaseType.MYSQL:
            # Validate MySQL-specific parameters
            if 'port' in params:
                port = params['port']
                if not isinstance(port, int) or port <= 0 or port > 65535:
                    raise ValueError(f"MySQL port must be between 1-65535, got {port}")
    
    async def create_sqlite_backend(self, 
                                  database: str = ":memory:",
                                  **config_kwargs) -> DatabaseBackend:
        """
        Convenience method to create SQLite backend
        
        Args:
            database: Database path or ':memory:' for in-memory
            **config_kwargs: Additional configuration
            
        Returns:
            SQLite backend instance
        """
        return await self.create_backend(
            'sqlite',
            connection_params={'database': database},
            **config_kwargs
        )
    
    async def create_trino_backend(self,
                                 host: str,
                                 port: int = 8080,
                                 catalog: str = "default",
                                 schema: str = "default",
                                 username: Optional[str] = None,
                                 **config_kwargs) -> DatabaseBackend:
        """
        Convenience method to create Trino backend
        
        Args:
            host: Trino coordinator host
            port: Trino coordinator port
            catalog: Default catalog
            schema: Default schema
            username: Username for authentication
            **config_kwargs: Additional configuration
            
        Returns:
            Trino backend instance
        """
        connection_params = {
            'host': host,
            'port': port,
            'catalog': catalog,
            'schema': schema
        }
        
        if username:
            connection_params['username'] = username
        
        return await self.create_backend(
            'trino',
            connection_params=connection_params,
            **config_kwargs
        )
    
    async def create_postgres_backend(self,
                                    host: str,
                                    database: str,
                                    port: int = 5432,
                                    username: Optional[str] = None,
                                    password: Optional[str] = None,
                                    **config_kwargs) -> DatabaseBackend:
        """
        Convenience method to create PostgreSQL backend
        
        Args:
            host: PostgreSQL server host
            database: Database name
            port: Server port
            username: Username for authentication
            password: Password for authentication
            **config_kwargs: Additional configuration
            
        Returns:
            PostgreSQL backend instance
        """
        connection_params = {
            'host': host,
            'database': database,
            'port': port
        }
        
        if username:
            connection_params['username'] = username
        if password:
            connection_params['password'] = password
        
        return await self.create_backend(
            'postgres',
            connection_params=connection_params,
            **config_kwargs
        )
    
    async def create_mysql_backend(self,
                                 host: str,
                                 database: str,
                                 port: int = 3306,
                                 username: Optional[str] = None,
                                 password: Optional[str] = None,
                                 **config_kwargs) -> DatabaseBackend:
        """
        Convenience method to create MySQL backend
        
        Args:
            host: MySQL server host
            database: Database name
            port: Server port
            username: Username for authentication
            password: Password for authentication
            **config_kwargs: Additional configuration
            
        Returns:
            MySQL backend instance
        """
        connection_params = {
            'host': host,
            'database': database,
            'port': port
        }
        
        if username:
            connection_params['username'] = username
        if password:
            connection_params['password'] = password
        
        return await self.create_backend(
            'mysql',
            connection_params=connection_params,
            **config_kwargs
        )
    
    def get_available_backends(self) -> Dict[str, Dict]:
        """
        Get information about all available backends
        
        Returns:
            Dictionary mapping backend names to their metadata
        """
        return self.registry.get_all_backend_info()
    
    def validate_backend_config(self, backend_type: str, connection_params: Dict[str, Any]) -> bool:
        """
        Validate configuration without creating backend instance
        
        Args:
            backend_type: Backend type to validate for
            connection_params: Connection parameters to validate
            
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
            BackendNotFoundError: If backend type is not available
        """
        try:
            config = self._create_config(backend_type, connection_params)
            self._validate_config(config)
            return True
        except Exception:
            raise
    
    def __repr__(self) -> str:
        """String representation"""
        backends = len(self.registry)
        return f"DatabaseBackendFactory(registry_backends={backends})" 