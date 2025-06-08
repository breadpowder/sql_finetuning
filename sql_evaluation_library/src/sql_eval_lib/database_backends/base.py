"""
Abstract base class and data models for database backends
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import time

from pydantic import BaseModel, Field, validator


class DatabaseType(str, Enum):
    """Supported database types"""
    SQLITE = "sqlite"
    TRINO = "trino"
    POSTGRES = "postgres"
    MYSQL = "mysql"
    CUSTOM = "custom"


class QueryStatus(str, Enum):
    """Query execution status"""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class QueryResult:
    """Result of a SQL query execution"""
    status: QueryStatus
    data: Optional[List[Tuple[Any, ...]]] = None
    columns: Optional[List[str]] = None
    row_count: Optional[int] = None
    execution_time_ms: Optional[float] = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def success(self) -> bool:
        """Check if query executed successfully"""
        return self.status == QueryStatus.SUCCESS
    
    @property
    def failed(self) -> bool:
        """Check if query execution failed"""
        return self.status == QueryStatus.ERROR


class DatabaseConfig(BaseModel):
    """Configuration for database connections"""
    backend_type: DatabaseType = Field(description="Type of database backend")
    connection_params: Dict[str, Any] = Field(default_factory=dict, description="Database-specific connection parameters")
    timeout: int = Field(default=30, ge=1, le=300, description="Query timeout in seconds")
    pool_size: int = Field(default=1, ge=1, le=20, description="Connection pool size")
    ssl_enabled: bool = Field(default=False, description="Enable SSL/TLS connections")
    readonly: bool = Field(default=True, description="Open connections in read-only mode")
    
    @validator('connection_params')
    def validate_connection_params(cls, v, values):
        """Validate connection parameters based on backend type"""
        backend_type = values.get('backend_type')
        
        if backend_type == DatabaseType.SQLITE:
            # SQLite-specific validation
            if 'database' not in v and 'path' not in v:
                v['database'] = ':memory:'  # Default to in-memory
                
        elif backend_type == DatabaseType.TRINO:
            # Trino-specific validation
            required_params = ['host', 'port', 'catalog']
            for param in required_params:
                if param not in v:
                    raise ValueError(f"Trino backend requires '{param}' in connection_params")
                    
        elif backend_type == DatabaseType.POSTGRES:
            # PostgreSQL-specific validation
            required_params = ['host', 'database']
            for param in required_params:
                if param not in v:
                    raise ValueError(f"PostgreSQL backend requires '{param}' in connection_params")
                    
        elif backend_type == DatabaseType.MYSQL:
            # MySQL-specific validation  
            required_params = ['host', 'database']
            for param in required_params:
                if param not in v:
                    raise ValueError(f"MySQL backend requires '{param}' in connection_params")
        
        return v


class DatabaseBackend(ABC):
    """
    Abstract base class for database backends
    
    This defines the interface that all database backends must implement.
    Backends are responsible for:
    - Establishing and managing connections
    - Executing SQL queries
    - Setting up database schemas and data
    - Cleanup and resource management
    """
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._connection = None
        self._is_connected = False
        
    @property
    def backend_type(self) -> DatabaseType:
        """Get the backend type"""
        return self.config.backend_type
        
    @property
    def is_connected(self) -> bool:
        """Check if backend is connected"""
        return self._is_connected
    
    @abstractmethod
    async def connect(self) -> None:
        """
        Establish connection to the database
        
        Raises:
            ConnectionError: If connection cannot be established
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """
        Close database connection and cleanup resources
        """
        pass
    
    @abstractmethod
    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> QueryResult:
        """
        Execute a SQL query
        
        Args:
            query: SQL query string
            params: Optional query parameters for prepared statements
            
        Returns:
            QueryResult containing execution results
        """
        pass
    
    @abstractmethod
    async def setup_database(self, schema_sql: str) -> QueryResult:
        """
        Set up database schema and initial data
        
        Args:
            schema_sql: SQL statements to create tables and insert data
            
        Returns:
            QueryResult indicating setup success/failure
        """
        pass
    
    @abstractmethod
    async def validate_connection(self) -> bool:
        """
        Validate that the database connection is healthy
        
        Returns:
            True if connection is valid, False otherwise
        """
        pass
    
    @abstractmethod
    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get information about the current connection
        
        Returns:
            Dictionary with connection details (sanitized, no passwords)
        """
        pass
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
    
    def __repr__(self) -> str:
        """String representation"""
        return f"{self.__class__.__name__}(backend_type={self.backend_type}, connected={self.is_connected})"


class DatabaseBackendError(Exception):
    """Base exception for database backend errors"""
    pass


class ConnectionError(DatabaseBackendError):
    """Raised when database connection fails"""
    pass


class QueryExecutionError(DatabaseBackendError):
    """Raised when query execution fails"""
    pass


class SchemaSetupError(DatabaseBackendError):
    """Raised when database schema setup fails"""
    pass


class BackendNotFoundError(DatabaseBackendError):
    """Raised when requested backend is not found in registry"""
    pass 