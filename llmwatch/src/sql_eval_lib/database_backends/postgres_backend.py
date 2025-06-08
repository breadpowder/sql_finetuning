"""
PostgreSQL database backend implementation
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List, Tuple, Union

from .base import DatabaseBackend, DatabaseConfig, QueryResult, QueryStatus, ConnectionError, QueryExecutionError, SchemaSetupError

# Try to import psycopg2, but make it optional
try:
    import psycopg2
    import psycopg2.extras
    from psycopg2 import sql, OperationalError, ProgrammingError
    PostgresConnection = psycopg2.extensions.connection
except ImportError:
    psycopg2 = None
    OperationalError = None
    ProgrammingError = None
    PostgresConnection = None

logger = logging.getLogger(__name__)


class PostgreSQLBackend(DatabaseBackend):
    """
    PostgreSQL database backend implementation
    
    Connects to a PostgreSQL database and executes queries.
    Requires the 'psycopg2' or 'psycopg2-binary' Python package to be installed.
    """
    
    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        
        if psycopg2 is None:
            raise ImportError("PostgreSQL client library not found. Please install 'psycopg2' or 'psycopg2-binary'.")
            
        self._connection: Optional[Union[PostgresConnection, Any]] = None
        
        # Extract PostgreSQL-specific connection parameters
        self._host = config.connection_params.get('host', 'localhost')
        self._port = config.connection_params.get('port', 5432)
        self._database = config.connection_params.get('database', 'postgres')
        self._username = config.connection_params.get('username', 'postgres')
        self._password = config.connection_params.get('password', '')
        self._sslmode = config.connection_params.get('sslmode', 'prefer')
        
    async def connect(self) -> None:
        """Establish connection to PostgreSQL database"""
        try:
            loop = asyncio.get_event_loop()
            self._connection = await loop.run_in_executor(
                None, self._create_postgres_connection
            )
            self._is_connected = True
            logger.info(f"Connected to PostgreSQL database '{self._database}' at {self._host}:{self._port}")
            
        except OperationalError as e:
            logger.error(f"PostgreSQL connection error: {e}")
            raise ConnectionError(f"Failed to connect to PostgreSQL: {e}")
        except Exception as e:
            logger.error(f"Unexpected error connecting to PostgreSQL: {e}")
            raise ConnectionError(f"Unexpected error connecting to PostgreSQL: {e}")
            
    def _create_postgres_connection(self) -> Union[PostgresConnection, Any]:
        """Create PostgreSQL connection synchronously"""
        return psycopg2.connect(
            host=self._host,
            port=self._port,
            database=self._database,
            user=self._username,
            password=self._password,
            sslmode=self._sslmode,
            connect_timeout=self.config.timeout
        )
    
    async def disconnect(self) -> None:
        """Close PostgreSQL database connection"""
        if self._connection:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._connection.close)
                self._connection = None
                self._is_connected = False
                logger.info("Disconnected from PostgreSQL server")
            except Exception as e:
                logger.error(f"Error disconnecting from PostgreSQL: {e}")
        
    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Execute SQL query against PostgreSQL server"""
        if not self._is_connected or not self._connection:
            raise ConnectionError("Not connected to PostgreSQL database")
        
        start_time = time.time()
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._execute_sync_postgres_query, query, params
            )
            
            execution_time = (time.time() - start_time) * 1000
            result.execution_time_ms = execution_time
            return result
            
        except (ProgrammingError, OperationalError) as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"PostgreSQL query error: {e}")
            # Try to rollback if in a transaction
            if self._connection and self._connection.status != psycopg2.extensions.STATUS_READY:
                 try: self._connection.rollback() 
                 except: pass
            return QueryResult(
                status=QueryStatus.ERROR,
                execution_time_ms=execution_time,
                error_message=str(e.diag.message_primary if hasattr(e, 'diag') and e.diag else e),
                error_code=str(e.pgcode if hasattr(e, 'pgcode') else None)
            )
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Unexpected error during PostgreSQL query execution: {e}")
            return QueryResult(
                status=QueryStatus.ERROR,
                execution_time_ms=execution_time,
                error_message=str(e)
            )
            
    def _execute_sync_postgres_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Execute PostgreSQL query synchronously"""
        with self._connection.cursor() as cursor:
            try:
                cursor.execute(query, params)
                
                if cursor.description:
                    rows = cursor.fetchall()
                    columns = [desc.name for desc in cursor.description]
                    # RealDictCursor returns list of dicts, convert to list of tuples
                    data = [tuple(row.values()) for row in rows]
                    row_count = len(data)
                else:
                    # For non-SELECT (INSERT, UPDATE, DELETE, etc.)
                    data = None
                    columns = None
                    row_count = cursor.rowcount
                
                self._connection.commit() # Commit after successful execution
                
                return QueryResult(
                    status=QueryStatus.SUCCESS,
                    data=data,
                    columns=columns,
                    row_count=row_count,
                    metadata={'pg_status_message': cursor.statusmessage}
                )
                
            except (ProgrammingError, OperationalError) as e:
                self._connection.rollback() # Rollback on error
                raise # Re-raise to be caught by async wrapper
    
    async def setup_database(self, schema_sql: str) -> QueryResult:
        """Set up database schema in PostgreSQL"""
        if not self._is_connected or not self._connection:
            raise ConnectionError("Not connected to PostgreSQL database")
        
        try:
            statements = self._split_sql_statements(schema_sql)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._setup_database_sync_postgres, statements
            )
            return result
            
        except Exception as e:
            logger.error(f"PostgreSQL database setup failed: {e}")
            raise SchemaSetupError(f"Failed to set up PostgreSQL database: {e}")
            
    def _setup_database_sync_postgres(self, statements: List[str]) -> QueryResult:
        """Execute schema setup statements synchronously in PostgreSQL"""
        executed_statements = 0
        with self._connection.cursor() as cursor:
            try:
                for statement in statements:
                    if statement.strip():
                        cursor.execute(statement)
                        executed_statements += 1
                self._connection.commit()
                return QueryResult(
                    status=QueryStatus.SUCCESS,
                    row_count=executed_statements,
                    metadata={'executed_statements': executed_statements}
                )
            except (ProgrammingError, OperationalError) as e:
                self._connection.rollback()
                return QueryResult(
                    status=QueryStatus.ERROR,
                    error_message=str(e.diag.message_primary if hasattr(e, 'diag') and e.diag else e),
                    error_code=str(e.pgcode if hasattr(e, 'pgcode') else None),
                    metadata={'executed_statements': executed_statements}
                )
    
    def _split_sql_statements(self, sql_script: str) -> List[str]:
        """Split SQL script into individual statements"""
        # psycopg2 can often handle multi-statement strings if separated by semicolons,
        # but executing one by one gives more control for error reporting.
        return [stmt.strip() for stmt in sql_script.split(';') if stmt.strip()]
    
    async def validate_connection(self) -> bool:
        """Validate PostgreSQL connection"""
        if not self._is_connected or not self._connection or self._connection.closed:
            return False
        
        try:
            # Check connection status attribute
            # STATUS_READY means connection is idle and ready
            return self._connection.status == psycopg2.extensions.STATUS_READY
        except Exception:
            return False
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get information about PostgreSQL connection"""
        info = {
            'backend_type': 'postgres',
            'host': self._host,
            'port': self._port,
            'database': self._database,
            'username': self._username, # Be cautious with exposing username
            'connected': self._is_connected,
            'readonly': self.config.readonly,
            'ssl_mode': self._sslmode
        }
        
        if self._connection and not self._connection.closed:
            try:
                info['pg_server_version'] = self._connection.server_version
                info['pg_protocol_version'] = self._connection.protocol_version
                # info['pg_dsn_parameters'] = self._connection.get_dsn_parameters() # Contains password!
                info['pg_status'] = self._connection.status
            except Exception as e:
                info['info_error'] = str(e)
        
        return info
    
    def __repr__(self) -> str:
        """String representation"""
        return f"PostgreSQLBackend(database={self._database}, host={self._host}, connected={self._is_connected})" 