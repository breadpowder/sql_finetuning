"""
Trino database backend implementation
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List, Tuple, Union

from .base import DatabaseBackend, DatabaseConfig, QueryResult, QueryStatus, ConnectionError, QueryExecutionError, SchemaSetupError

# Try to import trino, but make it optional
try:
    import trino
    from trino.exceptions import TrinoUserError, TrinoConnectionError, TrinoInternalError
    TrinoConnection = trino.dbapi.Connection
except ImportError:
    trino = None # Placeholder if trino is not installed
    TrinoUserError = None
    TrinoConnectionError = None
    TrinoInternalError = None
    TrinoConnection = None

logger = logging.getLogger(__name__)


class TrinoBackend(DatabaseBackend):
    """
    Trino database backend implementation
    
    Connects to a Trino cluster and executes queries.
    Requires the 'trino' Python package to be installed.
    """
    
    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        
        if trino is None:
            raise ImportError("Trino client library not found. Please install 'trino' or 'sql-eval-lib[database]'.")
            
        self._connection: Optional[Union[TrinoConnection, Any]] = None
        
        # Extract Trino-specific connection parameters
        self._host = config.connection_params.get('host', 'localhost')
        self._port = config.connection_params.get('port', 8080)
        self._username = config.connection_params.get('username')
        self._catalog = config.connection_params.get('catalog', 'default')
        self._schema = config.connection_params.get('schema', 'default')
        self._http_scheme = config.connection_params.get('http_scheme', 'http')
        self._source = config.connection_params.get('source', 'sql-eval-lib')
        
    async def connect(self) -> None:
        """Establish connection to Trino cluster"""
        try:
            loop = asyncio.get_event_loop()
            self._connection = await loop.run_in_executor(
                None, self._create_trino_connection
            )
            self._is_connected = True
            logger.info(f"Connected to Trino cluster at {self._host}:{self._port}")
            
        except TrinoConnectionError as e:
            logger.error(f"Trino connection error: {e}")
            raise ConnectionError(f"Failed to connect to Trino: {e}")
        except Exception as e:
            logger.error(f"Unexpected error connecting to Trino: {e}")
            raise ConnectionError(f"Unexpected error connecting to Trino: {e}")
            
    def _create_trino_connection(self) -> Union[TrinoConnection, Any]:
        """Create Trino connection synchronously"""
        return trino.dbapi.connect(
            host=self._host,
            port=self._port,
            user=self._username,
            catalog=self._catalog,
            schema=self._schema,
            http_scheme=self._http_scheme,
            source=self._source,
            # Add other parameters like auth, session_properties, etc. as needed
            # request_timeout=self.config.timeout # Trino client handles timeout differently
        )
    
    async def disconnect(self) -> None:
        """Close Trino database connection"""
        if self._connection:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._connection.close)
                self._connection = None
                self._is_connected = False
                logger.info("Disconnected from Trino cluster")
            except Exception as e:
                logger.error(f"Error disconnecting from Trino: {e}")
        
    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Execute SQL query against Trino cluster"""
        if not self._is_connected or not self._connection:
            raise ConnectionError("Not connected to Trino database")
        
        start_time = time.time()
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._execute_sync_trino_query, query, params
            )
            
            execution_time = (time.time() - start_time) * 1000
            result.execution_time_ms = execution_time
            return result
            
        except TrinoUserError as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Trino query error: {e}")
            return QueryResult(
                status=QueryStatus.ERROR,
                execution_time_ms=execution_time,
                error_message=str(e),
                error_code=str(getattr(e, 'error_code', None))
            )
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Unexpected error during Trino query execution: {e}")
            return QueryResult(
                status=QueryStatus.ERROR,
                execution_time_ms=execution_time,
                error_message=str(e)
            )
            
    def _execute_sync_trino_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Execute Trino query synchronously"""
        cursor = self._connection.cursor()
        
        try:
            # Trino client uses 'parameters' for prepared statements
            cursor.execute(query, parameters=params)
            
            # Fetch results
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            data = [tuple(row) for row in rows]
            row_count = len(data)
            
            return QueryResult(
                status=QueryStatus.SUCCESS,
                data=data,
                columns=columns,
                row_count=row_count,
                metadata={
                    'query_id': cursor.query_id,
                    'warnings': cursor.warnings
                }
            )
            
        finally:
            cursor.close()
    
    async def setup_database(self, schema_sql: str) -> QueryResult:
        """Set up database schema in Trino (executes provided SQL)"""
        # Trino setup often involves external tools or specific catalog configurations.
        # This method executes the provided SQL, which might include CREATE TABLE AS, etc.
        
        if not self._is_connected or not self._connection:
            raise ConnectionError("Not connected to Trino database")
        
        try:
            statements = self._split_sql_statements(schema_sql)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._setup_database_sync_trino, statements
            )
            return result
            
        except Exception as e:
            logger.error(f"Trino database setup failed: {e}")
            raise SchemaSetupError(f"Failed to set up Trino database: {e}")
            
    def _setup_database_sync_trino(self, statements: List[str]) -> QueryResult:
        """Execute schema setup statements synchronously in Trino"""
        cursor = self._connection.cursor()
        executed_statements = 0
        errors = []
        
        try:
            for statement in statements:
                if statement.strip():
                    try:
                        cursor.execute(statement)
                        executed_statements += 1
                    except TrinoUserError as e:
                        logger.warning(f"Error executing Trino setup statement: {statement[:100]}... Error: {e}")
                        errors.append(str(e))
            
            if errors:
                return QueryResult(
                    status=QueryStatus.ERROR,
                    error_message="One or more schema setup statements failed: " + "; ".join(errors),
                    metadata={'executed_statements': executed_statements, 'errors': errors}
                )
                
            return QueryResult(
                status=QueryStatus.SUCCESS,
                row_count=executed_statements,
                metadata={'executed_statements': executed_statements}
            )
            
        finally:
            cursor.close()
    
    def _split_sql_statements(self, sql_script: str) -> List[str]:
        """Split SQL script into individual statements (Trino specific if needed)"""
        # Trino handles multi-statement queries differently in some clients.
        # For dbapi, individual execution is safer.
        # This basic splitter can be improved for complex cases.
        return [stmt.strip() for stmt in sql_script.split(';') if stmt.strip()]
    
    async def validate_connection(self) -> bool:
        """Validate Trino connection by executing a simple query"""
        if not self._is_connected or not self._connection:
            return False
        
        try:
            result = await self.execute_query("SELECT 1")
            return result.success
        except Exception:
            return False
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get information about Trino connection"""
        info = {
            'backend_type': 'trino',
            'host': self._host,
            'port': self._port,
            'username': self._username,
            'catalog': self._catalog,
            'schema': self._schema,
            'connected': self._is_connected,
        }
        
        if self._connection:
            try:
                # Access Trino-specific connection attributes if available
                # (e.g., server version, cluster info)
                # This might require additional API calls not directly supported by dbapi
                pass
            except Exception as e:
                info['info_error'] = str(e)
        
        return info
    
    def __repr__(self) -> str:
        """String representation"""
        return f"TrinoBackend(host={self._host}, port={self._port}, connected={self._is_connected})" 