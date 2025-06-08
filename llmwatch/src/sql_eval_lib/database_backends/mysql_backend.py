"""
MySQL database backend implementation
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List, Tuple, Union

from .base import DatabaseBackend, DatabaseConfig, QueryResult, QueryStatus, ConnectionError, QueryExecutionError, SchemaSetupError

# Try to import pymysql, but make it optional
try:
    import pymysql
    from pymysql import OperationalError, ProgrammingError, InterfaceError
    MySQLConnection = pymysql.connections.Connection
except ImportError:
    pymysql = None
    OperationalError = None
    ProgrammingError = None
    InterfaceError = None
    MySQLConnection = None

logger = logging.getLogger(__name__)


class MySQLBackend(DatabaseBackend):
    """
    MySQL database backend implementation
    
    Connects to a MySQL database and executes queries.
    Requires the 'pymysql' Python package to be installed.
    """
    
    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        
        if pymysql is None:
            raise ImportError("MySQL client library not found. Please install 'pymysql'.")
            
        self._connection: Optional[Union[MySQLConnection, Any]] = None
        
        # Extract MySQL-specific connection parameters
        self._host = config.connection_params.get('host', 'localhost')
        self._port = config.connection_params.get('port', 3306)
        self._database = config.connection_params.get('database', 'mysql')
        self._username = config.connection_params.get('username', 'root')
        self._password = config.connection_params.get('password', '')
        self._charset = config.connection_params.get('charset', 'utf8mb4')
        
    async def connect(self) -> None:
        """Establish connection to MySQL database"""
        try:
            loop = asyncio.get_event_loop()
            self._connection = await loop.run_in_executor(
                None, self._create_mysql_connection
            )
            self._is_connected = True
            logger.info(f"Connected to MySQL database '{self._database}' at {self._host}:{self._port}")
            
        except OperationalError as e:
            logger.error(f"MySQL connection error: {e}")
            raise ConnectionError(f"Failed to connect to MySQL: {e}")
        except Exception as e:
            logger.error(f"Unexpected error connecting to MySQL: {e}")
            raise ConnectionError(f"Unexpected error connecting to MySQL: {e}")
            
    def _create_mysql_connection(self) -> Union[MySQLConnection, Any]:
        """Create MySQL connection synchronously"""
        return pymysql.connect(
            host=self._host,
            port=self._port,
            database=self._database,
            user=self._username,
            password=self._password,
            charset=self._charset,
            connect_timeout=self.config.timeout,
            autocommit=False
        )
    
    async def disconnect(self) -> None:
        """Close MySQL database connection"""
        if self._connection:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._connection.close)
                self._connection = None
                self._is_connected = False
                logger.info("Disconnected from MySQL server")
            except Exception as e:
                logger.error(f"Error disconnecting from MySQL: {e}")
        
    async def execute_query(self, query: str, params: Optional[Union[Dict[str, Any], Tuple]] = None) -> QueryResult:
        """Execute SQL query against MySQL server"""
        if not self._is_connected or not self._connection:
            raise ConnectionError("Not connected to MySQL database")
        
        start_time = time.time()
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._execute_sync_mysql_query, query, params
            )
            
            execution_time = (time.time() - start_time) * 1000
            result.execution_time_ms = execution_time
            return result
            
        except (ProgrammingError, OperationalError, InterfaceError) as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"MySQL query error: {e}")
            if self._connection: 
                try: self._connection.rollback() 
                except: pass # Ignore rollback errors if connection is already bad
            return QueryResult(
                status=QueryStatus.ERROR,
                execution_time_ms=execution_time,
                error_message=str(e.args[1] if len(e.args) > 1 else e), # PyMySQL error messages often in args[1]
                error_code=str(e.args[0] if len(e.args) > 0 else None) # PyMySQL error codes often in args[0]
            )
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Unexpected error during MySQL query execution: {e}")
            return QueryResult(
                status=QueryStatus.ERROR,
                execution_time_ms=execution_time,
                error_message=str(e)
            )
            
    def _execute_sync_mysql_query(self, query: str, params: Optional[Union[Dict[str, Any], Tuple]] = None) -> QueryResult:
        """Execute MySQL query synchronously"""
        with self._connection.cursor() as cursor:
            try:
                cursor.execute(query, args=params)
                
                if cursor.description:
                    rows = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
                    # DictCursor returns list of dicts, convert to list of tuples
                    data = [tuple(row.values()) for row in rows]
                    row_count = len(data)
                else:
                    data = None
                    columns = None
                    row_count = cursor.rowcount
                
                self._connection.commit()
                
                return QueryResult(
                    status=QueryStatus.SUCCESS,
                    data=data,
                    columns=columns,
                    row_count=row_count,
                    metadata={
                        'mysql_affected_rows': cursor.rowcount,
                        'mysql_last_row_id': cursor.lastrowid,
                        'mysql_warnings_count': cursor.warning_count
                    }
                )
                
            except (ProgrammingError, OperationalError, InterfaceError) as e:
                self._connection.rollback()
                raise # Re-raise to be caught by async wrapper
    
    async def setup_database(self, schema_sql: str) -> QueryResult:
        """Set up database schema in MySQL"""
        if not self._is_connected or not self._connection:
            raise ConnectionError("Not connected to MySQL database")
        
        try:
            statements = self._split_sql_statements(schema_sql)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._setup_database_sync_mysql, statements
            )
            return result
            
        except Exception as e:
            logger.error(f"MySQL database setup failed: {e}")
            raise SchemaSetupError(f"Failed to set up MySQL database: {e}")
            
    def _setup_database_sync_mysql(self, statements: List[str]) -> QueryResult:
        """Execute schema setup statements synchronously in MySQL"""
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
            except (ProgrammingError, OperationalError, InterfaceError) as e:
                self._connection.rollback()
                return QueryResult(
                    status=QueryStatus.ERROR,
                    error_message=str(e.args[1] if len(e.args) > 1 else e),
                    error_code=str(e.args[0] if len(e.args) > 0 else None),
                    metadata={'executed_statements': executed_statements}
                )
    
    def _split_sql_statements(self, sql_script: str) -> List[str]:
        """Split SQL script into individual statements"""
        # PyMySQL can handle multi-statement execution with `multi=True` in cursor,
        # but for consistency and error reporting, we split and execute individually.
        return [stmt.strip() for stmt in sql_script.split(';') if stmt.strip()]
    
    async def validate_connection(self) -> bool:
        """Validate MySQL connection by pinging the server"""
        if not self._is_connected or not self._connection:
            return False
        
        try:
            # PyMySQL's ping(reconnect=False) checks connection liveness
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._connection.ping, False)
            return True
        except (OperationalError, InterfaceError):
            self._is_connected = False # Mark as disconnected on ping failure
            return False
        except Exception:
            return False # Other unexpected errors
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get information about MySQL connection"""
        info = {
            'backend_type': 'mysql',
            'host': self._host,
            'port': self._port,
            'database': self._database,
            'username': self._username,
            'connected': self._is_connected,
            'readonly': self.config.readonly,
            'charset': self._charset
        }
        
        if self._connection and self._connection.open:
            try:
                info['mysql_server_version'] = self._connection.get_server_info()
                info['mysql_connection_id'] = self._connection.thread_id()
            except Exception as e:
                info['info_error'] = str(e)
        
        return info
    
    def __repr__(self) -> str:
        """String representation"""
        return f"MySQLBackend(database={self._database}, host={self._host}, connected={self._is_connected})" 