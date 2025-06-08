"""
SQLite database backend implementation
"""

import sqlite3
import asyncio
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from .base import DatabaseBackend, DatabaseConfig, QueryResult, QueryStatus, ConnectionError, QueryExecutionError, SchemaSetupError

logger = logging.getLogger(__name__)


class SQLiteBackend(DatabaseBackend):
    """
    SQLite database backend implementation
    
    Supports both in-memory (:memory:) and file-based SQLite databases.
    Provides SQL execution with proper error handling and result formatting.
    """
    
    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        self._connection: Optional[sqlite3.Connection] = None
        self._database_path = config.connection_params.get('database', ':memory:')
        
    async def connect(self) -> None:
        """Establish connection to SQLite database"""
        try:
            # SQLite operations are synchronous, so we run in thread pool
            loop = asyncio.get_event_loop()
            self._connection = await loop.run_in_executor(
                None, self._create_connection
            )
            self._is_connected = True
            logger.info(f"Connected to SQLite database: {self._database_path}")
            
        except Exception as e:
            logger.error(f"Failed to connect to SQLite database: {e}")
            raise ConnectionError(f"Failed to connect to SQLite: {e}")
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create SQLite connection with proper configuration"""
        # Set connection parameters
        timeout = self.config.timeout
        readonly = self.config.readonly
        
        # Create connection
        conn = sqlite3.connect(
            self._database_path,
            timeout=timeout,
            check_same_thread=False  # Allow use from different threads
        )
        
        # Configure connection
        conn.row_factory = sqlite3.Row  # Enable column name access
        
        if readonly:
            # Set read-only mode (SQLite doesn't have true read-only, 
            # but we can prevent modifications)
            conn.execute("PRAGMA query_only = ON")
        
        # Set reasonable defaults
        conn.execute("PRAGMA foreign_keys = ON")  # Enable FK constraints
        conn.execute("PRAGMA journal_mode = WAL")  # Enable WAL mode for better concurrency
        
        return conn
    
    async def disconnect(self) -> None:
        """Close SQLite database connection"""
        if self._connection:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._connection.close)
                self._connection = None
                self._is_connected = False
                logger.info("Disconnected from SQLite database")
            except Exception as e:
                logger.error(f"Error disconnecting from SQLite: {e}")
        
    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Execute SQL query against SQLite database"""
        if not self._is_connected or not self._connection:
            raise ConnectionError("Not connected to database")
        
        start_time = time.time()
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._execute_sync_query, query, params
            )
            
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            result.execution_time_ms = execution_time
            
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Query execution failed: {e}")
            
            return QueryResult(
                status=QueryStatus.ERROR,
                execution_time_ms=execution_time,
                error_message=str(e),
                error_code=getattr(e, 'sqlite_errorcode', None)
            )
    
    def _execute_sync_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Execute query synchronously (runs in thread pool)"""
        cursor = self._connection.cursor()
        
        try:
            # Execute query with or without parameters
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # Fetch results for SELECT queries
            if query.strip().upper().startswith(('SELECT', 'WITH', 'EXPLAIN')):
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                data = [tuple(row) for row in rows]
                row_count = len(data)
            else:
                # For non-SELECT queries (INSERT, UPDATE, DELETE, etc.)
                data = None
                columns = None
                row_count = cursor.rowcount
            
            return QueryResult(
                status=QueryStatus.SUCCESS,
                data=data,
                columns=columns,
                row_count=row_count,
                metadata={'sqlite_changes': self._connection.total_changes}
            )
            
        except sqlite3.Error as e:
            error_code = getattr(e, 'sqlite_errorcode', None)
            return QueryResult(
                status=QueryStatus.ERROR,
                error_message=str(e),
                error_code=str(error_code) if error_code else None
            )
        finally:
            cursor.close()
    
    async def setup_database(self, schema_sql: str) -> QueryResult:
        """Set up database schema and initial data"""
        if not self._is_connected or not self._connection:
            raise ConnectionError("Not connected to database")
        
        try:
            # Split schema SQL into individual statements
            statements = self._split_sql_statements(schema_sql)
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._setup_database_sync, statements
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            raise SchemaSetupError(f"Failed to set up database schema: {e}")
    
    def _setup_database_sync(self, statements: List[str]) -> QueryResult:
        """Set up database synchronously"""
        cursor = self._connection.cursor()
        executed_statements = 0
        
        try:
            # Begin transaction
            cursor.execute("BEGIN TRANSACTION")
            
            for statement in statements:
                if statement.strip():
                    cursor.execute(statement)
                    executed_statements += 1
            
            # Commit transaction
            cursor.execute("COMMIT")
            
            return QueryResult(
                status=QueryStatus.SUCCESS,
                row_count=executed_statements,
                metadata={'executed_statements': executed_statements}
            )
            
        except sqlite3.Error as e:
            # Rollback on error
            try:
                cursor.execute("ROLLBACK")
            except:
                pass
                
            return QueryResult(
                status=QueryStatus.ERROR,
                error_message=str(e),
                error_code=getattr(e, 'sqlite_errorcode', None),
                metadata={'executed_statements': executed_statements}
            )
        finally:
            cursor.close()
    
    def _split_sql_statements(self, sql_script: str) -> List[str]:
        """Split SQL script into individual statements"""
        # Simple splitting by semicolon - can be enhanced for complex cases
        statements = []
        current_statement = ""
        
        for line in sql_script.split('\n'):
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('--') or line.startswith('#'):
                continue
            
            current_statement += line + " "
            
            # Check if statement is complete (ends with semicolon)
            if line.endswith(';'):
                statements.append(current_statement.strip())
                current_statement = ""
        
        # Add any remaining statement
        if current_statement.strip():
            statements.append(current_statement.strip())
        
        return statements
    
    async def validate_connection(self) -> bool:
        """Validate that database connection is healthy"""
        if not self._is_connected or not self._connection:
            return False
        
        try:
            result = await self.execute_query("SELECT 1")
            return result.success
        except Exception:
            return False
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get information about current connection"""
        info = {
            'backend_type': 'sqlite',
            'database_path': self._database_path,
            'is_memory': self._database_path == ':memory:',
            'connected': self._is_connected,
            'readonly': self.config.readonly
        }
        
        if self._connection:
            try:
                # Get SQLite version and other info
                cursor = self._connection.cursor()
                cursor.execute("SELECT sqlite_version()")
                version = cursor.fetchone()[0]
                info['sqlite_version'] = version
                
                # Get database size for file-based databases
                if self._database_path != ':memory:':
                    path = Path(self._database_path)
                    if path.exists():
                        info['database_size_bytes'] = path.stat().st_size
                
                cursor.close()
                
            except Exception as e:
                info['info_error'] = str(e)
        
        return info
    
    def __repr__(self) -> str:
        """String representation"""
        return f"SQLiteBackend(database={self._database_path}, connected={self._is_connected})" 