"""
Langfuse project and API key management utilities.

This module provides tools for managing Langfuse projects, API keys,
and project-specific configurations for the SQL evaluation library.
"""

import os
import json
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

try:
    from langfuse import Langfuse
except ImportError:
    print("Warning: langfuse package not installed. Project management will not work.")
    Langfuse = None

try:
    import requests
except ImportError:
    print("Warning: requests package not installed. HTTP operations will not work.")
    requests = None


@dataclass
class ProjectConfig:
    """Configuration for a Langfuse project."""
    name: str
    description: str = ""
    public_key: Optional[str] = None
    secret_key: Optional[str] = None
    host: str = "localhost"
    port: int = 3000
    base_url: Optional[str] = None
    created_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set derived configuration values."""
        if self.base_url is None:
            self.base_url = f"http://{self.host}:{self.port}"
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class ApiKeyPair:
    """A Langfuse API key pair."""
    public_key: str
    secret_key: str
    project_name: str
    created_at: datetime
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "public_key": self.public_key,
            "secret_key": self.secret_key,
            "project_name": self.project_name,
            "created_at": self.created_at.isoformat(),
            "description": self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApiKeyPair":
        """Create from dictionary."""
        return cls(
            public_key=data["public_key"],
            secret_key=data["secret_key"],
            project_name=data["project_name"],
            created_at=datetime.fromisoformat(data["created_at"]),
            description=data.get("description", "")
        )


class LangfuseProjectManager:
    """
    Manages Langfuse projects and API keys.
    
    This class provides tools for creating, configuring, and managing
    Langfuse projects and their associated API keys.
    """
    
    def __init__(self, 
                 base_url: str = "http://localhost:3000",
                 config_file: Optional[Path] = None):
        """
        Initialize the project manager.
        
        Args:
            base_url: Base URL of Langfuse instance
            config_file: Optional path to configuration file
        """
        self.base_url = base_url
        self.config_file = config_file or Path("langfuse_projects.json")
        self.projects: Dict[str, ProjectConfig] = {}
        
        # Load existing configuration
        self._load_configuration()
    
    def create_project(self, 
                      name: str,
                      description: str = "",
                      tags: Optional[List[str]] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> ProjectConfig:
        """
        Create a new Langfuse project configuration.
        
        Note: This creates a local project configuration. Actual project
        creation in Langfuse requires manual setup or admin API access.
        
        Args:
            name: Project name
            description: Project description
            tags: Optional tags for the project
            metadata: Optional metadata for the project
            
        Returns:
            Created project configuration
        """
        if name in self.projects:
            raise ValueError(f"Project '{name}' already exists")
        
        project = ProjectConfig(
            name=name,
            description=description,
            base_url=self.base_url,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        self.projects[name] = project
        self._save_configuration()
        
        print(f"Created project configuration: {name}")
        return project
    
    def add_api_keys(self, 
                    project_name: str,
                    public_key: str,
                    secret_key: str) -> bool:
        """
        Add API keys to an existing project.
        
        Args:
            project_name: Name of the project
            public_key: Public API key
            secret_key: Secret API key
            
        Returns:
            True if keys were added successfully, False otherwise
        """
        if project_name not in self.projects:
            raise ValueError(f"Project '{project_name}' not found")
        
        project = self.projects[project_name]
        project.public_key = public_key
        project.secret_key = secret_key
        
        self._save_configuration()
        
        # Validate the keys
        if self._validate_api_keys(project):
            print(f"API keys added and validated for project: {project_name}")
            return True
        else:
            print(f"Warning: API keys added but validation failed for project: {project_name}")
            return False
    
    def get_project(self, name: str) -> Optional[ProjectConfig]:
        """
        Get a project configuration by name.
        
        Args:
            name: Project name
            
        Returns:
            Project configuration or None if not found
        """
        return self.projects.get(name)
    
    def list_projects(self) -> List[str]:
        """
        List all project names.
        
        Returns:
            List of project names
        """
        return list(self.projects.keys())
    
    def get_project_details(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed project information.
        
        Args:
            name: Project name
            
        Returns:
            Dictionary with project details or None if not found
        """
        project = self.projects.get(name)
        if project is None:
            return None
        
        has_keys = bool(project.public_key and project.secret_key)
        keys_valid = False
        
        if has_keys:
            keys_valid = self._validate_api_keys(project)
        
        return {
            "name": project.name,
            "description": project.description,
            "base_url": project.base_url,
            "created_at": project.created_at.isoformat(),
            "tags": project.tags,
            "metadata": project.metadata,
            "has_api_keys": has_keys,
            "api_keys_valid": keys_valid,
            "public_key": project.public_key[:8] + "..." if project.public_key else None
        }
    
    def update_project(self,
                      name: str,
                      description: Optional[str] = None,
                      tags: Optional[List[str]] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update project configuration.
        
        Args:
            name: Project name
            description: New description (optional)
            tags: New tags (optional)
            metadata: New metadata (optional)
            
        Returns:
            True if project was updated, False if not found
        """
        if name not in self.projects:
            return False
        
        project = self.projects[name]
        
        if description is not None:
            project.description = description
        if tags is not None:
            project.tags = tags
        if metadata is not None:
            project.metadata = metadata
        
        self._save_configuration()
        print(f"Updated project: {name}")
        return True
    
    def delete_project(self, name: str) -> bool:
        """
        Delete a project configuration.
        
        Args:
            name: Project name
            
        Returns:
            True if project was deleted, False if not found
        """
        if name not in self.projects:
            return False
        
        del self.projects[name]
        self._save_configuration()
        print(f"Deleted project: {name}")
        return True
    
    def get_client(self, project_name: str) -> Optional[Langfuse]:
        """
        Get a Langfuse client for a project.
        
        Args:
            project_name: Name of the project
            
        Returns:
            Langfuse client or None if project not found or keys missing
        """
        if Langfuse is None:
            print("Langfuse package not available")
            return None
        
        project = self.projects.get(project_name)
        if project is None:
            print(f"Project '{project_name}' not found")
            return None
        
        if not project.public_key or not project.secret_key:
            print(f"Project '{project_name}' does not have API keys configured")
            return None
        
        try:
            client = Langfuse(
                public_key=project.public_key,
                secret_key=project.secret_key,
                host=project.base_url
            )
            return client
        except Exception as e:
            print(f"Failed to create Langfuse client for '{project_name}': {e}")
            return None
    
    def validate_all_projects(self) -> Dict[str, bool]:
        """
        Validate API keys for all projects.
        
        Returns:
            Dictionary mapping project names to validation results
        """
        results = {}
        
        for name, project in self.projects.items():
            if project.public_key and project.secret_key:
                results[name] = self._validate_api_keys(project)
            else:
                results[name] = False
        
        return results
    
    def export_configuration(self, file_path: Path) -> bool:
        """
        Export project configuration to a file.
        
        Args:
            file_path: Path to export file
            
        Returns:
            True if export was successful, False otherwise
        """
        try:
            export_data = {
                "export_timestamp": datetime.utcnow().isoformat(),
                "base_url": self.base_url,
                "projects": {}
            }
            
            for name, project in self.projects.items():
                export_data["projects"][name] = {
                    "name": project.name,
                    "description": project.description,
                    "base_url": project.base_url,
                    "created_at": project.created_at.isoformat(),
                    "tags": project.tags,
                    "metadata": project.metadata,
                    # Note: API keys are not exported for security
                    "has_api_keys": bool(project.public_key and project.secret_key)
                }
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"Configuration exported to {file_path}")
            return True
            
        except Exception as e:
            print(f"Failed to export configuration: {e}")
            return False
    
    def import_configuration(self, file_path: Path, merge: bool = True) -> bool:
        """
        Import project configuration from a file.
        
        Args:
            file_path: Path to import file
            merge: Whether to merge with existing projects or replace
            
        Returns:
            True if import was successful, False otherwise
        """
        try:
            with open(file_path, 'r') as f:
                import_data = json.load(f)
            
            if not merge:
                self.projects.clear()
            
            projects_data = import_data.get("projects", {})
            
            for name, project_data in projects_data.items():
                if merge and name in self.projects:
                    print(f"Skipping existing project: {name}")
                    continue
                
                project = ProjectConfig(
                    name=project_data["name"],
                    description=project_data.get("description", ""),
                    base_url=project_data.get("base_url", self.base_url),
                    created_at=datetime.fromisoformat(project_data["created_at"]),
                    tags=project_data.get("tags", []),
                    metadata=project_data.get("metadata", {})
                )
                
                self.projects[name] = project
            
            self._save_configuration()
            print(f"Configuration imported from {file_path}")
            return True
            
        except Exception as e:
            print(f"Failed to import configuration: {e}")
            return False
    
    def setup_project_from_env(self, 
                              project_name: str = "default",
                              description: str = "Project created from environment variables") -> Optional[ProjectConfig]:
        """
        Set up a project using environment variables.
        
        Expected environment variables:
        - LANGFUSE_PUBLIC_KEY: API public key
        - LANGFUSE_SECRET_KEY: API secret key
        - LANGFUSE_HOST: Host (default: localhost)
        - LANGFUSE_PORT: Port (default: 3000)
        
        Args:
            project_name: Name for the project
            description: Description for the project
            
        Returns:
            Created project configuration or None if env vars not found
        """
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_HOST", "localhost")
        port = int(os.getenv("LANGFUSE_PORT", "3000"))
        
        if not public_key or not secret_key:
            print("Environment variables LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are required")
            return None
        
        # Create or update project
        if project_name in self.projects:
            project = self.projects[project_name]
        else:
            project = ProjectConfig(
                name=project_name,
                description=description,
                host=host,
                port=port
            )
            self.projects[project_name] = project
        
        # Set API keys
        project.public_key = public_key
        project.secret_key = secret_key
        project.host = host
        project.port = port
        project.base_url = f"http://{host}:{port}"
        
        self._save_configuration()
        
        # Validate
        if self._validate_api_keys(project):
            print(f"Project '{project_name}' configured successfully from environment")
            return project
        else:
            print(f"Warning: Project '{project_name}' configured but API keys validation failed")
            return project
    
    def _validate_api_keys(self, project: ProjectConfig) -> bool:
        """Validate API keys for a project."""
        if not project.public_key or not project.secret_key:
            return False
        
        if Langfuse is None:
            print("Cannot validate API keys: langfuse package not available")
            return False
        
        try:
            client = Langfuse(
                public_key=project.public_key,
                secret_key=project.secret_key,
                host=project.base_url
            )
            
            # Try to create a test trace
            trace = client.trace(name="validation_test")
            client.flush()
            
            return True
            
        except Exception as e:
            print(f"API key validation failed for '{project.name}': {e}")
            return False
    
    def _load_configuration(self) -> None:
        """Load configuration from file."""
        if not self.config_file.exists():
            return
        
        try:
            with open(self.config_file, 'r') as f:
                data = json.load(f)
            
            projects_data = data.get("projects", {})
            
            for name, project_data in projects_data.items():
                project = ProjectConfig(
                    name=project_data["name"],
                    description=project_data.get("description", ""),
                    public_key=project_data.get("public_key"),
                    secret_key=project_data.get("secret_key"),
                    host=project_data.get("host", "localhost"),
                    port=project_data.get("port", 3000),
                    base_url=project_data.get("base_url"),
                    created_at=datetime.fromisoformat(project_data["created_at"]),
                    tags=project_data.get("tags", []),
                    metadata=project_data.get("metadata", {})
                )
                
                self.projects[name] = project
            
            print(f"Loaded {len(self.projects)} projects from {self.config_file}")
            
        except Exception as e:
            print(f"Failed to load configuration: {e}")
    
    def _save_configuration(self) -> None:
        """Save configuration to file."""
        try:
            config_data = {
                "saved_at": datetime.utcnow().isoformat(),
                "base_url": self.base_url,
                "projects": {}
            }
            
            for name, project in self.projects.items():
                config_data["projects"][name] = {
                    "name": project.name,
                    "description": project.description,
                    "public_key": project.public_key,
                    "secret_key": project.secret_key,
                    "host": project.host,
                    "port": project.port,
                    "base_url": project.base_url,
                    "created_at": project.created_at.isoformat(),
                    "tags": project.tags,
                    "metadata": project.metadata
                }
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
        except Exception as e:
            print(f"Failed to save configuration: {e}")


def create_project_manager(base_url: str = "http://localhost:3000",
                          config_file: Optional[Path] = None) -> LangfuseProjectManager:
    """
    Create a Langfuse project manager.
    
    Args:
        base_url: Base URL of Langfuse instance
        config_file: Optional path to configuration file
        
    Returns:
        Project manager instance
    """
    return LangfuseProjectManager(base_url, config_file)


def setup_default_project(project_manager: LangfuseProjectManager) -> Optional[ProjectConfig]:
    """
    Set up a default project from environment variables.
    
    Args:
        project_manager: Project manager instance
        
    Returns:
        Created project configuration or None if setup failed
    """
    return project_manager.setup_project_from_env(
        project_name="sql_evaluation_default",
        description="Default project for SQL evaluation library"
    )


def print_project_summary(project_manager: LangfuseProjectManager) -> None:
    """
    Print a summary of all projects.
    
    Args:
        project_manager: Project manager instance
    """
    projects = project_manager.list_projects()
    
    if not projects:
        print("No projects configured")
        return
    
    print("\n" + "="*60)
    print("LANGFUSE PROJECTS SUMMARY")
    print("="*60)
    
    for project_name in projects:
        details = project_manager.get_project_details(project_name)
        if details:
            print(f"\nProject: {details['name']}")
            print(f"  Description: {details['description']}")
            print(f"  Base URL: {details['base_url']}")
            print(f"  Created: {details['created_at']}")
            print(f"  API Keys: {'✅ Valid' if details['api_keys_valid'] else '❌ Invalid/Missing'}")
            print(f"  Tags: {', '.join(details['tags']) if details['tags'] else 'None'}")
    
    print("="*60) 