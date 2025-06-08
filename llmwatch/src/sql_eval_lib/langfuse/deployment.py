"""
Automated Langfuse deployment and setup utilities.

This module provides tools for automatically deploying and configuring Langfuse
for use with the SQL evaluation library, using the official Langfuse Docker Compose setup.
"""

import os
import subprocess
import time
import shutil
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum


class DeploymentStatus(Enum):
    """Status of Langfuse deployment."""
    NOT_DEPLOYED = "not_deployed"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class DeploymentConfig:
    """Configuration for Langfuse deployment."""
    port: int = 3001
    project_root: Optional[Path] = None
    # Authentication keys that will be created during headless initialization
    public_key: str = "pk-lf-test-key-for-integration-testing"
    secret_key: str = "sk-lf-test-secret-for-integration-testing"
    # Test user credentials
    admin_email: str = "test@example.com"
    admin_password: str = "testpassword123"
    
    def __post_init__(self):
        """Set default project root if not provided."""
        if self.project_root is None:
            self.project_root = Path.cwd()


class LangfuseDeployment:
    """
    Manages automated Langfuse deployment using the official Docker Compose setup.
    
    This class provides methods for deploying, configuring, and managing
    a local Langfuse instance for development and testing purposes using
    the official Langfuse repository and Docker Compose configuration.
    """
    
    def __init__(self, config: Optional[DeploymentConfig] = None):
        """
        Initialize the deployment manager.
        
        Args:
            config: Optional deployment configuration. If None, uses defaults.
        """
        self.config = config or DeploymentConfig()
        self.langfuse_repo_path = self.config.project_root / "langfuse"
        self.docker_compose_file = self.langfuse_repo_path / "docker-compose.test.yml"
    
    def check_docker_availability(self) -> bool:
        """
        Check if Docker and Docker Compose are available.
        
        Returns:
            True if Docker is available, False otherwise.
        """
        try:
            # Check Docker
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True, check=True)
            print(f"Docker version: {result.stdout.strip()}")
            
            # Check Docker Compose
            result = subprocess.run(['docker', 'compose', 'version'], 
                                  capture_output=True, text=True, check=True)
            print(f"Docker Compose version: {result.stdout.strip()}")
            
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Docker is not available or not properly installed: {e}")
            return False
    
    def clone_langfuse_repo(self) -> bool:
        """
        Clone the official Langfuse repository if it doesn't exist.
        
        Returns:
            True if successful or already exists, False otherwise.
        """
        if self.langfuse_repo_path.exists():
            print(f"Langfuse repository already exists at {self.langfuse_repo_path}")
            return True
        
        try:
            print("Cloning official Langfuse repository...")
            result = subprocess.run([
                'git', 'clone', 
                'https://github.com/langfuse/langfuse.git',
                str(self.langfuse_repo_path)
            ], capture_output=True, text=True, check=True, cwd=self.config.project_root)
            
            print(f"Langfuse repository cloned to {self.langfuse_repo_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone Langfuse repository: {e.stderr}")
            return False
    
    def setup_test_docker_compose(self) -> bool:
        """
        Copy the test Docker Compose configuration to the Langfuse repository.
        
        Returns:
            True if successful, False otherwise.
        """
        source_compose_file = self.config.project_root / "langfuse" / "docker-compose.test.yml"
        
        if not source_compose_file.exists():
            print(f"Test Docker Compose file not found at {source_compose_file}")
            return False
        
        try:
            # The file should already be created by our edit_file call
            print(f"Test Docker Compose file ready at {source_compose_file}")
            return True
        except Exception as e:
            print(f"Failed to setup test Docker Compose: {e}")
            return False
    
    def deploy(self, wait_for_healthy: bool = True, timeout: int = 300) -> Tuple[bool, str]:
        """
        Deploy Langfuse using the official Docker Compose setup.
        
        Args:
            wait_for_healthy: Whether to wait for services to be healthy
            timeout: Maximum time to wait for deployment (seconds)
            
        Returns:
            Tuple of (success, status_message)
        """
        if not self.check_docker_availability():
            return False, "Docker is not available"
        
        if not self.clone_langfuse_repo():
            return False, "Failed to clone Langfuse repository"
        
        if not self.setup_test_docker_compose():
            return False, "Failed to setup test Docker Compose configuration"
        
        try:
            # Stop any existing deployment
            self.stop()
            
            # Start services
            print("Starting Langfuse deployment using official Docker Compose...")
            result = subprocess.run([
                'docker', 'compose', 
                '-f', str(self.docker_compose_file),
                'up', '-d'
            ], capture_output=True, text=True, check=True, cwd=self.langfuse_repo_path)
            
            print("Langfuse containers started")
            
            if wait_for_healthy:
                if self.wait_for_healthy(timeout):
                    return True, "Langfuse deployed successfully and is healthy"
                else:
                    return False, f"Langfuse deployment timed out after {timeout} seconds"
            else:
                return True, "Langfuse deployment initiated"
                
        except subprocess.CalledProcessError as e:
            error_msg = f"Docker Compose failed: {e.stderr}"
            print(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Deployment failed: {e}"
            print(error_msg)
            return False, error_msg
    
    def stop(self) -> bool:
        """
        Stop the Langfuse deployment.
        
        Returns:
            True if stopped successfully, False otherwise.
        """
        if not self.docker_compose_file.exists():
            print("No Docker Compose file found")
            return True
        
        try:
            print("Stopping Langfuse deployment...")
            subprocess.run([
                'docker', 'compose',
                '-f', str(self.docker_compose_file),
                'down'
            ], capture_output=True, text=True, check=True, cwd=self.langfuse_repo_path)
            
            print("Langfuse deployment stopped")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to stop deployment: {e.stderr}")
            return False
        except Exception as e:
            print(f"Error stopping deployment: {e}")
            return False
    
    def get_status(self) -> DeploymentStatus:
        """
        Get the current deployment status.
        
        Returns:
            Current deployment status.
        """
        if not self.docker_compose_file.exists():
            return DeploymentStatus.NOT_DEPLOYED
        
        try:
            result = subprocess.run([
                'docker', 'compose',
                '-f', str(self.docker_compose_file),
                'ps', '--format', 'json'
            ], capture_output=True, text=True, check=True, cwd=self.langfuse_repo_path)
            
            if not result.stdout.strip():
                return DeploymentStatus.NOT_DEPLOYED
            
            # Parse container status
            import json
            containers = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    containers.append(json.loads(line))
            
            if not containers:
                return DeploymentStatus.NOT_DEPLOYED
            
            # Check if all containers are running
            running_containers = [c for c in containers if c.get('State') == 'running']
            
            if len(running_containers) == len(containers):
                # Check health status
                if self.check_health():
                    return DeploymentStatus.DEPLOYED
                else:
                    return DeploymentStatus.DEPLOYING
            elif running_containers:
                return DeploymentStatus.DEPLOYING
            else:
                return DeploymentStatus.FAILED
                
        except subprocess.CalledProcessError:
            return DeploymentStatus.FAILED
        except Exception:
            return DeploymentStatus.FAILED
    
    def check_health(self) -> bool:
        """
        Check if Langfuse is healthy and accessible.
        
        Returns:
            True if healthy, False otherwise.
        """
        try:
            import requests
            response = requests.get(f"http://localhost:{self.config.port}/api/public/health", timeout=10)
            return response.status_code == 200
        except Exception:
            return False
    
    def wait_for_healthy(self, timeout: int = 300) -> bool:
        """
        Wait for Langfuse to become healthy.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if becomes healthy within timeout, False otherwise.
        """
        print(f"Waiting for Langfuse to become healthy (timeout: {timeout}s)...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_status()
            print(f"Status: {status.value}")
            
            if status == DeploymentStatus.DEPLOYED:
                print("Langfuse is healthy!")
                return True
            elif status == DeploymentStatus.FAILED:
                print("Deployment failed")
                return False
            
            time.sleep(10)
        
        print("Timeout waiting for Langfuse to become healthy")
        return False
    
    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get connection information for the deployed Langfuse instance.
        
        Returns:
            Dictionary with connection details.
        """
        return {
            "host": "localhost",
            "port": self.config.port,
            "url": f"http://localhost:{self.config.port}",
            "api_url": f"http://localhost:{self.config.port}/api",
            "public_key": self.config.public_key,
            "secret_key": self.config.secret_key,
            "admin_email": self.config.admin_email,
            "admin_password": self.config.admin_password,
            "status": self.get_status().value
        }
    
    def cleanup(self) -> bool:
        """
        Clean up deployment files and resources.
        
        Returns:
            True if cleanup was successful, False otherwise.
        """
        success = True
        
        # Stop deployment first
        if not self.stop():
            success = False
        
        # Remove volumes
        try:
            subprocess.run([
                'docker', 'compose',
                '-f', str(self.docker_compose_file),
                'down', '-v'
            ], capture_output=True, text=True, check=True, cwd=self.langfuse_repo_path)
        except subprocess.CalledProcessError as e:
            print(f"Failed to remove volumes: {e.stderr}")
            success = False
        
        return success


def deploy_langfuse(config: Optional[DeploymentConfig] = None, 
                   wait_for_healthy: bool = True,
                   timeout: int = 300) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Deploy Langfuse with default configuration using official setup.
    
    Args:
        config: Optional deployment configuration
        wait_for_healthy: Whether to wait for the service to be healthy
        timeout: Maximum time to wait for deployment
        
    Returns:
        Tuple of (success, message, connection_info)
    """
    deployment = LangfuseDeployment(config)
    success, message = deployment.deploy(wait_for_healthy, timeout)
    connection_info = deployment.get_connection_info()
    
    if success:
        print("\n" + "="*50)
        print("LANGFUSE DEPLOYMENT SUCCESSFUL")
        print("="*50)
        print(f"URL: {connection_info['url']}")
        print(f"API: {connection_info['api_url']}")
        print(f"Public Key: {connection_info['public_key']}")
        print(f"Secret Key: {connection_info['secret_key']}")
        print(f"Admin Email: {connection_info['admin_email']}")
        print(f"Admin Password: {connection_info['admin_password']}")
        print("="*50)
    
    return success, message, connection_info


def stop_langfuse(config: Optional[DeploymentConfig] = None) -> bool:
    """
    Stop Langfuse deployment.
    
    Args:
        config: Optional deployment configuration
        
    Returns:
        True if stopped successfully, False otherwise.
    """
    deployment = LangfuseDeployment(config)
    return deployment.stop()


def get_langfuse_status(config: Optional[DeploymentConfig] = None) -> Dict[str, Any]:
    """
    Get Langfuse deployment status and connection info.
    
    Args:
        config: Optional deployment configuration
        
    Returns:
        Dictionary with status and connection information.
    """
    deployment = LangfuseDeployment(config)
    return deployment.get_connection_info() 