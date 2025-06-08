"""Configuration for integration tests with Docker Langfuse deployment."""

import pytest
import time
import os
import subprocess
import shutil
from pathlib import Path
from sql_eval_lib.langfuse.deployment import LangfuseDeployment, DeploymentConfig


@pytest.fixture(scope="session", autouse=True)
def ensure_langfuse_repo():
    """Ensure Langfuse repository is available at sql_evaluation_library root."""
    project_root = Path(__file__).parent.parent.parent.parent  # sql_evaluation_library root
    langfuse_path = project_root / "langfuse"
    
    if not langfuse_path.exists():
        print(f"\n📥 Langfuse repository not found at {langfuse_path}")
        print("🔄 Cloning Langfuse repository...")
        
        try:
            # Clone the Langfuse repository
            result = subprocess.run([
                'git', 'clone', 
                'https://github.com/langfuse/langfuse.git',
                str(langfuse_path)
            ], capture_output=True, text=True, cwd=project_root)
            
            if result.returncode == 0:
                print(f"✅ Langfuse repository cloned successfully to {langfuse_path}")
            else:
                print(f"❌ Failed to clone Langfuse repository: {result.stderr}")
                pytest.skip("Failed to clone Langfuse repository")
                
        except FileNotFoundError:
            print("❌ Git command not found. Please install git.")
            pytest.skip("Git not available for cloning Langfuse repository")
        except Exception as e:
            print(f"❌ Unexpected error cloning Langfuse repository: {e}")
            pytest.skip("Failed to clone Langfuse repository")
    else:
        print(f"✅ Langfuse repository found at {langfuse_path}")
    
    return langfuse_path


@pytest.fixture(scope="session")
def test_deployment_config():
    """Create test-specific deployment configuration."""
    config = DeploymentConfig(
        port=3001,  # Test port to avoid conflicts
        project_root=Path(__file__).parent.parent.parent.parent  # sql_evaluation_library root
    )
    return config


@pytest.fixture(scope="session")
def langfuse_deployment(test_deployment_config):
    """Deploy Langfuse for testing session and clean up afterward."""
    deployment = LangfuseDeployment(test_deployment_config)
    
    # Check if Docker is available
    if not deployment.check_docker_availability():
        pytest.skip("Docker not available for integration testing")
    
    # Deploy Langfuse using official setup
    print("\n🚀 Deploying test Langfuse instance using official Docker Compose...")
    success, message = deployment.deploy(wait_for_healthy=True, timeout=300)
    
    if not success:
        pytest.skip(f"Failed to deploy test Langfuse: {message}")
    
    print(f"✅ Test Langfuse deployed successfully: {message}")
    
    # Provide connection information
    connection_info = deployment.get_connection_info()
    print(f"📡 Langfuse available at: {connection_info['url']}")
    print(f"🔑 Public Key: {connection_info['public_key']}")
    print(f"🔐 Secret Key: {connection_info['secret_key']}")
    
    yield {
        'deployment': deployment,
        'config': test_deployment_config,
        'connection_info': connection_info,
        'host': f"http://localhost:{test_deployment_config.port}",
        'public_key': connection_info['public_key'],
        'secret_key': connection_info['secret_key']
    }
    
    # Cleanup
    print("\n🧹 Cleaning up test Langfuse deployment...")
    cleanup_success = deployment.cleanup()
    if cleanup_success:
        print("✅ Test Langfuse cleaned up successfully")
    else:
        print("⚠️  Warning: Test Langfuse cleanup may have failed")


@pytest.fixture
def langfuse_test_client(langfuse_deployment):
    """Create a Langfuse client configured for testing."""
    from langfuse import Langfuse
    
    # Wait a bit more to ensure service is fully ready
    time.sleep(5)
    
    client = Langfuse(
        public_key=langfuse_deployment['public_key'],
        secret_key=langfuse_deployment['secret_key'],
        host=langfuse_deployment['host']
    )
    
    # Test connection
    try:
        # Create a test trace to verify connection
        trace = client.trace(name="test_connection_trace")
        client.flush()
        print(f"✅ Langfuse client connected successfully (trace: {trace.id})")
        return client
    except Exception as e:
        print(f"⚠️  Langfuse client connection issue: {e}")
        # Don't skip, let the test decide how to handle connection issues
        return client


@pytest.fixture
def test_dataset_small():
    """Create a small test dataset for integration testing."""
    from sql_eval_lib.dataset import HuggingFaceLoader
    
    loader = HuggingFaceLoader()
    # Load a very small dataset for fast testing
    dataset = loader.load('gretelai/synthetic_text_to_sql', split='train[:2]')
    return dataset 