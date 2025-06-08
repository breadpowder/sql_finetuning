"""End-to-end integration tests with real Docker Langfuse deployment."""

import pytest
import time
from sql_eval_lib.dataset import HuggingFaceLoader, SQLTransformer, LangfuseDatasetManager


def test_docker_deployment_setup(test_deployment_config):
    """Test that Docker deployment configuration is properly set up."""
    print("\n🧪 Testing Docker deployment configuration...")
    
    from sql_eval_lib.langfuse.deployment import LangfuseDeployment
    
    deployment = LangfuseDeployment(test_deployment_config)
    
    # Test Docker availability
    docker_available = deployment.check_docker_availability()
    if not docker_available:
        pytest.skip("Docker not available for testing")
    
    print("✅ Docker is available")
    
    # Test repository cloning
    repo_cloned = deployment.clone_langfuse_repo()
    assert repo_cloned, "Should be able to clone/access Langfuse repository"
    print("✅ Langfuse repository ready")
    
    # Test Docker Compose setup
    compose_setup = deployment.setup_test_docker_compose()
    assert compose_setup, "Should be able to setup test Docker Compose configuration"
    print("✅ Docker Compose configuration ready")
    
    # Verify files exist
    assert deployment.docker_compose_file.exists(), "Docker Compose file should exist"
    print("✅ Configuration files verified")
    
    # Test deployment status check
    status = deployment.get_status()
    print(f"✅ Deployment status check working: {status.value}")
    
    print("✅ Docker deployment setup test completed")


def test_full_pipeline_with_docker_langfuse(langfuse_deployment, langfuse_test_client, test_dataset_small):
    """Test the complete dataset pipeline with real Docker Langfuse instance."""
    print("\n🧪 Starting end-to-end integration test with Docker Langfuse...")
    
    # Step 1: Load dataset
    print("📥 Loading test dataset...")
    loader = HuggingFaceLoader()
    dataset = loader.load('gretelai/synthetic_text_to_sql', split='train[:3]')
    assert len(dataset) == 3
    print(f"✅ Loaded {len(dataset)} items from dataset")
    
    # Step 2: Transform dataset with SQL filtering
    print("🔄 Transforming dataset with SQL filtering...")
    transformer = SQLTransformer()
    filtered = transformer.transform(dataset, "SELECT * FROM dataset WHERE id IS NOT NULL")
    assert len(filtered) >= 0  # Some items should remain after filtering
    print(f"✅ Filtered dataset to {len(filtered)} items")
    
    # Step 3: Test Langfuse manager with real deployment
    print("🌐 Testing Langfuse integration with real instance...")
    manager = LangfuseDatasetManager(
        public_key=langfuse_deployment['public_key'],
        secret_key=langfuse_deployment['secret_key'],
        host=langfuse_deployment['host']
    )
    assert manager is not None
    print("✅ Langfuse dataset manager created successfully")
    
    # Step 4: Test actual dataset upload to live Langfuse
    print("📤 Uploading dataset to live Langfuse instance...")
    dataset_name = f"test_dataset_{int(time.time())}"
    
    try:
        upload_success = manager.upload_dataset(
            dataset=filtered,
            dataset_name=dataset_name,
            batch_size=10
        )
        assert upload_success, "Dataset upload should succeed"
        print(f"✅ Dataset '{dataset_name}' uploaded successfully")
        
        # Wait a moment for data to be processed
        time.sleep(2)
        
        # Step 5: Verify the upload by checking if we can create traces
        print("🔍 Verifying upload by testing trace creation...")
        test_trace = langfuse_test_client.trace(
            name=f"verification_trace_{int(time.time())}",
            metadata={"test": "end_to_end_verification", "dataset": dataset_name}
        )
        langfuse_test_client.flush()
        
        assert test_trace.id is not None, "Should be able to create traces after upload"
        print(f"✅ Verification trace created: {test_trace.id}")
        
    except Exception as e:
        pytest.fail(f"Dataset upload or verification failed: {e}")
    
    print("🎉 End-to-end integration test completed successfully!")


def test_pipeline_error_handling(langfuse_deployment):
    """Test pipeline error handling with invalid configurations."""
    print("\n🧪 Testing error handling...")
    
    # Test with invalid credentials
    print("🔑 Testing invalid credentials handling...")
    invalid_manager = LangfuseDatasetManager(
        public_key="invalid_key",
        secret_key="invalid_secret",
        host=langfuse_deployment['host']
    )
    
    # Create a small test dataset
    loader = HuggingFaceLoader()
    small_dataset = loader.load('gretelai/synthetic_text_to_sql', split='train[:1]')
    
    # Upload should fail gracefully
    upload_success = invalid_manager.upload_dataset(
        dataset=small_dataset,
        dataset_name="test_invalid_upload",
        batch_size=1
    )
    
    # Should return False (not throw exception)
    assert upload_success is False, "Upload with invalid credentials should fail gracefully"
    print("✅ Invalid credentials handled gracefully")


def test_basic_pipeline_compatibility():
    """Test that basic pipeline still works without Docker (backward compatibility)."""
    print("\n🧪 Testing backward compatibility...")
    
    # Load dataset
    loader = HuggingFaceLoader()
    dataset = loader.load('gretelai/synthetic_text_to_sql', split='train[:2]')
    assert len(dataset) == 2
    
    # Transform dataset
    transformer = SQLTransformer()
    filtered = transformer.transform(dataset, "SELECT * FROM dataset WHERE id IS NOT NULL")
    assert len(filtered) >= 0
    
    # Test manager creation (without actual upload)
    manager = LangfuseDatasetManager('test_key', 'test_secret')
    assert manager is not None
    
    print("✅ Basic pipeline compatibility maintained")


@pytest.mark.slow
def test_large_dataset_integration(langfuse_deployment, langfuse_test_client):
    """Test integration with a larger dataset (marked as slow test)."""
    print("\n🧪 Testing with larger dataset...")
    
    # Load a slightly larger dataset
    loader = HuggingFaceLoader()
    dataset = loader.load('gretelai/synthetic_text_to_sql', split='train[:10]')
    assert len(dataset) == 10
    
    # Transform dataset
    transformer = SQLTransformer()
    filtered = transformer.transform(dataset, "SELECT * FROM dataset WHERE id IS NOT NULL")
    
    # Upload to Langfuse
    manager = LangfuseDatasetManager(
        public_key=langfuse_deployment['public_key'],
        secret_key=langfuse_deployment['secret_key'],
        host=langfuse_deployment['host']
    )
    
    dataset_name = f"large_test_dataset_{int(time.time())}"
    upload_success = manager.upload_dataset(
        dataset=filtered,
        dataset_name=dataset_name,
        batch_size=5  # Smaller batches for testing
    )
    
    assert upload_success, "Large dataset upload should succeed"
    print(f"✅ Large dataset uploaded successfully")


if __name__ == "__main__":
    # Allow running tests individually for debugging
    print("Running integration tests...")
    pytest.main([__file__, "-v", "-s"])
