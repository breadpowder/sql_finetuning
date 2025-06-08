"""Langfuse dataset manager implementation."""

from typing import Any, Dict, List, Optional
import time
from langfuse import Langfuse
from ..interfaces import Dataset
from ..models import DatasetItem, DatasetMetadata


class LangfuseDatasetManager:
    """Wrapper for Langfuse dataset operations."""

    def __init__(self, public_key: str, secret_key: str, host: Optional[str] = None):
        self.langfuse = Langfuse(public_key=public_key, secret_key=secret_key, host=host)

    def upload_dataset(self, dataset: Dataset, dataset_name: str, batch_size: int = 100) -> bool:
        """Upload dataset to Langfuse with batch processing."""
        try:
            items = list(dataset)
            
            # First, create the dataset if it doesn't exist
            try:
                print(f"Creating dataset '{dataset_name}' in Langfuse...")
                langfuse_dataset = self.langfuse.create_dataset(
                    name=dataset_name,
                    description=f"Dataset uploaded via SQL evaluation library at {time.strftime('%Y-%m-%d %H:%M:%S')}",
                    metadata={"source": "sql_evaluation_library", "items_count": len(items)}
                )
                print(f"✅ Dataset '{dataset_name}' created successfully")
            except Exception as e:
                # Dataset might already exist, try to get it
                print(f"Dataset creation failed (might already exist): {e}")
                try:
                    langfuse_dataset = self.langfuse.get_dataset(dataset_name)
                    print(f"✅ Using existing dataset '{dataset_name}'")
                except Exception as get_error:
                    print(f"❌ Failed to create or get dataset: {get_error}")
                    return False
            
            # Process items in batches
            print(f"Uploading {len(items)} items in batches of {batch_size}...")
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                success = self._upload_batch(batch, dataset_name)
                if not success:
                    print(f"❌ Failed to upload batch {i//batch_size + 1}")
                    return False
                print(f"✅ Uploaded batch {i//batch_size + 1}/{(len(items) + batch_size - 1)//batch_size}")
                time.sleep(0.1)  # Rate limiting
            
            # Flush to ensure all data is sent
            self.langfuse.flush()
            print(f"✅ Successfully uploaded {len(items)} items to dataset '{dataset_name}'")
            return True
            
        except Exception as e:
            print(f'Error uploading dataset: {e}')
            return False

    def _upload_batch(self, batch: List[DatasetItem], dataset_name: str) -> bool:
        """Upload a batch of items to Langfuse."""
        try:
            for item in batch:
                # Convert DatasetItem to Langfuse format
                langfuse_item = self._convert_to_langfuse_format(item)
                
                # Create dataset item in Langfuse
                self.langfuse.create_dataset_item(
                    dataset_name=dataset_name,
                    **langfuse_item
                )
            return True
        except Exception as e:
            print(f"Error uploading batch: {e}")
            return False

    def _convert_to_langfuse_format(self, item: DatasetItem) -> Dict[str, Any]:
        """Convert DatasetItem to Langfuse format."""
        return {
            'input': item.input,
            'expected_output': item.expected_output,
            'metadata': item.metadata,
            'id': item.item_id
        }
