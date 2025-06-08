"""HuggingFace dataset loader implementation."""

from typing import Any, Dict
from datasets import load_dataset
from ..interfaces import DatasetLoader, Dataset
from ..models import DatasetItem, DatasetMetadata
from ..base_dataset import BaseDataset


class HuggingFaceLoader(DatasetLoader):
    def load(self, source: str, **kwargs) -> Dataset:
        hf_dataset = load_dataset(source, **kwargs)
        items = []
        if hasattr(hf_dataset, 'items'):
            for split_name, split_data in hf_dataset.items():
                for item in split_data:
                    items.append(self._convert_item(item))
        else:
            for item in hf_dataset:
                items.append(self._convert_item(item))
        metadata = DatasetMetadata(name=source, source=source)
        return BaseDataset(items, metadata)

    def _convert_item(self, item: Dict[str, Any]) -> DatasetItem:
        return DatasetItem(input=item, expected_output=None, metadata={})
