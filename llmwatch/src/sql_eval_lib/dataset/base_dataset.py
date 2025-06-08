"""Base dataset implementation."""

from typing import Iterator, List
from .interfaces import Dataset
from .models import DatasetItem, DatasetMetadata


class BaseDataset(Dataset):
    """Concrete implementation of Dataset interface."""

    def __init__(self, items: List[DatasetItem], metadata: DatasetMetadata):
        self.items = items
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[DatasetItem]:
        return iter(self.items)
