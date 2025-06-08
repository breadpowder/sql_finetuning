"""Abstract base classes for the dataset management module."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional
from .models import DatasetItem, DatasetMetadata


class Dataset(ABC):
    """Abstract dataset interface."""

    @abstractmethod
    def __len__(self) -> int:
        """Return number of items in dataset."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[DatasetItem]:
        """Iterate over dataset items."""
        pass


class DatasetLoader(ABC):
    """Abstract loader interface."""

    @abstractmethod
    def load(self, source: str, **kwargs) -> Dataset:
        """Load dataset from source."""
        pass


class DatasetTransformer(ABC):
    """Abstract transformation interface."""

    @abstractmethod
    def transform(self, dataset: Dataset, **kwargs) -> Dataset:
        """Transform dataset."""
        pass
