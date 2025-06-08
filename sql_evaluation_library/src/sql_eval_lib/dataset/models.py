"""Core data models for the dataset management module."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid


@dataclass
class DatasetItem:
    """Represents a single dataset item with input, expected output, and metadata."""
    input: Dict[str, Any]
    expected_output: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    item_id: Optional[str] = None

    def __post_init__(self):
        """Generate a unique ID if not provided."""
        if self.item_id is None:
            self.item_id = str(uuid.uuid4())
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DatasetMetadata:
    """Metadata information about a dataset."""
    name: str
    description: Optional[str] = None
    source: Optional[str] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        """Set creation time if not provided."""
        if self.created_at is None:
            self.created_at = datetime.utcnow()
