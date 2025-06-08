"""Tests for dataset models."""

import pytest
from sql_eval_lib.dataset.models import DatasetItem, DatasetMetadata


def test_dataset_item_creation():
    """Test DatasetItem creation and auto-generation of ID."""
    item = DatasetItem(input={'query': 'SELECT * FROM users'})
    assert item.input == {'query': 'SELECT * FROM users'}
    assert item.item_id is not None
    assert item.metadata == {}
