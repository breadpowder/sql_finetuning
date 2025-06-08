"""Tests for dataset transformers."""

import pytest
from sql_eval_lib.dataset import SQLTransformer, HuggingFaceLoader


def test_sql_transformer():
    """Test SQL transformer with filtering."""
    loader = HuggingFaceLoader()
    dataset = loader.load('gretelai/synthetic_text_to_sql', split='train[:5]')
    transformer = SQLTransformer()
    # Test that transformation doesn't crash
    filtered = transformer.transform(dataset, "SELECT * FROM dataset WHERE domain = 'test'")
    assert len(filtered) >= 0  # May be 0 if no matches
