"""Tests for dataset loaders."""

import pytest
from sql_eval_lib.dataset import HuggingFaceLoader


def test_huggingface_loader():
    """Test HuggingFace loader with small dataset."""
    loader = HuggingFaceLoader()
    dataset = loader.load('gretelai/synthetic_text_to_sql', split='train[:2]')
    assert len(dataset) == 2
    assert dataset.metadata.name == 'gretelai/synthetic_text_to_sql'
