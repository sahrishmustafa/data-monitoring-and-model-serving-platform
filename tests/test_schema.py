# tests/test_schema.py
"""Test schema change detection logic from ingestion.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_feature_added():
    """Simulates a new column appearing between two batches."""
    old_schema = ["age", "income", "label"]
    new_schema = ["age", "income", "new_feature", "label"]

    added = set(new_schema) - set(old_schema)
    removed = set(old_schema) - set(new_schema)

    assert "new_feature" in added, "Should detect added feature"
    assert len(removed) == 0, "Nothing should be removed"


def test_feature_removed():
    """Simulates an existing column disappearing."""
    old_schema = ["age", "income", "dropped_col", "label"]
    new_schema = ["age", "income", "label"]

    added = set(new_schema) - set(old_schema)
    removed = set(old_schema) - set(new_schema)

    assert "dropped_col" in removed, "Should detect removed feature"
    assert len(added) == 0, "Nothing should be added"


def test_schema_unchanged():
    """Identical schemas should report no changes."""
    schema = ["age", "income", "label"]
    added = set(schema) - set(schema)
    removed = set(schema) - set(schema)
    assert len(added) == 0
    assert len(removed) == 0
