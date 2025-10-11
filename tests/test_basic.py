"""
Simple test to verify testbook setup
"""
import os

import pytest
from testbook import testbook


def test_testbook_basic():
    """Basic test to verify testbook is working"""
    notebook_path = os.path.join(os.path.dirname(__file__), "..", "notebook", "1_Embedding_with_OpenAI.ipynb")

    with testbook(notebook_path, execute=False) as tb:
        # Test that notebook loads
        assert tb is not None
        assert len(tb.cells) > 0
        print(f"Notebook has {len(tb.cells)} cells")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
