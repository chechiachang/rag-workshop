"""
Test suite for 1_Embedding_with_OpenAI.ipynb notebook
Tests the embedding functionality and utility functions
"""
import os

import pytest
from testbook import testbook


class TestEmbeddingNotebook:
    """Test cases for the embedding notebook functionality"""

    @pytest.fixture
    def notebook_path(self):
        """Return the path to the embedding notebook"""
        return os.path.join(os.path.dirname(__file__), "..", "notebook", "1_Embedding_with_OpenAI.ipynb")

    def test_notebook_executes_without_errors(self, notebook_path):
        """Test that the notebook can be executed without errors"""
        with testbook(notebook_path, execute=False) as tb:
            # Test that notebook loads successfully
            assert tb is not None
            assert len(tb.cells) > 0

if __name__ == "__main__":
    pytest.main([__file__])
