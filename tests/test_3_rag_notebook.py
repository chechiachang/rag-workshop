"""
Test suite for 3_RAG_with_OpenAI.ipynb notebook
Tests the RAG functionality and question-answering system
"""
import os

import pytest
from testbook import testbook


class TestRAGNotebook:
    """Test cases for the RAG notebook functionality"""

    @pytest.fixture
    def notebook_path(self):
        """Return the path to the RAG notebook"""
        return os.path.join(os.path.dirname(__file__), "..", "notebook", "3_RAG_with_OpenAI.ipynb")

    def test_notebook_loads_successfully(self, notebook_path):
        """Test that the notebook loads without errors"""
        with testbook(notebook_path, execute=False) as tb:
            assert tb is not None
            assert len(tb.cells) > 0

if __name__ == "__main__":
    pytest.main([__file__])
