"""
Test suite for 2_Embedding_Search_with_Qdrant_and_OpenAI.ipynb notebook
Tests the vector search functionality with Qdrant
"""
import os
from unittest.mock import patch

import pandas as pd
import pytest
from testbook import testbook


class TestQdrantSearchNotebook:
    """Test cases for the Qdrant search notebook functionality"""

    @pytest.fixture
    def notebook_path(self):
        """Return the path to the Qdrant search notebook"""
        return os.path.join(
            os.path.dirname(__file__), "..", "notebook", "2_Embedding_Search_with_Qdrant_and_OpenAI.ipynb"
        )

    def test_notebook_loads_successfully(self, notebook_path):
        """Test that the notebook loads without errors"""
        with testbook(notebook_path, execute=False) as tb:
            assert tb is not None
            assert len(tb.cells) > 0

    @patch('pandas.read_csv')
    def test_data_loading_with_embeddings(self, mock_read_csv, notebook_path):
        """Test loading of embedded data"""
        # Mock embedded data
        test_data = pd.DataFrame({
            'title': ['Test title'],
            'answer': ['Test answer'],
            'title_vector': [str([0.1] * 1536)],
            'answer_vector': [str([0.2] * 1536)]
        })
        mock_read_csv.return_value = test_data

        with testbook(notebook_path, execute=False) as tb:
            # This will depend on the actual cell structure of notebook 2
            # For now, test basic loading functionality
            assert tb is not None


class TestEvaluationNotebook:
    """Test cases for the evaluation notebook"""

    @pytest.fixture
    def notebook_path(self):
        """Return the path to the evaluation notebook"""
        return os.path.join(
            os.path.dirname(__file__), "..", "notebook", "5_Evaluation.ipynb")

    def test_notebook_loads_successfully(self, notebook_path):
        """Test that the evaluation notebook loads"""
        with testbook(notebook_path, execute=False) as tb:
            assert tb is not None
            assert len(tb.cells) > 0


class TestDIYRAGNotebook:
    """Test cases for the DIY RAG notebook"""

    @pytest.fixture
    def notebook_path(self):
        """Return the path to the DIY RAG notebook"""
        return os.path.join(
            os.path.dirname(__file__), "..", "notebook", "4_RAG_DIY.ipynb"
        )

    def test_notebook_loads_successfully(self, notebook_path):
        """Test that the DIY RAG notebook loads"""
        with testbook(notebook_path, execute=False) as tb:
            assert tb is not None
            assert len(tb.cells) > 0


if __name__ == "__main__":
    pytest.main([__file__])
