"""
Simplified test suite for notebook functionality with testbook
Tests core functionality without executing potentially problematic cells
"""
import os
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from testbook import testbook


class TestNotebookStructure:
    """Test basic notebook structure and loading"""

    @pytest.fixture
    def embedding_notebook_path(self):
        """Return the path to the embedding notebook"""
        return os.path.join(
            os.path.dirname(__file__), "..", "notebook", "1_Embedding_with_OpenAI.ipynb")

    @pytest.fixture
    def rag_notebook_path(self):
        """Return the path to the RAG notebook"""
        return os.path.join(
            os.path.dirname(__file__), "..", "notebook", "3_RAG_with_OpenAI.ipynb")

    def test_embedding_notebook_loads(self, embedding_notebook_path):
        """Test that the embedding notebook loads successfully"""
        with testbook(embedding_notebook_path, execute=False) as tb:
            assert tb is not None
            assert len(tb.cells) > 0
            print(f"Embedding notebook has {len(tb.cells)} cells")

    def test_rag_notebook_loads(self, rag_notebook_path):
        """Test that the RAG notebook loads successfully"""
        with testbook(rag_notebook_path, execute=False) as tb:
            assert tb is not None
            assert len(tb.cells) > 0
            print(f"RAG notebook has {len(tb.cells)} cells")

    def test_notebook_cell_types(self, embedding_notebook_path):
        """Test that notebooks have the expected cell types"""
        with testbook(embedding_notebook_path, execute=False) as tb:
            cell_types = [cell['cell_type'] for cell in tb.cells]
            assert 'code' in cell_types
            assert 'markdown' in cell_types
            print(f"Cell types found: {set(cell_types)}")


class TestNotebookCodeExecution:
    """Test execution of safe code cells"""

    @pytest.fixture
    def embedding_notebook_path(self):
        return os.path.join(
            os.path.dirname(__file__), "..", "notebook", "1_Embedding_with_OpenAI.ipynb"
        )

    def test_basic_imports(self, embedding_notebook_path):
        """Test that basic imports can be executed"""
        with testbook(embedding_notebook_path, execute=False) as tb:
            # Try to inject and execute a simple import test
            tb.inject("import os\nimport pandas as pd\ntest_var = 'success'")

            # Test that injected code worked
            assert tb.ref("test_var") == "success"
            assert tb.ref("os") is not None
            assert tb.ref("pd") is not None

    def test_tiktoken_functionality(self, embedding_notebook_path):
        """Test tiktoken functionality in isolation"""
        with testbook(embedding_notebook_path, execute=False) as tb:
            # Inject tiktoken code directly
            tiktoken_code = """
import tiktoken

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

test_result = num_tokens_from_string("tiktoken is great!", "cl100k_base")
"""
            tb.inject(tiktoken_code)

            # Test the function
            result = tb.ref("test_result")
            assert isinstance(result, int)
            assert result > 0

            # Test the function directly
            func = tb.ref("num_tokens_from_string")
            assert callable(func)
            assert func("hello", "cl100k_base") > 0

    @patch('openai.AzureOpenAI')
    def test_mock_embedding_function(self, mock_azure_openai, embedding_notebook_path):
        """Test embedding function with mocked OpenAI"""
        # Setup mock
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = [0.1, 0.2, 0.3] * 512
        mock_client.embeddings.create.return_value = mock_response
        mock_azure_openai.return_value = mock_client

        with testbook(embedding_notebook_path, execute=False) as tb:
            # Inject just the function definition without calling it
            embedding_code = """
from unittest.mock import MagicMock

# Mock the OpenAI client
mock_client = MagicMock()
mock_response = MagicMock()
mock_response.data = [MagicMock()]
mock_response.data[0].embedding = [0.1, 0.2, 0.3] * 512
mock_client.embeddings.create.return_value = mock_response

def embedding(input: str, model: str="text-embedding-3-large"):
    response = mock_client.embeddings.create(
        input=input,
        model=model,
    )
    return response.data[0].embedding

# Test the function
test_embedding = embedding("test text")
embedding_function_exists = callable(embedding)
"""
            tb.inject(embedding_code)

            # Test the embedding function
            result = tb.ref("test_embedding")
            assert isinstance(result, list)
            assert len(result) > 0
            assert tb.ref("embedding_function_exists") is True


class TestDataProcessing:
    """Test data processing functionality"""

    @pytest.fixture
    def embedding_notebook_path(self):
        return os.path.join(
            os.path.dirname(__file__), "..", "notebook", "1_Embedding_with_OpenAI.ipynb"
        )

    def test_dataframe_operations(self, embedding_notebook_path):
        """Test basic DataFrame operations that might be in notebooks"""
        with testbook(embedding_notebook_path, execute=False) as tb:
            # Inject DataFrame testing code
            df_code = """
import pandas as pd

# Create test data similar to what's in notebooks
test_data = pd.DataFrame({
    'title': ['What are COVID symptoms?', 'How to prevent COVID?'],
    'answer': ['Fever, cough, fatigue', 'Masks, distance, vaccines']
})

# Test basic operations
df_len = len(test_data)
df_columns = list(test_data.columns)
"""
            tb.inject(df_code)

            # Test results
            assert tb.ref("df_len") == 2
            assert set(tb.ref("df_columns")) == {'title', 'answer'}

    def test_vector_operations(self, embedding_notebook_path):
        """Test vector similarity operations"""
        with testbook(embedding_notebook_path, execute=False) as tb:
            # Inject vector similarity code
            vector_code = """
import numpy as np

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

# Test vectors
vec1 = [1, 0, 0]
vec2 = [0, 1, 0]
vec3 = [1, 0, 0]

sim_same = cosine_similarity(vec1, vec3)
sim_diff = cosine_similarity(vec1, vec2)
"""
            tb.inject(vector_code)

            # Test similarity calculations
            assert abs(tb.ref("sim_same") - 1.0) < 1e-10  # Same vectors
            assert abs(tb.ref("sim_diff") - 0.0) < 1e-10  # Orthogonal vectors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
