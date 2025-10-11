"""
Configuration for pytest
"""
import os
import sys
from unittest.mock import patch

import pytest

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture(autouse=True)
def mock_environment():
    """Automatically mock environment variables for all tests"""
    test_env = {
        'AZURE_OPENAI_API_KEY': 'test_api_key_12345',
        'AZURE_OPENAI_ENDPOINT': 'https://test-resource.openai.azure.com/',
        'OPENAI_API_VERSION': '2024-12-01-preview',
        'OPENAI_MODEL': 'text-embedding-3-large'
    }

    with patch.dict(os.environ, test_env, clear=False):
        yield


@pytest.fixture
def sample_data():
    """Sample data for testing"""
    import pandas as pd
    return pd.DataFrame({
        'title': ['Test question 1', 'Test question 2'],
        'answer': ['Test answer 1', 'Test answer 2']
    })


@pytest.fixture
def sample_embedded_data():
    """Sample data with embeddings for testing"""
    import pandas as pd
    return pd.DataFrame({
        'title': ['Test question 1', 'Test question 2'],
        'answer': ['Test answer 1', 'Test answer 2'],
        'title_vector': [str([0.1] * 1536), str([0.2] * 1536)],
        'answer_vector': [str([0.3] * 1536), str([0.4] * 1536)]
    })
