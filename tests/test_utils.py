"""
Utility functions for testing notebooks
"""
import os
import tempfile

import pandas as pd


def create_test_csv_data():
    """Create test CSV data for notebook testing"""
    return pd.DataFrame({
        'title': [
            'What are the symptoms of COVID-19?',
            'How to prevent COVID-19?',
            'COVID-19 vaccine information'
        ],
        'answer': [
            'Common symptoms include fever, cough, fatigue, and loss of taste or smell.',
            'Wear masks, maintain social distance, wash hands frequently, and get vaccinated.',
            'COVID-19 vaccines are safe and effective in preventing severe illness.'
        ]
    })


def create_test_embedded_data():
    """Create test data with embeddings for testing"""
    df = create_test_csv_data()
    # Add mock embedding vectors
    df['title_vector'] = [str([0.1] * 1536), str([0.2] * 1536), str([0.3] * 1536)]
    df['answer_vector'] = [str([0.4] * 1536), str([0.5] * 1536), str([0.6] * 1536)]
    return df


def setup_test_environment():
    """Setup test environment variables"""
    test_env = {
        'AZURE_OPENAI_API_KEY': 'test_key_12345',
        'AZURE_OPENAI_ENDPOINT': 'https://test-resource.openai.azure.com/',
        'OPENAI_API_VERSION': '2024-12-01-preview',
        'OPENAI_MODEL': 'text-embedding-3-large'
    }
    return test_env


def create_mock_openai_client():
    """Create a mock OpenAI client for testing"""
    from unittest.mock import MagicMock

    mock_client = MagicMock()

    # Mock embedding response
    mock_embed_response = MagicMock()
    mock_embed_response.data = [MagicMock()]
    mock_embed_response.data[0].embedding = [0.1] * 1536
    mock_client.embeddings.create.return_value = mock_embed_response

    # Mock chat completion response
    mock_chat_response = MagicMock()
    mock_chat_response.choices = [MagicMock()]
    mock_chat_response.choices[0].message.content = "This is a test response from the AI assistant."
    mock_client.chat.completions.create.return_value = mock_chat_response

    return mock_client


def save_test_data_to_temp_file(data, filename='test_data.csv'):
    """Save test data to a temporary file and return the path"""
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, filename)
    data.to_csv(file_path, index=False)
    return file_path
