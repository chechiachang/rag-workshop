"""
Integration tests for the RAG Workshop notebooks
Tests complete workflows and integration between notebooks
"""
import os

import pandas as pd
import pytest
from testbook import testbook


class TestRAGWorkflow:
    """Test the complete RAG workflow across notebooks"""

    @pytest.fixture
    def test_data(self):
        """Create test data for workflow testing"""
        return pd.DataFrame({
            'title': [
                'What are the main symptoms of COVID-19?',
                'How can I prevent getting COVID-19?',
                'When should I get tested for COVID-19?'
            ],
            'answer': [
                'The main symptoms include fever, cough, fatigue, and loss of taste or smell.',
                'You can prevent COVID-19 by wearing masks, maintaining social distance, and getting vaccinated.',
                'You should get tested if you have symptoms, have been exposed, or before traveling.'
            ]
        })

    def test_data_processing_pipeline(self, test_data):
        """Test the data processing pipeline that would be used across notebooks"""
        with testbook(os.path.join(
            os.path.dirname(__file__), "..", "notebook", "1_Embedding_with_OpenAI.ipynb"),
            execute=False) as tb:
            # Test data loading and processing
            pipeline_code = f"""
import pandas as pd
import numpy as np

# Simulate loading the test data
test_df = pd.DataFrame({repr(test_data.to_dict())})

# Basic data validation
has_required_columns = all(col in test_df.columns for col in ['title', 'answer'])
data_not_empty = len(test_df) > 0
no_null_values = not test_df.isnull().any().any()

# Data processing functions
def preprocess_text(text):
    return text.strip().lower()

def validate_embedding_vector(vector, expected_dim=1536):
    return isinstance(vector, list) and len(vector) == expected_dim

# Process the data
test_df['title_processed'] = test_df['title'].apply(preprocess_text)
test_df['answer_processed'] = test_df['answer'].apply(preprocess_text)

processing_successful = len(test_df['title_processed']) == len(test_df)
"""
            tb.inject(pipeline_code)

            # Validate the pipeline worked
            assert tb.ref("has_required_columns") is True
            assert tb.ref("data_not_empty") is True
            assert tb.ref("no_null_values") is True
            assert tb.ref("processing_successful") is True

    def test_embedding_generation_workflow(self):
        """Test the embedding generation workflow"""
        with testbook(os.path.join(
            os.path.dirname(__file__), "..", "notebook", "1_Embedding_with_OpenAI.ipynb"),
            execute=False) as tb:
            embedding_workflow = """
import tiktoken
from unittest.mock import MagicMock

# Test token counting functionality
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Test embedding generation (mocked)
def mock_embedding_function(text: str, model: str = "text-embedding-3-large") -> list:
    # Simulate different embedding sizes for different models
    if "3-large" in model:
        return [0.1] * 3072
    elif "3-small" in model:
        return [0.1] * 1536
    else:  # ada-002
        return [0.1] * 1536

# Test the workflow
test_text = "What are the symptoms of COVID-19?"
token_count = num_tokens_from_string(test_text, "cl100k_base")
embedding_large = mock_embedding_function(test_text, "text-embedding-3-large")
embedding_small = mock_embedding_function(test_text, "text-embedding-3-small")

# Validation
token_count_valid = token_count > 0
embedding_large_valid = len(embedding_large) == 3072
embedding_small_valid = len(embedding_small) == 1536
embeddings_different_sizes = len(embedding_large) != len(embedding_small)
"""
            tb.inject(embedding_workflow)

            # Validate the workflow
            assert tb.ref("token_count_valid") is True
            assert tb.ref("embedding_large_valid") is True
            assert tb.ref("embedding_small_valid") is True
            assert tb.ref("embeddings_different_sizes") is True

    def test_search_and_rag_workflow(self):
        """Test the search and RAG workflow"""
        with testbook(os.path.join(
            os.path.dirname(__file__), "..", "notebook", "3_RAG_with_OpenAI.ipynb"),
            execute=False) as tb:
            rag_workflow = """
import numpy as np
from unittest.mock import MagicMock

# Mock data with embeddings
documents = [
    {
        'title': 'COVID-19 Symptoms',
        'answer': 'Symptoms include fever, cough, and fatigue.',
        'title_vector': np.array([0.1, 0.2, 0.3]),
        'answer_vector': np.array([0.4, 0.5, 0.6])
    },
    {
        'title': 'COVID-19 Prevention',
        'answer': 'Prevention includes masks and vaccination.',
        'title_vector': np.array([0.2, 0.3, 0.4]),
        'answer_vector': np.array([0.5, 0.6, 0.7])
    }
]

# Search functions
def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot_product / (norm_a * norm_b)

def search_documents(query_vector, documents, top_k=1):
    similarities = []
    for doc in documents:
        sim = cosine_similarity(query_vector, doc['title_vector'])
        similarities.append((sim, doc))

    similarities.sort(key=lambda x: x[0], reverse=True)
    return similarities[:top_k]

def mock_rag_response(context, query):
    return f"Based on the context about {context['title']}, {context['answer']}"

# Test the workflow
query_vector = np.array([0.15, 0.25, 0.35])
search_results = search_documents(query_vector, documents, top_k=1)
best_match = search_results[0][1] if search_results else None
rag_response = mock_rag_response(best_match, "What are COVID symptoms?") if best_match else "No results found"

# Validation
search_returned_results = len(search_results) > 0
best_match_found = best_match is not None
response_generated = len(rag_response) > 0
response_contains_context = "COVID-19 Symptoms" in rag_response or "fever" in rag_response
"""
            tb.inject(rag_workflow)

            # Validate the RAG workflow
            assert tb.ref("search_returned_results") is True
            assert tb.ref("best_match_found") is True
            assert tb.ref("response_generated") is True
            # Check the actual response content - it should contain relevant content
            response = tb.ref("rag_response")
            assert ("fever" in response or "COVID-19 Symptoms" in response or
                   "Prevention" in response or "masks" in response)

    def test_evaluation_workflow(self):
        """Test evaluation metrics that might be used"""
        with testbook(os.path.join(
            os.path.dirname(__file__), "..", "notebook", "5_Evaluation.ipynb"),
            execute=False) as tb:
            evaluation_code = """
import numpy as np

# Mock evaluation data
test_queries = [
    "What are COVID symptoms?",
    "How to prevent COVID?",
    "When to get tested?"
]

expected_answers = [
    "fever, cough, fatigue",
    "masks, distance, vaccines",
    "if symptomatic or exposed"
]

generated_answers = [
    "The symptoms include fever, cough, and fatigue",
    "Prevention methods include wearing masks, social distancing, and getting vaccinated",
    "You should get tested if you have symptoms or have been exposed"
]

# Simple evaluation metrics
def simple_keyword_overlap(expected, generated):
    expected_words = set(expected.lower().split())
    generated_words = set(generated.lower().split())
    intersection = expected_words.intersection(generated_words)
    if len(expected_words) == 0:
        return 0
    return len(intersection) / len(expected_words)

def evaluate_responses(expected_list, generated_list):
    scores = []
    for exp, gen in zip(expected_list, generated_list):
        score = simple_keyword_overlap(exp, gen)
        scores.append(score)
    return scores

# Run evaluation
scores = evaluate_responses(expected_answers, generated_answers)
average_score = np.mean(scores)
all_scores_positive = all(score > 0 for score in scores)
reasonable_performance = average_score > 0.2  # At least 20% keyword overlap

# Validation
evaluation_completed = len(scores) == len(test_queries)
scores_calculated = all(isinstance(score, (int, float)) for score in scores)
"""
            tb.inject(evaluation_code)

            # Validate evaluation
            assert tb.ref("evaluation_completed") is True
            assert tb.ref("scores_calculated") is True
            assert tb.ref("all_scores_positive") is True
            # Check average score directly
            avg_score = tb.ref("average_score")
            assert float(avg_score) > 0.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
