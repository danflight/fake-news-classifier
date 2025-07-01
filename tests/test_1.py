import pytest

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import clean_text, vectorize
from src.predict import predict

# Test clean_text basic functionality
def test_clean_text_removes_urls():
    text = "Check this out: https://example.com"
    cleaned = clean_text(text)
    assert "http" not in cleaned and "example" not in cleaned

def test_clean_text_lowercase():
    text = "This Is A Test"
    cleaned = clean_text(text)
    assert cleaned == cleaned.lower()

def test_clean_text_removes_stopwords():
    text = "This is a simple test of the stopword removal."
    cleaned = clean_text(text)
    assert "is" not in cleaned and "the" not in cleaned

def test_vectorize_output_shape():
    texts = ["Fake news here", "Real news there"]
    features, vectorizer = vectorize(texts)
    assert features.shape[0] == 2
    assert hasattr(vectorizer, 'transform')

# Test predict (requires model.pkl and vectorizer.pkl to exist)
def test_predict_runs(monkeypatch):
    # Monkeypatch joblib.load to avoid loading real model
    import joblib
    class DummyModel:
        def predict(self, X):
            return [1]
    class DummyVectorizer:
        def transform(self, X):
            return [[0, 1]]
    monkeypatch.setattr(joblib, 'load', lambda path: DummyModel() if 'model' in path else DummyVectorizer())
    result = predict("Some news text")
    assert result in ["REAL", "FAKE"]
