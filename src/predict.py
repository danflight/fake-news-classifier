import joblib
from .preprocess import clean_text
import argparse
import os
import sys

def predict(text):
    """
    Predict whether a news article or headline is REAL or FAKE.

    Args:
        text (str): The input news text or headline to classify.

    Returns:
        str: "REAL" if the model predicts real news, "FAKE" otherwise.
    Raises:
        FileNotFoundError: If model or vectorizer files are missing.
    """
    model_path = "model.pkl"
    vectorizer_path = "vectorizer.pkl"
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("Model or vectorizer file not found. Please train the model first.")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    pred = model.predict(features)[0]
    return "REAL" if pred == 1 else "FAKE"


def predict_batch(file_path):
    """
    Predict labels for a batch of news texts from a file.

    Args:
        file_path (str): Path to a text file with one news article or headline per line.

    Returns:
        list of tuples: Each tuple contains (input_text, prediction or error message).
    """
    if not os.path.exists(file_path):
        print(f"Input file '{file_path}' not found.")
        sys.exit(1)
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    results = []
    for line in lines:
        try:
            result = predict(line)
        except Exception as e:
            result = f"ERROR: {str(e)}"
        results.append((line, result))
    return results


if __name__ == "__main__":
    """
    Command-line interface for predicting news labels.
    Use --text for a single prediction or --file for batch prediction.
    """
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Single news text to classify.")
    group.add_argument("--file", type=str, help="Path to file with one news text per line.")
    args = parser.parse_args()
    try:
        if args.text:
            print(predict(args.text))
        elif args.file:
            results = predict_batch(args.file)
            for text, label in results:
                print(f"{label}\t{text}")
    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        sys.exit(1)
