# Fake News Classifier

A machine learning project that classifies news articles or headlines as **real** or **fake** using natural language processing and supervised learning.

**NOTE**: This is not a perfect model, and works best for U.S. politcs headlines, as it is trained on such data.

## Features
- End-to-end ML pipeline with `scikit-learn`
- Advanced text cleaning and lemmatization
- TF-IDF vectorization
- Supports both Logistic Regression and Neural Network classifiers (requires more processing power and memory)
- Model evaluation with precision, recall, and F1-score
- Interactive web demo using Streamlit
- Batch and single prediction from the command line

## File Structure
- `src/` — Preprocessing, training, and prediction scripts
  - `preprocess.py` — Text cleaning and vectorization
  - `model.py` — Model training and evaluation
  - `predict.py` — Command-line prediction
- `app.py` — Streamlit web app interface
- `data/` — Place your dataset CSVs here (`True.csv`, `Fake.csv`)
- `requirements.txt` — Python dependencies
- `classification_report.txt` — Model evaluation metrics

## Dataset
Uses the [Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset) from Kaggle. Place `True.csv` and `Fake.csv` in the `data/` directory. Each file should have at least `title` and `text` columns.

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
- For logistic regression (default):
  ```bash
  python src/model.py
  ```
- For neural network:
  ```bash
  python src/model.py --nn
  ```

### 3. Predict from the command line
- Single prediction:
  ```bash
  python src/predict.py --text "Your news headline or article here"
  ```
- Batch prediction:
  ```bash
  python src/predict.py --file test_news.txt
  ```

### 4. Launch the web app
```bash
streamlit run app.py
```

## Notes
- By default, both the news `title` and `text` are used for classification.
- You can switch between Logistic Regression and Neural Network by using the `--nn` flag when running `model.py`.
- Model performance metrics are saved in `classification_report.txt` after training.

