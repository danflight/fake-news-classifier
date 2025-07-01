# Fake News Classifier

A small machine learning project that classifies news articles as **real** or **fake** based on their content.

## Features
- End-to-end ML pipeline using `scikit-learn`
- Text cleaning and vectorization
- Model training and evaluation
- Interactive web demo with Streamlit or Gradio

## File Structure
- `notebooks/`: EDA and prototyping
- `src/`: Preprocessing, training, prediction scripts
- `app.py`: Interactive app interface
- `data/`: Dataset storage (not committed to Git)
- `requirements.txt`: Python dependencies

## Dataset
Uses the [Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset).

## To Run
```bash
# Preprocess and train
python src/model.py

# Predict
python src/predict.py --text "Your news headline here"

# Launch demo
streamlit run app.py

