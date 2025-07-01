import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import joblib
import os
from datetime import datetime
import csv
import argparse

from preprocess import clean_text, vectorize

# Configurable parameters
REAL_PATH = "data/True.csv"
FAKE_PATH = "data/Fake.csv"
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
REPORT_PATH = "classification_report.txt"
HISTORY_PATH = "utils/metrics_history.csv"

# Argument parser for model selection
parser = argparse.ArgumentParser(description="Train fake news classifier.")
parser.add_argument('--nn', action='store_true', help='Use neural network (MLPClassifier) instead of logistic regression')
args = parser.parse_args()
use_NN = args.nn

# Load and validate data
if not os.path.exists(REAL_PATH) or not os.path.exists(FAKE_PATH):
    raise FileNotFoundError(f"Data files not found: {REAL_PATH}, {FAKE_PATH}")

real_df = pd.read_csv(REAL_PATH)
fake_df = pd.read_csv(FAKE_PATH)

# avoid imbalanced dataset
min_count = min(real_df.shape[0], fake_df.shape[0])
real_df = real_df.sample(n=min_count, random_state=42)
fake_df = fake_df.sample(n=min_count, random_state=42)

# Concatenate title and text for richer features
for df_ in [real_df, fake_df]:
    df_['title'] = df_['title'].fillna('')
    df_['text'] = df_['title'] + ' ' + df_['text'].fillna('')
    # print(df_['text'].value_counts())
    # print(df_['title'].value_counts())
if 'text' not in real_df.columns or 'text' not in fake_df.columns:
    raise ValueError("Both CSVs must contain a 'text' column.")

real_df = real_df.dropna(subset=['text'])
fake_df = fake_df.dropna(subset=['text'])

real_df['label'] = 'REAL'
fake_df['label'] = 'FAKE'

# Load user feedback if available
feedback_file = "user_feedback.csv"
if os.path.exists(feedback_file):
    feedback_df = pd.read_csv(feedback_file)

    # Upsample feedback to give it more influence
    if not feedback_df.empty:
        feedback_df = pd.concat([feedback_df]*10, ignore_index=True)  # Use a higher number for more weight

    # Only proceed if required columns exist
    if set(["text", "user_label"]).issubset(feedback_df.columns):
        feedback_df = feedback_df.rename(columns={"user_label": "label"})
        feedback_df['title'] = ''  # For compatibility with the rest of the pipeline
        # Only keep necessary columns
        feedback_df = feedback_df[["text", "label", "title"]]
        # Optionally, drop duplicates
        feedback_df = feedback_df.drop_duplicates(subset=["text", "label"])
    else:
        print("Warning: user_feedback.csv missing required columns. Skipping feedback data.")
        feedback_df = pd.DataFrame(columns=["text", "label", "title"])
else:
    feedback_df = pd.DataFrame(columns=["text", "label", "title"])

# Combine and shuffle
df = pd.concat([real_df, fake_df, feedback_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

# Preprocess
df['clean'] = df['text'].apply(clean_text)
X_features, vectorizer = vectorize(df['clean'])
y = df['label'].map({"FAKE": 0, "REAL": 1})

# Split
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42, stratify=y)

# Model selection
if use_NN:
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
else:
    model = LogisticRegression(max_iter=1000)

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
report_dict = classification_report(y_test, y_pred, target_names=['FAKE', 'REAL'], output_dict=True)
report = classification_report(y_test, y_pred, target_names=['FAKE', 'REAL'])
print(report)
with open(REPORT_PATH, 'w') as f:
    f.write(report)




# Save metrics history
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
model_type = "NN" if use_NN else "LogReg"
row = {
    'datetime': now,
    'model': model_type,
    'accuracy': report_dict['accuracy'],
    'f1_fake': report_dict['FAKE']['f1-score'],
    'f1_real': report_dict['REAL']['f1-score'],
    'precision_fake': report_dict['FAKE']['precision'],
    'precision_real': report_dict['REAL']['precision'],
    'recall_fake': report_dict['FAKE']['recall'],
    'recall_real': report_dict['REAL']['recall'],
    'support_fake': report_dict['FAKE']['support'],
    'support_real': report_dict['REAL']['support']
}
header = list(row.keys())
write_header = not os.path.exists(HISTORY_PATH)
with open(HISTORY_PATH, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=header)
    if write_header:
        writer.writeheader()
    writer.writerow(row)

# Save model and vectorizer
joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)

# Plot metrics history
try:
    import matplotlib.pyplot as plt
    hist_df = pd.read_csv(HISTORY_PATH)
    plt.figure(figsize=(10,6))
    plt.plot(hist_df['datetime'], hist_df['accuracy'], marker='o', label='Accuracy')
    plt.plot(hist_df['datetime'], hist_df['f1_fake'], marker='x', label='F1 FAKE')
    plt.plot(hist_df['datetime'], hist_df['f1_real'], marker='x', label='F1 REAL')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Run Time')
    plt.ylabel('Score')
    plt.title('Model Accuracy and F1-score History')
    plt.legend()
    plt.tight_layout()
    plt.savefig('utils/metrics_history.png')
    plt.close()
except Exception as e:
    print(f"Could not plot metrics history: {e}")
