import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load Dataset
data = pd.read_csv(r"D:\Data Science\ML Model\Resume Analyser\Resume.csv")

# Remove unwanted column if present
if "Resume_html" in data.columns:
    data.drop("Resume_html", axis=1, inplace=True)

# Rename column
if "Resume_str" in data.columns:
    data.rename(
        columns={"Resume_str": "Resume"},
        inplace=True
    )

# Text Cleaning
def clean_resume(text):

    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)

    text = text.lower()

    text = re.sub(r"\s+", " ", text)

    return text.strip()

data["Resume"] = data["Resume"].apply(clean_resume)

# Features and Target
x = data["Resume"]
y = data["Category"]

# Label Encoding
encoder = LabelEncoder()

y = encoder.fit_transform(y)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2
)

x_vec = vectorizer.fit_transform(x)

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x_vec,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# SVM Model
svm_model = LinearSVC(
    C=1.0
)

svm_model.fit(
    x_train,
    y_train
)

# Prediction
pred = svm_model.predict(
    x_test
)

# Evaluation
print("\n===== SVM Results =====\n")

print(
    "Accuracy:",
    accuracy_score(
        y_test,
        pred
    )
)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        pred
    )
)

print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        y_test,
        pred
    )
)

# Save Artifacts
joblib.dump(
    svm_model,
    "resume_classifier.pkl"
)

joblib.dump(
    vectorizer,
    "tfidf_vectorizer.pkl"
)

joblib.dump(
    encoder,
    "label_encoder.pkl"
)

print("\nModel Saved Successfully!")