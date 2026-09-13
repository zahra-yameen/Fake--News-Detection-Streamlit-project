
from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "dataset" / "processed_fake-news.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD DATA
# ==========================================

print("Loading processed dataset...")

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}\n"
        "Please check the filename inside the dataset folder."
    )

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ==========================================
# 3. CHECK REQUIRED COLUMNS
# ==========================================

if "clean_text" not in df.columns:
    raise ValueError(
        "Column 'clean_text' not found in processed dataset."
    )

if "label" not in df.columns:
    raise ValueError(
        "Column 'label' not found in processed dataset."
    )


# ==========================================
# 4. REMOVE MISSING VALUES
# ==========================================

print("\nRemoving missing values...")

df = df.dropna(subset=["clean_text", "label"]).copy()

print("Shape after removing missing values:", df.shape)


# ==========================================
# 5. CONVERT LABELS
# ==========================================

print("\nConverting labels...")

# Convert labels safely.
# Supports both numeric labels and text labels.

def convert_label(value):

    value = str(value).strip().upper()

    if value in ["FAKE", "0"]:
        return 0

    elif value in ["REAL", "1"]:
        return 1

    else:
        return None


df["label"] = df["label"].apply(convert_label)

# Remove invalid labels
df = df.dropna(subset=["label"]).copy()

df["label"] = df["label"].astype(int)


# ==========================================
# 6. CHECK LABEL DISTRIBUTION
# ==========================================

print("\nLabel distribution:")

print(df["label"].value_counts())

print("\nUnique labels:")

print(df["label"].unique())


# Stop if dataset becomes empty
if len(df) == 0:
    raise ValueError(
        "Dataset contains 0 rows after label conversion. "
        "Please check your label values."
    )

if df["label"].nunique() < 2:
    raise ValueError(
        "Dataset must contain both classes: 0 and 1."
    )


# ==========================================
# 7. INPUT AND TARGET
# ==========================================

X = df["clean_text"]
y = df["label"]


# ==========================================
# 8. TRAIN / TEST SPLIT
# ==========================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 9. TF-IDF VECTORIZATION
# ==========================================

print("\nCreating TF-IDF features...")

tfidf = TfidfVectorizer(
    max_features=50000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = tfidf.fit_transform(X_train)

X_test_tfidf = tfidf.transform(X_test)

print("TF-IDF training shape:", X_train_tfidf.shape)

print("TF-IDF testing shape:", X_test_tfidf.shape)


# ==========================================
# 10. DEFINE MODELS
# ==========================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Naive Bayes": MultinomialNB(),

    "Linear SVM": LinearSVC(
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

}


# ==========================================
# 11. TRAIN AND COMPARE MODELS
# ==========================================

results = {}

best_model = None

best_model_name = None

best_f1 = -1


for name, model in models.items():

    print("\n================================")
    print("Training:", name)
    print("================================")

    model.fit(
        X_train_tfidf,
        y_train
    )

    predictions = model.predict(
        X_test_tfidf
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    results[name] = {

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1": f1

    }

    print("Accuracy :", round(accuracy, 4))

    print("Precision:", round(precision, 4))

    print("Recall   :", round(recall, 4))

    print("F1 Score :", round(f1, 4))


    # Find best model

    if f1 > best_f1:

        best_f1 = f1

        best_model = model

        best_model_name = name


# ==========================================
# 12. DISPLAY BEST MODEL
# ==========================================

print("\n================================")
print("BEST MODEL")
print("================================")

print("Model:", best_model_name)

print("F1 Score:", round(best_f1, 4))


# ==========================================
# 13. SAVE BEST MODEL
# ==========================================

model_path = MODEL_DIR / "best_model.pkl"

joblib.dump(
    best_model,
    model_path
)

print("\nBest model saved successfully!")

print(model_path)


# ==========================================
# 14. SAVE TF-IDF VECTORIZER
# ==========================================

vectorizer_path = MODEL_DIR / "tfidf_vectorizer.pkl"

joblib.dump(
    tfidf,
    vectorizer_path
)

print("\nTF-IDF vectorizer saved successfully!")

print(vectorizer_path)


# ==========================================
# 15. SAVE RESULTS
# ==========================================

results_df = pd.DataFrame(results).T

results_path = MODEL_DIR / "model_results.csv"

results_df.to_csv(results_path)

print("\nModel comparison saved successfully!")

print(results_path)


# ==========================================
# 16. FINAL MESSAGE
# ==========================================

print("\n================================")
print("TRAINING COMPLETED!")
print("================================")

print("\nYour models folder now contains:")

print("1. best_model.pkl")

print("2. tfidf_vectorizer.pkl")

print("3. model_results.csv")