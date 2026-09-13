import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("dataset/processed_fake-news.csv")

df = df.dropna(
    subset=["clean_text", "label"]
)

df = df.drop_duplicates()

# ==========================================
# LABEL CONVERSION
# ==========================================

df["label"] = df["label"].astype(int)

print("\nLabel distribution:")
print(df["label"].value_counts())


X = df["clean_text"]
y = df["label"]


# ==========================================
# SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# LOAD TF-IDF
# ==========================================

tfidf = joblib.load(
    "../models/tfidf_vectorizer.pkl"
)


X_test_tfidf = tfidf.transform(
    X_test
)


# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load(
    "../models/best_model.pkl"
)


# ==========================================
# PREDICTION
# ==========================================

predictions = model.predict(
    X_test_tfidf
)


# ==========================================
# METRICS
# ==========================================

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


print("\n========== FINAL RESULTS ==========")

print(
    "Accuracy:",
    round(accuracy, 4)
)

print(
    "Precision:",
    round(precision, 4)
)

print(
    "Recall:",
    round(recall, 4)
)

print(
    "F1 Score:",
    round(f1, 4)
)


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=[
            "FAKE",
            "REAL"
        ],
        zero_division=0
    )
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
)


plt.figure(
    figsize=(6, 5)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=[
        "FAKE",
        "REAL"
    ],
    yticklabels=[
        "FAKE",
        "REAL"
    ]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Fake News Detection - Confusion Matrix")

plt.show()