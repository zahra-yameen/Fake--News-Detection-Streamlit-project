from pathlib import Path
import pandas as pd
import re

# ==========================================
# 1. SET PROJECT PATHS
# ==========================================

# Get the main project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset paths
INPUT_FILE = BASE_DIR / "dataset" / "fake-news.csv"
OUTPUT_FILE = BASE_DIR / "dataset" / "processed_fake-news.csv"


# ==========================================
# 2. LOAD DATASET
# ==========================================

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset loaded successfully!")
print("Original shape:", df.shape)


# ==========================================
# 3. CHECK REQUIRED COLUMNS
# ==========================================

required_columns = ["text", "label"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Required column '{column}' was not found in the dataset."
        )


# ==========================================
# 4. KEEP IMPORTANT COLUMNS
# ==========================================

# Keep title if it exists
if "title" in df.columns:
    df = df[["title", "text", "label"]]
else:
    df = df[["text", "label"]]


# ==========================================
# 5. REMOVE MISSING VALUES
# ==========================================

print("\nRemoving missing values...")

df = df.dropna(subset=["text", "label"])

print("Shape after removing missing values:", df.shape)


# ==========================================
# 6. REMOVE DUPLICATE NEWS ARTICLES
# ==========================================

print("\nRemoving duplicate articles...")

df = df.drop_duplicates(subset=["text"])

print("Shape after removing duplicates:", df.shape)


# ==========================================
# 7. TEXT CLEANING FUNCTION
# ==========================================

def clean_text(text):
    """
    Clean news article text for machine learning.
    """

    # Convert to string
    text = str(text)

    # Convert text to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ==========================================
# 8. CLEAN NEWS TEXT
# ==========================================

print("\nCleaning news text...")

df["clean_text"] = df["text"].apply(clean_text)


# ==========================================
# 9. CLEAN TITLE IF AVAILABLE
# ==========================================

if "title" in df.columns:
    df["title"] = df["title"].fillna("")
    df["clean_title"] = df["title"].apply(clean_text)


# ==========================================
# 10. REMOVE EMPTY TEXT
# ==========================================

df = df[df["clean_text"].str.len() > 0]


# ==========================================
# 11. CLEAN LABELS
# ==========================================

df["label"] = df["label"].astype(str).str.strip().str.upper()


# ==========================================
# 12. CONVERT NUMERIC LABELS
# ==========================================

print("\nConverting labels...")

# Dataset labels:
# 0 = Fake News
# 1 = Real News

df["label"] = df["label"].astype(int)

# Make sure only 0 and 1 are kept
df = df[df["label"].isin([0, 1])]

print("\nLabel distribution:")
print(df["label"].value_counts())

# Convert numeric labels into readable names
df["label_name"] = df["label"].map({
    0: "FAKE",
    1: "REAL"
})

print("\nLabels AFTER filtering:")
print(df["label"].value_counts(dropna=False))


# ==========================================
# 13. RESET INDEX
# ==========================================

df = df.reset_index(drop=True)


# ==========================================
# 14. SAVE PROCESSED DATASET
# ==========================================

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

print("\n==========================================")
print("PREPROCESSING COMPLETED SUCCESSFULLY!")
print("==========================================")

print("\nFinal dataset shape:", df.shape)

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nFinal columns:")
print(df.columns.tolist())

print("\nProcessed dataset saved at:")
print(OUTPUT_FILE)