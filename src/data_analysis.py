import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("dataset/fake-news.csv")

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns)

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATES ==========")
print(df.duplicated().sum())

print("\n========== LABEL COUNTS ==========")
print(df["label"].value_counts())

# Remove duplicate rows
df = df.drop_duplicates()

print("\n========== SHAPE AFTER REMOVING DUPLICATES ==========")
print(df.shape)

# Plot label distribution
plt.figure(figsize=(6, 4))

sns.countplot(data=df, x="label")

plt.title("Fake vs Real News")
plt.xlabel("News Label")
plt.ylabel("Number of Articles")

plt.show()