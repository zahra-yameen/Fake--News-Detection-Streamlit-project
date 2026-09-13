import joblib


# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load(
    "../models/best_model.pkl"
)

tfidf = joblib.load(
    "../models/tfidf_vectorizer.pkl"
)


# ==========================================
# GET NEWS FROM USER
# ==========================================

print("===================================")
print("FAKE NEWS DETECTION")
print("===================================")

news = input(
    "\nEnter news article: "
)


# ==========================================
# TRANSFORM TEXT
# ==========================================

news_tfidf = tfidf.transform(
    [news]
)


# ==========================================
# PREDICT
# ==========================================

prediction = model.predict(
    news_tfidf
)[0]


# ==========================================
# RESULT
# ==========================================

if prediction == 0:

    print("\n🚨 Prediction: FAKE NEWS")

else:

    print("\n✅ Prediction: REAL NEWS")