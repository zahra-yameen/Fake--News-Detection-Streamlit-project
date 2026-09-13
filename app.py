import streamlit as st
import pandas as pd
import joblib

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    model = joblib.load(
        "models/best_model.pkl"
    )

    tfidf = joblib.load(
        "models/tfidf_vectorizer.pkl"
    )

    return model, tfidf


model, tfidf = load_model()

# ==========================================
# TITLE
# ==========================================

st.title(
    "📰 Fake News Detection System"
)

st.write(
    "Machine Learning based news text classification."
)

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title(
    "Navigation"
)

page = st.sidebar.radio(
    "Choose a page:",
    [
        "Single Prediction",
        "Batch Prediction",
        "About"
    ]
)

# ==========================================
# SINGLE PREDICTION
# ==========================================

if page == "Single Prediction":

    st.header(
        "🔍 Check News"
    )

    news_text = st.text_area(
        "Enter news article:",
        height=250,
        placeholder="Paste the news article here..."
    )

    if st.button(
            "Predict",
            type="primary"
    ):

        if not news_text.strip():

            st.warning(
                "Please enter a news article."
            )

        else:

            # Convert text
            text_vector = tfidf.transform(
                [news_text]
            )

            # Prediction
            prediction = model.predict(
                text_vector
            )[0]

            if prediction == 0:

                st.error(
                    "🚨 Model Prediction: FAKE NEWS"
                )

            else:

                st.success(
                    "✅ Model Prediction: REAL NEWS"
                )


# ==========================================
# BATCH PREDICTION
# ==========================================

elif page == "Batch Prediction":

    st.header(
        "📁 Batch News Prediction"
    )

    st.write(
        "Upload a CSV containing a 'text' column."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(
            uploaded_file
        )

        st.subheader(
            "Uploaded Data"
        )

        st.dataframe(
            df.head()
        )

        if "text" not in df.columns:

            st.error(
                "CSV must contain a 'text' column."
            )

        else:

            text_vectors = tfidf.transform(
                df["text"].fillna("")
            )

            predictions = model.predict(
                text_vectors
            )

            df["Prediction"] = [
                "REAL" if value == 1 else "FAKE"
                for value in predictions
            ]

            st.subheader(
                "Predictions"
            )

            st.dataframe(
                df
            )

            csv_data = df.to_csv(
                index=False
            )

            st.download_button(
                "⬇️ Download Predictions",
                csv_data,
                "fake_news_predictions.csv",
                "text/csv"
            )


# ==========================================
# ABOUT
# ==========================================

elif page == "About":

    st.header(
        "ℹ️ About This Project"
    )

    st.write(
        """
        This project uses machine learning and natural
        language processing to classify news articles
        based on patterns learned from a labeled dataset.
        """
    )

    st.write(
        """
        Technologies:

        • Python
        • Pandas
        • NumPy
        • scikit-learn
        • TF-IDF
        • Logistic Regression
        • Naive Bayes
        • Linear SVM
        • Random Forest
        • Streamlit
        """
    )

    st.warning(
        "This system is a text-classification model and "
        "should not be treated as an authoritative fact-checker."
    )