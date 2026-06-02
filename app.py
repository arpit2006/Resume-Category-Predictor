import streamlit as st
import joblib
import re
import pdfplumber

# Load saved artifacts
model = joblib.load("resume_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
encoder = joblib.load("label_encoder.pkl")

# Resume cleaning
def clean_resume(text):

    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)

    text = text.lower()

    text = re.sub(r"\s+", " ", text)

    return text.strip()

# PDF text extraction
def extract_text(pdf_file):

    text = ""

    with pdfplumber.open(pdf_file) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text

    return text

# UI
st.set_page_config(
    page_title="Resume Category Predictor",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Resume Category Predictor")

st.markdown(
    "Upload a resume and predict its professional category using Machine Learning."
)

uploaded_file = st.file_uploader(
    "Upload Resume",
    type=["pdf"]
)

if uploaded_file:

    with st.spinner("Analyzing Resume..."):

        resume_text = extract_text(
            uploaded_file
        )

        cleaned_text = clean_resume(
            resume_text
        )

        vectorized_text = vectorizer.transform(
            [cleaned_text]
        )

        prediction = model.predict(
            vectorized_text
        )

        category = encoder.inverse_transform(
            prediction
        )[0]

    st.success(
        f"Predicted Category: {category}"
    )

    st.subheader("Extracted Resume Text")

    st.text_area(
        "",
        resume_text[:3000],
        height=300
    )