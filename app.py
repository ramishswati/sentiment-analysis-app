from transformers import pipeline
import streamlit as st

# Load Hugging Face sentiment model
sentiment_pipeline = pipeline("sentiment-analysis")

def predict_sentiment(text):
    result = sentiment_pipeline(text)[0]
    return result['label'], result['score']

# Streamlit UI
st.title("Sentiment Analysis Web App")
user_input = st.text_area("Enter text here:")

if st.button("Analyze"):
    if user_input.strip() != "":
        label, score = predict_sentiment(user_input)
        st.write(f"Sentiment: {label}")
        st.write(f"Confidence: {score:.2f}")
    else:
        st.warning("Please enter some text!")
