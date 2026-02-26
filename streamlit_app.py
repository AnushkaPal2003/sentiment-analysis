import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

st.title("🎬 Sentiment Analysis App")
st.write("Enter a movie review and get sentiment prediction.")

user_input = st.text_area("Enter Review")

if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        transformed = vectorizer.transform([user_input])
        prediction = model.predict(transformed)[0]

        if prediction == 1:
            st.success("Positive Sentiment 😊")
        else:
            st.error("Negative Sentiment 😞")