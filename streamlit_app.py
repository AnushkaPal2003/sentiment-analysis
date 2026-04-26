import streamlit as st
import joblib

model = joblib.load("models/model.pkl")

st.title("🎬 Sentiment Analysis App")
st.write("Enter a movie review and get sentiment prediction.")

user_input = st.text_area("Enter Review")

if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        prediction = model.predict([user_input])[0]

        if prediction == 1:
            st.success("Positive Sentiment 😊")
        else:
            st.error("Negative Sentiment 😞")