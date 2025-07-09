import streamlit as st
import pandas as pd 
import numpy as np 
import pickle 
import os

model = pickle.load(open(os.path.join(os.path.dirname(__file__), 'fake_news_model.pkl'), 'rb'))
vectorizer = pickle.load(open(os.path.join(os.path.dirname(__file__), 'tfidf_vectorizer.pkl'), 'rb'))

st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake vs Real News Detector")
st.markdown("Made by **Utkarsh ❤️**")


user_input = st.text_area("Enter the News Article :")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        Vec_transform = vectorizer.transform([user_input])
        prediction = model.predict(Vec_transform)

        if prediction[0] == 1 :
            st.success("✅ This news article is *Real*.")
        else:
            st.error("❌ This news article is *Fake*.")

st.markdown("---")
st.markdown(
    "<div style='text-align: center;'>App created with ❤️ by <b>Pandit Utkarsh</b></div>",
    unsafe_allow_html=True
)
