import streamlit as st
import joblib

# 🔹 Load model & vectorizer
model = joblib.load(r"D:\sriram\Projects\Fake news detection\models\fake_news_model.pkl")
vectorizer = joblib.load(r"D:\sriram\Projects\Fake news detection\models\tfidf_vectorizer.pkl")

# 🔹 Streamlit UI
st.title("📰 Fake News Detection App")
st.write("Paste any news article or headline below and check if it's **Fake** or **Real**.")

# Text input
user_input = st.text_area("Enter news text here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        # Transform input
        input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(input_tfidf)[0]

        if prediction == 1:
            st.error("❌ This looks like **FAKE News**")
        else:
            st.success("✅ This looks like **REAL News**")
