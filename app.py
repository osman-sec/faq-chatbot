import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# App Configuration
# -------------------------
st.set_page_config(
    page_title="FAQ Chatbot",
    page_icon="💬",
    layout="centered",
)

st.title("💬 FAQ Chatbot")
st.write("Ask me a question about AI, ML, or Python!")

# -------------------------
# FAQ Dataset
# -------------------------
faqs = {
    "What is AI?": "AI stands for Artificial Intelligence. It enables machines to mimic human intelligence.",
    "What is Machine Learning?": "ML is a subset of AI that uses algorithms to learn from data.",
    "What is Python?": "Python is a high-level programming language popular for AI and ML.",
}

# Preprocessing
questions = list(faqs.keys())
answers = list(faqs.values())

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# -------------------------
# User Input
# -------------------------
user_input = st.text_input("👉 Type your question:")

if st.button("Get Answer"):
    if user_input.strip():
        # Transform user input
        user_vec = vectorizer.transform([user_input])
        sim = cosine_similarity(user_vec, X)
        best_match = sim.argmax()
        
        # Show result
        st.success(answers[best_match])
    else:
        st.warning("⚠️ Please enter a question!")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("🚀 Built with Streamlit & Scikit-learn (TF-IDF + Cosine Similarity)")
