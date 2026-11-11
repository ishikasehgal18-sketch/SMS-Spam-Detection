
import streamlit as st
import joblib

# Load trained model
model = joblib.load("spam_model.pkl")

# Page configuration
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
.title {
    text-align: center;
    color: #243b64;
    font-size: 36px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #555;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# Heading
st.markdown(
    '<p class="title">📩 SMS Spam Detection System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Machine Learning Based Message Classification</p>',
    unsafe_allow_html=True
)

st.divider()

st.write("### 🔍 Check Your Message")
st.write("Enter any message below to identify whether it is Spam or Not Spam.")

# Input
message = st.text_area(
    "Your Message",
    placeholder="Type or paste your message here...",
    height=150
)

# Prediction
if st.button("🔎 Predict Message", use_container_width=True):

    if message.strip() == "":
        st.warning("Please enter a message first.")

    else:
        prediction = model.predict([message])[0]

        if prediction == "spam":
            st.error("⚠️ This message is SPAM!")
        else:
            st.success("✅ This message is NOT SPAM!")

st.divider()

st.caption("Developed using Python, Scikit-learn and Streamlit")


