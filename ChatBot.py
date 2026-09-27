import os
import streamlit as st
from huggingface_hub import InferenceClient

st.set_page_config(
    page_title="NoteBot",
    page_icon="📚"
)

st.title("📚 NoteBot")

HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    st.error("HF_TOKEN not found")
    st.stop()

st.success("HF_TOKEN found!")

try:
    client = InferenceClient(
        token=HF_TOKEN,
        provider="auto"
    )

    st.write("Testing Hugging Face...")

    result = client.feature_extraction(
        "This is a test sentence.",
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    st.success("Hugging Face embedding API is working!")

    st.write("Embedding generated successfully.")

except Exception as e:
    st.error(f"Hugging Face embedding error: {e}")
