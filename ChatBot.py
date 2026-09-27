import os
import streamlit as st

st.set_page_config(
    page_title="NoteBot",
    page_icon="📚"
)

st.title("📚 NoteBot")

HF_TOKEN = os.environ.get("HF_TOKEN")

if HF_TOKEN:
    st.success("Hugging Face token found!")
else:
    st.error("Hugging Face token NOT found!")

st.write("Application is running successfully.")
