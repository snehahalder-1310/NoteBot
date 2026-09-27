import os
import streamlit as st
from PyPDF2 import PdfReader

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

file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if file is not None:

    st.success("PDF uploaded successfully!")

    try:
        pdf = PdfReader(file)

        st.write("Number of pages:", len(pdf.pages))

        text = ""

        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text

        if text.strip():
            st.success("PDF text extracted successfully!")
            st.write(text[:1000])
        else:
            st.warning("No readable text found in this PDF.")

    except Exception as e:
        st.error(f"PDF error: {e}")
