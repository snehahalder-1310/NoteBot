import streamlit as st

st.set_page_config(
    page_title="NoteBot",
    page_icon="📚"
)

st.title("📚 NoteBot")

st.success("Streamlit interface is working!")

st.write("If you can see this page, Render and Streamlit are working correctly.")

st.header("Test Section")

name = st.text_input("Enter your name")

if name:
    st.write(f"Hello, {name}!")

st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)
