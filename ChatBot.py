import streamlit as st
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="NoteBot",
    page_icon="📚"
)

st.title("📚 NoteBot")

st.write("Loading embedding model...")

try:

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    st.success("Embedding model loaded successfully!")

    text = st.text_input(
        "Enter some text to test:"
    )

    if text:

        embedding = model.encode(
            text,
            convert_to_numpy=True
        )

        st.success("Embedding generated successfully!")

        st.write(
            "Embedding size:",
            len(embedding)
        )

except Exception as e:

    st.error(
        f"Embedding model error: {e}"
    )
