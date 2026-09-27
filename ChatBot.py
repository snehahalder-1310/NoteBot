import os
import streamlit as st
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="NoteBot",
    page_icon="📚",
    layout="wide"
)


# =========================================================
# HUGGING FACE TOKEN
# =========================================================

HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    st.error("Hugging Face token is not configured.")
    st.stop()


# =========================================================
# TITLE
# =========================================================

st.title("📚 NoteBot")
st.write("Upload your PDF notes and ask questions about them.")


# =========================================================
# LOAD EMBEDDING MODEL ONLY WHEN NEEDED
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


# =========================================================
# CUSTOM HUGGING FACE EMBEDDINGS
# =========================================================

class HuggingFaceEmbeddings(Embeddings):

    def embed_documents(self, texts):

        model = load_embedding_model()

        embeddings = model.encode(
            texts,
            convert_to_numpy=True
        )

        return embeddings.tolist()


    def embed_query(self, text):

        model = load_embedding_model()

        embedding = model.encode(
            text,
            convert_to_numpy=True
        )

        return embedding.tolist()


embeddings = HuggingFaceEmbeddings()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("My Notes")

    file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"]
    )


# =========================================================
# PDF PROCESSING
# =========================================================

if file is not None:

    # -----------------------------------------------------
    # READ PDF
    # -----------------------------------------------------

    try:

        my_pdf = PdfReader(file)

        text = ""

        for page in my_pdf.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text


        # -------------------------------------------------
        # CHECK PDF TEXT
        # -------------------------------------------------

        if not text.strip():

            st.error(
                "No readable text was found in this PDF."
            )

            st.stop()


        # -------------------------------------------------
        # SPLIT TEXT INTO CHUNKS
        # -------------------------------------------------

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            length_function=len
        )

        chunks = splitter.split_text(text)


        if not chunks:

            st.error(
                "Could not create text chunks from the PDF."
            )

            st.stop()


        # -------------------------------------------------
        # CREATE FAISS VECTOR DATABASE
        # -------------------------------------------------

        with st.spinner(
            "Processing your PDF..."
        ):

            vector_store = FAISS.from_texts(
                chunks,
                embeddings
            )


        st.success(
            "PDF uploaded successfully!"
        )


        # =================================================
        # QUESTION INPUT
        # =================================================

        user_query = st.text_input(
            "Ask a question about your notes:"
        )


        if user_query:

            # ---------------------------------------------
            # SEMANTIC SEARCH
            # ---------------------------------------------

            with st.spinner(
                "Searching your notes..."
            ):

                matching_chunks = vector_store.similarity_search(
                    user_query,
                    k=3
                )


            # ---------------------------------------------
            # CREATE CONTEXT
            # ---------------------------------------------

            context = "\n\n".join(
                document.page_content
                for document in matching_chunks
            )


            # ---------------------------------------------
            # HUGGING FACE CLIENT
            # ---------------------------------------------

            client = InferenceClient(
                token=HF_TOKEN,
                provider="auto"
            )


            # ---------------------------------------------
            # PROMPT
            # ---------------------------------------------

            prompt = f"""
You are my assistant tutor.

Answer the question based only on the following context.

If the answer cannot be found in the context, simply say:
"I don't know."

Context:
{context}

Question:
{user_query}

Answer:
"""


            # ---------------------------------------------
            # GENERATE RESPONSE
            # ---------------------------------------------

            try:

                with st.spinner(
                    "Generating answer..."
                ):

                    response = client.chat.completions.create(

                        model="openai/gpt-oss-120b",

                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],

                        max_tokens=300,

                        temperature=0.2
                    )


                # -----------------------------------------
                # GET RESPONSE
                # -----------------------------------------

                output = (
                    response
                    .choices[0]
                    .message
                    .content
                )


                # -----------------------------------------
                # DISPLAY RESPONSE
                # -----------------------------------------

                st.subheader("Answer")

                st.write(output)


            except Exception as e:

                st.error(
                    f"Error while generating response: {e}"
                )


    except Exception as e:

        st.error(
            f"Error while processing the PDF: {e}"
        )
