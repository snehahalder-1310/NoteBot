import os
import streamlit as st
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from huggingface_hub import InferenceClient



# PAGE CONFIG
st.set_page_config(
    page_title="NoteBot",
    page_icon="📚",
    layout="wide"
)



# TITLE
st.title("📚 NoteBot")
st.write("Upload your PDF notes and ask questions about them.")



# HUGGING FACE TOKEN
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    st.error("Hugging Face token is not configured.")
    st.stop()



# HUGGING FACE EMBEDDINGS
class HuggingFaceEmbeddings(Embeddings):

    def __init__(self):
        self.client = InferenceClient(
            token=HF_TOKEN,
            provider="auto"
        )

        self.model = (
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def embed_documents(self, texts):

        embeddings = []

        for text in texts:

            result = self.client.feature_extraction(
                text,
                model=self.model
            )

            # Convert result to a normal list
            if hasattr(result, "tolist"):
                result = result.tolist()

            embeddings.append(result)

        return embeddings

    def embed_query(self, text):

        result = self.client.feature_extraction(
            text,
            model=self.model
        )

        if hasattr(result, "tolist"):
            result = result.tolist()

        return result


embeddings = HuggingFaceEmbeddings()



# SIDEBAR
with st.sidebar:

    st.header("📄 My Notes")

    file = st.file_uploader(
        "Upload your notes PDF",
        type=["pdf"]
    )



# PDF PROCESSING

if file is not None:

    try:

       
        # READ PDF
        pdf = PdfReader(file)

        text = ""

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text


        if not text.strip():

            st.error(
                "No readable text was found in this PDF."
            )

            st.stop()


       # SPLIT TEXT
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            length_function=len
        )

        chunks = splitter.split_text(text)


        if not chunks:

            st.error(
                "Could not create text chunks."
            )

            st.stop()


        st.info(
            f"PDF processed into {len(chunks)} text chunks."
        )


        
       # CREATE FAISS VECTOR STORE
        with st.spinner(
            "Creating document embeddings..."
        ):

            vector_store = FAISS.from_texts(
                chunks,
                embeddings
            )


        st.success(
            "PDF uploaded and processed successfully!"
        )


       
        # QUESTION
        user_query = st.text_input(
            "Ask a question about your notes:"
        )


        if user_query:

            # SEARCH RELEVANT CHUNKS
            with st.spinner(
                "Searching your notes..."
            ):

                matching_chunks = (
                    vector_store.similarity_search(
                        user_query,
                        k=3
                    )
                )


            context = "\n\n".join(
                document.page_content
                for document in matching_chunks
            )


            # HUGGING FACE LLM
            client = InferenceClient(
                token=HF_TOKEN,
                provider="auto"
            )


            prompt = f"""
You are NoteBot, an assistant that answers questions
about uploaded study notes.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context, say:

"I don't know Sneha."

Do not invent information.

Context:
{context}

Question:
{user_query}

Answer:
"""


             # GENERATE ANSWER
            try:

                with st.spinner(
                    "Generating answer..."
                ):

                    response = (
                        client.chat.completions.create(

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
                    )


                output = (
                    response
                    .choices[0]
                    .message
                    .content
                )


              
                # DISPLAY ANSWER
                st.subheader("💡 Answer")

                st.write(output)


            except Exception as e:

                st.error(
                    f"Error while generating answer: {e}"
                )


    except Exception as e:

        st.error(
            f"Error while processing PDF: {e}"
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        text-align: center;
        padding: 8px;
        font-size: 14px;
        color: #888;
        background-color: transparent;
    }
    </style>

    <div class="footer">
        © 2026 Sneha Halder | NoteBot
    </div>
    """,
    unsafe_allow_html=True
)
