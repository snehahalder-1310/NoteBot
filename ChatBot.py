import os
import streamlit as st
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient


# HUGGING FACE TOKEN
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    st.error("Hugging Face token is not configured.")
    st.stop()



# PAGE TITLE
st.header("NoteBot")



# HUGGING FACE EMBEDDING MODEL
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


class HuggingFaceEmbeddings(Embeddings):

    def embed_documents(self, texts):

        embeddings = embedding_model.encode(
            texts,
            convert_to_numpy=True
        )

        return embeddings.tolist()

    def embed_query(self, text):

        embedding = embedding_model.encode(
            text,
            convert_to_numpy=True
        )

        return embedding.tolist()

embeddings = HuggingFaceEmbeddings()



# SIDEBAR
with st.sidebar:

    st.title("My Notes")

    file = st.file_uploader(
        "Upload notes PDF and start asking questions",
        type="pdf"
    )



# PDF PROCESSING
if file is not None:
    my_pdf = PdfReader(file)    # READ PDF
    text = ""

    for page in my_pdf.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text



    # SPLIT TEXT INTO CHUNKS
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len
    )

    chunks = splitter.split_text(text)


    # CREATE FAISS VECTOR DATABASE
    vector_store = FAISS.from_texts(
        chunks,
        embeddings
    )


    st.success("PDF uploaded successfully!")



    # USER QUERY
    user_query = st.text_input(
        "Type your query here"
    )


    if user_query:


        # SEMANTIC SEARCH
        matching_chunks = vector_store.similarity_search(
            user_query,
            k=3
        )



        # CREATE CONTEXT

        context = "\n\n".join(
            document.page_content
            for document in matching_chunks
        )



        # HUGGING FACE LLM

        client = InferenceClient(
            token=HF_TOKEN,
            provider="auto"
        )



        # PROMPT

        prompt = f"""
You are my assistant tutor.

Answer the question based only on the following context.

If the answer cannot be found in the context, simply say:
"I don't know Sneha."

Context:
{context}

Question:
{user_query}

Answer:
"""



        # GENERATE RESPONSE

        try:

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


            # GET RESPONSE TEXT

            output = response.choices[0].message.content



            # DISPLAY RESPONSE

            st.subheader("Answer")
            st.write(output)


        except Exception as e:

            st.error(
                f"Error while generating response: {e}"
            )