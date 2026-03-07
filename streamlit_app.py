
import streamlit as st
import tempfile
import re
import os
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq


st.set_page_config(page_title="AI Document Assistant", layout="wide")

st.title("🤖 AI Document Assistant")

st.markdown("""
Upload a document and analyze it using AI.

Features:
• Summarize documents  
• Extract key insights  
• Generate quiz questions  
• Explain concepts simply  
• Chat with your document
""")


# -----------------------------
# Sidebar
# -----------------------------

tool = st.sidebar.selectbox(
    "Choose AI Function",
    [
        "Chat with Document",
        "Summarize Document",
        "Extract Key Insights",
        "Generate Quiz Questions",
        "Explain Simply"
    ]
)


# -----------------------------
# Upload File
# -----------------------------

uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt"])

if not uploaded_file:
    st.info("Upload a document to begin.")
    st.stop()


# -----------------------------
# Clean Text
# -----------------------------

def clean_text(text):

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"Page \d+", "", text)
    text = re.sub(r"[•●►■□◆]", "", text)

    return text.strip()


# -----------------------------
# Extract PDF text
# -----------------------------

def load_pdf_text(path):

    reader = PdfReader(path)
    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# -----------------------------
# Save uploaded file
# -----------------------------

with tempfile.NamedTemporaryFile(delete=False) as tmp:

    tmp.write(uploaded_file.read())
    file_path = tmp.name


# -----------------------------
# Read file
# -----------------------------

if uploaded_file.type == "application/pdf":

    raw_text = load_pdf_text(file_path)

else:

    raw_text = open(file_path).read()


if not raw_text or len(raw_text.strip()) < 20:

    st.error("Could not extract readable text from this file.")
    st.stop()


text = clean_text(raw_text)


# -----------------------------
# Split document
# -----------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200
)

docs = splitter.create_documents([text])

docs = [d for d in docs if len(d.page_content.strip()) > 30]

if not docs:

    st.error("Document contains no usable content.")
    st.stop()


# -----------------------------
# Embeddings
# -----------------------------

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


embeddings = load_embeddings()


# -----------------------------
# Reset vector store when new file uploaded
# -----------------------------

if "vectorstore" not in st.session_state or st.session_state.get("current_file") != uploaded_file.name:

    vectorstore = Chroma.from_documents(docs, embeddings)

    st.session_state.vectorstore = vectorstore
    st.session_state.current_file = uploaded_file.name

else:

    vectorstore = st.session_state.vectorstore


# -----------------------------
# Retriever
# -----------------------------

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 12
    }
)


# -----------------------------
# Load LLM
# -----------------------------

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)


# -----------------------------
# Layout
# -----------------------------

col1, col2 = st.columns([1,2])


# -----------------------------
# Document Info
# -----------------------------

with col1:

    st.subheader("📄 Document Info")

    st.write("File:", uploaded_file.name)
    st.write("Chunks:", len(docs))
    st.write("Text length:", len(text))


# -----------------------------
# AI Assistant
# -----------------------------

with col2:

    st.subheader("🤖 AI Assistant")


    if tool == "Summarize Document":

        context = "\n\n".join([d.page_content for d in docs[:6]])

        prompt = f"""
Use ONLY the document context to produce a summary.

Context:
{context}
"""

        response = llm.invoke(prompt)

        st.subheader("Summary")
        st.write(response.content)


    elif tool == "Extract Key Insights":

        context = "\n\n".join([d.page_content for d in docs[:6]])

        prompt = f"""
Use the document context to extract key insights.

Context:
{context}
"""

        response = llm.invoke(prompt)

        st.subheader("Key Insights")
        st.write(response.content)


    elif tool == "Generate Quiz Questions":

        retrieved_docs = retriever.invoke("main concepts definitions topics")

        context = "\n\n".join([doc.page_content for doc in retrieved_docs[:3]])

        prompt = f"""
Create 5 multiple-choice quiz questions using ONLY the document context.

Context:
{context}
"""

        response = llm.invoke(prompt)

        st.subheader("Quiz Questions")
        st.write(response.content)


    elif tool == "Explain Simply":

        retrieved_docs = retriever.invoke("important concepts")

        context = "\n\n".join([doc.page_content for doc in retrieved_docs[:3]])

        prompt = f"""
Explain the following document content in simple terms.

Context:
{context}
"""

        response = llm.invoke(prompt)

        st.write(response.content)


    elif tool == "Chat with Document":

        query = st.chat_input("Ask something about the document")

        if query:

            with st.chat_message("user"):
                st.write(query)

            retrieved_docs = retriever.invoke(query)

            context = "\n\n".join([doc.page_content for doc in retrieved_docs[:3]])

            prompt = f"""
Answer the question using ONLY the document context.

Context:
{context}

Question:
{query}
"""

            response = llm.invoke(prompt)

            with st.chat_message("assistant"):
                st.write(response.content)
# latest update
