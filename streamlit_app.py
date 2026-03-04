
import os
import streamlit as st
import tempfile

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(page_title="AI Document Assistant")

st.title("🤖 AI Document Assistant")
st.markdown("Upload a document and let AI analyze it.")


# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.title("AI Tools")

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

st.sidebar.markdown("---")

st.sidebar.markdown("### Try asking")

st.sidebar.write("""
• Summarize this document  
• What are the key points?  
• Explain this in simple terms  
• Generate quiz questions  
""")


# -----------------------------
# Upload File
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["txt", "pdf"]
)


# -----------------------------
# Stop if no file uploaded
# -----------------------------

if uploaded_file is None:
    st.info("Please upload a document to begin.")
    st.stop()


# -----------------------------
# Load Document
# -----------------------------

with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
    tmp_file.write(uploaded_file.read())
    temp_path = tmp_file.name

if uploaded_file.type == "application/pdf":
    loader = PyPDFLoader(temp_path)
else:
    loader = TextLoader(temp_path)

documents = loader.load()

st.sidebar.write(f"Pages loaded: {len(documents)}")


# -----------------------------
# Text Splitting
# -----------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)


# -----------------------------
# Embeddings
# -----------------------------

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------
# Vector Database
# -----------------------------

vectorstore = Chroma.from_documents(
    chunks,
    embedding
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)


# -----------------------------
# LLM Setup
# -----------------------------

groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY not found. Please set it in environment variables or Streamlit secrets.")
    st.stop()

llm = ChatGroq(
    api_key=groq_api_key,
    model="llama-3.1-8b-instant"
)


# -----------------------------
# User Query
# -----------------------------

query = st.text_input("Ask something about the document")

if not query:
    st.stop()


# -----------------------------
# Retrieve Context
# -----------------------------

retrieved_docs = retriever.invoke(query)

context = "\n\n".join([doc.page_content for doc in retrieved_docs])


# -----------------------------
# Prompt Selection
# -----------------------------

if tool == "Chat with Document":

    prompt = f"""
You are an AI assistant that answers ONLY using the provided context.

If the answer is not in the context, say:
"I could not find that information in the document."

Context:
{context}

Question:
{query}
"""


elif tool == "Extract Key Insights":

    prompt = f"""
Extract the key insights from this document.

Document:
{context}
"""


elif tool == "Generate Quiz Questions":

    prompt = f"""
Generate 5 quiz questions based on this document.

Document:
{context}
"""


elif tool == "Explain Simply":

    prompt = f"""
Explain this content in simple terms for beginners.

Content:
{context}
"""


else:

    prompt = f"""
Summarize this document clearly.

Document:
{context}
"""


# -----------------------------
# LLM Response
# -----------------------------

with st.spinner("AI is analyzing the document..."):
    response = llm.invoke(prompt)

st.subheader("AI Response")
st.write(response.content)


# -----------------------------
# Sources
# -----------------------------

sources = set()

for doc in retrieved_docs:
    sources.add(doc.metadata.get("source", "Uploaded document"))

st.markdown("### Sources")

for src in sources:
    st.write(src)


