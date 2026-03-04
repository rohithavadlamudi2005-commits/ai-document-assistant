
import os
import streamlit as st
import tempfile

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq


# -----------------------------
# Page Title
# -----------------------------

st.title("🤖 AI Document Assistant")

st.markdown("""
Upload a document or use the built-in knowledge base and let AI analyze it.
""")


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
• What is blockchain technology?  
• Explain machine learning basics  
• Summarize the climate change report  
• What are common cybersecurity threats?  
• Generate quiz questions from AI overview  
""")


# -----------------------------
# Upload file
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["txt", "pdf"]
)


# -----------------------------
# Load documents
# -----------------------------

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    if uploaded_file.type == "application/pdf":
        loader = PyPDFLoader(temp_path)
    else:
        loader = TextLoader(temp_path)

    documents = loader.load()

else:

    loader = DirectoryLoader(
        "data",
        glob="*.txt",
        loader_cls=TextLoader
    )

    documents = loader.load()


st.sidebar.markdown("### Knowledge Base")
st.sidebar.write(f"Documents loaded: {len(documents)}")


# -----------------------------
# Split documents into chunks
# -----------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)


# -----------------------------
# Create Vector Database
# -----------------------------

def create_vectorstore(docs):

    embedding = HuggingFaceEmbeddings()

    vectorstore = Chroma.from_documents(
        docs,
        embedding
    )

    return vectorstore


vectorstore = create_vectorstore(chunks)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3}
)


# -----------------------------
# LLM Setup
# -----------------------------

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)


# -----------------------------
# User Query
# -----------------------------

query = st.text_input("Ask something about the document:")


if query:

    retrieved_docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])


    # -----------------------------
    # Prompt Selection
    # -----------------------------

    if tool == "Chat with Document":

        prompt = f"""
You are an intelligent AI assistant.

Use the context below to answer the user's question.

If the answer is not present in the context, say:
"I could not find that information in the provided documents."

Context:
{context}

Question:
{query}
"""


    elif tool == "Extract Key Insights":

        prompt = f"""
Extract the key insights from the following document.

Document:
{context}
"""


    elif tool == "Generate Quiz Questions":

        prompt = f"""
Generate 5 quiz questions based on the following document.

Document:
{context}
"""


    elif tool == "Explain Simply":

        prompt = f"""
Explain the following content in simple terms for beginners.

Content:
{context}
"""


    else:

        prompt = f"""
Summarize the following document clearly.

Document:
{context}
"""


    # -----------------------------
    # AI Response
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
        sources.add(doc.metadata.get("source", "Unknown"))

    st.markdown("### Sources")

    for src in sources:
        st.write(src)

    st.markdown("---")


