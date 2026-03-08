
import streamlit as st
import tempfile
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_groq import ChatGroq


# -----------------------------
# Page Config
# -----------------------------

st.set_page_config(page_title="AI Document Assistant", layout="wide")

st.title("🤖 AI Document Assistant")

st.markdown(
"""
Upload a document or use built-in sample documents and interact with them using AI.

**Features**
- Chat with documents
- Summarize documents
- Extract key insights
- Generate quiz questions
- Explain concepts simply
"""
)

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


# -----------------------------
# Extract PDF Text
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
# Load Documents
# -----------------------------

documents = []

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    if uploaded_file.type == "application/pdf":
        raw_text = load_pdf_text(file_path)
    else:
        raw_text = open(file_path).read()

    documents = [raw_text]

else:

    st.info("Using sample documents from data folder.")

    loader = DirectoryLoader(
        "data",
        glob="*.txt",
        loader_cls=TextLoader
    )

    loaded_docs = loader.load()

    documents = [doc.page_content for doc in loaded_docs]


# -----------------------------
# Split Documents
# -----------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

docs = splitter.create_documents(documents)


# -----------------------------
# Load Embeddings
# -----------------------------

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()


# -----------------------------
# Create Vectorstore
# -----------------------------

@st.cache_resource
def create_vectorstore(docs):

    return Chroma.from_documents(docs, embeddings)

vectorstore = create_vectorstore(docs)


# -----------------------------
# Retriever
# -----------------------------

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 10}
)


# -----------------------------
# LLM (Groq)
# -----------------------------

GROQ_API_KEY = "YOUR_GROQ_API_KEY"

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant"
)


# -----------------------------
# Layout
# -----------------------------

col1, col2 = st.columns([1,2])


with col1:

    st.subheader("📄 Document Info")

    st.write("Total chunks:", len(docs))


with col2:

    st.subheader("🤖 AI Assistant")


    # -----------------------------
    # Summarize
    # -----------------------------

    if tool == "Summarize Document":

        context = "\n\n".join([d.page_content for d in docs[:6]])

        prompt = f"""
Summarize the following document.

Context:
{context}
"""

        response = llm.invoke(prompt)

        st.subheader("Summary")
        st.write(response.content)


    # -----------------------------
    # Key Insights
    # -----------------------------

    elif tool == "Extract Key Insights":

        context = "\n\n".join([d.page_content for d in docs[:6]])

        prompt = f"""
Extract key insights from the document.

Context:
{context}
"""

        response = llm.invoke(prompt)

        st.subheader("Key Insights")
        st.write(response.content)


    # -----------------------------
    # Quiz Questions
    # -----------------------------

    elif tool == "Generate Quiz Questions":

        retrieved_docs = retriever.invoke("important concepts definitions")

        context = "\n\n".join([doc.page_content for doc in retrieved_docs[:3]])

        prompt = f"""
Generate 5 quiz questions based on the document.

Context:
{context}
"""

        response = llm.invoke(prompt)

        st.subheader("Quiz Questions")
        st.write(response.content)


    # -----------------------------
    # Explain Simply
    # -----------------------------

    elif tool == "Explain Simply":

        retrieved_docs = retriever.invoke("important concepts")

        context = "\n\n".join([doc.page_content for doc in retrieved_docs[:3]])

        prompt = f"""
Explain the following content in simple terms.

Context:
{context}
"""

        response = llm.invoke(prompt)

        st.write(response.content)


    # -----------------------------
    # Chat with Document
    # -----------------------------

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

