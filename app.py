import os

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq


# ---------------------------
# Load all text files in data folder
# ---------------------------
loader = DirectoryLoader(
    "data",
    glob="*.txt",
    loader_cls=TextLoader
)

documents = loader.load()


# ---------------------------
# Split documents into chunks
# ---------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)


# ---------------------------
# Create embeddings
# ---------------------------
embedding = HuggingFaceEmbeddings()


# ---------------------------
# Store embeddings in Chroma
# ---------------------------
vectorstore = Chroma.from_documents(docs, embedding)


# ---------------------------
# Retriever (search system)
# ---------------------------
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3}
)


# ---------------------------
# Load LLM (Groq)
# ---------------------------
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)


print("\nAI Assistant Ready!")
print("Ask any question about the documents.")
print("Type 'exit' to stop.\n")


# ---------------------------
# Chat Loop
# ---------------------------
while True:

    query = input("You: ")

    if query.lower() == "exit":
        print("Goodbye!")
        break


    # Retrieve relevant docs
    retrieved_docs = retriever.invoke(query)


    # Combine context
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])


    prompt = f"""
You are an intelligent AI assistant.

Use the provided context to answer the user's question.

If the answer is not present in the context, say:
"I could not find that information in the provided documents."

Context:
{context}

Question:
{query}
"""


    response = llm.invoke(prompt)


    print("\nBot:", response.content)


    # ---------------------------
    # Show Sources
    # ---------------------------
    sources = set()

    for doc in retrieved_docs:
        sources.add(doc.metadata.get("source", "Unknown"))


    print("\nSources:")
    for src in sources:
        print(src)

    print("\n-----------------------------\n")