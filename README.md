# 🤖 AI Document Assistant

An AI-powered **RAG (Retrieval-Augmented Generation) based document assistant** built using **Streamlit, LangChain, Groq (Llama 3), HuggingFace Embeddings, and ChromaDB**.
Users can upload documents or use built-in dataset files and interact with them using AI.

---

## 🚀 Features

* Chat with documents
* Summarize documents
* Extract key insights
* Generate quiz questions
* Explain concepts simply
* Works with **uploaded files or built-in data**

---

## 🧠 RAG Pipeline

1. Documents are loaded from uploaded files or the `data` folder
2. Text is split into chunks
3. Chunks are converted into **vector embeddings**
4. Embeddings are stored in **ChromaDB vector database**
5. Relevant chunks are retrieved and sent to **Groq Llama 3** to generate answers

---

## 🧰 Tech Stack

* Python
* Streamlit
* LangChain
* Groq (Llama 3.1)
* HuggingFace Embeddings
* ChromaDB

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## 🌐 Live Demo

https://ai-document-assistant-gkmbvlqwsdiaxo5ibjshqz.streamlit.app/

---

## 👨‍💻 Author

Rohitha Vadlamudi
