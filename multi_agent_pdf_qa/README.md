# 🧠 Multi-Agent PDF QA Assistant

This project is a Multi-Agent Reasoning System designed to process and answer questions from uploaded PDF files using Azure OpenAI and Pinecone.

---

## ✅ Features

- 📎 **Multi-file PDF upload**
- 🔍 **Question answering via Retrieval-Augmented Generation (RAG)**
- 📝 **Per-file summarization**
- 🧠 **Azure OpenAI Agents SDK for intelligent routing**
- 🧹 **Auto-cleanup of uploaded embeddings when session ends**

---

## 🚀 How It Works

1. **Upload PDFs** → Text is extracted, chunked, and embedded.
2. **User Query**:
   - **If it’s a question** → uses vector search via `rag_tool`.
   - **If it’s a summary request** → user provides file name (e.g., `"Summarize resume.pdf"`).
3. **Routing Agent** decides which tool to call based on query.
4. **Embeddings** are auto-deleted from Pinecone when the session ends.

---

## 🛠 Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
