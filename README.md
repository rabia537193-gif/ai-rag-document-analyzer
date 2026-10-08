# 🤖 AI RAG Document Analyzer & Prompt Engineering Dashboard

An interactive, production-ready **FastAPI** web application leveraging **Retrieval-Augmented Generation (RAG)**, **ChromaDB Vector Store**, **SentenceTransformers**, and **Google Gemini / Groq LLMs** to analyze custom documents and benchmark dynamic prompt engineering strategies.

---

## 🌟 Key Features

* 📄 **Multi-Format Document Ingestion:** Full support for uploading and parsing `.pdf` and `.txt` files.
* ⚡ **Vector Indexing & Embeddings:** Embeds document chunks using `sentence-transformers/all-MiniLM-L6-v2` stored in ChromaDB.
* 🎯 **Prompt Engineering Benchmarks:** Evaluates Role-Based + Chain-of-Thought (CoT), Few-Shot Exemplar, and Standard Direct Prompting techniques.
* 📊 **Interactive Evaluation Dashboard:** Bootstrap 5 dual-tab UI with live RAG execution, retrieved context visualization, and accuracy/latency metrics.

---

## 🛠️ Tech Stack

* **Backend:** Python, FastAPI, Jinja2, Uvicorn
* **Vector Database:** ChromaDB
* **Embeddings:** HuggingFace SentenceTransformers (`all-MiniLM-L6-v2`)
* **LLM Engines:** Google Gemini API (`gemini-3.8-flash`) / Groq API (`llama-3.3-70b-versatile`)
* **Frontend:** Bootstrap 5, HTML5, CSS3

---

## 🚀 Live Demo & Repository

* **GitHub Repository:** [https://github.com/rabia537193-gif/ai-rag-document-analyzer](https://github.com/rabia537193-gif/ai-rag-document-analyzer)
* **Live Vercel Application:** [https://ai-rag-document-analyzer.vercel.app](https://ai-rag-document-analyzer.vercel.app)

---

## 💻 Local Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/rabia537193-gif/ai-rag-document-analyzer.git](https://github.com/rabia537193-gif/ai-rag-document-analyzer.git)
   cd ai-rag-document-analyzer