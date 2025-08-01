# 🔷 Zuri Knowledge Base  
_Chat with your research papers using GPT-4 and RAG_

Zuri Knowledge Base is an interactive tool that lets you upload academic PDFs and ask questions using GPT-4. It uses LlamaIndex for semantic chunking, ChromaDB for vector storage, and Gradio for the frontend — all orchestrated through OpenAI’s function calling.

---

## 🧠 Features

- Upload academic PDFs
- Ask natural language questions
- GPT-4-powered responses grounded in your documents
- Uses ChromaDB, OpenAI embeddings, and LlamaIndex
- Clean Gradio UI with custom styling

---

## 🚀 Getting Started

```bash
git clone https://github.com/yourusername/zuri-knowledge-base.git
cd zuri-knowledge-base
cp .env.example .env  # Add your OpenAI key to .env
python chat_builder.py
