# 📄 Chat with PDF using Gemini API

This is a Streamlit web app that lets users upload PDF documents and ask questions about their content using Google’s Gemini large language models via LangChain. It also includes user forms for contact and appointment booking.

---

## 🚀 Features

- Upload multiple PDF files and process their text
- Automatically chunk PDF text and create vector embeddings with Google Generative AI Embeddings
- Store and search vectors using FAISS index locally
- Ask natural language questions about uploaded PDFs and get detailed answers using Gemini LLM
- Simple user forms to collect user info and book appointments
- Validation for email and phone number inputs

---

## 🧰 Tech Stack & Libraries

- **Streamlit** — Frontend UI framework
- **PyPDF2** — PDF text extraction
- **LangChain** — Text splitting, vector stores, chain loading
- **Google Generative AI** — Embeddings and chat model (Gemini)
- **FAISS** — Efficient similarity search index
- **dotenv** — Environment variable management
- **re (regex)** — Input validation for email and phone

---



