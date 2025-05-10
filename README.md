# RAG-Powered Multi-Agent Knowledge Assistant

This project implements a Retrieval-Augmented Generation (RAG) system integrated with a simple agentic workflow. It dynamically chooses between tools like a calculator, dictionary, or a retrieval-based LLM QA pipeline to answer queries. The system is exposed via a minimal Streamlit web interface.

---

## Objective

Design and implement a simple “knowledge assistant” that:

1. Retrieves relevant information from a small document collection (RAG)
2. Generates natural-language answers via an LLM
3. Orchestrates retrieval + generation steps with a basic agentic workflow

---

## Project Structure

.

├── app.py

├── build_vectorstore.py

├── rag_model.py

├── documents/

├── vectorstore.pkl

├── requirements.txt

└── README.md


---

## Components

### 1. Data Ingestion
- Text documents are loaded from documents/
- Split into chunks using RecursiveCharacterTextSplitter
- Embeddings generated using sentence-transformers/all-MiniLM-L6-v2

### 2. Vector Store & Retrieval
- Vector index built using FAISS
- Top 3 most relevant chunks retrieved for each query

### 3. LLM Integration
- Question answering model: valhalla/t5-base-qa-qg-hl
- Uses Hugging Face pipeline to generate answers from context

### 4. Agentic Workflow

| Query Type                            | Routed Tool            |
|--------------------------------------|------------------------|
| Contains math expression             | Calculator             |
| Includes "define ..."                | Dictionary (stub)      |
| Contains technical numbers (e.g. ISO)| Number Theory RAG      |
| All other queries                    | Standard RAG → LLM     |

Each query logs:

- Selected Tool/Agent Branch  
- Retrieved Context  
- Final Answer  

### 5. Streamlit UI

- One input field for queries
- Output includes tool used, context, and answer
- Screenshot placeholders provided below

---

## Getting Started

```bash
git clone https://github.com/yourusername/rag-multi-agent-assistant.git
cd rag-multi-agent-assistant
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install faiss-cpu==1.7.4 --no-build-isolation --no-cache-dir
python build_vectorstore.py
streamlit run app.py
```

## Tools Used

- FAISS — In-memory vector store for fast similarity search

- LangChain — Agent routing and retrieval abstraction

- Transformers — LLMs for question answering

- Streamlit — Front-end web UI

## Future Enhancements

- Integrate real dictionary APIs (e.g., Wordnik)

- Add memory and chat history

- Containerize the app using Docker

- Implement user feedback and rating for LLM responses
