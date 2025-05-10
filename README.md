# 🧠 RAG-Powered Multi-Agent Knowledge Assistant

This project implements a **Retrieval-Augmented Generation (RAG)** system integrated with a simple agentic workflow. It dynamically chooses between tools like a calculator, dictionary, or a retrieval-based LLM QA pipeline to answer queries. The system is exposed via a minimal **Streamlit** web interface.

---

## 🎯 Objective

Design and implement a simple “knowledge assistant” that:

1. Retrieves relevant information from a small document collection (RAG)
2. Generates natural-language answers via an LLM
3. Orchestrates retrieval + generation steps with a basic agentic workflow

---

## 🧱 Project Structure

.
├── app.py # Streamlit app interface
├── build_vectorstore.py # Preprocess & vectorize documents
├── rag_model.py # Core logic: routing, retrieval, LLM
├── documents/ # Folder with .txt files
├── vectorstore.pkl # Serialized FAISS index
├── requirements.txt # All dependencies
├── README.md # This file
└── screenshots/ # Screenshots for demo UI

markdown
Copy
Edit

---

## 🔍 Components

### 1. 📄 Data Ingestion

- Text documents are loaded from `documents/`.
- Split into chunks using `RecursiveCharacterTextSplitter`.
- Embeddings generated using `sentence-transformers/all-MiniLM-L6-v2`.

### 2. 🔎 Vector Store & Retrieval

- Vector index built using **FAISS**.
- Top 3 most relevant chunks retrieved for each query.

### 3. 🧠 LLM Integration

- Question answering model: `valhalla/t5-base-qa-qg-hl`.
- Uses Hugging Face `pipeline` to generate answers from context.

### 4. 🤖 Agentic Workflow

Query routing logic:

| Query Type                              | Route to Tool          |
|----------------------------------------|------------------------|
| Contains math expression               | Calculator             |
| Includes "define ..."                  | Dictionary (stub)      |
| Has technical numbers (e.g. 5G, ISO)   | Number Theory RAG      |
| Everything else                        | Standard RAG → LLM     |

Each query logs:

- **Selected Branch**
- **Retrieved Context**
- **Final Answer**

### 5. 🌐 Streamlit UI

- Single input box for queries
- Displays tool selected, retrieved context, and final output

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/rag-multi-agent-assistant.git
cd rag-multi-agent-assistant
2️⃣ Set Up Virtual Environment
bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
3️⃣ Install Dependencies
bash
Copy
Edit
pip install --upgrade pip
pip install -r requirements.txt
pip install faiss-cpu==1.7.4 --no-build-isolation --no-cache-dir
4️⃣ Prepare Your Documents
Add 3–5 .txt files to the documents/ folder.

Then run:

bash
Copy
Edit
python build_vectorstore.py
5️⃣ Run the Assistant
bash
Copy
Edit
streamlit run app.py
🧪 Sample Queries
text
Copy
Edit
Define throughput
Calculate (14 + 6) * 7 / 2
What does ISO 27001 mean?
What is the refund policy?
🖼️ Screenshots
Input + Tool Selection	Retrieved Context + Final Answer

Replace these with your own screenshots after running the app.

📦 Requirements
txt
Copy
Edit
streamlit==1.33.0
langchain==0.1.13
langchain-community==0.0.29
transformers[sentencepiece]==4.40.1
sentence-transformers==2.6.1
faiss-cpu==1.7.4
torch==2.2.2
scikit-learn==1.4.2
protobuf==4.25.3
🧰 Tools Used
FAISS – In-memory vector store for fast document similarity search

LangChain – Agent routing and retriever abstraction

Transformers – Question answering with a local LLM

Streamlit – Rapid UI prototyping

📌 Limitations
Dictionary tool is currently stubbed (placeholder).

Local inference only (no OpenAI/GPT-4).

Retrieval quality depends on small doc sample.

🚧 Future Enhancements
Integrate real dictionary APIs (e.g., Wordnik)

Add memory and chat history

Containerize via Docker for deployment

Add user feedback or ranking on LLM answers

🪪 License
This project is licensed under the MIT License.