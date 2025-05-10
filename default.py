# rag_assistant.py

import os
import re
import pickle
import streamlit as st
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# ---------- CONFIG ---------- #
st.set_page_config(page_title="RAG Multi-Agent Assistant")
st.title("🤗 RAG-Powered Multi-Agent Q&A Assistant")

VECTORSTORE_PATH = "vectorstore.pkl"
DOCS_FOLDER = "/home/shadow-monarch/Documents/Jupyter/Intern_project/documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QA_MODEL_NAME = "valhalla/t5-base-qa-qg-hl"

# ---------- VECTORSTORE BUILD OR LOAD ---------- #
def load_documents_from_folder(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
                text = file.read()
                docs.append(Document(page_content=text, metadata={"source": filename}))
    return docs

def get_vectorstore():
    if os.path.exists(VECTORSTORE_PATH):
        with open(VECTORSTORE_PATH, "rb") as f:
            return pickle.load(f)
    documents = load_documents_from_folder(DOCS_FOLDER)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    with open(VECTORSTORE_PATH, "wb") as f:
        pickle.dump(vectorstore, f)
    return vectorstore

vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ---------- QA PIPELINE ---------- #
tokenizer = T5Tokenizer.from_pretrained(QA_MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(QA_MODEL_NAME)
qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# ---------- UTILITIES / AGENTS ---------- #
def extract_math_expression(query):
    matches = re.findall(r"[\d\.\+\-\*/\(\)\s]+", query)
    for match in matches:
        cleaned = match.strip()
        if cleaned and re.fullmatch(r"[0-9\.\+\-\*/\(\)\s]+", cleaned):
            return cleaned
    return None

def use_calculator(query):
    expr = extract_math_expression(query)
    if not expr:
        return "No valid expression found in the query."
    try:
        result = eval(expr)
        return str(result)
    except Exception as e:
        return f"Error evaluating: {expr} — {e}"

def define_term(term):
    return f"Definition of '{term}' (stubbed): [Use a dictionary API here]"

def answer_with_qa_pipeline(query, retriever):
    docs = retriever.get_relevant_documents(query)
    context = " ".join([doc.page_content for doc in docs])
    prompt = f"question: {query}  context: {context}"
    result = qa_pipeline(prompt, max_length=256, do_sample=False)[0]["generated_text"]
    return result, docs

def number_with_theory(query, retriever):
    docs = retriever.get_relevant_documents(query)
    context = " ".join([doc.page_content for doc in docs])
    prompt = f"Extract detailed technical information based on the question: {query}\n\nUse the following context if relevant:\n{context}"
    result = qa_pipeline(prompt, max_length=256, do_sample=False)[0]["generated_text"]
    return result, docs

def contains_number_theory_query(query):
    return bool(re.search(r"\b(?:\d+[a-zA-Z]*|[a-zA-Z]*\d+)\b", query))

def route_query(query):
    expr = extract_math_expression(query)
    if expr and len(expr.strip()) >= 3:
        return "calculator"
    if re.search(r"\bdefine\b", query, re.I):
        return "dictionary"
    if contains_number_theory_query(query):
        return "number_theory"
    return "rag_llm"

# ---------- STREAMLIT UI ---------- #
query = st.text_input("Ask me anything...")

if query:
    decision = route_query(query)

    st.markdown("## 🔍 Tool/Agent Selection")
    st.markdown(f"**Selected Branch:** {decision}")

    if decision == "calculator":
        answer = use_calculator(query)
        st.markdown("## 📄 Retrieved Context")
        st.write("_N/A (calculation tool used)_")
        st.markdown("## 🧠 Final Answer")
        st.write(answer)

    elif decision == "dictionary":
        answer = define_term(query)
        st.markdown("## 📄 Retrieved Context")
        st.write("_N/A (dictionary tool used)_")
        st.markdown("## 🧠 Final Answer")
        st.write(answer)

    elif decision == "number_theory":
        answer, retrieved_docs = number_with_theory(query, retriever)
        st.markdown("## 📄 Retrieved Context")
        for i, doc in enumerate(retrieved_docs):
            st.markdown(f"**Snippet {i+1}** from {doc.metadata['source']}:")
            st.code(doc.page_content[:300] + "...")
        st.markdown("## 🧠 Final Answer")
        st.write(answer)

    else:
        answer, retrieved_docs = answer_with_qa_pipeline(query, retriever)
        st.markdown("## 📄 Retrieved Context")
        for i, doc in enumerate(retrieved_docs):
            st.markdown(f"**Snippet {i+1}** from {doc.metadata['source']}:")
            st.code(doc.page_content[:300] + "...")
        st.markdown("## 🧠 Final Answer")
        st.write(answer)
