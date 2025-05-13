import os
import re
import pickle
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


with open("vectorstore.pkl", "rb") as f:
    vectorstore = pickle.load(f)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def extract_math_expression(query):
    matches = re.findall(r"[\d\.\+\-\*/\(\)\s]+", query)
    
    for match in matches:
        cleaned = match.strip()
    
        if cleaned and re.fullmatch(r"[0-9\.\+\-\*/\(\)\s]+", cleaned):
            return cleaned
    return None

def route_query(query):
   
    expr = extract_math_expression(query)
    if expr and len(expr.strip()) >= 3: 
        return "calculator"
    if re.search(r"\bdefine\b", query, re.I):
        return "dictionary"
    return "rag_llm"

def contains_number_theory_query(query):
    
    return bool(re.search(r"\b(?:\d+[a-zA-Z]*|[a-zA-Z]*\d+)\b", query))

def number_with_theory(query, retriever):
    docs = retriever.get_relevant_documents(query)
    context = " ".join([doc.page_content for doc in docs])
    
    prompt = (
        f"Extract detailed technical information based on the question: {query}\n\n"
        f"Use the following context if relevant:\n{context}"
    )
    
    result = qa_pipeline(prompt, max_length=256, do_sample=False)[0]["generated_text"]
    return result, docs


def use_calculator(query):
    expr = extract_math_expression(query)
    if not expr:
        return "No valid expression found in the query."
    
    try:
        result = eval(expr)
        return str(result)
    except Exception as e:
        return f"Error evaluating: `{expr}` — {e}"

def define_term(term):
    return f"Definition of '{term}' (stubbed): [I have to add an dictionary API here later but not now for this model.]"

# QA Model
qa_model_name = "valhalla/t5-base-qa-qg-hl"
tokenizer = T5Tokenizer.from_pretrained(qa_model_name)
model = T5ForConditionalGeneration.from_pretrained(qa_model_name)
qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

def answer_with_qa_pipeline(query, retriever):
    docs = retriever.get_relevant_documents(query)
    context = " ".join([doc.page_content for doc in docs])
    prompt = f"question: {query}  context: {context}"
    result = qa_pipeline(prompt, max_length=256, do_sample=False)[0]["generated_text"]
    return result, docs
