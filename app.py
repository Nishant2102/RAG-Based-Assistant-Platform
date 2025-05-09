import streamlit as st
from rag_model import route_query, use_calculator, define_term, answer_with_qa_pipeline, number_with_theory, retriever


st.set_page_config(page_title="RAG Multi-Agent Assistant")
st.title("🤗 RAG-Powered Multi-Agent Q&A Assistant")

query = st.text_input("Ask me anything...")

if query:
    decision = route_query(query)
    
    st.markdown("## 🔍 Tool/Agent Selection")
    st.markdown(f"**Selected Branch:** `{decision}`")

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
            st.markdown(f"**Snippet {i+1}** from `{doc.metadata['source']}`:")
            st.code(doc.page_content[:300] + "...")
        st.markdown("## 🧠 Final Answer")
        st.write(answer)


    else:
        answer, retrieved_docs = answer_with_qa_pipeline(query, retriever)
        st.markdown("## 📄 Retrieved Context")
        for i, doc in enumerate(retrieved_docs):
            st.markdown(f"**Snippet {i+1}** from `{doc.metadata['source']}`:")
            st.code(doc.page_content[:300] + "...")
        st.markdown("## 🧠 Final Answer")
        st.write(answer)
