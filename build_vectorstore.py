import os
import pickle
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def load_documents_from_folder(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
                text = file.read()
                docs.append(Document(page_content=text, metadata={"source": filename}))
    return docs

folder_path = "documents"
documents = load_documents_from_folder(folder_path)

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(documents)

embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save to disk
with open("vectorstore.pkl", "wb") as f:
    pickle.dump(vectorstore, f)

print("✅ Vectorstore saved to 'vectorstore.pkl'")
