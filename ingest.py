import os
import pickle
from typing import List, Tuple
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

DATA_DIR = "data"
DB_DIR = "db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

def load_documents(file_paths: List[str]) -> List[Document]:
    docs = []
    for path in file_paths:
        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif path.endswith(".txt") or path.endswith(".md"):
            loader = TextLoader(path)
        else:
            continue
        docs.extend(loader.load())
    return docs

def split_documents(docs: List[Document], chunk_size: int = 600, chunk_overlap: int = 100) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)

def build_indices(chunks: List[Document]) -> Tuple[FAISS, BM25Okapi, List[Document]]:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(os.path.join(DB_DIR, "faiss_index"))

    tokenized_corpus = [doc.page_content.lower().split() for doc in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(os.path.join(DB_DIR, "bm25.pkl"), "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    return vector_store, bm25, chunks

def load_existing_indices():
    faiss_path = os.path.join(DB_DIR, "faiss_index")
    bm25_path = os.path.join(DB_DIR, "bm25.pkl")
    
    if not (os.path.exists(faiss_path) and os.path.exists(bm25_path)):
        return None, None, None

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    
    with open(bm25_path, "rb") as f:
        data = pickle.load(f)
        
    return vector_store, data["bm25"], data["chunks"]