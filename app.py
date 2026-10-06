import os
import streamlit as st
from ingest import load_documents, split_documents, build_indices, load_existing_indices, DATA_DIR
from retrieval import hybrid_search
from llm import generate_answer

st.set_page_config(page_title="RAG Hybrid Search", layout="wide")

st.sidebar.title("Search Controls")
top_k = st.sidebar.slider("Top K Results", min_value=1, max_value=10, value=4)
bm25_weight = st.sidebar.slider("Keyword Weight (BM25)", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
vector_weight = st.sidebar.slider("Vector Weight (Semantic)", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

st.sidebar.markdown("---")
uploaded_files = st.sidebar.file_uploader("Upload Documents (PDF, TXT, MD)", type=["pdf", "txt", "md"], accept_multiple_files=True)

if st.sidebar.button("Index Documents"):
    if uploaded_files:
        saved_paths = []
        for file in uploaded_files:
            file_path = os.path.join(DATA_DIR, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            saved_paths.append(file_path)

        with st.spinner("Processing & building hybrid indices..."):
            docs = load_documents(saved_paths)
            chunks = split_documents(docs)
            vector_store, bm25, stored_chunks = build_indices(chunks)
            st.session_state["vector_store"] = vector_store
            st.session_state["bm25"] = bm25
            st.session_state["chunks"] = stored_chunks
        st.sidebar.success("Indexing complete!")
    else:
        st.sidebar.warning("Please upload at least one document.")

if "vector_store" not in st.session_state:
    v_store, bm25_idx, stored_chunks = load_existing_indices()
    if v_store is not None:
        st.session_state["vector_store"] = v_store
        st.session_state["bm25"] = bm25_idx
        st.session_state["chunks"] = stored_chunks

st.title("RAG Pipeline with Hybrid Search")
st.caption("Combines BM25 Keyword Search + FAISS Semantic Vector Search")

query = st.text_input("Ask anything about your documents:")
if st.button("Send") and query:
    if "vector_store" not in st.session_state or st.session_state["vector_store"] is None:
        st.error("No indexed documents found! Please upload files in the sidebar and click 'Index Documents'.")
    else:
        with st.spinner("Executing hybrid retrieval and generating answer..."):
            results = hybrid_search(
                query=query,
                vector_store=st.session_state["vector_store"],
                bm25=st.session_state["bm25"],
                chunks=st.session_state["chunks"],
                top_k=top_k,
                bm25_weight=bm25_weight,
                vector_weight=vector_weight
            )
            answer = generate_answer(query, results)

        st.markdown("### Answer:")
        st.write(answer)

        st.markdown("### Sources:")
        for idx, (doc, score) in enumerate(results, 1):
            source = os.path.basename(doc.metadata.get("source", "Document"))
            page = doc.metadata.get("page", 0) + 1
            st.markdown(f"**{idx}. {source} (Page {page})** — *RRF Relevance Score: {score:.5f}*")
            with st.expander("View Retrieved Chunk"):
                st.write(doc.page_content)