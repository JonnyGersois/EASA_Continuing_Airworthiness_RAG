# app.py

import streamlit as st
import os
from dotenv import load_dotenv

from extractor import extract_clean_xml_from_package, convert_xml_to_documents

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


# --------------------
# Utility: Chunk docs
# --------------------
def chunk_documents(docs, threshold=1200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = []
    for doc in docs:
        if len(doc.page_content) > threshold:
            sub_docs = splitter.split_documents([doc])
            for sub in sub_docs:
                sub.metadata.update(doc.metadata)
            chunks.extend(sub_docs)
        else:
            chunks.append(doc)
    return chunks


# --------------------
# Retriever builder
# --------------------
def build_hybrid_retriever(docs, faiss_path="vectorstores/easa_airworthiness", k=5):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(faiss_path):
        vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(faiss_path)

    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = k

    return EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.6, 0.4]
    )


# --------------------
# Chain builder
# --------------------
def create_conversational_chain(retriever, model_name="llama-3.1-8b-instant"):
    llm = ChatGroq(model_name=model_name, temperature=0.2)
    system_prompt = """
    You are an expert assistant on EASA Continuing Airworthiness (Regulation (EU) No 1321/2014).

    - Answer in plain English first, then cite regulation IDs/sections.
    - Summarize retrieved text briefly; quote only short snippets.
    - Mark clearly if something is **outside regulation**.
    - Refuse irrelevant or non-aviation queries.

    Context:
    {context}
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    qa_chain = prompt | llm
    return create_retrieval_chain(retriever, qa_chain)


# --------------------
# Streamlit App
# --------------------
st.set_page_config(page_title="EASA Airworthiness RAG", layout="wide")
st.title("EASA Continuing Airworthiness Compliance Assistant")

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if "model_name" not in st.session_state:
    st.session_state.model_name = "llama-3.1-8b-instant"
if "retrieval_k" not in st.session_state:
    st.session_state.retrieval_k = 5

# Sidebar
with st.sidebar:
    st.header("Settings")
    st.selectbox(
        "Choose an LLM:",
        options=[
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
            "qwen/qwen3-32b",
            "openai/gpt-oss-20b"
            "openai/gpt-oss-120b",
        ],
        key="model_name"
    )
    st.slider(
        "Number of chunks to retrieve (k)",
        min_value=2, max_value=20,
        value=st.session_state.retrieval_k,
        key="retrieval_k"
    )


@st.cache_resource
def load_pipeline():
    xml_path = os.path.join("data", "Easy Access Rules for Continuing Airworthiness (Regulation (EU) No 13212014).xml")
    clean_path = os.path.join("data", "easa_clean.xml")

    clean_xml = extract_clean_xml_from_package(xml_path, save_clean_path=clean_path)
    docs = convert_xml_to_documents(clean_xml)
    chunks = chunk_documents(docs)

    retriever = build_hybrid_retriever(chunks, k=st.session_state.retrieval_k)
    rag_chain = create_conversational_chain(retriever, model_name=st.session_state.model_name)
    return rag_chain


if groq_api_key:
    rag_chain = load_pipeline()

    # Chat interface
    user_q = st.chat_input("Ask a question about Continuing Airworthiness Regulation (EU) 1321/2014...")
    if user_q:
        with st.spinner("Searching through the regulations..."):
            response = rag_chain.invoke({"input": user_q})

        # Results
        st.chat_message("user").markdown(user_q)
        # st.chat_message("assistant").markdown(response["answer"])
        answer_content = response["answer"].content if hasattr(response["answer"], 'content') else response["answer"]
        st.chat_message("assistant").markdown(answer_content)
else:
    st.warning("Please enter your GROQ_API_KEY in .env or environment variables.")