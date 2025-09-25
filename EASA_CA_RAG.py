# app.py

import streamlit as st
import os
from dotenv import load_dotenv
from datetime import datetime

from extractor import extract_clean_xml_from_package, convert_xml_to_documents

# LangChain imports
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings  # DEPRECATED
from langchain_huggingface import HuggingFaceEmbeddings
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
    os.environ["GROQ_API_KEY"] = st.session_state.groq_api_key
    llm = ChatGroq(model_name=model_name, temperature=0.2)
    system_prompt = """
    You are an expert assistant on EASA Continuing Airworthiness (Regulation (EU) No 1321/2014).

    - Answer in plain English first, then cite regulation IDs/sections.
    - Summarize retrieved text briefly; quote only short snippets.
    - Mark clearly if something is **outside regulation**.
    - Refuse irrelevant or non-aviation queries politely.
    - Limit answers to 500 words and offer further help if the answer would have been longer than 500 words

    Context:
    {context}
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),   # history threaded properly
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
if "groq_api_key" not in st.session_state:
    # Load from .env first if available, otherwise default ""
    st.session_state.groq_api_key = os.getenv("GROQ_API_KEY", "")

# if "model_name" not in st.session_state:
#     st.session_state.model_name = "llama-3.1-8b-instant"
# if "retrieval_k" not in st.session_state:
#     st.session_state.retrieval_k = 5
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Stores conversation

# Max chat turns to prevent token overflow
MAX_HISTORY_TURNS = 10  

def get_truncated_history():
    history = st.session_state.chat_history
    if len(history) > MAX_HISTORY_TURNS:
        # Keep only the last N turns
        truncated = history[-MAX_HISTORY_TURNS:]
        # Notify user
        st.warning(
            f"Conversation exceeded {MAX_HISTORY_TURNS} turns. "
            "Older messages have been truncated. "
            "If context seems incomplete, use 'Clear conversation' to start fresh."
        )
        return truncated
    return history

# Sidebar
with st.sidebar:
    st.header("Settings")

    if not st.session_state.get("groq_api_key", ""):
        st.warning("No GROQ_API_KEY found. Please enter it below:")

        entered_key = st.text_input(
            "Groq API Key",
            type="password",
            key="groq_api_input"
        )

        # Add short instructions with link (opens new tab)
        st.markdown(
            """
            To use this assistant you need a **Groq API key**.  
            You can obtain one for free here:  
            [Get a Groq API Key](https://console.groq.com/keys)
            """,
            unsafe_allow_html=True
        )

        if entered_key:
            st.session_state.groq_api_key = entered_key
            os.environ["GROQ_API_KEY"] = entered_key
            # st.success("Groq API Key configured successfully!")
            st.rerun()
    else:
        st.success("Groq API Key configured successfully.")

        # Only show detailed settings if API key is set
        st.selectbox(
            "Choose an LLM:",
            options=[
                "llama-3.1-8b-instant",
                "gemma2-9b-it",
                "qwen/qwen3-32b",
                "openai/gpt-oss-20b",
                "openai/gpt-oss-120b",
            ],
            index=0, # Default model selection
            key="model_name"
        )
        st.slider(
            "Number of chunks to retrieve (k)",
            min_value=2, max_value=20,
            value=st.session_state.get("retrieval_k",5),
            key="retrieval_k"
        )

        # Input for download filename prefix
        file_prefix = st.text_input("Filename prefix", value="conversation", key="fname_prefix")

        # Check if conversation exists
        history_exists = bool(st.session_state.chat_history)

        if history_exists:
            st.success(f"Conversation active. {len(st.session_state.chat_history)} turns recorded.")
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{file_prefix}_{timestamp}.md"
            md_text = "\n\n".join(
                [f"**User:** {turn['user']}\n\n**Assistant:** {turn['assistant']}" 
                for turn in st.session_state.chat_history]
            )
        else:
            st.info("No active conversation. Ask a question to begin.")
            filename, md_text = "conversation.md", "No conversation recorded yet."

        # Download button
        st.download_button(
            "Download conversation",
            data=md_text,
            file_name=filename,
            disabled=not history_exists,
            key="download_btn"
        )

        # Clear button
        if st.button("Clear conversation", disabled=not history_exists, key="clear_btn"):
            st.session_state.chat_history = []
            st.rerun()
        
# --------------------
# Cached Retriever
# --------------------
@st.cache_resource
def load_retriever(retrieval_k: int):
    """Heavy work (parse XML, chunk, build retriever) cached by k."""
    xml_path = os.path.join("data", "Easy Access Rules for Continuing Airworthiness (Regulation (EU) No 13212014).xml")
    clean_path = os.path.join("data", "easa_clean.xml")

    clean_xml = extract_clean_xml_from_package(xml_path, save_clean_path=clean_path)
    docs = convert_xml_to_documents(clean_xml)
    chunks = chunk_documents(docs)

    retriever = build_hybrid_retriever(chunks, k=retrieval_k)
    return retriever

# --------------------
# Main program logic
# --------------------
if st.session_state.groq_api_key:
    # Cached retriever stays based on retrieval_k
    retriever = load_retriever(st.session_state.get("retrieval_k",5))

    # Always build conversational chain with *current* model dropdown
    rag_chain = create_conversational_chain(
        retriever,
        model_name=st.session_state.model_name
    )

    # Chat interface
    user_q = st.chat_input("Ask a question about Continuing Airworthiness Regulation (EU) 1321/2014...")
    if user_q:
        with st.spinner("Searching through the regulations..."):
            # Get truncated history to avoid huge token usage
            truncated_history = get_truncated_history()

            # Convert structured chat history
            history_msgs = []
            for turn in truncated_history:
                history_msgs.append({"role": "human", "content": turn["user"]})
                history_msgs.append({"role": "ai", "content": turn["assistant"]})

            response = rag_chain.invoke({
                "chat_history": history_msgs,
                "input": user_q
            })

        # Extract assistant answer
        answer_content = response["answer"].content if hasattr(response["answer"], "content") else response["answer"]

        # Save to history
        st.session_state.chat_history.append({
            "user": user_q,
            "assistant": answer_content,
            "model": st.session_state.model_name
        })
        
        # Force re-render so sidebar picks up updated history immediately
        st.rerun()

    # Display full conversation so far
    for turn in st.session_state.chat_history:
        st.chat_message("user").markdown(turn["user"])
        with st.chat_message("assistant"):
            st.markdown(turn["assistant"])
            # show model used for this particular response
            if "model" in turn:
                st.caption(f"Response generated by: {turn['model']}")

else:
    st.warning("Please enter your GROQ_API_KEY in in the settings sidebar or otherwise set up an environment variable.")