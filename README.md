# EASA Continuing Airworthiness Compliance Assistant  

A **Streamlit Retrieval-Augmented Generation (RAG) app** for querying the **EASA Continuing Airworthiness Regulation (EU No 1321/2014)**.  

This project combines modern LLMs from [Groq](https://groq.com/) with a hybrid search pipeline (semantic + lexical) to provide precise, regulation‑grounded answers.  

---

## Features

- **Regulation‑aware assistant** trained on *EU No 1321/2014*.  
- **Hybrid retriever**: combines FAISS (vector search using sentence‑transformers) + BM25 (keyword search).  
- **Multiple model support** via Groq API: easily switch between LLMs in the sidebar.  
- **Conversation memory**: multi‑turn dialogue with full threading.  
- **Token‑safe truncation**: conversation buffer trimmed safely if too long.  
- **Conversation controls**:  
  - Download conversation as a `.md` file (with timestamped filenames).  
  - Save conversations locally on the server (to custom folders).  
  - Clear/reset chat.  
- **Transparent responses**:  
  - Model name displayed under each assistant response.  
  - Optional retrieval context for auditability.  
- **Flexible API key handling**: load from `.env` or enter directly in sidebar with one‑click Groq key link.

---

## Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get a Groq API Key  
- Sign up at [Groq Console](https://console.groq.com/keys).  
- Copy your key.  
- Store it in a `.env` file or paste it in the sidebar on first run.

`.env` file example:
```env
GROQ_API_KEY=your_api_key_here
```

### 4. Run the app
```bash
streamlit run app.py
```

Access in your browser at [http://localhost:8501](http://localhost:8501).

---

## Project Structure

```
.
├── app.py                # Main Streamlit app
├── extractor.py          # Utilities to clean and parse XML regulations
├── data/
│   └── Easy Access Rules for Continuing Airworthiness (Regulation (EU) No 13212014).xml
├── vectorstores/         # Local FAISS index storage
├── requirements.txt
└── README.md
```

---

## How It Works

1. **Data ingestion**
   - Parses the XML “Easy Access Rules” doc.
   - Converts to LangChain `Document` objects and chunks intelligently.

2. **Indexing & Retrieval**
   - FAISS + HuggingFace `sentence-transformers/all-MiniLM-L6-v2`.
   - BM25 retriever for keyword match.
   - Combined into an `EnsembleRetriever`.

3. **RAG flow**
   - Retrieved context + conversation history feed into a LangChain `ChatPromptTemplate`.
   - LLM (via Groq API) answers in plain English, citing official reg references.

---

## Configuration Options

Available in **sidebar**:
- LLM model choice (Groq‑hosted models).
- Retrieval depth `k` (number of chunks).  
- Filename prefix for exports.  
- Conversation management (download / clear).

---

## Example Workflow
1. Pick your preferred LLM in sidebar (`llama-3.1-8b-instant`, `gemma2-9b-it`, etc).  
2. Type:  
   *“What are the requirements for continuing airworthiness management organisations under Part‑CAMO?”*  
3. Assistant responds in plain English → followed by regulatory references.  
4. Export the conversation as `.md` for audit or record‑keeping.  

---

## Disclaimer
This tool provides **informational summaries only**.  
For compliance decisions always refer to the official **EASA regulations** and consult appropriate experts.  

---

## Contributing
Pull requests are welcome!  
For major changes, please open an issue first to discuss what you’d like to modify.  

---

## License
MIT License — free to use, modify, and distribute.  

---
