# AD Biomarker RAG Chatbot

This project implements a retrieval-augmented generation (RAG) system for scientific question answering using Alzheimer’s disease biomarker literature. The pipeline includes:

- Bulk download of PubMed Central full-text XMLs.
- Extraction and normalization of paper text.
- Section-aware chunking.
- Dense retrieval using FAISS.  
- Embedding generation using [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B).
- Question answering using Claude Sonnet 4.5.
- A Streamlit chat interface for interactive querying.

## Project Structure

- `scripts/`  
  - `download_papers.py`: Fetch and parse PMC articles  
  - `build_index.py`: Chunk text, embed, build FAISS index  
  - `query.py`: Retrieval and RAG logic  
  - `app.py`: Streamlit chatbot

- `data/` (ignored by .gitignore)  
  - `papers/`: Raw and processed text  
  - `processed/`: Chunks, metadata, FAISS index

- `models/` (ignored by .gitignore)  
  - HuggingFace model caches

## Running the Chatbot

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Build the index (once)
```bash
python scripts/download_papers.py
python scripts/build_index.py
```

### 4. Run the streamlit app
```bash
streamlit run scripts/app.py
```

## Notes
- No model weights or data are included in this repository.  
- The system is designed for local embedding generation and cloud LLM inference.