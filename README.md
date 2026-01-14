# FinancialChatbot

A complete RAG (Retrieval Augmented Generation) system for financial document analysis. This project fetches SEC 10-K filings, processes them into optimized chunks, generates embeddings, and provides an intelligent question-answering interface powered by Google Gemini.

## Features

- **Automated SEC filing download** with rate limiting and proper headers
- **Intelligent text cleaning** to remove viewer artifacts and extract narrative content
- **Smart document analysis** to determine optimal chunking strategies
- **Section-based chunking** that preserves SEC document structure
- **Vector embeddings** using Google Vertex AI (text-embedding-005)
- **FAISS vector database** for fast similarity search
- **RAG query interface** with Google Gemini for intelligent Q&A
- **Multi-company support** with balanced retrieval for comparison queries
- **Configurable pipeline** with YAML-based settings

## Project Structure

```
FinancialChatbot/
├── 01_download_filings.py      # Download SEC 10-K filings
├── 01a_download_annual.py      # Download annual reports specifically
├── 02_clean_sec_data.py        # Clean SEC filing text
├── 02_clean_yahoo_finance_data.py  # Clean Yahoo Finance market data
├── 03_analyze_documents.py     # Analyze document structure for chunking
├── 04_create_chunks.py         # Create text chunks for RAG
├── 05_create_embeddings.py     # Generate vector embeddings (Vertex AI)
├── 06_setup_vector_db.py       # Set up FAISS vector database
├── 07_generation_system.py     # RAG pipeline with Gemini LLM
├── chunk_utils.py              # Text chunking utilities
├── config.yaml                 # Configuration settings
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── sec_data/                   # Raw HTML filings (gitignored)
├── sec_txt/                    # Raw text extracts (gitignored)
├── sec_txt_clean/              # Cleaned text files (gitignored)
├── chunks/                     # Processed chunks (gitignored)
├── embeddings/                 # Generated embeddings (gitignored)
└── vector_db/                  # FAISS vector database (gitignored)
```

## Setup

```bash
git clone https://github.com/yourusername/FinancialChatbot.git
cd FinancialChatbot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Create environment file
cp .env.example .env
# Edit .env with your credentials (see Environment Variables below)
```

## Environment Variables

Create a `.env` file with:

```bash
# SEC API (from sec-api.io)
SEC_API_KEY=your_sec_api_key_here
CONTACT_EMAIL=your_email@example.com

# Google Cloud / Vertex AI
GCP_PROJECT_ID=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=your-google-cloud-location
```

**Google Cloud Setup:**
1. Create a GCP project with Vertex AI API enabled
2. Set up authentication: `gcloud auth application-default login`
3. Ensure your account has Vertex AI permissions

## Usage

### 1. Download SEC Filings
```bash
python3 01_download_filings.py
```
Downloads 10-K filings for companies in `config.yaml` (default: AAPL, MSFT, GOOGL, AMZN, TSLA, META, NFLX, NVDA, WMT).

### 2. Clean Filing Text
```bash
python3 02_clean_sec_data.py
```
Removes HTML artifacts and extracts clean narrative text.

### 3. Analyze Document Structure
```bash
python3 03_analyze_documents.py
```
Analyzes documents to determine optimal chunking strategy.

### 4. Create Chunks
```bash
python3 04_create_chunks.py
```
Creates section-based chunks optimized for RAG retrieval.

### 5. Generate Embeddings
```bash
python3 05_create_embeddings.py
```
Generates vector embeddings using Vertex AI text-embedding-005.

### 6. Set Up Vector Database
```bash
python3 06_setup_vector_db.py
```
Creates FAISS index for fast similarity search.

### 7. Run RAG System
```bash
python3 07_generation_system.py
```
Tests the complete RAG pipeline with sample queries.

## Example Queries

```python
from 07_generation_system import FinancialRAGSystem

rag = FinancialRAGSystem('vector_db/')

# Single company query
result = rag.query("What are Apple's main revenue sources?")
print(result['answer'])

# Comparison query (automatically balances retrieval)
result = rag.query("Compare Apple and Microsoft's business models")
print(result['answer'])

# Risk-focused query
result = rag.query("What are the key risk factors for Tesla?")
print(result['answer'])
```

## Configuration

Edit `config.yaml` to customize:

```yaml
tickers: ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NFLX", "NVDA", "WMT"]
form_type: "10-K"

embeddings:
  provider: "vertex-ai"
  model: "text-embedding-005"
  batch_size: 50

vertex_ai:
  project_id_env_var: "GCP_PROJECT_ID"
  location: "us-central1"
  embedding_model: "text-embedding-005"
  llm_model: "gemini-1.5-flash"
```

## Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│ Query Embedding │  (Vertex AI text-embedding-005)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FAISS Search   │  (Vector similarity search)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Context Builder │  (Format retrieved chunks)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Gemini LLM     │  (Generate response with citations)
└────────┬────────┘
         │
         ▼
    Response with Citations
```

## Requirements

- Python 3.8+
- SEC API key (from [sec-api.io](https://sec-api.io))
- Google Cloud account with Vertex AI enabled
- Valid email for SEC request headers

## Key Dependencies

- `google-cloud-aiplatform` - Vertex AI for embeddings and Gemini
- `faiss-cpu` - Vector similarity search
- `sec-api` - SEC filing access
- `beautifulsoup4` - HTML parsing
- `spacy` - Text processing
- `tiktoken` - Token counting
- `yfinance` - Market data
- `pandas` - Data manipulation

See `requirements.txt` for complete list.

## License

MIT License
