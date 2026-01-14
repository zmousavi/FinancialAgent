# Financial Agent

RAG-powered agent for querying SEC financial filings (10-K annual reports).

Ask questions about company finances, risks, business models, and more - with citations from official SEC documents.

## Installation

```bash
pip install financial-agent
```

## Quick Start

### 1. Get a Gemini API Key (free)

Get your free API key at: https://aistudio.google.com/app/apikey

### 2. Set Environment Variable

```bash
export GOOGLE_API_KEY=your-api-key-here
```

### 3. Download Vector Database

Download the pre-built vector database from [GitHub Releases](https://github.com/zmousavi/FinancialAgent/releases) and extract it.

### 4. Use the Agent

```python
from financial_agent import FinancialAgent

# Initialize with path to vector database
agent = FinancialAgent("path/to/vector_db")

# Ask a question
result = agent.query("What are Apple's main revenue sources?")

# Print the answer
print(result["answer"])

# Print citations
for citation in result["citations"]:
    print(f"[{citation['reference_number']}] {citation['user_friendly_format']}")
```

## Features

- **Natural Language Queries**: Ask questions in plain English
- **Multi-Company Support**: Query data from AAPL, MSFT, GOOGL, AMZN, TSLA, META, NFLX, NVDA, WMT
- **Citations**: Every answer includes references to source documents
- **Smart Retrieval**: Hybrid search combining semantic similarity and keyword matching

## Example Queries

```python
# Single company questions
agent.query("What are Apple's main revenue sources?")
agent.query("What risks does Tesla face?")

# Comparison questions
agent.query("Compare Apple and Microsoft's business models")

# Specific topics
agent.query("What does Amazon say about competition?")
```

## Response Format

```python
result = agent.query("What are Apple's revenues?")

# result contains:
{
    "query": "What are Apple's revenues?",
    "answer": "According to Apple's 2023 annual report...",
    "citations": [...],
    "retrieved_chunks": 5,
    "model_used": "gemini-2.0-flash-exp",
    "timestamp": "2024-01-15T10:30:00"
}
```

## Requirements

- Python 3.9+
- Google Gemini API key (free tier available)
- Pre-built vector database (download from releases)

---

# Building Your Own Vector Database (Advanced)

If you want to build your own vector database with fresh data or different companies, follow these instructions.

## Pipeline Setup

```bash
git clone https://github.com/zmousavi/FinancialAgent.git
cd FinancialAgent
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[pipeline]"
```

## Environment Variables

Create a `.env` file:

```bash
# SEC API (from sec-api.io)
SEC_API_KEY=your_sec_api_key_here
CONTACT_EMAIL=your_email@example.com

# Google Cloud / Vertex AI (for embeddings)
GCP_PROJECT_ID=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=us-central1
```

## Run the Pipeline

```bash
# 1. Download SEC filings
python scripts/01_download_filings.py

# 2. Clean filing text
python scripts/02_clean_sec_data.py

# 3. Analyze document structure
python scripts/03_analyze_documents.py

# 4. Create chunks
python scripts/04_create_chunks.py

# 5. Generate embeddings (requires GCP)
python scripts/05_create_embeddings.py

# 6. Set up vector database
python scripts/06_setup_vector_db.py
```

## Project Structure

```
financial-agent/
├── src/
│   └── financial_agent/        # The pip-installable library
│       ├── __init__.py
│       ├── agent.py            # FinancialAgent class
│       ├── vector_db.py        # Vector database search
│       └── chunk_utils.py      # Text chunking utilities
├── scripts/                    # Data pipeline scripts
│   ├── 01_download_filings.py
│   ├── 02_clean_sec_data.py
│   ├── 03_analyze_documents.py
│   ├── 04_create_chunks.py
│   ├── 05_create_embeddings.py
│   └── 06_setup_vector_db.py
├── vector_db/                  # Pre-built FAISS index
├── pyproject.toml
└── README.md
```

## License

MIT
