# ChefVision App

A recipe ingestion and retrieval system that extracts recipes from PDF cookbooks, detects recipe content using NLP heuristics, and stores them as embeddings in a local ChromaDB vector store for later retrieval and recipe suggestions.

## What it does

1. **PDF Extraction** — Reads uploaded PDF recipe books and extracts text page by page, skipping blank pages.
2. **Smart Chunking** — Splits extracted text into chunks using a two-tier strategy: first tries to detect recipe boundaries (titles, ingredient lists, metadata patterns), then falls back to character-based splitting with configurable overlap. Works across different cookbook formats and languages.
3. **Recipe Detection** — Scores each chunk for recipe relevance based on cooking verbs and quantity patterns, filtering out non-recipe content like introductions and table of contents.
4. **Vector Embedding** — Stores qualifying recipe chunks in ChromaDB with deterministic IDs and metadata for deduplication and targeted retrieval.
5. **Admin UI** — A Streamlit interface for uploading PDFs, viewing embedding status, and managing stored files.

## Project Structure

```
chefvision-app/
├── backend/
│   ├── config.py                 # App configuration (Pydantic Settings)
│   ├── main.py                   # App entrypoint (placeholder)
│   ├── model/
│   │   └── schema.py             # Data models (Chunk, PageText, ChunkingConfig, etc.)
│   └── services/
│       ├── chunking.py           # Chunking engine + recipe block detection
│       ├── error.py              # Custom error hierarchy
│       ├── ingestion.py          # PDF extraction + ingestion pipeline
│       └── vector_store.py       # ChromaDB operations (embed, delete, query)
├── frontend/
│   └── app.py                    # Streamlit admin interface
├── scripts/
│   └── ingest_pdf.py             # CLI script for PDF ingestion
├── data/                         # PDF recipe books (not tracked in git)
└── tests/                        # Property-based + unit tests (Hypothesis + pytest)
```

## Setup

Requires Python 3.12+.

```bash
# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install pypdf pydantic-settings hypothesis pytest chromadb streamlit
```

## Usage

### Admin UI

```bash
streamlit run frontend/app.py
```

Upload PDF recipe books, view embedding status, and delete files through the web interface.

### CLI

```bash
python scripts/ingest_pdf.py path/to/recipe_book.pdf
```

### Running Tests

```bash
python -m pytest tests/ -v
```

The test suite includes 23 tests covering 10 correctness properties validated through property-based testing with Hypothesis.

## Configuration

Chunking parameters can be configured via environment variables or a `.env` file:

| Variable           | Default | Description                              |
|--------------------|---------|------------------------------------------|
| `CHUNK_SIZE`       | 1000    | Max characters per chunk                 |
| `CHUNK_OVERLAP`    | 200     | Overlap characters between chunks        |
| `RECIPE_THRESHOLD` | 5       | Min score for recipe block classification|

## Tech Stack

- **Python 3.12** — Runtime
- **pypdf** — PDF text extraction
- **ChromaDB** — Local vector store for embeddings
- **Streamlit** — Admin web interface
- **Pydantic Settings** — Configuration management
- **Hypothesis + pytest** — Property-based and unit testing
