<p align="center">
  <img src="https://img.shields.io/badge/python-3.12+-3776AB?style=flat&logo=python&logoColor=white" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/streamlit-1.41-FF4B4B?style=flat&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/ChromaDB-local-orange?style=flat" alt="ChromaDB">
  <img src="https://img.shields.io/badge/license-proprietary-red?style=flat" alt="Proprietary License">
</p>

<h1 align="center">🍽️ ChefVision</h1>

<p align="center">
  <em>Tu asistente culinario inteligente — Your intelligent culinary assistant</em>
</p>

<p align="center">
  Capture ingredients with your camera, get personalized recipe suggestions powered by AI.
</p>

---

## ⚠️ License & Usage

> **This software is proprietary and NOT open source.**
>
> All rights reserved © 2026 ChefVision.
> Unauthorized copying, distribution, modification, or use of this software, in whole or in part, is strictly prohibited without prior written permission from the project owners.
> This repository is shared privately among authorized team members only.

---

## Overview

ChefVision is a full-stack application that combines computer vision and AI to turn photos of your ingredients into recipe suggestions. Point your camera at a fridge, cupboard, or countertop — a fine-tuned YOLO model detects fruits, vegetables, and groceries, then an LLM acting as a personal chef suggests 3–4 dishes you can prepare.

The platform also includes an admin module for managing a recipe knowledge base: upload PDF cookbooks, automatically chunk and embed them into a local ChromaDB vector store, and use that knowledge to enrich recipe suggestions.

### Key Capabilities

| Feature | Description |
|---|---|
| 📷 Ingredient Detection | Capture photos from mobile or desktop; YOLO model identifies ingredients |
| 🤖 AI Chef | LLM (local or cloud) suggests 3–4 recipes with detailed or quick instructions |
| 📚 Recipe Knowledge Base | Upload PDF cookbooks via admin UI; auto-chunked and embedded into ChromaDB |
| 🔍 Smart Chunking | Two-tier strategy: recipe-boundary detection → character-based fallback with overlap |
| 🏷️ Recipe Detection | NLP scoring filters non-recipe content (TOC, intros) before embedding |
| 🗂️ File Management | View embedding status per file, detect duplicates, remove files + embeddings |
| 🔐 Role-Based Access | Admin and user roles with separate interfaces |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (Streamlit)                  │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │  Login   │  │  User View   │  │   Admin Panel     │  │
│  │  Screen  │  │  (Photos +   │  │   (PDF Upload +   │  │
│  │          │  │   Recipes)   │  │    Management)    │  │
│  └──────────┘  └──────┬───────┘  └────────┬──────────┘  │
└────────────────────────┼──────────────────┼─────────────┘
                         │                  │
┌────────────────────────┼──────────────────┼─────────────┐
│                    Backend (Python)        │             │
│  ┌─────────────────────┘                  │             │
│  │  ┌──────────────┐    ┌─────────────────┘             │
│  │  │ YOLO Model   │    │  Ingestion Pipeline           │
│  │  │ (Detection)  │    │  ┌────────────────────────┐   │
│  │  └──────┬───────┘    │  │ PDF Extract → Chunk →  │   │
│  │         │            │  │ Recipe Detect → Embed   │   │
│  │         ▼            │  └───────────┬────────────┘   │
│  │  ┌──────────────┐    │              │                │
│  │  │ LLM Engine   │    │              ▼                │
│  │  │ (Chef Mode)  │    │  ┌────────────────────────┐   │
│  │  └──────────────┘    │  │   ChromaDB (Local)     │   │
│  │                      │  │   Vector Store         │   │
│  │                      │  └────────────────────────┘   │
│  └──────────────────────┘                               │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
chefvision/
├── backend/
│   ├── config.py                  # Pydantic Settings (env + defaults)
│   ├── main.py                    # Backend entrypoint
│   ├── model/
│   │   └── schema.py              # Data models: Chunk, PageText, ChunkingConfig, etc.
│   └── services/
│       ├── chunking.py            # Chunking engine + recipe block detection
│       ├── error.py               # Custom error hierarchy (5 error types)
│       ├── ingestion.py           # PDF extraction + full ingestion pipeline
│       ├── vector_store.py        # ChromaDB operations (embed, delete, query)
│       └── audit.py               # Audit logging (placeholder)
│
├── frontend/
│   ├── app.py                     # Streamlit app entry (router + session)
│   ├── components/
│   │   ├── auth.py                # Login screen with role-based auth
│   │   ├── admin.py               # Admin panel: PDF upload, library, file mgmt
│   │   ├── user.py                # User view: photo upload, meal selector, recipes
│   │   └── styles.py              # Global CSS theme (cream/amber/sage palette)
│   ├── .streamlit/config.toml     # Streamlit theme & server config
│   ├── Dockerfile                 # Multi-stage Docker build
│   ├── docker-compose.yml         # Docker Compose for frontend service
│   └── requirements.txt           # Frontend Python dependencies
│
├── scripts/
│   └── ingest_pdf.py              # CLI tool for PDF ingestion
│
├── data/                          # PDF recipe books + ChromaDB storage
│   ├── *.pdf                      # Recipe book PDFs
│   └── chroma/                    # ChromaDB persistent storage
│
├── tests/                         # Property-based + unit tests
│   ├── test_chunking.py           # Chunking engine tests
│   ├── test_chunk_ids.py          # Deterministic ID tests
│   ├── test_config.py             # Configuration validation tests
│   ├── test_ingestion_pipeline.py # Pipeline orchestration tests
│   ├── test_pdf_extractor.py      # PDF extraction tests
│   ├── test_recipe_detection.py   # Recipe scoring tests
│   └── test_vector_store.py       # ChromaDB operation tests
│
└── .kiro/specs/                   # Feature specs (requirements, design, tasks)
```

---

## Getting Started

### Prerequisites

- Python 3.12+
- pip or a virtual environment manager
- (Optional) Docker & Docker Compose for containerized frontend

### 1. Clone the Repository

```bash
git clone <repository-url>
cd chefvision
```

### 2. Create Virtual Environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install pypdf pydantic-settings chromadb streamlit hypothesis pytest pillow python-dotenv
```

### 4. Configure Environment

Create a `.env` file in the project root (optional — defaults are provided):

```env
# OpenAI (for LLM-powered recipe suggestions)
OPEN_API_KEY=your-api-key-here

# Chunking parameters
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RECIPE_THRESHOLD=5

# Models
EMBEDDING_MODEL=text-embedding3-small
LLM_MODEL=gpt-4-o-mini
```

---

## Usage

### Running the Frontend (Streamlit)

```bash
streamlit run frontend/app.py
```

The app opens at `http://localhost:8501` with two roles:

| Role | Credentials | What you can do |
|---|---|---|
| Admin | `admin` / `chef123` | Upload PDFs, manage recipe library, view stats |
| User | `maria` / `cocina1` | Upload ingredient photos, select meal type, get recipes |

### Running with Docker

```bash
cd frontend
docker compose up --build
```

Access at `http://localhost:8501`.

### CLI: Ingest a PDF

```bash
python scripts/ingest_pdf.py path/to/recipe_book.pdf
```

Output shows processing status, chunks processed, and any errors.

### Running Tests

```bash
python -m pytest tests/ -v
```

The test suite uses Hypothesis for property-based testing across 10 correctness properties, plus unit tests for edge cases and error handling.

---

## Configuration Reference

All parameters are configurable via environment variables or `.env`:

| Variable | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | `1000` | Maximum characters per text chunk |
| `CHUNK_OVERLAP` | `200` | Overlap characters between consecutive chunks |
| `RECIPE_THRESHOLD` | `5` | Minimum score for a chunk to be classified as recipe content |
| `OPEN_API_KEY` | `None` | OpenAI API key for LLM features |
| `EMBEDDING_MODEL` | `text-embedding3-small` | Model used for text embeddings |
| `LLM_MODEL` | `gpt-4-o-mini` | LLM model for recipe generation |
| `TOP_K` | `20` | Number of top results for vector similarity search |
| `MIN_RELEVANCE_SCORE` | `0.4` | Minimum relevance threshold for search results |

---

## How the Chunking Pipeline Works

```
PDF Upload
    │
    ▼
┌──────────────────┐
│  PDF Extraction   │  Extract text page-by-page (skip blank pages)
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Smart Chunking   │  1. Detect recipe boundaries (titles + metadata signals)
│                   │  2. Merge small adjacent sections
│                   │  3. Split oversized sections on word boundaries with overlap
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Recipe Detection  │  Score chunks: cooking verbs + quantity patterns
│                   │  Filter: keep only chunks above threshold
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Vector Embed     │  Upsert to ChromaDB with deterministic IDs
│                   │  Metadata: filename, page number, chunk index
└──────────────────┘
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | Streamlit 1.41 | Web UI with custom CSS theme |
| Backend | Python 3.12 | Core logic and services |
| PDF Processing | pypdf | Text extraction from PDFs |
| Vector Store | ChromaDB (local) | Persistent embedding storage |
| Configuration | Pydantic Settings | Type-safe config with `.env` support |
| Testing | Hypothesis + pytest | Property-based and unit testing |
| Containerization | Docker + Compose | Frontend deployment |
| AI/ML | YOLO (detection), OpenAI (LLM) | Ingredient detection + recipe generation |

---

## Team

This project is developed by Dialcha, Itakeb, Darien-o.

---

<p align="center">
  <strong>© 2026 ChefVision — All Rights Reserved</strong><br>
  <em>Proprietary software. Unauthorized use is prohibited.</em>
</p>
