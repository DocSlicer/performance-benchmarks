"""
Benchmark Suite Configuration

Shared settings for all benchmark scripts.
"""
import os
from pathlib import Path

# Base paths
BENCHMARKS_DIR = Path(__file__).parent
DATA_DIR = BENCHMARKS_DIR / "data"

# Dataset directories
CUAD_DIR = DATA_DIR / "cuad"
ACL_DIR = DATA_DIR / "acl"  # Replaces QASPER
QASPER_DIR = DATA_DIR / "qasper"  # Legacy, kept for compatibility
RFC_DIR = DATA_DIR / "rfc"

# Supabase tables for benchmark data
BENCHMARK_DOCS_TABLE = "benchmark_documents"
BENCHMARK_QUESTIONS_TABLE = "benchmark_questions"
BENCHMARK_CHUNKS_TABLE = "benchmark_chunks"

# Chunking methods to evaluate
CHUNKING_METHODS = [
    {
        "id": "fixed_token",
        "name": "Fixed Token (500)",
        "display_name": "Fixed Token",
        "library": None,
        "description": "Fixed 500-token chunks with 50-token overlap",
        "requires_pdf": False,
    },
    {
        "id": "recursive",
        "name": "LangChain RecursiveCharacterTextSplitter",
        "display_name": "Recursive",
        "library": "LangChain",
        "description": "Recursive splitting on separators",
        "requires_pdf": False,
    },
    {
        "id": "semantic",
        "name": "LangChain SemanticChunker",
        "display_name": "Semantic",
        "library": "LangChain",
        "description": "Embedding-based semantic boundaries",
        "requires_pdf": False,
    },
    {
        "id": "flat_header",
        "name": "Flat Header Splitter",
        "display_name": "Flat Header",
        "library": None,
        "description": "Split on markdown headers only",
        "requires_pdf": False,
    },
    {
        "id": "docling",
        "name": "Docling HierarchicalChunker",
        "display_name": "Docling",
        "library": "Docling",
        "description": "Document-aware hierarchical chunking",
        "requires_pdf": False,
    },
    {
        "id": "docslicer",
        "name": "DocSlicer",
        "display_name": "DocSlicer",
        "library": "DocSlicer",
        "description": "Smart document-aware chunking with layout analysis",
        "requires_pdf": True,  # DocSlicer requires PDF or HTML input
    },
]

# Dataset configurations
DATASETS = {
    "cuad": {
        "name": "CUAD",
        "domain": "legal",
        "description": "Contract Understanding Atticus Dataset - Legal contracts with PDFs",
        "source_url": "https://huggingface.co/datasets/theatricusproject/cuad",
        "docs_count": 510,
        "questions_count": 20910,
        "has_pdf": True,
        "has_html": False,
    },
    "acl": {
        "name": "ACL Anthology",
        "domain": "academic",
        "description": "Academic NLP papers with PDFs from ACL Anthology",
        "source_url": "https://aclanthology.org/",
        "docs_count": 40,  # Curated subset
        "questions_count": 0,  # To be generated
        "has_pdf": True,
        "has_html": False,
    },
    "rfc": {
        "name": "RFC Corpus",
        "domain": "technical",
        "description": "IETF Request for Comments - Technical standards with HTML",
        "source_url": "https://www.rfc-editor.org/",
        "docs_count": 155,
        "questions_count": 0,  # To be generated
        "has_pdf": False,
        "has_html": True,
    },
}

# Legacy dataset (kept for compatibility)
LEGACY_DATASETS = {
    "qasper": {
        "name": "QASPER",
        "domain": "academic",
        "description": "Question Answering on Scientific Papers (no PDFs available)",
        "source_url": "https://allenai.org/data/qasper",
        "docs_count": 1585,
        "questions_count": 4639,
        "has_pdf": False,
        "has_html": False,
    },
}

# Embedding settings
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# Evaluation settings
RETRIEVAL_K_VALUES = [1, 3, 5, 10]

# Environment
def get_supabase_url():
    return os.environ.get("SUPABASE_URL", "")

def get_supabase_key():
    return os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

def get_openai_key():
    return os.environ.get("OPENAI_API_KEY", "")

def get_docslicer_api_key():
    """Get DocSlicer API key for benchmark chunking."""
    return os.environ.get("DOCSLICER_API_KEY", "")

def get_docslicer_api_url():
    """Get DocSlicer API URL (default: localhost for dev)."""
    return os.environ.get("DOCSLICER_API_URL", "http://localhost:8000")
