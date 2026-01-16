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
QASPER_DIR = DATA_DIR / "qasper"
RFC_DIR = DATA_DIR / "rfc"

# Chunking methods to evaluate
CHUNKING_METHODS = [
    {
        "id": "fixed_token",
        "name": "Fixed Token (500)",
        "display_name": "Fixed Token",
        "library": None,
        "description": "Fixed 500-token chunks with 50-token overlap",
    },
    {
        "id": "recursive",
        "name": "LangChain RecursiveCharacterTextSplitter",
        "display_name": "Recursive",
        "library": "LangChain",
        "description": "Recursive splitting on separators",
    },
    {
        "id": "flat_header",
        "name": "Flat Header Splitter",
        "display_name": "Flat Header",
        "library": None,
        "description": "Split on markdown headers only",
    },
    {
        "id": "docling",
        "name": "Docling HierarchicalChunker",
        "display_name": "Docling",
        "library": "Docling",
        "description": "Document-aware hierarchical chunking",
    },
]

# Dataset configurations
DATASETS = {
    "cuad": {
        "name": "CUAD",
        "domain": "legal",
        "description": "Contract Understanding Atticus Dataset",
        "source_url": "https://github.com/TheAtticusProject/cuad",
        "docs_count": 510,
        "questions_count": 6702,
    },
    "qasper": {
        "name": "QASPER",
        "domain": "academic",
        "description": "Question Answering on Scientific Papers",
        "source_url": "https://allenai.org/data/qasper",
        "docs_count": 1585,
        "questions_count": 4639,
    },
    "rfc": {
        "name": "RFC Corpus",
        "domain": "technical",
        "description": "IETF Request for Comments",
        "source_url": "https://www.rfc-editor.org/",
        "docs_count": 155,
        "questions_count": 1078,
    },
}

# Embedding settings
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# Evaluation settings
RETRIEVAL_K_VALUES = [1, 3, 5, 10]

# Environment
def get_openai_key():
    return os.environ.get("OPENAI_API_KEY", "")
