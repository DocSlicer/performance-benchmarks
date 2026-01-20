#!/usr/bin/env python3
"""
04_run_chunking.py

Run all chunking methods on benchmark documents.
Generates chunks for each method and stores them for evaluation.

Includes DocSlicer which requires PDF/HTML files for layout-aware chunking.

Usage:
    python 04_run_chunking.py [--method METHOD] [--dataset DATASET]
"""
import argparse
import json
import os
import sys
import re
import hashlib
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_DIR, CUAD_DIR, ACL_DIR, RFC_DIR,
    CHUNKING_METHODS, DATASETS, get_openai_key,
    get_docslicer_api_key, get_docslicer_api_url
)

# Chunking parameters
CHUNK_SIZE = 500  # tokens
CHUNK_OVERLAP = 50  # tokens


def load_json(path: Path) -> list:
    with open(path) as f:
        return json.load(f)


def save_json(data: list, path: Path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (4 chars per token)."""
    return len(text) // 4


def chunk_hash(text: str) -> str:
    """Generate hash for chunk deduplication."""
    return hashlib.md5(text.encode()).hexdigest()[:12]


# ============================================================
# TEXT-BASED CHUNKING METHODS
# ============================================================

def chunk_fixed_token(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """Fixed token chunking with overlap."""
    char_size = chunk_size * 4
    char_overlap = overlap * 4
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + char_size
        chunk_text = text[start:end]
        
        if chunk_text.strip():
            chunks.append({
                "text": chunk_text,
                "start": start,
                "end": min(end, len(text)),
                "method": "fixed_token",
            })
        
        start = end - char_overlap
        if start >= len(text) - char_overlap:
            break
    
    return chunks


def chunk_recursive(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """LangChain RecursiveCharacterTextSplitter."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        print("Installing langchain-text-splitters...")
        os.system("pip install langchain-text-splitters")
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size * 4,
        chunk_overlap=overlap * 4,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    splits = splitter.split_text(text)
    
    chunks = []
    pos = 0
    for split in splits:
        start = text.find(split, pos)
        if start == -1:
            start = pos
        
        chunks.append({
            "text": split,
            "start": start,
            "end": start + len(split),
            "method": "recursive",
        })
        pos = start + 1
    
    return chunks


def chunk_semantic(text: str, api_key: str) -> List[Dict]:
    """LangChain SemanticChunker using embeddings."""
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        print("Installing langchain packages...")
        os.system("pip install langchain-experimental langchain-openai")
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_openai import OpenAIEmbeddings
    
    embeddings = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")
    splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
    
    splits = splitter.split_text(text)
    
    chunks = []
    pos = 0
    for split in splits:
        start = text.find(split, pos)
        if start == -1:
            start = pos
        
        chunks.append({
            "text": split,
            "start": start,
            "end": start + len(split),
            "method": "semantic",
        })
        pos = start + 1
    
    return chunks


def chunk_flat_header(text: str) -> List[Dict]:
    """Split on markdown/document headers."""
    header_pattern = r'(?:^|\n)(#{1,6}\s+.+|[A-Z][A-Za-z\s]+:?\n[=-]+|\d+\.?\s+[A-Z][A-Za-z\s]+)'
    
    headers = list(re.finditer(header_pattern, text))
    
    if not headers:
        return [{
            "text": text,
            "start": 0,
            "end": len(text),
            "method": "flat_header",
        }]
    
    chunks = []
    
    for i, match in enumerate(headers):
        start = match.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "start": start,
                "end": end,
                "method": "flat_header",
            })
    
    return chunks


def chunk_docling(text: str) -> List[Dict]:
    """Docling HierarchicalChunker."""
    try:
        from docling_core.transforms.chunker import HierarchicalChunker
        from docling_core.types.doc import DoclingDocument
        from docling_core.types.doc.labels import DocItemLabel
    except ImportError:
        print("Installing docling...")
        os.system("pip install docling-core")
        try:
            from docling_core.transforms.chunker import HierarchicalChunker
            from docling_core.types.doc import DoclingDocument
            from docling_core.types.doc.labels import DocItemLabel
        except ImportError:
            print("  Docling not available, falling back to flat_header")
            return chunk_flat_header(text)
    
    try:
        doc = DoclingDocument(name="benchmark_doc")
        
        paragraphs = text.split('\n\n')
        for para in paragraphs:
            para = para.strip()
            if para:
                if para.startswith('#') or (len(para) < 100 and para.isupper()):
                    doc.add_text(label=DocItemLabel.SECTION_HEADER, text=para)
                else:
                    doc.add_text(label=DocItemLabel.PARAGRAPH, text=para)
        
        chunker = HierarchicalChunker()
        chunks_result = list(chunker.chunk(doc))
        
        chunks = []
        for chunk in chunks_result:
            chunk_text = chunk.text if hasattr(chunk, 'text') else str(chunk)
            start = text.find(chunk_text)
            
            chunks.append({
                "text": chunk_text,
                "start": start if start >= 0 else 0,
                "end": (start + len(chunk_text)) if start >= 0 else len(chunk_text),
                "method": "docling",
            })
        
        return chunks if chunks else chunk_flat_header(text)
        
    except Exception as e:
        print(f"  Docling error: {e}, falling back to flat_header")
        return chunk_flat_header(text)


# ============================================================
# DOCSLICER - Layout-Aware Chunking (requires PDF/HTML)
# ============================================================

def chunk_docslicer_local(
    doc: dict,
    dataset_dir: Path,
) -> List[Dict]:
    """
    Use DocSlicer's local parsing pipeline for layout-aware chunking.
    Requires pdf_path or html_path in the document.
    
    Returns empty list on failure (NO FALLBACK - for accurate benchmarking).
    """
    # Add the backend to path for imports
    backend_path = Path(__file__).parent.parent.parent / "backend"
    if str(backend_path) not in sys.path:
        sys.path.insert(0, str(backend_path))
    
    try:
        from app.services.parsing.orchestrator import run_pipeline
    except ImportError as e:
        print(f"    DocSlicer import error: {e}")
        return []  # NO FALLBACK
    
    # Determine file path
    pdf_path = doc.get("pdf_path")
    html_path = doc.get("html_path")
    
    if pdf_path:
        file_path = dataset_dir / pdf_path
        content_type = "pdf"
    elif html_path:
        file_path = dataset_dir / html_path
        content_type = "html"
    else:
        print(f"    No PDF or HTML path for doc {doc['id']}, skipping docslicer")
        return []  # NO FALLBACK
    
    if not file_path.exists():
        print(f"    File not found: {file_path}, skipping docslicer")
        return []  # NO FALLBACK
    
    try:
        # Read file content
        with open(file_path, "rb") as f:
            content = f.read()
        
        # For HTML, decode to string
        if content_type == "html":
            content = content.decode("utf-8", errors="replace")
        
        # Run async pipeline
        result = asyncio.run(
            run_pipeline(
                content=content,
                content_type=content_type,
                source_url=None,
            )
        )
        
        # Unpack result - may be tuple of (df_chunks, df_table_cells, ...) or just df_chunks
        if isinstance(result, tuple):
            df_chunks = result[0]
        else:
            df_chunks = result
        
        # Convert DataFrame to chunk list
        chunks = []
        for i, row in df_chunks.iterrows():
            chunk_text = row.get("text", "")
            if not chunk_text:
                continue
            
            # Try to find position in original text (approximate)
            start = doc["text"].find(chunk_text[:100]) if len(chunk_text) > 100 else doc["text"].find(chunk_text)
            if start == -1:
                start = 0
            
            chunks.append({
                "text": chunk_text,
                "start": start,
                "end": start + len(chunk_text),
                "method": "docslicer",
                # Include DocSlicer-specific metadata
                "heading": row.get("chunk_heading", ""),
                "heading_level": row.get("heading_level", None),
                "page_number": row.get("page_number", None),
            })
        
        return chunks  # Return whatever we got (even if empty)
        
    except Exception as e:
        print(f"    DocSlicer error: {e}")
        return []  # NO FALLBACK


def chunk_docslicer_api(
    doc: dict,
    dataset_dir: Path,
    api_url: str,
    api_key: str,
) -> List[Dict]:
    """
    Use DocSlicer API for layout-aware chunking.
    Requires pdf_path or html_path in the document.
    
    Returns empty list on failure (NO FALLBACK - for accurate benchmarking).
    """
    import requests
    
    pdf_path = doc.get("pdf_path")
    html_path = doc.get("html_path")
    
    if pdf_path:
        file_path = dataset_dir / pdf_path
        content_type = "application/pdf"
    elif html_path:
        file_path = dataset_dir / html_path
        content_type = "text/html"
    else:
        print(f"    No PDF or HTML path for doc {doc['id']}, skipping docslicer")
        return []  # NO FALLBACK
    
    if not file_path.exists():
        print(f"    File not found: {file_path}, skipping docslicer")
        return []  # NO FALLBACK
    
    try:
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, content_type)}
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            
            response = requests.post(
                f"{api_url}/api/v1/chunk",
                files=files,
                headers=headers,
                timeout=120,
            )
            response.raise_for_status()
        
        result = response.json()
        chunks_data = result.get("chunks", [])
        
        chunks = []
        for i, chunk_data in enumerate(chunks_data):
            chunk_text = chunk_data.get("text", "")
            if not chunk_text:
                continue
            
            start = doc["text"].find(chunk_text[:100]) if len(chunk_text) > 100 else doc["text"].find(chunk_text)
            if start == -1:
                start = 0
            
            chunks.append({
                "text": chunk_text,
                "start": start,
                "end": start + len(chunk_text),
                "method": "docslicer",
                "heading": chunk_data.get("heading", ""),
                "heading_level": chunk_data.get("heading_level"),
                "page_number": chunk_data.get("page_number"),
            })
        
        return chunks  # Return whatever we got (even if empty)
        
    except Exception as e:
        print(f"    DocSlicer API error: {e}")
        return []  # NO FALLBACK


# ============================================================
# MAIN PROCESSING
# ============================================================

def process_document(
    doc: dict,
    methods: List[str],
    dataset_dir: Path,
    api_key: str = None,
    docslicer_api_url: str = None,
    docslicer_api_key: str = None,
    use_docslicer_local: bool = True,
) -> Dict[str, List[Dict]]:
    """Process a single document with all specified chunking methods."""
    text = doc["text"]
    results = {}
    
    for method in methods:
        try:
            if method == "fixed_token":
                chunks = chunk_fixed_token(text)
            elif method == "recursive":
                chunks = chunk_recursive(text)
            elif method == "semantic":
                if not api_key:
                    print(f"    Skipping semantic (no API key)")
                    continue
                chunks = chunk_semantic(text, api_key)
            elif method == "flat_header":
                chunks = chunk_flat_header(text)
            elif method == "docling":
                chunks = chunk_docling(text)
            elif method == "docslicer":
                # DocSlicer requires PDF/HTML
                if use_docslicer_local:
                    chunks = chunk_docslicer_local(doc, dataset_dir)
                else:
                    if not docslicer_api_url:
                        print(f"    Skipping docslicer (no API URL)")
                        continue
                    chunks = chunk_docslicer_api(doc, dataset_dir, docslicer_api_url, docslicer_api_key)
            else:
                print(f"    Unknown method: {method}")
                continue
            
            # Add metadata to each chunk
            for i, chunk in enumerate(chunks):
                chunk["chunk_id"] = f"{doc['id']}_{method}_{i}"
                chunk["document_id"] = doc["id"]
                chunk["chunk_index"] = i
                chunk["char_count"] = len(chunk["text"])
                chunk["token_estimate"] = estimate_tokens(chunk["text"])
                chunk["hash"] = chunk_hash(chunk["text"])
            
            results[method] = chunks
            
        except Exception as e:
            print(f"    Error with {method}: {e}")
            results[method] = []
    
    return results


def process_dataset(
    dataset_name: str,
    dataset_dir: Path,
    methods: List[str],
    api_key: str = None,
    docslicer_api_url: str = None,
    docslicer_api_key: str = None,
    use_docslicer_local: bool = True,
    max_docs: int = None
) -> Dict[str, Any]:
    """Process all documents in a dataset."""
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name.upper()}")
    print(f"{'='*60}")
    
    prepared_dir = dataset_dir / "prepared"
    docs_file = prepared_dir / "documents.json"
    
    if not docs_file.exists():
        print(f"  ERROR: Run 02_prepare_documents.py first")
        return None
    
    docs = load_json(docs_file)
    if max_docs:
        docs = docs[:max_docs]
    
    # Check PDF/HTML availability for docslicer
    if "docslicer" in methods:
        docs_with_files = sum(1 for d in docs if d.get("pdf_path") or d.get("html_path"))
        print(f"  Documents: {len(docs)} ({docs_with_files} with PDF/HTML for docslicer)")
    else:
        print(f"  Documents: {len(docs)}")
    print(f"  Methods: {', '.join(methods)}")
    
    # Process each document
    all_chunks = {method: [] for method in methods}
    stats = {method: {"total_chunks": 0, "total_chars": 0, "total_tokens": 0} for method in methods}
    
    for i, doc in enumerate(docs):
        doc_title = doc.get('title', doc['id'])[:40]
        print(f"  [{i+1}/{len(docs)}] {doc_title}...", end=" ", flush=True)
        
        results = process_document(
            doc, methods, dataset_dir, api_key,
            docslicer_api_url, docslicer_api_key, use_docslicer_local
        )
        
        chunk_counts = [f"{m}:{len(results.get(m, []))}" for m in methods if results.get(m)]
        print(f"-> {', '.join(chunk_counts)}")
        
        for method, chunks in results.items():
            all_chunks[method].extend(chunks)
            stats[method]["total_chunks"] += len(chunks)
            stats[method]["total_chars"] += sum(c["char_count"] for c in chunks)
            stats[method]["total_tokens"] += sum(c["token_estimate"] for c in chunks)
    
    # Save chunks for each method
    chunks_dir = prepared_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    
    for method, chunks in all_chunks.items():
        if chunks:
            save_json(chunks, chunks_dir / f"{method}.json")
            print(f"  Saved {len(chunks)} {method} chunks")
    
    # Calculate averages
    for method in methods:
        if stats[method]["total_chunks"] > 0:
            stats[method]["avg_chunk_chars"] = stats[method]["total_chars"] // stats[method]["total_chunks"]
            stats[method]["avg_chunk_tokens"] = stats[method]["total_tokens"] // stats[method]["total_chunks"]
            stats[method]["chunks_per_doc"] = stats[method]["total_chunks"] / len(docs)
    
    return {
        "dataset": dataset_name,
        "documents": len(docs),
        "methods": stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Run chunking methods on benchmark documents")
    parser.add_argument(
        "--method",
        choices=["fixed_token", "recursive", "semantic", "flat_header", "docling", "docslicer", "all", "all_with_docslicer"],
        default="all",
        help="Which method to run (default: all without docslicer)"
    )
    parser.add_argument(
        "--dataset",
        choices=["cuad", "acl", "rfc", "all"],
        default="all",
        help="Which dataset to process (default: all)"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Max documents per dataset (for testing)"
    )
    parser.add_argument(
        "--docslicer-api",
        action="store_true",
        help="Use DocSlicer API instead of local pipeline"
    )
    args = parser.parse_args()
    
    print("="*60)
    print("BENCHMARK CHUNKING")
    print("="*60)
    
    # Determine methods to run
    if args.method == "all":
        methods = ["fixed_token", "recursive", "flat_header", "docling"]
        print("Note: Use --method all_with_docslicer to include DocSlicer")
    elif args.method == "all_with_docslicer":
        methods = ["fixed_token", "recursive", "flat_header", "docling", "docslicer"]
    else:
        methods = [args.method]
    
    print(f"Methods: {', '.join(methods)}")
    
    # Get API keys
    openai_key = get_openai_key()
    docslicer_api_url = get_docslicer_api_url()
    docslicer_api_key = get_docslicer_api_key()
    use_docslicer_local = not args.docslicer_api
    
    if "semantic" in methods and not openai_key:
        print("WARNING: OPENAI_API_KEY not set, skipping semantic chunking")
        methods = [m for m in methods if m != "semantic"]
    
    if "docslicer" in methods:
        if use_docslicer_local:
            print("DocSlicer: Using local pipeline")
        else:
            print(f"DocSlicer: Using API at {docslicer_api_url}")
    
    # Determine datasets
    datasets = []
    if args.dataset in ["cuad", "all"]:
        datasets.append(("cuad", CUAD_DIR))
    if args.dataset in ["acl", "all"]:
        datasets.append(("acl", ACL_DIR))
    if args.dataset in ["rfc", "all"]:
        datasets.append(("rfc", RFC_DIR))
    
    all_stats = []
    
    for name, dir_path in datasets:
        stats = process_dataset(
            name, dir_path, methods, openai_key,
            docslicer_api_url, docslicer_api_key, use_docslicer_local,
            args.max_docs
        )
        if stats:
            all_stats.append(stats)
    
    # Summary
    print("\n" + "="*60)
    print("CHUNKING SUMMARY")
    print("="*60)
    
    print(f"\n{'Method':<15} {'Chunks':>10} {'Avg Tokens':>12} {'Chunks/Doc':>12}")
    print("-"*52)
    
    for method in methods:
        total_chunks = sum(s["methods"].get(method, {}).get("total_chunks", 0) for s in all_stats)
        total_tokens = sum(s["methods"].get(method, {}).get("total_tokens", 0) for s in all_stats)
        total_docs = sum(s["documents"] for s in all_stats)
        
        avg_tokens = total_tokens // total_chunks if total_chunks else 0
        chunks_per_doc = total_chunks / total_docs if total_docs else 0
        
        print(f"{method:<15} {total_chunks:>10,} {avg_tokens:>12} {chunks_per_doc:>12.1f}")
    
    # Save combined stats
    stats_file = DATA_DIR / "chunking_stats.json"
    save_json({
        "datasets": all_stats,
        "methods": methods,
    }, stats_file)
    print(f"\nStats saved to {stats_file}")


if __name__ == "__main__":
    main()
