#!/usr/bin/env python3
"""
05_evaluate_retrieval.py

Evaluate retrieval performance for each chunking method.
- Embed all chunks and questions
- Calculate Recall@k, MRR@k, nDCG@k for each method
- Generate comparison report

Usage:
    python 05_evaluate_retrieval.py [--dataset DATASET] [--method METHOD]
"""
import argparse
import json
import os
import sys
import time
import math
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_DIR, CUAD_DIR, QASPER_DIR, RFC_DIR,
    EMBEDDING_MODEL, RETRIEVAL_K_VALUES, get_openai_key
)

# Batch size for embeddings (smaller to avoid rate limits)
EMBEDDING_BATCH_SIZE = 50
MAX_CHUNK_TOKENS = 8000  # OpenAI limit is 8192
MAX_RETRIES = 3


def load_json(path: Path) -> list:
    with open(path) as f:
        return json.load(f)


def save_json(data: Any, path: Path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def truncate_text(text: str, max_tokens: int = MAX_CHUNK_TOKENS) -> str:
    """Truncate text to fit within token limit (rough estimate: 4 chars per token)."""
    max_chars = max_tokens * 4
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def get_embeddings(texts: List[str], client, model: str = EMBEDDING_MODEL, show_progress: bool = False, prefix: str = "") -> List[List[float]]:
    """Get embeddings for a list of texts with retry logic."""
    embeddings = []
    total = len(texts)
    
    # Truncate texts that are too long
    texts = [truncate_text(t) for t in texts]
    
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i:i + EMBEDDING_BATCH_SIZE]
        
        if show_progress:
            print(f"\r{prefix}Embedding {min(i + EMBEDDING_BATCH_SIZE, total):,}/{total:,}...", end="", flush=True)
        
        # Retry logic
        for retry in range(MAX_RETRIES):
            try:
                response = client.embeddings.create(
                    model=model,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                break  # Success
                
            except Exception as e:
                error_str = str(e)
                
                # Rate limit - wait and retry
                if "429" in error_str or "rate_limit" in error_str.lower():
                    wait_time = (retry + 1) * 2  # 2, 4, 6 seconds
                    if show_progress:
                        print(f"\n{prefix}Rate limited, waiting {wait_time}s...", end="", flush=True)
                    time.sleep(wait_time)
                    continue
                
                # Token limit - try smaller batch
                if "max_tokens" in error_str.lower() or "8192" in error_str:
                    if show_progress:
                        print(f"\n{prefix}Batch too large, processing individually...", end="", flush=True)
                    # Process one at a time
                    for single_text in batch:
                        try:
                            single_text = truncate_text(single_text, 7500)  # Extra margin
                            resp = client.embeddings.create(model=model, input=[single_text])
                            embeddings.append(resp.data[0].embedding)
                        except:
                            embeddings.append([0.0] * 1536)
                        time.sleep(0.1)
                    break
                
                # Other error
                if retry == MAX_RETRIES - 1:
                    print(f"\n{prefix}Error: {error_str[:100]}")
                    embeddings.extend([[0.0] * 1536 for _ in batch])
                else:
                    time.sleep(1)
        
        time.sleep(0.2)  # Rate limiting between batches
    
    if show_progress:
        print()  # New line after progress
    
    return embeddings


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def find_relevant_chunks(question: Dict, chunks: List[Dict], doc_text: str) -> List[str]:
    """
    Find which chunks contain the answer to a question.
    Returns list of relevant chunk_ids.
    """
    relevant = []
    
    # Get answer text and positions
    answers = question.get("answers", [])
    answer_positions = question.get("answer_positions", [])
    
    for chunk in chunks:
        chunk_start = chunk.get("start", 0)
        chunk_end = chunk.get("end", len(chunk.get("text", "")))
        chunk_text = chunk.get("text", "").lower()
        
        # Check if any answer overlaps with this chunk
        for i, answer in enumerate(answers):
            if not answer:
                continue
            
            answer_lower = answer.lower()
            
            # Method 1: Check if answer text is in chunk
            if answer_lower in chunk_text:
                relevant.append(chunk["chunk_id"])
                break
            
            # Method 2: Check position overlap (if available)
            if i < len(answer_positions) and answer_positions[i] >= 0:
                ans_start = answer_positions[i]
                ans_end = ans_start + len(answer)
                
                # Check overlap
                if chunk_start <= ans_start < chunk_end or chunk_start < ans_end <= chunk_end:
                    relevant.append(chunk["chunk_id"])
                    break
    
    return relevant


def calculate_metrics(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k_values: List[int]
) -> Dict[str, float]:
    """Calculate retrieval metrics."""
    metrics = {}
    
    relevant_set = set(relevant_ids)
    
    for k in k_values:
        top_k = retrieved_ids[:k]
        
        # Recall@k: Is any relevant doc in top k?
        recall = 1.0 if any(rid in relevant_set for rid in top_k) else 0.0
        metrics[f"recall@{k}"] = recall
        
        # MRR@k: Reciprocal rank of first relevant
        rr = 0.0
        for i, rid in enumerate(top_k):
            if rid in relevant_set:
                rr = 1.0 / (i + 1)
                break
        metrics[f"mrr@{k}"] = rr
        
        # nDCG@k
        dcg = 0.0
        for i, rid in enumerate(top_k):
            if rid in relevant_set:
                dcg += 1.0 / math.log2(i + 2)  # +2 because log2(1) = 0
        
        # Ideal DCG (all relevant at top)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_ids), k)))
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        metrics[f"ndcg@{k}"] = ndcg
    
    return metrics


def evaluate_method(
    method: str,
    dataset_dir: Path,
    questions: List[Dict],
    documents: List[Dict],
    client,
    k_values: List[int] = RETRIEVAL_K_VALUES
) -> Dict[str, Any]:
    """Evaluate a single chunking method."""
    
    chunks_file = dataset_dir / "prepared" / "chunks" / f"{method}.json"
    if not chunks_file.exists():
        print(f"    ❌ Chunks file not found: {chunks_file}")
        return None
    
    chunks = load_json(chunks_file)
    print(f"    Loaded {len(chunks):,} chunks")
    
    # Check for cached embeddings
    embeddings_file = dataset_dir / "prepared" / "chunks" / f"{method}_embeddings.json"
    
    if embeddings_file.exists():
        print(f"    Loading cached embeddings...")
        chunk_embeddings = load_json(embeddings_file)
        print(f"    Loaded {len(chunk_embeddings):,} cached embeddings")
        
        # Validate cache matches chunks
        if len(chunk_embeddings) != len(chunks):
            print(f"    WARNING: Cache mismatch ({len(chunk_embeddings)} embeddings vs {len(chunks)} chunks)")
            print(f"    Re-embedding...")
            chunk_texts = [c["text"] for c in chunks]
            chunk_embeddings = get_embeddings(chunk_texts, client, show_progress=True, prefix="      ")
            save_json(chunk_embeddings, embeddings_file)
            print(f"    Cached embeddings to {embeddings_file}")
    else:
        # Embed all chunks
        print(f"    Embedding {len(chunks):,} chunks...")
        chunk_texts = [c["text"] for c in chunks]
        
        chunk_embeddings = get_embeddings(chunk_texts, client, show_progress=True, prefix="      ")
        
        # Cache embeddings
        save_json(chunk_embeddings, embeddings_file)
        print(f"    Cached embeddings to {embeddings_file}")
    
    # Create doc_id -> chunks mapping
    doc_chunks = {}
    for i, chunk in enumerate(chunks):
        doc_id = chunk["document_id"]
        if doc_id not in doc_chunks:
            doc_chunks[doc_id] = []
        doc_chunks[doc_id].append((i, chunk))
    
    # Create doc_id -> text mapping
    doc_texts = {d["id"]: d["text"] for d in documents}
    
    # Evaluate each question
    all_metrics = {f"recall@{k}": [] for k in k_values}
    all_metrics.update({f"mrr@{k}": [] for k in k_values})
    all_metrics.update({f"ndcg@{k}": [] for k in k_values})
    
    questions_evaluated = 0
    questions_with_relevant = 0
    
    print(f"    Evaluating {len(questions):,} questions...")
    
    # Embed questions in batches
    question_texts = [q["question"] for q in questions]
    print(f"    Embedding {len(questions):,} questions...")
    question_embeddings = get_embeddings(question_texts, client, show_progress=True, prefix="      ")
    
    for q_idx, question in enumerate(questions):
        doc_id = question["document_id"]
        
        if doc_id not in doc_chunks:
            continue
        
        doc_text = doc_texts.get(doc_id, "")
        relevant_chunks = doc_chunks[doc_id]
        
        # Find which chunks are relevant (contain the answer)
        chunk_list = [c for _, c in relevant_chunks]
        relevant_ids = find_relevant_chunks(question, chunk_list, doc_text)
        
        if not relevant_ids:
            continue
        
        questions_with_relevant += 1
        
        # Get question embedding
        q_embedding = question_embeddings[q_idx]
        
        # Calculate similarity to all chunks in this document
        similarities = []
        for chunk_idx, chunk in relevant_chunks:
            # Safety check: ensure chunk_idx is within bounds
            if chunk_idx >= len(chunk_embeddings):
                print(f"      WARNING: chunk_idx {chunk_idx} out of range (max: {len(chunk_embeddings) - 1}), skipping")
                continue
            sim = cosine_similarity(q_embedding, chunk_embeddings[chunk_idx])
            similarities.append((chunk["chunk_id"], sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        retrieved_ids = [s[0] for s in similarities]
        
        # Calculate metrics
        metrics = calculate_metrics(retrieved_ids, relevant_ids, k_values)
        
        for metric_name, value in metrics.items():
            all_metrics[metric_name].append(value)
        
        questions_evaluated += 1
        
        if (q_idx + 1) % 200 == 0 or q_idx == len(questions) - 1:
            r5_so_far = sum(all_metrics["recall@5"]) / len(all_metrics["recall@5"]) if all_metrics["recall@5"] else 0
            print(f"      [{q_idx + 1:,}/{len(questions):,}] Recall@5 so far: {r5_so_far:.3f}")
    
    # Average metrics
    avg_metrics = {}
    for metric_name, values in all_metrics.items():
        avg_metrics[metric_name] = sum(values) / len(values) if values else 0.0
    
    # Additional stats
    avg_metrics["questions_evaluated"] = questions_evaluated
    avg_metrics["questions_with_relevant"] = questions_with_relevant
    avg_metrics["total_chunks"] = len(chunks)
    avg_metrics["avg_chunk_tokens"] = sum(c["token_estimate"] for c in chunks) // len(chunks) if chunks else 0
    
    return avg_metrics


def evaluate_dataset(
    dataset_name: str,
    dataset_dir: Path,
    methods: List[str],
    client,
    k_values: List[int] = RETRIEVAL_K_VALUES
) -> Dict[str, Any]:
    """Evaluate all methods on a dataset."""
    
    print(f"\n{'='*60}")
    print(f"Evaluating {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # Load questions and documents
    questions_file = dataset_dir / "prepared" / "questions.json"
    docs_file = dataset_dir / "prepared" / "documents.json"
    
    if not questions_file.exists() or not docs_file.exists():
        print(f"  ❌ Prepared data not found")
        return None
    
    questions = load_json(questions_file)
    documents = load_json(docs_file)
    
    print(f"  Documents: {len(documents)}")
    print(f"  Questions: {len(questions)}")
    
    results = {}
    
    for method in methods:
        print(f"\n  📊 {method}")
        metrics = evaluate_method(method, dataset_dir, questions, documents, client, k_values)
        if metrics:
            results[method] = metrics
            print(f"    Recall@5: {metrics['recall@5']:.3f}")
            print(f"    MRR@5: {metrics['mrr@5']:.3f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval performance")
    parser.add_argument(
        "--dataset",
        choices=["cuad", "qasper", "rfc", "all"],
        default="all",
        help="Which dataset to evaluate"
    )
    parser.add_argument(
        "--method",
        choices=["fixed_token", "recursive", "flat_header", "docling", "all"],
        default="all",
        help="Which method to evaluate"
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Max questions per dataset (for testing)"
    )
    args = parser.parse_args()
    
    print("="*60)
    print("RETRIEVAL EVALUATION")
    print("="*60)
    
    # Check API key
    api_key = get_openai_key()
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)
    
    # Initialize client
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    
    # Methods to evaluate
    if args.method == "all":
        methods = ["fixed_token", "recursive", "flat_header", "docling"]
    else:
        methods = [args.method]
    
    print(f"Methods: {', '.join(methods)}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    
    # Datasets
    datasets = []
    if args.dataset in ["cuad", "all"]:
        datasets.append(("cuad", CUAD_DIR))
    if args.dataset in ["qasper", "all"]:
        datasets.append(("qasper", QASPER_DIR))
    if args.dataset in ["rfc", "all"]:
        datasets.append(("rfc", RFC_DIR))
    
    all_results = {}
    
    for name, dir_path in datasets:
        results = evaluate_dataset(name, dir_path, methods, client)
        if results:
            all_results[name] = results
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    # Aggregate across datasets
    aggregated = {method: {
        "recall@1": [], "recall@5": [], "mrr@5": [], "ndcg@5": [],
        "total_chunks": 0, "avg_tokens": []
    } for method in methods}
    
    for dataset_name, dataset_results in all_results.items():
        for method, metrics in dataset_results.items():
            aggregated[method]["recall@1"].append(metrics.get("recall@1", 0))
            aggregated[method]["recall@5"].append(metrics.get("recall@5", 0))
            aggregated[method]["mrr@5"].append(metrics.get("mrr@5", 0))
            aggregated[method]["ndcg@5"].append(metrics.get("ndcg@5", 0))
            aggregated[method]["total_chunks"] += metrics.get("total_chunks", 0)
            aggregated[method]["avg_tokens"].append(metrics.get("avg_chunk_tokens", 0))
    
    print(f"\n{'Method':<15} {'Recall@1':>10} {'Recall@5':>10} {'MRR@5':>10} {'nDCG@5':>10} {'Chunks':>10} {'Avg Tok':>10}")
    print("-"*80)
    
    final_results = {}
    for method in methods:
        agg = aggregated[method]
        r1 = sum(agg["recall@1"]) / len(agg["recall@1"]) if agg["recall@1"] else 0
        r5 = sum(agg["recall@5"]) / len(agg["recall@5"]) if agg["recall@5"] else 0
        mrr = sum(agg["mrr@5"]) / len(agg["mrr@5"]) if agg["mrr@5"] else 0
        ndcg = sum(agg["ndcg@5"]) / len(agg["ndcg@5"]) if agg["ndcg@5"] else 0
        avg_tok = sum(agg["avg_tokens"]) // len(agg["avg_tokens"]) if agg["avg_tokens"] else 0
        
        print(f"{method:<15} {r1:>10.3f} {r5:>10.3f} {mrr:>10.3f} {ndcg:>10.3f} {agg['total_chunks']:>10,} {avg_tok:>10}")
        
        final_results[method] = {
            "recall@1": r1,
            "recall@5": r5,
            "mrr@5": mrr,
            "ndcg@5": ndcg,
            "total_chunks": agg["total_chunks"],
            "avg_tokens": avg_tok,
        }
    
    # Save results
    results_file = DATA_DIR / "evaluation_results.json"
    save_json({
        "by_dataset": all_results,
        "aggregated": final_results,
        "methods": methods,
        "k_values": RETRIEVAL_K_VALUES,
    }, results_file)
    print(f"\nResults saved to {results_file}")
    
    # Context efficiency
    print("\n" + "="*60)
    print("CONTEXT EFFICIENCY (Recall@5 / Avg Tokens)")
    print("="*60)
    
    efficiencies = []
    for method in methods:
        r5 = final_results[method]["recall@5"]
        avg_tok = final_results[method]["avg_tokens"]
        eff = (r5 / avg_tok * 1000) if avg_tok > 0 else 0  # Per 1000 tokens
        efficiencies.append((method, eff, r5, avg_tok))
    
    efficiencies.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n{'Method':<15} {'Efficiency':>12} {'Recall@5':>10} {'Avg Tokens':>12}")
    print("-"*52)
    for method, eff, r5, avg_tok in efficiencies:
        print(f"{method:<15} {eff:>12.4f} {r5:>10.3f} {avg_tok:>12}")


if __name__ == "__main__":
    main()

