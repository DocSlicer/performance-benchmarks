#!/usr/bin/env python3
"""
02_prepare_documents.py

Prepare and normalize downloaded benchmark documents:
- Validate data integrity
- Subsample if needed (for faster iteration)
- Create unified schema across datasets
- Generate stats and quality report

Usage:
    python 02_prepare_documents.py [--subsample N] [--validate-only]
"""
import argparse
import json
import sys
from pathlib import Path
from collections import Counter
import hashlib

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, CUAD_DIR, QASPER_DIR, RFC_DIR, DATASETS


def load_json(path: Path) -> list:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def save_json(data: list, path: Path):
    """Save JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def validate_documents(docs: list, dataset_name: str) -> dict:
    """Validate document data integrity."""
    issues = []
    
    # Check for required fields
    required_fields = ["id", "title", "text", "char_count"]
    for i, doc in enumerate(docs):
        for field in required_fields:
            if field not in doc:
                issues.append(f"Doc {i}: missing '{field}'")
        
        # Validate char_count matches text length
        if "text" in doc and "char_count" in doc:
            actual_len = len(doc["text"])
            if actual_len != doc["char_count"]:
                issues.append(f"Doc {doc.get('id', i)}: char_count mismatch ({doc['char_count']} vs {actual_len})")
        
        # Check for empty documents
        if doc.get("text", "").strip() == "":
            issues.append(f"Doc {doc.get('id', i)}: empty text")
    
    # Check for duplicate IDs
    ids = [doc["id"] for doc in docs if "id" in doc]
    duplicates = [id for id, count in Counter(ids).items() if count > 1]
    if duplicates:
        issues.append(f"Duplicate IDs: {duplicates[:5]}...")
    
    return {
        "dataset": dataset_name,
        "total_docs": len(docs),
        "issues": issues,
        "valid": len(issues) == 0,
    }


def validate_questions(questions: list, docs: list, dataset_name: str) -> dict:
    """Validate question data integrity."""
    issues = []
    doc_ids = {doc["id"] for doc in docs}
    
    required_fields = ["id", "document_id", "question"]
    for i, q in enumerate(questions):
        for field in required_fields:
            if field not in q:
                issues.append(f"Question {i}: missing '{field}'")
        
        # Check document reference exists
        if q.get("document_id") and q["document_id"] not in doc_ids:
            issues.append(f"Question {q.get('id', i)}: references non-existent doc '{q['document_id']}'")
        
        # Check for empty questions
        if not q.get("question", "").strip():
            issues.append(f"Question {q.get('id', i)}: empty question text")
    
    # Questions per document distribution
    q_per_doc = Counter(q.get("document_id") for q in questions)
    docs_with_questions = len([d for d in doc_ids if q_per_doc.get(d, 0) > 0])
    
    return {
        "dataset": dataset_name,
        "total_questions": len(questions),
        "docs_with_questions": docs_with_questions,
        "avg_questions_per_doc": len(questions) / len(docs) if docs else 0,
        "issues": issues[:20],  # Limit issues shown
        "valid": len(issues) == 0,
    }


def normalize_document(doc: dict, dataset: str) -> dict:
    """Normalize document to unified schema."""
    return {
        "id": doc["id"],
        "title": doc.get("title", doc["id"]),
        "text": doc["text"],
        "char_count": len(doc["text"]),
        "domain": DATASETS[dataset]["domain"],
        "dataset": dataset,
        # Preserve original fields
        **{k: v for k, v in doc.items() if k not in ["id", "title", "text", "char_count"]}
    }


def normalize_question(q: dict, dataset: str) -> dict:
    """Normalize question to unified schema."""
    normalized = {
        "id": q["id"],
        "document_id": q["document_id"],
        "question": q["question"],
        "dataset": dataset,
    }
    
    # Handle answers - unify format
    if "answers" in q:
        normalized["answers"] = q["answers"]
    
    # Handle answer positions
    if "answer_starts" in q:
        normalized["answer_positions"] = q["answer_starts"]
    
    # Handle evidence (QASPER)
    if "evidence" in q:
        normalized["evidence"] = q["evidence"]
    
    # Handle impossible flag (CUAD)
    if "is_impossible" in q:
        normalized["is_unanswerable"] = q["is_impossible"]
    
    return normalized


def subsample_dataset(docs: list, questions: list, n_docs: int) -> tuple:
    """
    Subsample dataset to n_docs documents.
    Keeps documents with the most questions for better evaluation.
    """
    if len(docs) <= n_docs:
        return docs, questions
    
    # Count questions per document
    q_per_doc = Counter(q["document_id"] for q in questions)
    
    # Sort documents by question count (descending)
    docs_sorted = sorted(docs, key=lambda d: q_per_doc.get(d["id"], 0), reverse=True)
    
    # Take top n_docs
    selected_docs = docs_sorted[:n_docs]
    selected_ids = {d["id"] for d in selected_docs}
    
    # Filter questions
    selected_questions = [q for q in questions if q["document_id"] in selected_ids]
    
    return selected_docs, selected_questions


def compute_text_hash(text: str) -> str:
    """Compute hash of document text for deduplication."""
    return hashlib.md5(text.encode()).hexdigest()[:12]


def analyze_dataset(docs: list, questions: list, name: str) -> dict:
    """Generate detailed stats for a dataset."""
    char_counts = [d["char_count"] for d in docs]
    q_per_doc = Counter(q["document_id"] for q in questions)
    
    # Question type analysis
    q_starters = Counter(q["question"].split()[0].lower() for q in questions if q.get("question"))
    
    # Answer analysis
    answerable = [q for q in questions if q.get("answers") and not q.get("is_unanswerable")]
    avg_answer_len = 0
    if answerable:
        all_answers = [a for q in answerable for a in q.get("answers", [])]
        if all_answers:
            avg_answer_len = sum(len(a) for a in all_answers) / len(all_answers)
    
    return {
        "name": name,
        "documents": {
            "count": len(docs),
            "total_chars": sum(char_counts),
            "avg_chars": int(sum(char_counts) / len(docs)) if docs else 0,
            "min_chars": min(char_counts) if char_counts else 0,
            "max_chars": max(char_counts) if char_counts else 0,
        },
        "questions": {
            "count": len(questions),
            "answerable": len(answerable),
            "unanswerable": len(questions) - len(answerable),
            "avg_per_doc": round(len(questions) / len(docs), 1) if docs else 0,
            "docs_with_questions": len([d for d in docs if q_per_doc.get(d["id"], 0) > 0]),
            "avg_answer_length": int(avg_answer_len),
        },
        "question_types": dict(q_starters.most_common(10)),
    }


def process_dataset(dataset_name: str, dataset_dir: Path, subsample: int = None) -> dict:
    """Process a single dataset."""
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name.upper()}")
    print(f"{'='*60}")
    
    docs_file = dataset_dir / "documents.json"
    questions_file = dataset_dir / "questions.json"
    
    if not docs_file.exists():
        print(f"  ERROR: {docs_file} not found")
        return None
    
    # Load data
    docs = load_json(docs_file)
    questions = load_json(questions_file) if questions_file.exists() else []
    
    print(f"  Loaded {len(docs)} documents, {len(questions)} questions")
    
    # Validate
    print("  Validating...")
    doc_validation = validate_documents(docs, dataset_name)
    if doc_validation["issues"]:
        print(f"  ⚠ Document issues: {len(doc_validation['issues'])}")
        for issue in doc_validation["issues"][:3]:
            print(f"    - {issue}")
    else:
        print(f"  ✓ Documents valid")
    
    if questions:
        q_validation = validate_questions(questions, docs, dataset_name)
        if q_validation["issues"]:
            print(f"  ⚠ Question issues: {len(q_validation['issues'])}")
            for issue in q_validation["issues"][:3]:
                print(f"    - {issue}")
        else:
            print(f"  ✓ Questions valid")
    
    # Normalize
    print("  Normalizing...")
    docs = [normalize_document(d, dataset_name) for d in docs]
    questions = [normalize_question(q, dataset_name) for q in questions]
    
    # Subsample if requested
    if subsample and len(docs) > subsample:
        print(f"  Subsampling to {subsample} documents...")
        docs, questions = subsample_dataset(docs, questions, subsample)
        print(f"    -> {len(docs)} documents, {len(questions)} questions")
    
    # Add text hashes for deduplication tracking
    for doc in docs:
        doc["text_hash"] = compute_text_hash(doc["text"])
    
    # Analyze
    stats = analyze_dataset(docs, questions, dataset_name)
    
    # Save prepared data
    prepared_dir = dataset_dir / "prepared"
    prepared_dir.mkdir(exist_ok=True)
    
    save_json(docs, prepared_dir / "documents.json")
    save_json(questions, prepared_dir / "questions.json")
    save_json(stats, prepared_dir / "stats.json")
    
    print(f"  Saved to {prepared_dir}/")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Prepare benchmark documents")
    parser.add_argument(
        "--subsample",
        type=int,
        default=None,
        help="Subsample each dataset to N documents (default: use all)"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate, don't process"
    )
    args = parser.parse_args()
    
    print("="*60)
    print("BENCHMARK DATA PREPARATION")
    print("="*60)
    
    datasets = [
        ("cuad", CUAD_DIR),
        ("qasper", QASPER_DIR),
        ("rfc", RFC_DIR),
    ]
    
    all_stats = []
    
    for name, dir_path in datasets:
        stats = process_dataset(name, dir_path, args.subsample)
        if stats:
            all_stats.append(stats)
    
    # Summary
    print("\n" + "="*60)
    print("PREPARATION SUMMARY")
    print("="*60)
    
    total_docs = sum(s["documents"]["count"] for s in all_stats)
    total_questions = sum(s["questions"]["count"] for s in all_stats)
    total_chars = sum(s["documents"]["total_chars"] for s in all_stats)
    
    print(f"\n{'Dataset':<12} {'Docs':>8} {'Questions':>10} {'Avg Chars':>12}")
    print("-"*45)
    for s in all_stats:
        print(f"{s['name']:<12} {s['documents']['count']:>8} {s['questions']['count']:>10} {s['documents']['avg_chars']:>12,}")
    print("-"*45)
    print(f"{'TOTAL':<12} {total_docs:>8} {total_questions:>10} {total_chars//total_docs:>12,}")
    
    # Save combined stats
    combined_stats = {
        "datasets": all_stats,
        "totals": {
            "documents": total_docs,
            "questions": total_questions,
            "characters": total_chars,
        },
        "subsample": args.subsample,
    }
    
    stats_file = DATA_DIR / "preparation_stats.json"
    save_json(combined_stats, stats_file)
    print(f"\nStats saved to {stats_file}")
    
    # Show question type distribution across all datasets
    print("\n" + "="*60)
    print("QUESTION TYPE DISTRIBUTION")
    print("="*60)
    all_q_types = Counter()
    for s in all_stats:
        all_q_types.update(s.get("question_types", {}))
    
    print(f"\n{'Starter':<12} {'Count':>8} {'%':>8}")
    print("-"*30)
    total_q = sum(all_q_types.values())
    for starter, count in all_q_types.most_common(10):
        pct = count / total_q * 100 if total_q else 0
        print(f"{starter:<12} {count:>8} {pct:>7.1f}%")


if __name__ == "__main__":
    main()

