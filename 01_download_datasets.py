#!/usr/bin/env python3
"""
01_download_datasets.py

Download benchmark datasets:
- CUAD (Contract Understanding Atticus Dataset) - Legal
- QASPER (Question Answering on Scientific Papers) - Academic
- RFC Corpus (IETF) - Technical

Usage:
    python 01_download_datasets.py [--dataset cuad|qasper|rfc|all]
"""
import argparse
import json
import os
import sys
import io
import zipfile
import tarfile
from pathlib import Path

# Add parent to path for config
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CUAD_DIR, QASPER_DIR, RFC_DIR,
    DATA_DIR
)


def download_file(url: str, timeout: int = 120) -> bytes:
    """Download a file from URL."""
    import urllib.request
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def download_cuad():
    """
    Download CUAD dataset from GitHub.
    
    CUAD is in SQuAD format with ~510 contracts and ~13,000 QA pairs.
    Questions are clause-extraction focused (e.g., "What is the termination clause?")
    """
    print("\n" + "="*60)
    print("Downloading CUAD (Contract Understanding Atticus Dataset)")
    print("="*60)
    
    CUAD_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download data.zip from GitHub
    url = 'https://github.com/TheAtticusProject/cuad/raw/main/data.zip'
    print(f"Downloading from: {url}")
    
    data = download_file(url)
    print(f"Downloaded {len(data) / 1024 / 1024:.1f} MB")
    
    # Extract CUADv1.json (full dataset)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        with zf.open('CUADv1.json') as f:
            cuad_data = json.load(f)
    
    # CUAD is in SQuAD format:
    # {version, data: [{title, paragraphs: [{context, qas: [{question, answers}]}]}]}
    
    documents = {}
    questions = []
    
    for entry in cuad_data['data']:
        doc_title = entry['title']
        
        for para in entry['paragraphs']:
            context = para['context']
            
            # Use title as document ID (one context per doc in CUAD)
            if doc_title not in documents:
                documents[doc_title] = {
                    "id": doc_title,
                    "title": doc_title,
                    "text": context,
                    "char_count": len(context),
                    "domain": "legal",
                }
            
            # Extract QA pairs
            for qa in para['qas']:
                question_text = qa['question']
                answers = qa.get('answers', [])
                
                # answers is a list of {text, answer_start}
                answer_texts = [a['text'] for a in answers if a['text']]
                answer_starts = [a['answer_start'] for a in answers if a['text']]
                
                questions.append({
                    "id": qa['id'],
                    "document_id": doc_title,
                    "question": question_text,
                    "answers": answer_texts,
                    "answer_starts": answer_starts,
                    "is_impossible": qa.get('is_impossible', len(answer_texts) == 0),
                })
    
    # Save
    docs_file = CUAD_DIR / "documents.json"
    questions_file = CUAD_DIR / "questions.json"
    
    with open(docs_file, "w") as f:
        json.dump(list(documents.values()), f, indent=2)
    
    with open(questions_file, "w") as f:
        json.dump(questions, f, indent=2)
    
    # Stats
    total_chars = sum(d["char_count"] for d in documents.values())
    answerable = sum(1 for q in questions if not q["is_impossible"])
    
    print(f"\nCUAD Summary:")
    print(f"  Documents: {len(documents)}")
    print(f"  Questions: {len(questions)} ({answerable} answerable)")
    print(f"  Total chars: {total_chars:,}")
    print(f"  Avg doc size: {total_chars // len(documents):,} chars")
    print(f"  Saved to: {CUAD_DIR}")
    
    return {
        "dataset": "cuad",
        "domain": "legal",
        "documents": len(documents),
        "questions": len(questions),
        "answerable_questions": answerable,
        "total_chars": total_chars,
    }


def download_qasper():
    """
    Download QASPER dataset from AllenAI S3.
    
    QASPER has ~1,500 NLP papers with ~5,000 questions.
    Questions are about paper content, methods, results.
    """
    print("\n" + "="*60)
    print("Downloading QASPER (Question Answering on Scientific Papers)")
    print("="*60)
    
    QASPER_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download train/dev and test splits
    train_dev_url = 'https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz'
    test_url = 'https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz'
    
    documents = []
    questions = []
    
    for name, url in [('train-dev', train_dev_url), ('test', test_url)]:
        print(f"Downloading {name}: {url}")
        
        data = download_file(url)
        print(f"  Downloaded {len(data) / 1024 / 1024:.1f} MB")
        
        with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tf:
            for member in tf.getmembers():
                if member.name.endswith('.json') and 'qasper' in member.name:
                    f = tf.extractfile(member)
                    if f:
                        papers = json.load(f)
                        split = 'test' if 'test' in member.name else ('dev' if 'dev' in member.name else 'train')
                        
                        for paper_id, paper in papers.items():
                            # Extract full text from sections
                            full_text_parts = []
                            
                            # Add title and abstract
                            if paper.get('title'):
                                full_text_parts.append(f"# {paper['title']}\n")
                            if paper.get('abstract'):
                                full_text_parts.append(f"## Abstract\n{paper['abstract']}\n")
                            
                            # Add sections
                            # full_text is a list of {section_name, paragraphs}
                            for section in paper.get('full_text', []):
                                section_name = section.get('section_name', '')
                                paragraphs = section.get('paragraphs', [])
                                
                                if section_name:
                                    full_text_parts.append(f"\n## {section_name}\n")
                                for para in paragraphs:
                                    full_text_parts.append(para + "\n")
                            
                            doc_text = "\n".join(full_text_parts)
                            
                            if not doc_text.strip():
                                continue
                            
                            documents.append({
                                "id": paper_id,
                                "title": paper.get('title', paper_id),
                                "text": doc_text,
                                "char_count": len(doc_text),
                                "domain": "academic",
                                "split": split,
                            })
                            
                            # Extract questions
                            for qa in paper.get('qas', []):
                                question_text = qa.get('question', '')
                                
                                # Collect answers from multiple annotators
                                # Structure: answers is a list of {answer: {unanswerable, extractive_spans, ...}}
                                answer_texts = []
                                evidence_texts = []
                                
                                for annotator_ans in qa.get('answers', []):
                                    ans = annotator_ans.get('answer', annotator_ans)
                                    
                                    if ans.get('unanswerable'):
                                        continue
                                    
                                    # Different answer types
                                    if ans.get('extractive_spans'):
                                        answer_texts.extend(ans['extractive_spans'])
                                    if ans.get('free_form_answer'):
                                        answer_texts.append(ans['free_form_answer'])
                                    if ans.get('yes_no') is not None:
                                        answer_texts.append('yes' if ans['yes_no'] else 'no')
                                    
                                    evidence_texts.extend(ans.get('evidence', []))
                                
                                if answer_texts:
                                    questions.append({
                                        "id": qa.get('question_id', f"{paper_id}_{len(questions)}"),
                                        "document_id": paper_id,
                                        "question": question_text,
                                        "answers": list(set(answer_texts)),
                                        "evidence": list(set(evidence_texts)),
                                        "split": split,
                                    })
    
    # Save
    docs_file = QASPER_DIR / "documents.json"
    questions_file = QASPER_DIR / "questions.json"
    
    with open(docs_file, "w") as f:
        json.dump(documents, f, indent=2)
    
    with open(questions_file, "w") as f:
        json.dump(questions, f, indent=2)
    
    total_chars = sum(d["char_count"] for d in documents)
    
    print(f"\nQASPER Summary:")
    print(f"  Documents: {len(documents)}")
    print(f"  Questions: {len(questions)}")
    print(f"  Total chars: {total_chars:,}")
    print(f"  Avg doc size: {total_chars // len(documents) if documents else 0:,} chars")
    print(f"  Saved to: {QASPER_DIR}")
    
    return {
        "dataset": "qasper",
        "domain": "academic",
        "documents": len(documents),
        "questions": len(questions),
        "total_chars": total_chars,
    }


def download_rfc():
    """
    Download RFC documents from IETF.
    
    RFCs are technical standards documents.
    Questions will be generated in step 03.
    """
    print("\n" + "="*60)
    print("Downloading RFC Corpus (IETF Technical Standards)")
    print("="*60)
    
    import time
    
    RFC_DIR.mkdir(parents=True, exist_ok=True)
    
    # Important RFCs covering major internet protocols and standards
    # Organized by category for diversity
    RFC_LIST = {
        # Core Internet Protocols
        "core": [791, 793, 768, 792, 826, 894],
        # HTTP/Web
        "http": [2616, 7230, 7231, 7232, 7233, 7234, 7235, 7540, 9110, 9111, 9112, 9113, 9114],
        # Security/TLS/Auth
        "security": [5246, 8446, 6749, 7519, 7617, 8725, 6125, 5280, 8017],
        # DNS
        "dns": [1035, 8484, 8499, 7858, 6891],
        # Email
        "email": [5321, 5322, 6376, 7208, 7489],
        # WebSocket/Realtime
        "websocket": [6455, 8441],
        # JSON/REST/APIs
        "api": [7159, 8259, 7807, 8288, 6570, 7946],
        # Networking
        "network": [3986, 4122, 5952, 6335, 8174],
        # Modern protocols (9114 is in http)
        "modern": [9000, 9001, 9002, 8999],
    }
    
    # Flatten list
    all_rfcs = []
    for category, rfcs in RFC_LIST.items():
        for rfc in rfcs:
            all_rfcs.append((rfc, category))
    
    # Add more RFCs to reach ~150 documents
    additional = list(range(8200, 8250)) + list(range(8700, 8750))
    for rfc in additional:
        if rfc not in [r[0] for r in all_rfcs]:
            all_rfcs.append((rfc, "additional"))
    
    documents = []
    failed = []
    
    print(f"Downloading {len(all_rfcs)} RFCs...")
    
    for i, (rfc_num, category) in enumerate(all_rfcs):
        url = f"https://www.rfc-editor.org/rfc/rfc{rfc_num}.txt"
        
        try:
            text = download_file(url, timeout=15).decode("utf-8", errors="replace")
            
            # Extract title from RFC header
            lines = text.split("\n")[:60]
            title = f"RFC {rfc_num}"
            for line in lines:
                line = line.strip()
                if line and not line.startswith(" "):
                    # Skip metadata lines
                    if any(x in line for x in ["Request for Comments", "Category:", "ISSN:", "Obsoletes:", "Updates:"]):
                        continue
                    if len(line) > 20:
                        title = line[:150]
                        break
            
            documents.append({
                "id": f"rfc{rfc_num}",
                "rfc_number": rfc_num,
                "title": title,
                "text": text,
                "char_count": len(text),
                "url": url,
                "category": category,
                "domain": "technical",
            })
            
            if (i + 1) % 25 == 0:
                print(f"  Downloaded {i + 1}/{len(all_rfcs)} RFCs...")
            
            time.sleep(0.05)  # Rate limiting
            
        except Exception as e:
            failed.append((rfc_num, str(e)))
    
    print(f"\nDownloaded {len(documents)} RFCs ({len(failed)} failed)")
    
    # Save documents
    docs_file = RFC_DIR / "documents.json"
    with open(docs_file, "w") as f:
        json.dump(documents, f, indent=2)
    
    # Empty questions file (to be generated in step 03)
    questions_file = RFC_DIR / "questions.json"
    with open(questions_file, "w") as f:
        json.dump([], f)
    
    total_chars = sum(d["char_count"] for d in documents)
    
    print(f"\nRFC Summary:")
    print(f"  Documents: {len(documents)}")
    print(f"  Total chars: {total_chars:,}")
    print(f"  Avg doc size: {total_chars // len(documents) if documents else 0:,} chars")
    print(f"  Categories: {list(RFC_LIST.keys())}")
    print(f"  Questions will be generated in step 03")
    print(f"  Saved to: {RFC_DIR}")
    
    if failed:
        print(f"  Failed RFCs: {[f[0] for f in failed[:10]]}...")
    
    return {
        "dataset": "rfc",
        "domain": "technical",
        "documents": len(documents),
        "questions": 0,  # To be generated
        "total_chars": total_chars,
    }


def main():
    parser = argparse.ArgumentParser(description="Download benchmark datasets")
    parser.add_argument(
        "--dataset",
        choices=["cuad", "qasper", "rfc", "all"],
        default="all",
        help="Which dataset to download (default: all)"
    )
    args = parser.parse_args()
    
    print("="*60)
    print("BENCHMARK DATASET DOWNLOADER")
    print("="*60)
    print(f"Data directory: {DATA_DIR}")
    
    results = []
    
    if args.dataset in ["cuad", "all"]:
        result = download_cuad()
        if result:
            results.append(result)
    
    if args.dataset in ["qasper", "all"]:
        result = download_qasper()
        if result:
            results.append(result)
    
    if args.dataset in ["rfc", "all"]:
        result = download_rfc()
        if result:
            results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    
    total_docs = sum(r["documents"] for r in results)
    total_questions = sum(r["questions"] for r in results)
    total_chars = sum(r.get("total_chars", 0) for r in results)
    
    for r in results:
        print(f"  {r['dataset']:12} ({r['domain']:10}): {r['documents']:>5} docs, {r['questions']:>6} questions")
    
    print("-"*60)
    print(f"  {'TOTAL':12} {' ':10}  {total_docs:>5} docs, {total_questions:>6} questions")
    print(f"  Total characters: {total_chars:,}")
    
    # Save summary
    summary_file = DATA_DIR / "download_summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "datasets": results,
            "total_documents": total_docs,
            "total_questions": total_questions,
            "total_chars": total_chars,
        }, f, indent=2)
    print(f"\nSummary saved to {summary_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    domains = list(set(r["domain"] for r in results))
    print(f'"Evaluated on {len(results)} public structured-document benchmarks')
    print(f' spanning {", ".join(domains)}.')
    print(f' Over {total_docs} documents and {total_questions}+ gold-standard questions."')


if __name__ == "__main__":
    main()
