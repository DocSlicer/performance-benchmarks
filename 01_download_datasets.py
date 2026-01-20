#!/usr/bin/env python3
"""
01_download_datasets.py

Download benchmark datasets:
- CUAD (Contract Understanding Atticus Dataset) - Legal (with PDFs)
- ACL Anthology (Academic NLP Papers) - Academic (with PDFs)
- RFC Corpus (IETF) - Technical (with HTML)

Usage:
    python 01_download_datasets.py [--dataset cuad|acl|rfc|all]
"""
import argparse
import json
import os
import sys
import io
import base64
import zipfile
from pathlib import Path
from typing import Optional

# Add parent to path for config
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CUAD_DIR, QASPER_DIR, RFC_DIR,
    DATA_DIR
)

# ACL directory (replaces QASPER)
ACL_DIR = DATA_DIR / "acl"


def download_file(url: str, timeout: int = 120) -> bytes:
    """Download a file from URL."""
    import urllib.request
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def download_cuad():
    """
    Download CUAD dataset with PDFs from Hugging Face.
    
    CUAD has ~510 contracts with ~13,000 QA pairs.
    Questions are clause-extraction focused.
    
    Source: https://huggingface.co/datasets/theatricusproject/cuad
    """
    print("\n" + "="*60)
    print("Downloading CUAD (Contract Understanding Atticus Dataset)")
    print("="*60)
    
    CUAD_DIR.mkdir(parents=True, exist_ok=True)
    pdf_dir = CUAD_DIR / "pdfs"
    pdf_dir.mkdir(exist_ok=True)
    
    # First, download the JSON data from GitHub for QA pairs
    print("Step 1: Downloading QA annotations from GitHub...")
    url = 'https://github.com/TheAtticusProject/cuad/raw/main/data.zip'
    data = download_file(url)
    print(f"  Downloaded {len(data) / 1024 / 1024:.1f} MB")
    
    # Extract CUADv1.json (full dataset)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        with zf.open('CUADv1.json') as f:
            cuad_data = json.load(f)
    
    # Parse documents and questions from SQuAD format
    documents = {}
    questions = []
    
    for entry in cuad_data['data']:
        doc_title = entry['title']
        
        for para in entry['paragraphs']:
            context = para['context']
            
            if doc_title not in documents:
                documents[doc_title] = {
                    "id": doc_title,
                    "title": doc_title,
                    "text": context,
                    "char_count": len(context),
                    "domain": "legal",
                    "pdf_path": None,  # Will be set if PDF downloaded
                }
            
            for qa in para['qas']:
                question_text = qa['question']
                answers = qa.get('answers', [])
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
    
    # Step 2: Download PDFs from Hugging Face
    print("\nStep 2: Downloading PDFs from Hugging Face...")
    print("  Using dataset: dvgodoy/CUAD_v1_Contract_Understanding_PDF")
    
    try:
        from datasets import load_dataset
        
        # Load the dataset with PDFs (base64 encoded)
        ds = load_dataset("dvgodoy/CUAD_v1_Contract_Understanding_PDF", split="train")
        
        pdf_count = 0
        for item in ds:
            file_name = item.get("file_name", f"contract_{pdf_count}.pdf")
            pdf_bytes_b64 = item.get("pdf_bytes_base64")  # Base64 encoded string
            
            if pdf_bytes_b64:
                # Decode base64 to actual PDF bytes
                try:
                    if isinstance(pdf_bytes_b64, str):
                        pdf_bytes = base64.b64decode(pdf_bytes_b64)
                    elif isinstance(pdf_bytes_b64, bytes):
                        # Try to decode as base64 if it looks like base64
                        try:
                            pdf_bytes = base64.b64decode(pdf_bytes_b64)
                        except:
                            pdf_bytes = pdf_bytes_b64  # Already raw bytes
                    else:
                        continue
                except Exception as e:
                    print(f"    Failed to decode PDF for {file_name}: {e}")
                    continue
                
                safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in file_name)
                if not safe_name.endswith(".pdf"):
                    safe_name += ".pdf"
                pdf_path = pdf_dir / safe_name
                
                with open(pdf_path, "wb") as f:
                    f.write(pdf_bytes)
                
                # Try to match with document by title
                # The file_name format is like: CompanyName_Date_DocType_...
                base_name = file_name.rsplit('.', 1)[0]  # Remove .pdf
                for doc_id, doc in documents.items():
                    if base_name in doc_id or doc_id in base_name:
                        doc["pdf_path"] = str(pdf_path.relative_to(CUAD_DIR))
                        break
                
                pdf_count += 1
                if pdf_count % 100 == 0:
                    print(f"    Downloaded {pdf_count} PDFs...")
        
        print(f"  Downloaded {pdf_count} PDFs to {pdf_dir}")
        
    except ImportError:
        print("  WARNING: 'datasets' library not installed. Skipping PDF download.")
        print("  Install with: pip install datasets")
        print("  PDFs will not be available for docslicer.")
    except Exception as e:
        print(f"  WARNING: Failed to download PDFs: {e}")
        print("  Continuing without PDFs...")
    
    # Save documents and questions
    docs_file = CUAD_DIR / "documents.json"
    questions_file = CUAD_DIR / "questions.json"
    
    with open(docs_file, "w") as f:
        json.dump(list(documents.values()), f, indent=2)
    
    with open(questions_file, "w") as f:
        json.dump(questions, f, indent=2)
    
    # Stats
    total_chars = sum(d["char_count"] for d in documents.values())
    answerable = sum(1 for q in questions if not q["is_impossible"])
    pdfs_available = sum(1 for d in documents.values() if d.get("pdf_path"))
    
    print(f"\nCUAD Summary:")
    print(f"  Documents: {len(documents)}")
    print(f"  PDFs available: {pdfs_available}")
    print(f"  Questions: {len(questions)} ({answerable} answerable)")
    print(f"  Total chars: {total_chars:,}")
    print(f"  Saved to: {CUAD_DIR}")
    
    return {
        "dataset": "cuad",
        "domain": "legal",
        "documents": len(documents),
        "pdfs": pdfs_available,
        "questions": len(questions),
        "answerable_questions": answerable,
        "total_chars": total_chars,
    }


def download_acl():
    """
    Download ACL Anthology papers (NLP academic papers with PDFs).
    
    Replaces QASPER since we need actual PDFs for docslicer.
    Downloads a curated subset of influential NLP papers.
    
    Source: https://aclanthology.org/
    """
    print("\n" + "="*60)
    print("Downloading ACL Anthology (Academic NLP Papers)")
    print("="*60)
    
    import time
    
    ACL_DIR.mkdir(parents=True, exist_ok=True)
    pdf_dir = ACL_DIR / "pdfs"
    pdf_dir.mkdir(exist_ok=True)
    
    # Curated list of influential NLP papers from ACL Anthology
    # Format: (paper_id, title, year, venue)
    # Paper IDs follow ACL Anthology format: venue-year.number
    ACL_PAPERS = [
        # Transformers & Attention
        ("N19-1423", "BERT: Pre-training of Deep Bidirectional Transformers", 2019, "NAACL"),
        ("P17-1017", "Attention Is All You Need (Transformer)", 2017, "ACL"),
        ("2020.acl-main.747", "Language Models are Few-Shot Learners (GPT-3)", 2020, "ACL"),
        ("2020.emnlp-main.346", "Exploring the Limits of Transfer Learning with T5", 2020, "EMNLP"),
        
        # Question Answering
        ("D16-1264", "SQuAD: 100,000+ Questions for Machine Comprehension", 2016, "EMNLP"),
        ("P18-1078", "Know What You Don't Know: Unanswerable Questions for SQuAD", 2018, "ACL"),
        ("D17-1070", "TriviaQA: A Large Scale Distantly Supervised Challenge Dataset", 2017, "EMNLP"),
        ("N18-1202", "HotpotQA: A Dataset for Diverse, Explainable Multi-hop QA", 2018, "NAACL"),
        
        # Named Entity Recognition & Information Extraction
        ("W03-0419", "Introduction to the CoNLL-2003 Shared Task", 2003, "CoNLL"),
        ("D14-1181", "Glove: Global Vectors for Word Representation", 2014, "EMNLP"),
        ("P13-1045", "Recursive Deep Models for Semantic Compositionality", 2013, "ACL"),
        
        # Summarization
        ("D04-1010", "ROUGE: A Package for Automatic Evaluation of Summaries", 2004, "ACL"),
        ("P17-1099", "Get To The Point: Summarization with Pointer-Generator Networks", 2017, "ACL"),
        ("D19-1387", "Text Summarization with Pretrained Encoders", 2019, "EMNLP"),
        
        # Machine Translation
        ("P02-1040", "BLEU: a Method for Automatic Evaluation of Machine Translation", 2002, "ACL"),
        ("D14-1179", "On the Properties of Neural Machine Translation: Encoder-Decoder", 2014, "EMNLP"),
        ("W14-4012", "A Systematic Comparison of Smoothing Techniques for Language Models", 2014, "WMT"),
        
        # Sentiment Analysis
        ("D13-1170", "Recursive Deep Models for Sentiment Analysis", 2013, "EMNLP"),
        ("S14-2004", "SemEval-2014 Task 4: Aspect Based Sentiment Analysis", 2014, "SemEval"),
        ("P15-1001", "Deep Unordered Composition Rivals Syntactic Methods", 2015, "ACL"),
        
        # Language Understanding & Reasoning
        ("N18-1101", "A Broad-Coverage Challenge Corpus for Sentence Understanding", 2018, "NAACL"),
        ("D19-1514", "SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding", 2019, "EMNLP"),
        ("2020.acl-main.164", "Beyond Accuracy: Behavioral Testing of NLP Models with CheckList", 2020, "ACL"),
        
        # Document Understanding
        ("D19-1006", "Document-Level Neural Machine Translation", 2019, "EMNLP"),
        ("2020.findings-emnlp.76", "Longformer: The Long-Document Transformer", 2020, "EMNLP"),
        ("2021.naacl-main.108", "LayoutLM: Pre-training of Visual-Textual Representations", 2021, "NAACL"),
        
        # Information Retrieval
        ("2020.emnlp-main.550", "Dense Passage Retrieval for Open-Domain QA", 2020, "EMNLP"),
        ("2021.naacl-main.466", "Retrieval-Augmented Generation for Knowledge-Intensive Tasks", 2021, "NAACL"),
        
        # Recent Papers (2022-2024)
        ("2022.acl-long.1", "Finetuned Language Models Are Zero-Shot Learners", 2022, "ACL"),
        ("2022.emnlp-main.174", "Chain-of-Thought Prompting Elicits Reasoning", 2022, "EMNLP"),
        ("2023.acl-long.1", "LLaMA: Open and Efficient Foundation Language Models", 2023, "ACL"),
        ("2023.emnlp-main.1", "Direct Preference Optimization (DPO)", 2023, "EMNLP"),
        
        # Additional diverse papers
        ("P16-1162", "Neural Machine Translation of Rare Words with Subword Units", 2016, "ACL"),
        ("D15-1166", "A Neural Attention Model for Sentence Summarization", 2015, "EMNLP"),
        ("N16-1030", "Hierarchical Attention Networks for Document Classification", 2016, "NAACL"),
        ("P18-1082", "Deep Contextualized Word Representations (ELMo)", 2018, "ACL"),
        ("D18-1259", "Contextual Word Representations: A Contextual Introduction", 2018, "EMNLP"),
        ("P19-1285", "XLNet: Generalized Autoregressive Pretraining", 2019, "ACL"),
        ("2020.acl-main.703", "Language Models as Knowledge Bases?", 2020, "ACL"),
    ]
    
    documents = []
    questions = []  # Will be generated in step 03
    failed = []
    
    print(f"Downloading {len(ACL_PAPERS)} papers from ACL Anthology...")
    
    for i, (paper_id, title, year, venue) in enumerate(ACL_PAPERS):
        # ACL Anthology PDF URL format
        pdf_url = f"https://aclanthology.org/{paper_id}.pdf"
        
        try:
            # Download PDF
            pdf_bytes = download_file(pdf_url, timeout=30)
            
            # Save PDF
            safe_id = paper_id.replace("/", "_").replace(".", "_")
            pdf_path = pdf_dir / f"{safe_id}.pdf"
            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)
            
            # Extract text from PDF (basic extraction)
            text = f"[PDF document: {title}]\n\nThis is a PDF document. Full text extraction requires PDF processing."
            
            # Try to extract text if PyMuPDF is available
            try:
                import fitz
                doc = fitz.open(pdf_path)
                text_parts = []
                for page in doc:
                    text_parts.append(page.get_text())
                text = "\n".join(text_parts)
                doc.close()
            except ImportError:
                pass
            except Exception:
                pass
            
            documents.append({
                "id": paper_id,
                "title": title,
                "year": year,
                "venue": venue,
                "text": text,
                "char_count": len(text),
                "domain": "academic",
                "pdf_path": str(pdf_path.relative_to(ACL_DIR)),
                "pdf_url": pdf_url,
            })
            
            if (i + 1) % 10 == 0:
                print(f"  Downloaded {i + 1}/{len(ACL_PAPERS)} papers...")
            
            time.sleep(0.2)  # Rate limiting
            
        except Exception as e:
            failed.append((paper_id, str(e)))
            print(f"  Failed: {paper_id} - {e}")
    
    print(f"\nDownloaded {len(documents)} papers ({len(failed)} failed)")
    
    # Save documents
    docs_file = ACL_DIR / "documents.json"
    questions_file = ACL_DIR / "questions.json"
    
    with open(docs_file, "w") as f:
        json.dump(documents, f, indent=2)
    
    with open(questions_file, "w") as f:
        json.dump([], f)  # Questions generated in step 03
    
    total_chars = sum(d["char_count"] for d in documents)
    
    print(f"\nACL Anthology Summary:")
    print(f"  Documents: {len(documents)}")
    print(f"  PDFs downloaded: {len(documents)}")
    print(f"  Years covered: {min(d['year'] for d in documents)}-{max(d['year'] for d in documents)}")
    print(f"  Total chars: {total_chars:,}")
    print(f"  Questions will be generated in step 03")
    print(f"  Saved to: {ACL_DIR}")
    
    if failed:
        print(f"  Failed papers: {[f[0] for f in failed[:5]]}...")
    
    return {
        "dataset": "acl",
        "domain": "academic",
        "documents": len(documents),
        "pdfs": len(documents),
        "questions": 0,  # To be generated
        "total_chars": total_chars,
    }


def download_rfc():
    """
    Download RFC documents from IETF with HTML versions.
    
    RFCs are technical standards documents.
    Questions will be generated in step 03.
    
    Now downloads both TXT and HTML formats for docslicer compatibility.
    """
    print("\n" + "="*60)
    print("Downloading RFC Corpus (IETF Technical Standards)")
    print("="*60)
    
    import time
    
    RFC_DIR.mkdir(parents=True, exist_ok=True)
    html_dir = RFC_DIR / "html"
    html_dir.mkdir(exist_ok=True)
    
    # Important RFCs covering major internet protocols and standards
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
        # Modern protocols
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
    html_count = 0
    
    print(f"Downloading {len(all_rfcs)} RFCs (text + HTML)...")
    
    for i, (rfc_num, category) in enumerate(all_rfcs):
        txt_url = f"https://www.rfc-editor.org/rfc/rfc{rfc_num}.txt"
        html_url = f"https://www.rfc-editor.org/rfc/rfc{rfc_num}.html"
        
        try:
            # Download TXT version
            text = download_file(txt_url, timeout=15).decode("utf-8", errors="replace")
            
            # Extract title from RFC header
            lines = text.split("\n")[:60]
            title = f"RFC {rfc_num}"
            for line in lines:
                line = line.strip()
                if line and not line.startswith(" "):
                    if any(x in line for x in ["Request for Comments", "Category:", "ISSN:", "Obsoletes:", "Updates:"]):
                        continue
                    if len(line) > 20:
                        title = line[:150]
                        break
            
            # Try to download HTML version
            html_path = None
            try:
                html_bytes = download_file(html_url, timeout=15)
                html_path = html_dir / f"rfc{rfc_num}.html"
                with open(html_path, "wb") as f:
                    f.write(html_bytes)
                html_count += 1
            except Exception:
                pass  # HTML not available for all RFCs
            
            documents.append({
                "id": f"rfc{rfc_num}",
                "rfc_number": rfc_num,
                "title": title,
                "text": text,
                "char_count": len(text),
                "url": txt_url,
                "html_url": html_url,
                "html_path": str(html_path.relative_to(RFC_DIR)) if html_path else None,
                "category": category,
                "domain": "technical",
            })
            
            if (i + 1) % 25 == 0:
                print(f"  Downloaded {i + 1}/{len(all_rfcs)} RFCs...")
            
            time.sleep(0.05)  # Rate limiting
            
        except Exception as e:
            failed.append((rfc_num, str(e)))
    
    print(f"\nDownloaded {len(documents)} RFCs ({len(failed)} failed)")
    print(f"  HTML versions: {html_count}")
    
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
    print(f"  HTML available: {html_count}")
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
        "html_count": html_count,
        "questions": 0,  # To be generated
        "total_chars": total_chars,
    }


def main():
    parser = argparse.ArgumentParser(description="Download benchmark datasets")
    parser.add_argument(
        "--dataset",
        choices=["cuad", "acl", "rfc", "all"],
        default="all",
        help="Which dataset to download (default: all)"
    )
    args = parser.parse_args()
    
    print("="*60)
    print("BENCHMARK DATASET DOWNLOADER")
    print("="*60)
    print(f"Data directory: {DATA_DIR}")
    print("\nDatasets:")
    print("  - CUAD: Legal contracts with PDFs")
    print("  - ACL: Academic NLP papers with PDFs (replaces QASPER)")
    print("  - RFC: Technical standards with HTML")
    
    results = []
    
    if args.dataset in ["cuad", "all"]:
        result = download_cuad()
        if result:
            results.append(result)
    
    if args.dataset in ["acl", "all"]:
        result = download_acl()
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
    total_questions = sum(r.get("questions", 0) for r in results)
    total_chars = sum(r.get("total_chars", 0) for r in results)
    total_pdfs = sum(r.get("pdfs", 0) for r in results)
    total_html = sum(r.get("html_count", 0) for r in results)
    
    print(f"\n{'Dataset':<12} {'Domain':<12} {'Docs':>6} {'PDFs':>6} {'HTML':>6} {'Questions':>10}")
    print("-"*60)
    for r in results:
        print(f"  {r['dataset']:<10} {r['domain']:<12} {r['documents']:>6} {r.get('pdfs', 0):>6} {r.get('html_count', 0):>6} {r.get('questions', 0):>10}")
    print("-"*60)
    print(f"  {'TOTAL':<10} {' ':<12} {total_docs:>6} {total_pdfs:>6} {total_html:>6} {total_questions:>10}")
    print(f"\n  Total characters: {total_chars:,}")
    
    # Save summary
    summary_file = DATA_DIR / "download_summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "datasets": results,
            "total_documents": total_docs,
            "total_pdfs": total_pdfs,
            "total_html": total_html,
            "total_questions": total_questions,
            "total_chars": total_chars,
        }, f, indent=2)
    print(f"\nSummary saved to {summary_file}")
    
    # Print message for Why DocSlicer page
    print("\n" + "="*60)
    print("FOR WHY DOCSLICER PAGE:")
    print("="*60)
    domains = list(set(r["domain"] for r in results))
    print(f'"Evaluated on {len(results)} public structured-document benchmarks')
    print(f' spanning {", ".join(domains)}.')
    print(f' Over {total_docs} documents with {total_pdfs} PDFs and {total_html} HTML files."')


if __name__ == "__main__":
    main()
