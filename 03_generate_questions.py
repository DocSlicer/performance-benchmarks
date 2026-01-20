#!/usr/bin/env python3
"""
03_generate_questions.py

Generate questions for ACL and RFC corpora using LLM.
Creates gold-standard QA pairs with answer spans.

CUAD already has questions from the dataset.

Usage:
    python 03_generate_questions.py [--dataset acl|rfc|all] [--n-questions N] [--model MODEL]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, ACL_DIR, RFC_DIR, get_openai_key

# Default settings
DEFAULT_QUESTIONS_PER_DOC = 7
DEFAULT_MODEL = "gpt-4o-mini"


def load_json(path: Path) -> list:
    with open(path) as f:
        return json.load(f)


def save_json(data: list, path: Path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def find_best_match(answer: str, doc_text: str, threshold: float = 0.6) -> tuple:
    """
    Find the best matching span in the document for the given answer.
    Returns (start_position, matched_text, score) or (-1, "", 0) if no match.
    """
    # First try exact match
    exact_pos = doc_text.find(answer)
    if exact_pos >= 0:
        return exact_pos, answer, 1.0
    
    # Try case-insensitive match
    lower_pos = doc_text.lower().find(answer.lower())
    if lower_pos >= 0:
        return lower_pos, doc_text[lower_pos:lower_pos + len(answer)], 0.95
    
    # Try to find substantial substring match
    answer_clean = " ".join(answer.split())  # Normalize whitespace
    
    # Try finding a significant portion (first 100 chars)
    if len(answer_clean) > 50:
        snippet = answer_clean[:100]
        doc_normalized = " ".join(doc_text.split())
        pos = doc_normalized.lower().find(snippet.lower()[:50])
        if pos >= 0:
            return pos, snippet, 0.8
    
    # Try word overlap matching for shorter answers
    answer_words = set(answer.lower().split())
    if len(answer_words) >= 3:
        window_size = len(answer) + 100
        best_score = 0
        best_pos = -1
        
        for i in range(0, len(doc_text) - window_size, 50):
            window = doc_text[i:i + window_size]
            window_words = set(window.lower().split())
            overlap = len(answer_words & window_words) / len(answer_words)
            if overlap > best_score and overlap >= threshold:
                best_score = overlap
                best_pos = i
        
        if best_pos >= 0:
            return best_pos, doc_text[best_pos:best_pos + len(answer)], best_score
    
    return -1, "", 0


def generate_questions_for_rfc(
    doc: dict,
    client,
    n_questions: int = DEFAULT_QUESTIONS_PER_DOC,
    model: str = DEFAULT_MODEL
) -> list:
    """Generate questions for a single RFC document."""
    
    text = doc["text"]
    if len(text) > 25000:
        text = text[:25000] + "\n\n[... document truncated ...]"
    
    prompt = f"""You are creating a benchmark dataset for evaluating document retrieval systems.

Given this RFC technical document, generate {n_questions} diverse questions that:
1. Can be answered using EXACT text spans copied directly from the document
2. Cover different sections and topics in the document
3. Mix question types: factual (what/which), procedural (how), definitional (what is)
4. Are specific enough that the answer appears in a limited portion of the document

CRITICAL: The "answer" field MUST be an EXACT verbatim quote from the document - copy and paste the relevant text directly. Do NOT paraphrase or summarize.

DOCUMENT:
Title: {doc['title']}
RFC Number: {doc.get('rfc_number', 'N/A')}

{text}

Respond in this exact JSON format:
{{
  "questions": [
    {{
      "question": "What is the maximum size of X?",
      "answer": "EXACT QUOTE FROM DOCUMENT - copy paste only, no paraphrasing",
      "section": "Section 4.2" 
    }},
    ...
  ]
}}

Generate exactly {n_questions} questions. ANSWERS MUST BE EXACT QUOTES - if you cannot find an exact quote, skip that question."""

    return _call_llm_and_format(doc, client, prompt, model, n_questions)


def generate_questions_for_acl(
    doc: dict,
    client,
    n_questions: int = DEFAULT_QUESTIONS_PER_DOC,
    model: str = DEFAULT_MODEL
) -> list:
    """Generate questions for a single ACL paper."""
    
    text = doc["text"]
    if len(text) > 25000:
        text = text[:25000] + "\n\n[... document truncated ...]"
    
    prompt = f"""You are creating a benchmark dataset for evaluating document retrieval systems.

Given this academic NLP paper, generate {n_questions} diverse questions that:
1. Can be answered using EXACT text spans copied directly from the document
2. Cover different aspects: methodology, results, background, conclusions
3. Mix question types: factual (what/which), technical (how does X work), comparative (how does X compare to Y)
4. Are specific enough that the answer appears in a limited portion of the document

CRITICAL: The "answer" field MUST be an EXACT verbatim quote from the document - copy and paste the relevant text directly. Do NOT paraphrase or summarize.

PAPER:
Title: {doc['title']}
Year: {doc.get('year', 'N/A')}
Venue: {doc.get('venue', 'N/A')}

{text}

Respond in this exact JSON format:
{{
  "questions": [
    {{
      "question": "What baseline models were compared against?",
      "answer": "EXACT QUOTE FROM DOCUMENT - copy paste only, no paraphrasing",
      "section": "Experiments" 
    }},
    ...
  ]
}}

Generate exactly {n_questions} questions. ANSWERS MUST BE EXACT QUOTES - if you cannot find an exact quote, skip that question."""

    return _call_llm_and_format(doc, client, prompt, model, n_questions)


def _call_llm_and_format(doc: dict, client, prompt: str, model: str, n_questions: int) -> list:
    """Call LLM and format the response into questions."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        
        result = json.loads(response.choices[0].message.content)
        questions = result.get("questions", [])
        
        formatted = []
        for i, q in enumerate(questions):
            if not q.get("question") or not q.get("answer"):
                continue
            
            answer_text = q["answer"]
            pos, matched_text, match_score = find_best_match(answer_text, doc["text"])
            
            formatted.append({
                "id": f"{doc['id']}_q{i+1}",
                "document_id": doc["id"],
                "question": q["question"],
                "answers": [matched_text if pos >= 0 else answer_text],
                "answer_positions": [pos] if pos >= 0 else [],
                "section": q.get("section", ""),
                "answer_found_in_doc": pos >= 0,
                "match_score": match_score,
            })
        
        return formatted
        
    except Exception as e:
        print(f"    Error generating questions: {e}")
        return []


def process_dataset(
    dataset_name: str,
    dataset_dir: Path,
    client,
    generate_fn,
    n_questions: int,
    model: str,
    max_docs: int = None,
    resume: bool = False
) -> dict:
    """Process a dataset and generate questions."""
    print(f"\n{'='*60}")
    print(f"Generating questions for {dataset_name.upper()}")
    print(f"{'='*60}")
    
    prepared_dir = dataset_dir / "prepared"
    docs_file = prepared_dir / "documents.json"
    
    if not docs_file.exists():
        print(f"ERROR: Run 02_prepare_documents.py first")
        return None
    
    docs = load_json(docs_file)
    print(f"Loaded {len(docs)} documents")
    
    # Load existing questions if resuming
    questions_file = prepared_dir / "questions.json"
    existing_questions = []
    processed_doc_ids = set()
    
    if resume and questions_file.exists():
        existing_questions = load_json(questions_file)
        processed_doc_ids = {q["document_id"] for q in existing_questions}
        print(f"Resuming: {len(processed_doc_ids)} docs already processed")
    
    # Filter docs
    docs_to_process = [d for d in docs if d["id"] not in processed_doc_ids]
    if max_docs:
        docs_to_process = docs_to_process[:max_docs]
    
    print(f"Processing {len(docs_to_process)} documents...")
    print(f"Model: {model}")
    print(f"Questions per doc: {n_questions}")
    
    all_questions = list(existing_questions)
    answers_found = 0
    answers_not_found = 0
    
    for i, doc in enumerate(docs_to_process):
        title_preview = doc['title'][:50] if doc.get('title') else doc['id'][:50]
        print(f"[{i+1}/{len(docs_to_process)}] {title_preview}...")
        
        questions = generate_fn(doc, client, n_questions=n_questions, model=model)
        
        for q in questions:
            if q.get("answer_found_in_doc"):
                answers_found += 1
            else:
                answers_not_found += 1
        
        all_questions.extend(questions)
        print(f"    Generated {len(questions)} questions")
        
        # Save progress periodically
        if (i + 1) % 10 == 0:
            save_json(all_questions, questions_file)
            print(f"    Progress saved ({len(all_questions)} total questions)")
        
        time.sleep(0.5)  # Rate limiting
    
    # Final save
    save_json(all_questions, questions_file)
    
    # Update stats
    stats_file = prepared_dir / "stats.json"
    if stats_file.exists():
        stats = load_json(stats_file)
        stats["questions"]["count"] = len(all_questions)
        stats["questions"]["generated"] = True
        stats["questions"]["model"] = model
        total_answers = answers_found + answers_not_found
        stats["questions"]["answer_match_rate"] = answers_found / total_answers if total_answers > 0 else 0
        save_json(stats, stats_file)
    
    return {
        "dataset": dataset_name,
        "total_questions": len(all_questions),
        "answers_found": answers_found,
        "answers_not_found": answers_not_found,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate questions for benchmark datasets")
    parser.add_argument(
        "--dataset",
        choices=["acl", "rfc", "all"],
        default="all",
        help="Which dataset to generate questions for (default: all)"
    )
    parser.add_argument(
        "--n-questions",
        type=int,
        default=DEFAULT_QUESTIONS_PER_DOC,
        help=f"Questions per document (default: {DEFAULT_QUESTIONS_PER_DOC})"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"OpenAI model to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Max documents to process (for testing)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from where we left off"
    )
    args = parser.parse_args()
    
    print("="*60)
    print("BENCHMARK QUESTION GENERATION")
    print("="*60)
    print("\nNote: CUAD already has questions from the dataset.")
    print("This script generates questions for ACL and RFC only.")
    
    # Check for API key
    api_key = get_openai_key()
    if not api_key:
        print("\nERROR: OPENAI_API_KEY not set")
        print("Set it with: export OPENAI_API_KEY=your-key")
        sys.exit(1)
    
    # Initialize OpenAI client
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except ImportError:
        print("Installing openai...")
        os.system("pip install openai")
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    
    results = []
    
    # Process ACL
    if args.dataset in ["acl", "all"]:
        result = process_dataset(
            "acl", ACL_DIR, client, generate_questions_for_acl,
            args.n_questions, args.model, args.max_docs, args.resume
        )
        if result:
            results.append(result)
    
    # Process RFC
    if args.dataset in ["rfc", "all"]:
        result = process_dataset(
            "rfc", RFC_DIR, client, generate_questions_for_rfc,
            args.n_questions, args.model, args.max_docs, args.resume
        )
        if result:
            results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    
    for r in results:
        total = r["answers_found"] + r["answers_not_found"]
        match_rate = r["answers_found"] / total * 100 if total > 0 else 0
        print(f"\n{r['dataset'].upper()}:")
        print(f"  Total questions: {r['total_questions']}")
        print(f"  Answer spans found: {r['answers_found']} ({match_rate:.1f}%)")
        print(f"  Answer spans not found: {r['answers_not_found']}")


if __name__ == "__main__":
    main()
