#!/usr/bin/env python3
"""
03_generate_questions.py

Generate questions for RFC corpus using LLM.
Creates gold-standard QA pairs with answer spans.

Usage:
    python 03_generate_questions.py [--n-questions N] [--model MODEL]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, RFC_DIR, get_openai_key

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
    # Split answer into sentences/phrases and find them
    answer_clean = " ".join(answer.split())  # Normalize whitespace
    
    # Try finding a significant portion (first 100 chars)
    if len(answer_clean) > 50:
        snippet = answer_clean[:100]
        # Normalize the document text similarly
        doc_normalized = " ".join(doc_text.split())
        pos = doc_normalized.lower().find(snippet.lower()[:50])
        if pos >= 0:
            # Map back to original position (approximate)
            return pos, snippet, 0.8
    
    # Try word overlap matching for shorter answers
    answer_words = set(answer.lower().split())
    if len(answer_words) >= 3:
        # Sliding window search
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


def generate_questions_for_doc(
    doc: dict,
    client,
    n_questions: int = DEFAULT_QUESTIONS_PER_DOC,
    model: str = DEFAULT_MODEL
) -> list:
    """Generate questions for a single RFC document."""
    
    # Truncate text if too long (keep first ~20k chars for context)
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

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        
        result = json.loads(response.choices[0].message.content)
        questions = result.get("questions", [])
        
        # Validate and format questions
        formatted = []
        for i, q in enumerate(questions):
            if not q.get("question") or not q.get("answer"):
                continue
                
            # Try to find answer position in document using fuzzy matching
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


def main():
    parser = argparse.ArgumentParser(description="Generate questions for RFC corpus")
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
    print("RFC QUESTION GENERATION")
    print("="*60)
    
    # Check for API key
    api_key = get_openai_key()
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
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
    
    # Load prepared RFC documents
    prepared_dir = RFC_DIR / "prepared"
    docs_file = prepared_dir / "documents.json"
    
    if not docs_file.exists():
        print(f"ERROR: Run 02_prepare_documents.py first")
        sys.exit(1)
    
    docs = load_json(docs_file)
    print(f"Loaded {len(docs)} RFC documents")
    
    # Load existing questions if resuming
    questions_file = prepared_dir / "questions.json"
    existing_questions = []
    processed_doc_ids = set()
    
    if args.resume and questions_file.exists():
        existing_questions = load_json(questions_file)
        processed_doc_ids = {q["document_id"] for q in existing_questions}
        print(f"Resuming: {len(processed_doc_ids)} docs already processed")
    
    # Filter docs
    docs_to_process = [d for d in docs if d["id"] not in processed_doc_ids]
    if args.max_docs:
        docs_to_process = docs_to_process[:args.max_docs]
    
    print(f"Processing {len(docs_to_process)} documents...")
    print(f"Model: {args.model}")
    print(f"Questions per doc: {args.n_questions}")
    print(f"Expected total: ~{len(docs_to_process) * args.n_questions} new questions")
    print()
    
    all_questions = list(existing_questions)
    answers_found = 0
    answers_not_found = 0
    
    for i, doc in enumerate(docs_to_process):
        print(f"[{i+1}/{len(docs_to_process)}] RFC {doc.get('rfc_number', 'N/A')}: {doc['title'][:50]}...")
        
        questions = generate_questions_for_doc(
            doc, 
            client, 
            n_questions=args.n_questions,
            model=args.model
        )
        
        # Track answer matching stats
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
        
        # Rate limiting
        time.sleep(0.5)
    
    # Final save
    save_json(all_questions, questions_file)
    
    # Update stats
    stats_file = prepared_dir / "stats.json"
    if stats_file.exists():
        stats = load_json(stats_file)
        stats["questions"]["count"] = len(all_questions)
        stats["questions"]["generated"] = True
        stats["questions"]["model"] = args.model
        stats["questions"]["answer_match_rate"] = answers_found / (answers_found + answers_not_found) if (answers_found + answers_not_found) > 0 else 0
        save_json(stats, stats_file)
    
    # Summary
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"Total questions: {len(all_questions)}")
    print(f"Answer spans found in doc: {answers_found} ({answers_found/(answers_found+answers_not_found)*100:.1f}%)")
    print(f"Answer spans not found: {answers_not_found}")
    print(f"Saved to: {questions_file}")
    
    # Show sample
    print("\n" + "="*60)
    print("SAMPLE QUESTIONS")
    print("="*60)
    for q in all_questions[:5]:
        print(f"\nQ: {q['question']}")
        print(f"A: {q['answers'][0][:100]}..." if q['answers'] else "A: (no answer)")
        print(f"Found in doc: {q.get('answer_found_in_doc', 'N/A')}")


if __name__ == "__main__":
    main()

