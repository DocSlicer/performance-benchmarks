# RAG Chunking Benchmark Suite

Comprehensive evaluation of document chunking methods for retrieval-augmented generation (RAG) pipelines.

## Summary

**697 documents** | **26,000+ questions** | **3 domains** | **77.5M characters**

## Results

### Retrieval Performance (Recall@5)

| Method | Legal (CUAD) | Academic (ACL) | Technical (RFC) | **Average** |
|--------|--------------|----------------|-----------------|-------------|
| Fixed Token (500) | 70% | 67% | 65% | 67% |
| RecursiveCharacterTextSplitter | 69% | 67% | 63% | 66% |
| Flat Header Splitter | 90% | 76% | 86% | 84% |
| Docling HierarchicalChunker | 65% | 85% | 26% | 59% |
| **DocSlicer** | **82%** | **85%** | **91%** | **86%** |

### Full Metrics (Averaged)

| Method | Recall@1 | Recall@5 | MRR@5 | nDCG@5 | Avg Tokens | Efficiency |
|--------|----------|----------|-------|--------|------------|------------|
| Fixed Token (500) | 0.34 | 0.67 | 0.46 | 0.48 | 492 | 1.37 |
| RecursiveCharacterTextSplitter | 0.31 | 0.66 | 0.44 | 0.46 | 398 | 1.67 |
| Docling HierarchicalChunker | 0.28 | 0.59 | 0.39 | 0.41 | 376 | 1.56 |
| Flat Header Splitter | 0.47 | 0.84 | 0.61 | 0.66 | 1083 | 0.78 |
| **DocSlicer** | **0.58** | **0.86** | **0.69** | **0.70** | 374 | **2.30** |

### Context Efficiency (Recall@5 / Avg Tokens × 1000)

Higher is better - measures retrieval quality per token of context used.

| Method | Efficiency | Recall@5 | Avg Tokens |
|--------|------------|----------|------------|
| **DocSlicer** | **2.30** | 0.86 | 374 |
| RecursiveCharacterTextSplitter | 1.67 | 0.66 | 398 |
| Docling HierarchicalChunker | 1.56 | 0.59 | 376 |
| Fixed Token (500) | 1.37 | 0.67 | 492 |
| Flat Header Splitter | 0.78 | 0.84 | 1083 |

## Datasets

| Dataset | Domain | Documents | Questions | Source |
|---------|--------|-----------|-----------|--------|
| **CUAD** | Legal | 510 | 20,910 | [TheAtticusProject/cuad](https://github.com/TheAtticusProject/cuad) |
| **ACL** | Academic | 33 | 264 | [ACL Anthology](https://aclanthology.org/) |
| **RFC** | Technical | 154 | 1,070 | [IETF RFC Editor](https://www.rfc-editor.org/) |

## Chunking Methods Evaluated

| Method | Library | Description |
|--------|---------|-------------|
| Fixed Token (500) | - | Naive 500-token chunks with 50-token overlap |
| RecursiveCharacterTextSplitter | LangChain | Recursive separator-based splitting |
| Flat Header Splitter | - | Split on markdown/document headers |
| HierarchicalChunker | Docling | Document-aware hierarchical chunking |
| **DocSlicer** | [docslicer.com](https://docslicer.com) | Layout-aware chunking preserving document structure |

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Environment Variables

```bash
export OPENAI_API_KEY="sk-..."
```

### Run the Pipeline

```bash
# Step 1: Download datasets
python 01_download_datasets.py --dataset all

# Step 2: Prepare documents
python 02_prepare_documents.py --dataset all

# Step 3: Generate questions (uses OpenAI for RFC)
python 03_generate_questions.py --dataset all

# Step 4: Run chunking methods
python 04_run_chunking.py --method all --dataset all

# Step 5: Evaluate retrieval
python 05_evaluate_retrieval.py --method all --dataset all
```

## Metrics

| Metric | Description |
|--------|-------------|
| **Recall@k** | Is any relevant chunk in top k results? |
| **MRR@k** | Mean Reciprocal Rank - how early is the first relevant chunk? |
| **nDCG@k** | Normalized Discounted Cumulative Gain - ranking quality |
| **Context Efficiency** | Recall@5 ÷ (Avg Tokens / 1000) - retrieval quality per token |

## Methodology

All methods are evaluated under identical conditions:
- **Embeddings**: OpenAI `text-embedding-3-small`
- **Similarity**: Cosine similarity
- **Retrieval**: Top-k (k=1, 3, 5, 10)
- **Variable**: Chunking strategy only

A chunk is considered "relevant" if it contains any answer span from the gold annotations.

## Directory Structure

```
.
├── config.py                    # Shared configuration
├── 01_download_datasets.py      # Download benchmark datasets
├── 02_prepare_documents.py      # Prepare/normalize documents
├── 03_generate_questions.py     # Generate questions (LLM for RFC)
├── 04_run_chunking.py           # Run chunking methods
├── 05_evaluate_retrieval.py     # Evaluate retrieval metrics
├── requirements.txt             # Python dependencies
└── data/                        # Downloaded data (gitignored)
    ├── cuad/
    ├── acl/
    └── rfc/
```

## Data Format

### documents.json

```json
{
  "id": "unique_document_id",
  "title": "Document Title",
  "text": "Full document text...",
  "char_count": 52563,
  "domain": "legal|academic|technical"
}
```

### questions.json

```json
{
  "id": "unique_question_id",
  "document_id": "references documents.json id",
  "question": "The question text",
  "answers": ["answer1", "answer2"],
  "answer_starts": [position_in_doc]
}
```

## License

MIT
