# RAG Chunking Benchmark Suite

Comprehensive evaluation of document chunking methods for retrieval-augmented generation (RAG) pipelines.

## Summary

**2,250 documents** | **12,400+ questions** | **3 domains** | **77.5M characters**

## Datasets

| Dataset | Domain | Documents | Questions | Source |
|---------|--------|-----------|-----------|--------|
| **CUAD** | Legal | 510 | 6,702 | [TheAtticusProject/cuad](https://github.com/TheAtticusProject/cuad) |
| **QASPER** | Academic | 1,585 | 4,639 | [AllenAI QASPER](https://allenai.org/data/qasper) |
| **RFC** | Technical | 155 | 1,078 | [IETF RFC Editor](https://www.rfc-editor.org/) |

## Chunking Methods Evaluated

| Method | Library | Description |
|--------|---------|-------------|
| Fixed Token (500) | - | Naive 500-token chunks with 50-token overlap |
| RecursiveCharacterTextSplitter | LangChain | Recursive separator-based splitting |
| Flat Header Splitter | - | Split on markdown/document headers |
| HierarchicalChunker | Docling | Document-aware hierarchical chunking |

## Results (Recall@5)

| Method | Legal (CUAD) | Academic (QASPER) | Technical (RFC) | Average |
|--------|--------------|-------------------|-----------------|---------|
| Fixed Token | 70% | 87% | 65% | 74% |
| Recursive | 69% | 87% | 61% | 72% |
| Flat Header | **90%** | **88%** | **87%** | **88%** |
| Docling | 65% | 75% | 28% | 56% |

## Quick Start

### Prerequisites

```bash
pip install datasets langchain langchain-openai docling tiktoken openai numpy
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
python 02_prepare_documents.py

# Step 3: Generate RFC questions (uses OpenAI)
python 03_generate_questions.py

# Step 4: Run chunking methods
python 04_run_chunking.py

# Step 5: Evaluate retrieval
python 05_evaluate_retrieval.py
```

## Metrics

| Metric | Description |
|--------|-------------|
| **Recall@k** | Is any relevant chunk in top k results? |
| **MRR@k** | Mean Reciprocal Rank - how early is the first relevant chunk? |
| **nDCG@k** | Normalized Discounted Cumulative Gain - ranking quality |
| **Context Efficiency** | Recall per token retrieved (cost-effectiveness) |

## Directory Structure

```
.
├── config.py                    # Shared configuration
├── 01_download_datasets.py      # Download benchmark datasets
├── 02_prepare_documents.py      # Prepare/normalize documents
├── 03_generate_questions.py     # Generate RFC questions (LLM)
├── 04_run_chunking.py           # Run chunking methods
├── 05_evaluate_retrieval.py     # Evaluate retrieval metrics
└── data/                        # Downloaded data (gitignored)
    ├── cuad/
    │   ├── documents.json
    │   └── questions.json
    ├── qasper/
    │   ├── documents.json
    │   └── questions.json
    └── rfc/
        ├── documents.json
        └── questions.json
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

## Methodology

All methods are evaluated under identical conditions:
- **Embeddings**: OpenAI `text-embedding-3-small`
- **Similarity**: Cosine similarity
- **Retrieval**: Top-k (k=1, 3, 5, 10)
- **Variable**: Chunking strategy only

A chunk is considered "relevant" if it contains any answer span from the gold annotations.

## License

MIT
