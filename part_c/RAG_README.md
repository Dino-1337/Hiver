# Part C: Mini-RAG for Knowledge Base Answering

## Overview
RAG (Retrieval-Augmented Generation) system that uses KB articles to answer user queries about Hiver.

## Technology Used
- **LangChain:** Document loading, text splitting, embeddings, vector store
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2 (free, local, 80MB)
- **Vector Store:** FAISS (free, local, fast)
- **LLM:** Mistral 7B via Ollama (free, local)
- **Language:** Python 3.8+

## Implementation Steps
1. Load KB articles from `kb_articles/` folder (`.txt` files)
2. Create embeddings using HuggingFace sentence-transformers
3. Split documents into chunks (500 chars, 50 overlap)
4. Build FAISS vector store with embeddings
5. Retrieve top 3 relevant chunks for each query
6. Generate answer using Ollama Mistral 7B with retrieved context
7. Calculate confidence score based on answer quality

## Results

### Query 1: "How do I configure automations in Hiver?"
- **Retrieved:** automation_setup.txt (perfect), email_rules.txt, tagging_system.txt
- **Answer:** Accurate step-by-step instructions
- **Confidence:** 88%

### Query 2: "Why is CSAT not appearing?"
- **Retrieved:** csat_visibility.txt (2 chunks), dashboard_analytics.txt
- **Answer:** Comprehensive troubleshooting guide (5 causes listed)
- **Confidence:** 93%

**Overall:** Both queries retrieved relevant articles and generated accurate, helpful answers.

## 5 Retrieval Improvement Ideas

1. **Hybrid Search:** Combine semantic embeddings with BM25 keyword search for better exact matches
2. **Re-ranking:** Use cross-encoder model to re-rank top 10-20 candidates for better accuracy
3. **Query Expansion:** Expand queries with synonyms (e.g., "CSAT not showing" → "CSAT visibility dashboard scores")
4. **Metadata Filtering:** Add metadata (article type, category) and filter before semantic search
5. **Confidence Calibration:** Improve confidence using answer quality metrics and verify grounding in context

## Failure Case & Debugging

**Scenario:** Query "Why is CSAT not appearing?" retrieves wrong article

**Debugging Steps:**
1. Check embedding similarity scores for all documents
2. Inspect chunking - verify CSAT content is properly chunked
3. Test query reformulation with different phrasings
4. Verify vector store - check if CSAT article appears in top results
5. Inspect retrieved context passed to LLM

**Root Causes:** Query phrasing mismatch, chunk boundaries splitting context, embedding model not prioritizing keywords

**Fixes:** Query expansion, increase chunk overlap, add keyword boost (hybrid search)

## Files
- `rag.py` - Main RAG script (runnable end-to-end)
- `rag_results.json` - Query results with answers and retrieved articles
- `kb_articles/` - Knowledge base articles folder (5 articles provided)

## Usage
```bash
python part_c/rag.py
```
