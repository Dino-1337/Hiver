# Hiver AI Intern Evaluation Assignment

## Overview
This project implements three AI/ML systems for Hiver's customer support platform:
- **Part A:** Email Tagging with customer isolation
- **Part B:** Sentiment Analysis prompt evaluation
- **Part C:** Mini-RAG for Knowledge Base answering

All implementations use free, local tools (Mistral 7B via Ollama, sentence-transformers, FAISS).

## Project Structure

```
Hiver/
├── requirements.txt          # Dependencies
├── data/
│   ├── small_dataset.csv    # 12 emails
│   └── large_dataset.csv    # 60 emails
├── part_a/
│   ├── Email_README.md      # Documentation
│   ├── classifier.py        # Email tagging system
│   └── results.csv          # Classification results
├── part_b/
│   ├── SENTIMENT_README.md  # Documentation
│   ├── sentiment.py         # Prompt v1
│   ├── sentiment_v2.py      # Prompt v2 (improved)
│   └── results_v1.csv, results_v2.csv
└── part_c/
    ├── RAG_README.md        # Documentation
    ├── rag.py               # RAG system
    ├── kb_articles/         # Knowledge base articles
    └── rag_results.json     # Query results
```

## Quick Start

### Setup
```bash
pip install -r requirements.txt
```

### Part A: Email Tagging
```bash
python part_a/classifier.py
```
- Classifies emails with customer isolation
- Uses patterns and anti-patterns for accuracy
- Results: 100% accuracy (small dataset), 86.67% (large dataset)

### Part B: Sentiment Analysis
```bash
python part_b/sentiment.py      # Prompt v1
python part_b/sentiment_v2.py   # Prompt v2
```
- Evaluates sentiment (positive/negative/neutral)
- Compares two prompt versions
- Results: v1 (89.5% confidence), v2 (70.5% confidence, better reasoning)

### Part C: RAG System
```bash
python part_c/rag.py
```
- Answers queries using KB articles
- Retrieves relevant articles and generates answers
- Results: 88-93% confidence on test queries

## Technology Stack

- **LLM:** Mistral 7B (local via Ollama)
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2
- **Vector Store:** FAISS
- **Framework:** LangChain (for Part C)
- **Language:** Python 3.8+

## Key Features

### Part A: Customer Isolation
- Ensures tags from one customer don't leak to another
- Uses few-shot learning (patterns) and guardrails (anti-patterns)
- Achieves high accuracy with customer-specific tag filtering

### Part B: Prompt Engineering
- Evaluates two prompt versions systematically
- Compares consistency, reasoning quality, and reliability
- Documents what failed and what improved

### Part C: RAG Pipeline
- Retrieves relevant KB articles using semantic search
- Generates answers with confidence scores
- Includes failure case analysis and improvement suggestions

## Results Summary

- **Part A:** 100% accuracy (small), 86.67% (large dataset)
- **Part B:** v1 - 89.5% avg confidence, v2 - 70.5% (better reasoning)
- **Part C:** 88-93% confidence on test queries

## Documentation

Each part has detailed README:
- `part_a/Email_README.md` - Email tagging approach and results
- `part_b/SENTIMENT_README.md` - Sentiment analysis evaluation
- `part_c/RAG_README.md` - RAG system and improvements

## Requirements

See `requirements.txt` for full list. Main dependencies:
- pandas
- ollama
- langchain
- langchain-community
- langchain-text-splitters
- sentence-transformers
- faiss-cpu

