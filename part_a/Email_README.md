# Part A: Email Tagging Mini-System

## Overview
LLM-based text classification system for customer support emails with customer isolation and accuracy improvements using patterns and anti-patterns.

## Technology Used
- **Model:** Mistral 7B (local via Ollama)
- **Language:** Python 3.8+
- **Libraries:** pandas, ollama
- **Approach:** Prompt-based classification with few-shot learning

## Implementation Steps

### 1. Baseline Classifier
- Built LLM prompt-based classifier using Mistral 7B
- Input: email subject + body
- Output: JSON with tag, confidence, reasoning

### 2. Customer Isolation
- Extracted unique tags per customer from dataset
- Filtered available tags by `customer_id` before classification
- Only customer-specific tags shown in prompt (prevents tag leakage)
- Implementation: `get_customer_tags()` function

### 3. Accuracy Improvements
**Patterns (Few-shot Learning):**
- Added 2-3 example emails per customer in prompt
- Shows correct tag examples before classification
- Helps model learn customer-specific patterns

**Anti-Patterns (Guardrails):**
- Added warnings for misleading words
- Examples:
  - "stuck + pending" → `status_bug`, not `access_issue`
  - "not visible + CSAT" → `analytics_issue`, not `performance`
  - "billing" word → only tag as billing if actual payment issue

## Results Analysis

### Small Dataset (12 emails)
- **Accuracy:** 100% (12/12 correct)
- **Errors Fixed:** 2
  - Email #3: `status_bug` (was `access_issue`) - fixed by guardrail
  - Email #7: `analytics_issue` (was `performance`) - fixed by guardrail

### Large Dataset (60 emails)
- **Accuracy:** 86.67% (52/60 correct)
- **Per-Customer Performance:**
  - CUST_A: 100% (10/10)
  - CUST_B: 100% (10/10)
  - CUST_C: 80% (8/10)
  - CUST_D: 50% (5/10) - needs attention
  - CUST_E: 90% (9/10)
  - CUST_F: 100% (10/10)

**Error Patterns:**
- Similar tags confusion: `analytics_bug` vs `analytics_accuracy` (2 errors)
- Misleading words: "stuck" → wrong tag (1 error)
- Context misunderstanding: server timeout → UI bug (2 errors)

## Improvements Made

1. **Patterns:** Few-shot examples improved model understanding
2. **Anti-Patterns:** Guardrails prevented common mistakes
3. **Customer Isolation:** Ensured no tag leakage between customers

## Production Improvement Ideas

1. **Dynamic Guardrails:** Learn from errors to automatically add guardrails for new patterns
2. **Customer-Specific Fine-tuning:** Fine-tune prompts per customer based on their tag distribution and common patterns
3. **Confidence-Based Routing:** Route low-confidence predictions to human review, use high-confidence for automation

## Files
- `classifier.py` - Main classification script (runnable end-to-end)
- `results.csv` - Large dataset results
- `results_small.csv` - Small dataset results

