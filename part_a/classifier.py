"""
Part A: Email Tagging Mini-System
Uses local Mistral 7B via Ollama for classification
"""

import pandas as pd
import json
import ollama
import os
from typing import Dict, List, Tuple
from collections import defaultdict

MODEL_NAME = "mistral:latest"
DATA_PATH = "Z:/AI PROJECTS/Hiver/data/large_dataset.csv"


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df


def get_customer_tags(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Extract unique tags per customer for isolation"""
    customer_tags = defaultdict(set)
    for _, row in df.iterrows():
        customer_tags[row['customer_id']].add(row['tag'])
    
    # Convert sets to sorted lists for consistency
    return {cust: sorted(list(tags)) for cust, tags in customer_tags.items()}


def get_pattern_examples(df: pd.DataFrame, customer_id: str, available_tags: List[str], 
                        exclude_email_id: int = None) -> str:
    """Get pattern examples (few-shot learning) for this customer's tags"""
    customer_data = df[df['customer_id'] == customer_id].copy()
    
    if exclude_email_id is not None:
        customer_data = customer_data[customer_data['email_id'] != exclude_email_id]
    
    examples = []
    for tag in available_tags:
        tag_examples = customer_data[customer_data['tag'] == tag]
        if len(tag_examples) > 0:
            example = tag_examples.iloc[0]
            examples.append(f"""Example:
Subject: {example['subject']}
Body: {example['body']}
Tag: {example['tag']}""")
    
    if examples:
        return "\n\n".join(examples[:3])
    return ""


def get_anti_patterns(available_tags: List[str]) -> str:
    """Get anti-pattern guardrails to prevent common mistakes"""
    guardrails = []
    
    if 'status_bug' in available_tags and 'access_issue' in available_tags:
        guardrails.append("⚠️ WARNING: Words like 'stuck' or 'pending' with status-related context → usually 'status_bug', not 'access_issue'")
    
    if 'analytics_issue' in available_tags and 'performance' in available_tags:
        guardrails.append("⚠️ WARNING: 'not visible', 'disappeared', or 'missing' with dashboard/CSAT/analytics → usually 'analytics_issue', not 'performance'")
    
    if 'billing' in available_tags:
        guardrails.append("⚠️ WARNING: Only tag as 'billing' if customer mentions being charged incorrectly, invoice problems, or payment issues. NOT if they just mention 'billing' as a feature.")
    
    if 'workflow_issue' in available_tags:
        guardrails.append("⚠️ WARNING: 'rule not working', 'automation stopped', 'workflow broken' → usually 'workflow_issue'")
    
    if 'feature_request' in available_tags:
        guardrails.append("⚠️ WARNING: Phrases like 'would help', 'please consider', 'feature request' → usually 'feature_request'")
    
    if guardrails:
        return "\n".join(guardrails)
    return ""


def create_prompt(subject: str, body: str, available_tags: List[str], 
                 df: pd.DataFrame = None, customer_id: str = None, 
                 exclude_email_id: int = None) -> str:
    """Create classification prompt with patterns and anti-patterns"""
    tags_str = ", ".join(available_tags)
    
    patterns_section = ""
    if df is not None and customer_id is not None:
        examples = get_pattern_examples(df, customer_id, available_tags, exclude_email_id)
        if examples:
            patterns_section = f"""
\n## Examples (Patterns):
{examples}
"""
    
    anti_patterns_section = ""
    guardrails = get_anti_patterns(available_tags)
    if guardrails:
        anti_patterns_section = f"""
\n## Guardrails (Anti-Patterns):
{guardrails}
"""
    
    prompt = f"""You are a customer support email classifier.

Available tags: {tags_str}
{patterns_section}{anti_patterns_section}

## Email to Classify:
Subject: {subject}
Body: {body}

Return ONLY valid JSON in this exact format:
{{
  "tag": "chosen_tag",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}

Choose the tag that best matches the email content. Pay attention to the examples and guardrails above."""
    
    return prompt


def classify_email(subject: str, body: str, available_tags: List[str], 
                  df: pd.DataFrame = None, customer_id: str = None, 
                  exclude_email_id: int = None) -> Dict:
    """Classify a single email using Mistral 7B with patterns and anti-patterns"""
    prompt = create_prompt(subject, body, available_tags, df, customer_id, exclude_email_id)
    
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response['message']['content'].strip()
        
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        start_idx = content.find("{")
        end_idx = content.rfind("}") + 1
        if start_idx != -1 and end_idx > start_idx:
            content = content[start_idx:end_idx]
        
        result = json.loads(content)
        return result
        
    except Exception as e:
        print(f"Error classifying email: {e}")
        return {
            "tag": available_tags[0],
            "confidence": 0.0,
            "reasoning": f"Error: {str(e)}"
        }


def evaluate_predictions(df: pd.DataFrame, predictions: List[Dict]) -> Dict:
    """Evaluate classifier performance"""
    correct = 0
    total = len(df)
    errors = []
    
    for idx, (_, row) in enumerate(df.iterrows()):
        predicted_tag = predictions[idx].get('tag', '')
        actual_tag = row['tag']
        
        if predicted_tag == actual_tag:
            correct += 1
        else:
            errors.append({
                "email_id": row['email_id'],
                "customer_id": row['customer_id'],
                "subject": row['subject'],
                "actual": actual_tag,
                "predicted": predicted_tag,
                "confidence": predictions[idx].get('confidence', 0.0)
            })
    
    accuracy = correct / total if total > 0 else 0.0
    
    customer_accuracy = defaultdict(lambda: {"correct": 0, "total": 0})
    for idx, (_, row) in enumerate(df.iterrows()):
        cust = row['customer_id']
        customer_accuracy[cust]["total"] += 1
        if predictions[idx].get('tag') == row['tag']:
            customer_accuracy[cust]["correct"] += 1
    
    customer_acc = {
        cust: acc["correct"] / acc["total"] 
        for cust, acc in customer_accuracy.items()
    }
    
    return {
        "overall_accuracy": accuracy,
        "customer_accuracy": customer_acc,
        "errors": errors
    }


def main():
    """Main execution"""
    print("=" * 60)
    print("Email Tagging Classifier - Part A")
    print("=" * 60)
    print(f"\nDataset: {DATA_PATH}")
    print("Loading data...")
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} emails")
    
    customer_tags = get_customer_tags(df)
    print("\nCustomer tags (for isolation):")
    for cust, tags in customer_tags.items():
        print(f"  {cust}: {tags}")
    
    print("\n\nClassifying emails...")
    predictions = []
    
    for idx, row in df.iterrows():
        print(f"Processing email {row['email_id']} ({row['customer_id']})...")
        
        available_tags = customer_tags[row['customer_id']]
        
        result = classify_email(
            row['subject'],
            row['body'],
            available_tags,
            df,
            row['customer_id'],
            row['email_id']
        )
        
        predictions.append(result)
        print(f"  Predicted: {result.get('tag')} (confidence: {result.get('confidence', 0.0):.2f})")
    
    print("\n\nEvaluating results...")
    evaluation = evaluate_predictions(df, predictions)
    
    print(f"\nOverall Accuracy: {evaluation['overall_accuracy']:.2%}")
    print("\nPer-Customer Accuracy:")
    for cust, acc in evaluation['customer_accuracy'].items():
        print(f"  {cust}: {acc:.2%}")
    
    print(f"\nErrors: {len(evaluation['errors'])}")
    if evaluation['errors']:
        print("\nError Details:")
        for error in evaluation['errors'][:5]:
            print(f"  Email {error['email_id']}: {error['actual']} -> {error['predicted']}")
    
    results_df = df.copy()
    results_df['predicted_tag'] = [p.get('tag') for p in predictions]
    results_df['confidence'] = [p.get('confidence', 0.0) for p in predictions]
    results_df['reasoning'] = [p.get('reasoning', '') for p in predictions]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(script_dir, 'results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")
    
    return evaluation, predictions


if __name__ == "__main__":
    main()

