"""
Part B: Sentiment Analysis Prompt Evaluation - Version 2 (Improved)
Uses local Mistral 7B via Ollama for sentiment analysis
Only the prompt is changed from v1 - everything else is identical
"""

import pandas as pd
import json
import ollama
import os
from typing import Dict, List

# Configuration
MODEL_NAME = "mistral:latest"
DATA_PATH = "Z:/AI PROJECTS/Hiver/data/small_dataset.csv"  # We'll select 10 emails from here
VERSION = "v2"  # Version 2 - improved prompt


def load_data(filepath: str) -> pd.DataFrame:
    """Load email dataset from CSV"""
    df = pd.read_csv(filepath)
    return df


def create_sentiment_prompt_v2(subject: str, body: str) -> str:
    """Create improved sentiment analysis prompt (v2)"""
    prompt = f"""You are a sentiment analysis expert for customer support emails.

Analyze the sentiment of this email carefully:

Subject: {subject}
Body: {body}

## Sentiment Guidelines:
- **Positive**: Expresses gratitude, satisfaction, appreciation, or positive feedback. Examples: "Thank you!", "Great feature!", "This helped a lot"
- **Negative**: Expresses frustration, complaints, problems, errors, or dissatisfaction. Examples: "This is broken", "Not working", "Very frustrating"
- **Neutral**: Factual questions, information requests, or statements without emotional tone. Examples: "How do I...?", "Can you help with...?", "We need to configure..."

## Important Considerations:
1. **Feature requests** can be neutral (simple request) or positive (appreciation/enthusiasm)
2. **Problem reports** are typically negative, but check for positive framing (e.g., "Thanks for fixing this!")
3. **Questions without emotional language** are neutral
4. **Consider both subject and body together** - don't judge by subject alone
5. **Be consistent** - similar emails should get similar sentiment
6. **Look for key phrases**: 
   - Positive: "thank", "appreciate", "great", "helpful", "love"
   - Negative: "broken", "not working", "error", "frustrated", "problem"
   - Neutral: "how", "can you", "need help", "want to configure"

## Consistency Check:
Before finalizing, ask: "Would another similar email get the same sentiment?"

Return ONLY valid JSON in this exact format:
{{
  "sentiment": "positive" or "negative" or "neutral",
  "confidence": 0.0-1.0,
  "reasoning": "detailed explanation including: (1) key phrases that influenced the decision, (2) why this sentiment over alternatives, (3) confidence justification"
}}"""
    
    return prompt


def analyze_sentiment(subject: str, body: str) -> Dict:
    """Analyze sentiment of an email using improved prompt v2"""
    prompt = create_sentiment_prompt_v2(subject, body)
    
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Extract JSON from response
        content = response['message']['content'].strip()
        
        # Try to parse JSON (handle cases where model adds extra text)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        # Find JSON object in response
        start_idx = content.find("{")
        end_idx = content.rfind("}") + 1
        if start_idx != -1 and end_idx > start_idx:
            content = content[start_idx:end_idx]
        
        result = json.loads(content)
        
        # Normalize sentiment to lowercase
        if 'sentiment' in result:
            result['sentiment'] = result['sentiment'].lower()
        
        return result
        
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return {
            "sentiment": "neutral",
            "confidence": 0.0,
            "reasoning": f"Error: {str(e)}"
        }


def main():
    """Main execution"""
    print("=" * 60)
    print(f"Part B: Sentiment Analysis - Prompt {VERSION.upper()} (Improved)")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} emails")
    
    # Select 10 diverse emails for testing
    # Mix of different customers and tag types
    test_emails = df.head(10).copy()  # Using first 10 for consistency
    print(f"\nSelected {len(test_emails)} emails for sentiment analysis")
    
    # Analyze sentiment for each email
    print(f"\nAnalyzing sentiment using improved prompt {VERSION}...")
    results = []
    
    for idx, row in test_emails.iterrows():
        print(f"\nProcessing email {row['email_id']}...")
        print(f"  Subject: {row['subject']}")
        
        result = analyze_sentiment(
            row['subject'],
            row['body']
        )
        
        results.append({
            'email_id': row['email_id'],
            'customer_id': row['customer_id'],
            'subject': row['subject'],
            'body': row['body'],
            'tag': row['tag'],
            'sentiment': result.get('sentiment', 'neutral'),
            'confidence': result.get('confidence', 0.0),
            'reasoning': result.get('reasoning', '')
        })
        
        print(f"  Sentiment: {result.get('sentiment', 'neutral')} (confidence: {result.get('confidence', 0.0):.2f})")
    
    # Save results
    results_df = pd.DataFrame(results)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(script_dir, f'results_{VERSION}.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n\nResults saved to {results_path}")
    
    # Summary statistics
    sentiment_counts = results_df['sentiment'].value_counts()
    avg_confidence = results_df['confidence'].mean()
    
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"\nSentiment Distribution:")
    for sentiment, count in sentiment_counts.items():
        print(f"  {sentiment.capitalize()}: {count}")
    print(f"\nAverage Confidence: {avg_confidence:.2%}")
    
    return results_df


if __name__ == "__main__":
    main()

