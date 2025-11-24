# Part B: Sentiment Analysis Prompt Evaluation

## Overview
Evaluation of two sentiment analysis prompts (v1 and v2) to determine which produces more consistent sentiment classification and better debugging explanations.

## Technology Used
- **Model:** Mistral 7B (local via Ollama)
- **Language:** Python 3.8+
- **Libraries:** pandas, ollama
- **Approach:** Prompt-based sentiment analysis

## Prompts Used

### Prompt v1 (Baseline)
```
Analyze the sentiment of the following customer support email.

Email:
Subject: {subject}
Body: {body}

Determine the sentiment and return ONLY valid JSON in this exact format:
{
  "sentiment": "positive" or "negative" or "neutral",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation of why this sentiment was chosen"
}

Consider:
- Positive: Appreciation, thanks, satisfaction, feature requests
- Negative: Complaints, frustration, problems, errors
- Neutral: Questions, information requests, neutral statements
```

### Prompt v2 (Improved)
```
You are a sentiment analysis expert for customer support emails.

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
{
  "sentiment": "positive" or "negative" or "neutral",
  "confidence": 0.0-1.0,
  "reasoning": "detailed explanation including: (1) key phrases that influenced the decision, (2) why this sentiment over alternatives, (3) confidence justification"
}
```

## Implementation Steps

### 1. Prompt v1 (Baseline)
- Simple prompt asking for sentiment analysis
- Output: sentiment (positive/negative/neutral), confidence, brief reasoning
- Tested on 10 emails

### 2. Evaluation of v1
- Reviewed results for consistency and accuracy
- Identified issues: brief reasoning, overly conservative classification

### 3. Prompt v2 (Improved)
- Enhanced prompt with detailed guidelines
- Added key phrase identification
- Improved reasoning requirements (cites specific phrases)
- Added consistency check instruction
- Tested on same 10 emails

## Results Analysis

### v1 Results
- **Sentiment Distribution:** 9 negative, 1 neutral, 0 positive
- **Average Confidence:** 89.50%
- **Errors:** 0 (all processed successfully)
- **Issues:** Brief reasoning, all problems classified as negative

### v2 Results
- **Sentiment Distribution:** 7 negative, 3 neutral, 0 positive
- **Average Confidence:** 70.50%
- **Errors:** 2 JSON parsing errors (emails #1, #7)
- **Improvements:** Better reasoning quality, more nuanced classification

## What Failed in v1
1. **Overly Conservative:** All problem reports → negative (missed neutral questions)
2. **Brief Reasoning:** Generic explanations, not helpful for debugging
3. **No Consistency Checks:** No explicit instruction for similar emails

## What Improved in v2
1. **Better Reasoning:** Detailed explanations with key phrases cited
2. **More Nuanced:** Better distinction between problems and neutral questions
3. **Structured Reasoning:** Key phrases, alternatives considered, confidence justification

## What Regressed in v2
1. **JSON Parsing Errors:** 2 emails failed (model returned dict syntax)
2. **Lower Confidence:** Dropped from 89.50% to 70.50%

## How to Evaluate Prompts Systematically

1. **Consistency Check:** Group similar emails, verify same sentiment
2. **Reasoning Quality:** Check if reasoning cites specific phrases
3. **Edge Case Handling:** Test feature requests, polite problems, questions
4. **Technical Reliability:** Check for parsing errors, valid JSON
5. **Confidence Calibration:** Verify confidence aligns with certainty

## Recommendation
Use v2's reasoning approach with simplified JSON structure to avoid parsing errors. The improved reasoning quality is valuable for debugging, but reliability must be maintained.

## Files
- `sentiment.py` - Prompt v1 script
- `sentiment_v2.py` - Prompt v2 script (improved)
- `results_v1.csv` - v1 results
- `results_v2.csv` - v2 results

