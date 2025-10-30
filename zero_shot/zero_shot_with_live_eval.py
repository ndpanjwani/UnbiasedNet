import json
import time
import re
from groq import Groq
from sklearn.metrics import f1_score, precision_recall_fscore_support

# ---------------- CONFIG ----------------
INPUT_FILE = "../data-preprocessing/basil_consolidated_all.json"
OUTPUT_FILE = "zero_shot_outputs.json"
MODEL = "llama-3.3-70b-versatile"  # Better model with larger context
LIMIT = 25
DELAY = 2

client = Groq(
)

# ---------------- METRICS TRACKER ----------------

class MetricsTracker:
    def __init__(self):
        self.y_true_binary = []
        self.y_pred_binary = []
        self.articles_processed = 0
        self.sentences_processed = 0
        self.parse_failures = 0
    
    def add_predictions(self, gt_sentences, pred_sentences):
        """Add predictions for one article."""
        for i, gt_sent in enumerate(gt_sentences):
            if i >= len(pred_sentences):
                break
            
            # Binary bias labels
            gt_label = 0 if gt_sent.get("bias") == "No" else 1
            pred_label = 0 if pred_sentences[i].get("bias") == "none" else 1
            
            self.y_true_binary.append(gt_label)
            self.y_pred_binary.append(pred_label)
            self.sentences_processed += 1
        
        self.articles_processed += 1
    
    def get_current_metrics(self):
        """Calculate current F1 scores."""
        if not self.y_true_binary:
            return None
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_true_binary, 
            self.y_pred_binary, 
            average='binary',
            zero_division=0
        )
        
        f1_macro = f1_score(self.y_true_binary, self.y_pred_binary, average='macro', zero_division=0)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_binary": f1,
            "f1_macro": f1_macro,
            "sentences": self.sentences_processed,
            "articles": self.articles_processed
        }
    
    def print_progress(self):
        """Print current metrics."""
        metrics = self.get_current_metrics()
        if metrics:
            print(f"  Running Stats - Articles: {metrics['articles']}, Sentences: {metrics['sentences']}")
            print(f"  F1 (Binary): {metrics['f1_binary']:.3f} | Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f}")


# ---------------- HELPER FUNCTIONS ----------------

def extract_json_from_response(text):
    """Extract and parse JSON from model response."""
    if not text:
        return None
    
    # Try markdown blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try whole text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try finding JSON object
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    return None


def format_article_prompt(article):
    meta = article.get("article_metadata", {})
    sentences = [s["text"] for s in article.get("sentences", [])]
    sentences = sentences[:30]

    instruction = (
        "You are a linguistic analyst specializing in detecting subtle political bias in writing.\n"
        "Carefully read each sentence of the following article and analyze the language\n\n"
        "For each sentence, decide:\n"
        "1. bias: 'none' (neutral/objective tone) or 'bias' (subjective, emotionally loaded, or framing tone)\n"
        "2. bias_type (if bias exists):\n"
        "   - 'lex' (lexical bias): subjective word choice, emotional tone, labeling ('reckless', 'heroic', 'radical', etc.)\n"
        "   - 'inf' (informational bias): framing, selective emphasis, implication, or omission (e.g., suggesting blame, causality, or moral judgment)\n\n"
        "Then determine the overall political orientation of the article text based only on linguistic patterns:\n"
        "article_bias: 'liberal', 'conservative', or 'neutral'.\n\n"
        "Be cautious but analytical: a sentence can appear factual yet still show framing or tonal bias.\n\n"
        "Return valid JSON only:\n"
        '{"sentences": [{"text": "...", "bias": "none"}, ...], "article_bias": "neutral"}'
    )

    article_info = (
        f"\nArticle Title: {meta.get('title', 'unknown')}\n"
        f"Source: {meta.get('source', 'unknown')}\n\n"
        f"Sentences:\n{json.dumps(sentences, indent=2)}"
    )

    return f"{instruction}\n{article_info}"


def analyze_article(article):
    """Send article to Groq and return parsed response."""
    prompt = format_article_prompt(article)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,  # Increased from 0.3 to be less conservative
            max_tokens=8000,  # Increased from 4000
        )

        if not response.choices:
            return None, None
        
        if response.choices[0].finish_reason == 'length':
            print("  [WARNING] Response truncated - trying to parse anyway...")
        
        output_text = response.choices[0].message.content
        if not output_text:
            return None, None
        
        output_text = output_text.strip()
        parsed_json = extract_json_from_response(output_text)
        
        # Even if truncated, if we got valid JSON, use it
        if not parsed_json and response.choices[0].finish_reason == 'length':
            print("  [ERROR] Truncated response couldn't be parsed as JSON")
        
        return output_text, parsed_json

    except Exception as e:
        print(f"  [ERROR] {e}")
        return None, None


# ---------------- MAIN LOOP ----------------

def run_zero_shot_with_eval():
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} articles from {INPUT_FILE}")
    print(f"Processing up to {LIMIT} articles with live evaluation...\n")

    results = []
    tracker = MetricsTracker()
    
    for i, article in enumerate(data[:LIMIT]):
        title = article.get("article_metadata", {}).get("title", "Untitled")
        print(f"\n[{i+1}/{LIMIT}] {title[:60]}...")

        raw_output, parsed_output = analyze_article(article)
        
        if raw_output and parsed_output:
            # Add to results
            record = {
                "uuid": article.get("uuid"),
                "triplet_uuid": article.get("triplet_uuid"),
                "source": article.get("article_metadata", {}).get("source"),
                "title": title,
                "model_output_raw": raw_output,
                "model_output_parsed": parsed_output
            }
            results.append(record)
            
            # Update metrics
            if "sentences" in parsed_output:
                gt_sentences = article.get("sentences", [])
                pred_sentences = parsed_output["sentences"]
                tracker.add_predictions(gt_sentences, pred_sentences)
                tracker.print_progress()
            else:
                tracker.parse_failures += 1
                print("  [WARNING] No sentences in parsed output")
        else:
            tracker.parse_failures += 1
            print("  ✗ Failed to get valid response")

        if i < LIMIT - 1:
            time.sleep(DELAY)

    # Save results
    with open(OUTPUT_FILE, "w") as out_f:
        json.dump(results, out_f, indent=2, ensure_ascii=False)

    # Final summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    final_metrics = tracker.get_current_metrics()
    if final_metrics:
        print(f"\nArticles Processed: {final_metrics['articles']}")
        print(f"Sentences Evaluated: {final_metrics['sentences']}")
        print(f"Parse Failures: {tracker.parse_failures}")
        print(f"\nFinal Metrics (Sentence-Level Binary Bias Detection):")
        print(f"  Precision: {final_metrics['precision']:.3f}")
        print(f"  Recall:    {final_metrics['recall']:.3f}")
        print(f"  F1 Score:  {final_metrics['f1_binary']:.3f}")
        print(f"  F1 Macro:  {final_metrics['f1_macro']:.3f}")
    
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    run_zero_shot_with_eval()