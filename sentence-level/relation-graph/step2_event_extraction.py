import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification
from textblob import TextBlob

# ----------------------------------------------------
# Load Semantic Role Labeling model (AllenNLP-style)
# ----------------------------------------------------
# Model: "ShiLab/srl-bert-base"
tokenizer = AutoTokenizer.from_pretrained("ShiLab/srl-bert-base")
model = AutoModelForTokenClassification.from_pretrained("ShiLab/srl-bert-base")

id2label = model.config.id2label

def run_srl(sentence):
    """Run SRL model and return predicate-argument structures."""
    tokens = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens).logits
    predictions = outputs.argmax(dim=-1)[0].tolist()

    word_tokens = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
    tags = [id2label[p] for p in predictions]
    return word_tokens, tags


def extract_events_srl(sentence):
    """Extract events based on SRL predicates."""
    word_tokens, tags = run_srl(sentence)

    events = []
    current_event = None

    for tok, tag in zip(word_tokens, tags):

        # SRL begins with "B-V" => verb predicate
        if tag == "B-V":
            if current_event:
                events.append(current_event)
            current_event = {
                "trigger": tok,
                "args": {"ARG0": [], "ARG1": [], "ARG2": []}
            }

        # Collect arguments
        elif tag.startswith("B-ARG") and current_event:
            argtype = tag.split("-")[-1]
            current_event["args"][argtype] = [tok]

        elif tag.startswith("I-ARG") and current_event:
            argtype = tag.split("-")[-1]
            current_event["args"][argtype].append(tok)

    if current_event:
        events.append(current_event)

    # Clean formatting: join subwords
    for ev in events:
        for role in ev["args"]:
            ev["args"][role] = " ".join(ev["args"][role]).replace("##", "")
        ev["trigger"] = ev["trigger"].replace("##", "")

    return events


def attach_bias_info(event, bias_info):
    """Find bias metadata based on trigger/arguments."""
    result = {
        "has_bias": False,
        "bias_type": None,
        "polarity": None,
        "target": None,
        "biased_word": None
    }

    for target, info in bias_info.items():
        if info["text"].lower() in event["trigger"].lower():
            return {
                "has_bias": True,
                "bias_type": info["type"],
                "polarity": info["polarity"],
                "target": target,
                "biased_word": info["text"]
            }

    return result


def build_bias_lookup(annotations):
    lookup = {}
    for ann in annotations:
        target = ann.get("target", "")
        lookup[target] = {
            "text": ann.get("txt", ""),
            "type": ann.get("bias", ""),
            "polarity": ann.get("polarity", "")
        }
    return lookup


# ----------------------------------------------------
# Main
# ----------------------------------------------------
def refine_step2(input_path, output_path):
    data = json.load(open(input_path))
    print(f"⚙️ Running SRL-based event extraction on {len(data)} sentences...")

    for sent in tqdm(data):
        sentence = sent["text"]
        bias_info = build_bias_lookup(sent.get("annotations", []))

        srl_events = extract_events_srl(sentence)
        refined_events = []

        for ev in srl_events:
            actor = ev["args"].get("ARG0", None)
            obj = ev["args"].get("ARG1", None)
            sentiment = round(TextBlob(sentence).sentiment.polarity, 3)
            bias = attach_bias_info(ev, bias_info)

            refined_events.append({
                "trigger": ev["trigger"],
                "actor": actor,
                "object": obj,
                "sentiment": sentiment,
                **bias
            })

        sent["events"] = refined_events

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print("✅ Done! Saved refined events.")


if __name__ == "__main__":
    refine_step2(
        input_path = "../news-discourse/discourse_data.json",
        output_path="discourse_with_events.json"
    )
    