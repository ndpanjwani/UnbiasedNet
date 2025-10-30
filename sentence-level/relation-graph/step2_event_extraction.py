import spacy
import json
from textblob import TextBlob
from tqdm import tqdm
from collections import Counter

#NEXT STEP -- MAKE MORE DYNAMIC 

# Article text
#    ↓
# Sentence segmentation (spaCy or Stanza)
#    ↓
# [Dynamic model]
#    ↳  (a) trigger detection  → find event words
#    ↳  (b) argument labeling → who did what, to whom
#    ↳  (c) event coref / merging
#    ↓
# JSON: [{trigger, actor, object, sentiment, role, ...}]
# -------------------------------
# Load model
# -------------------------------
nlp = spacy.load("en_core_web_sm")

# -------------------------------
# Helper functions
# -------------------------------
def build_bias_lookup(annotations):
    """Create dict target->bias info from annotations."""
    lookup = {}
    for ann in annotations:
        target = ann.get("target", "")
        lookup[target] = {
            "text": ann.get("txt", ""),
            "type": ann.get("bias", ""),
            "polarity": ann.get("polarity", ""),
            "aim": ann.get("aim", ""),
            "is_quote": ann.get("quote", "no") == "yes",
            "speaker": ann.get("speaker", "")
        }
    return lookup


def get_event_bias(trigger_token, actor, obj, bias_info):
    """Match event parts to known bias annotations."""
    result = {
        "has_bias": False,
        "bias_type": None,
        "polarity": None,
        "target": None,
        "biased_word": None
    }

    trigger_text = trigger_token.text.lower()
    for target, info in bias_info.items():
        word = info["text"].lower()
        if word and (word in trigger_text or trigger_text in word):
            result.update({
                "has_bias": True,
                "bias_type": info["type"],
                "polarity": info["polarity"],
                "target": target,
                "biased_word": info["text"]
            })
            return result

    for arg in (actor, obj):
        if not arg:
            continue
        arg_clean = arg.replace("_", " ")
        if arg_clean in bias_info:
            info = bias_info[arg_clean]
            result.update({
                "has_bias": True,
                "bias_type": info["type"],
                "polarity": info["polarity"],
                "target": arg_clean,
                "biased_word": info["text"]
            })
            return result
    return result


def is_event_trigger(token):
    """Select verbs likely to represent events."""
    if token.pos_ != "VERB":
        return False
    if token.dep_ in ("aux", "auxpass") or token.lemma_ in (
        "be", "have", "do", "will", "would", "can", "could",
        "may", "might", "should"
    ):
        return False
    if token.dep_ not in ("ROOT", "conj", "xcomp", "ccomp", "advcl", "relcl"):
        return False
    return True


def extract_actor(verb_token):
    """Return subject (actor) of the verb."""
    for child in verb_token.children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            return get_full_np(child)
    for child in verb_token.children:
        if child.dep_ == "agent":
            for gc in child.children:
                if gc.dep_ == "pobj":
                    return get_full_np(gc)
    return None


def extract_object(verb_token):
    """Return object/patient of the verb."""
    for child in verb_token.children:
        if child.dep_ in ("dobj", "attr", "oprd"):
            return get_full_np(child)
        if child.dep_ == "prep" and verb_token.lemma_ in ("call", "refer", "describe"):
            for gc in child.children:
                if gc.dep_ == "pobj":
                    return get_full_np(gc)
    return None


def get_full_np(token):
    """Return full noun phrase including modifiers."""
    exclude = {"punct", "cc"}
    words = sorted(
        [t for t in token.subtree if t.dep_ not in exclude],
        key=lambda t: t.i
    )
    return " ".join([t.text for t in words])


# -------------------------------
# Main
# -------------------------------
def main(input_path, output_path, stats_path=None):
    print("📘 Loading dataset...")
    data = json.load(open(input_path))

    print(f"⚙️ Extracting events for {len(data)} sentences...")
    all_events = []
    for sent in tqdm(data, desc="Processing sentences"):
        sent_text = sent["text"]
        doc = nlp(sent_text)
        bias_info = build_bias_lookup(sent.get("annotations", []))
        sent_events = []

        for token in doc:
            if not is_event_trigger(token):
                continue
            actor = extract_actor(token)
            obj = extract_object(token)
            event_bias = get_event_bias(token, actor, obj, bias_info)

            sent_events.append({
                "trigger": token.lemma_,
                "actor": actor,
                "object": obj,
                "sentiment": round(TextBlob(sent_text).sentiment.polarity, 3),
                "has_bias": event_bias["has_bias"],
                "bias_type": event_bias["bias_type"],
                "bias_polarity": event_bias["polarity"],
                "bias_target": event_bias["target"],
                "biased_word": event_bias["biased_word"]
            })

        sent["events"] = sent_events
        all_events.extend(sent_events)

    print("💾 Saving updated dataset...")
    json.dump(data, open(output_path, "w"), indent=2, ensure_ascii=False)

    stats = compute_statistics(all_events)
    print_statistics(stats)
    if stats_path:
        json.dump(stats, open(stats_path, "w"), indent=2)

    print(f"✅ Done! Updated dataset saved to {output_path}")


# -------------------------------
# Stats & print
# -------------------------------
def compute_statistics(events):
    stats = {
        "total_events": len(events),
        "events_with_actors": sum(1 for e in events if e["actor"]),
        "events_with_objects": sum(1 for e in events if e["object"]),
        "events_with_bias": sum(1 for e in events if e["has_bias"]),
        "top_triggers": dict(Counter(e["trigger"] for e in events).most_common(10)),
        "top_actors": dict(Counter(e["actor"] for e in events if e["actor"]).most_common(10))
    }
    return stats


def print_statistics(stats):
    print("\n" + "=" * 60)
    print("📊 EVENT EXTRACTION STATS")
    print("=" * 60)
    print(f"Total events: {stats['total_events']}")
    print(f"Events with actors: {stats['events_with_actors']}")
    print(f"Events with objects: {stats['events_with_objects']}")
    print(f"Events with bias: {stats['events_with_bias']}")
    print("\nTop triggers:")
    for t, c in stats["top_triggers"].items():
        print(f"  {t:15s}: {c}")
    print("\nTop actors:")
    for a, c in stats["top_actors"].items():
        print(f"  {a:25s}: {c}")
    print("=" * 60 + "\n")


# -------------------------------
# Entry point
# -------------------------------

if __name__ == "__main__":
    main(
        input_path="/Users/nainapanjwani/444/UnbiasedNet/sentence-level/news-discourse/discourse_data.json",
        output_path="/Users/nainapanjwani/444/UnbiasedNet/sentence-level/relation-graph/discourse_with_events.json",
        stats_path="/Users/nainapanjwani/444/UnbiasedNet/sentence-level/relation-graph/event_stats.json"
    )