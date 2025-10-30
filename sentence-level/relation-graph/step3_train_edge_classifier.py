import json
import re
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
from sentence_transformers import SentenceTransformer

# -------------------------------
# Global configs
# -------------------------------
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

DISCOURSE_WEIGHTS = {
    "Main": 1.0,
    "Main_Consequence": 0.9,
    "Cause_General": 0.8,
    "Cause_Specific": 0.8,
    "Distant_Historical": 0.5,
    "Distant_Anecdotal": 0.5,
    "Distant_Evaluation": 0.6,
    "Distant_Expectation": 0.6,
    "NONE": 0.3
}

# -------------------------------
# Feature extraction for event pairs
# -------------------------------
def extract_pair_features(e1, e2, sent1, sent2, sent_distance):
    """Extract features for an event pair."""
    features = {}

    # === Distance features ===
    features["sent_distance"] = sent_distance
    features["is_adjacent"] = 1 if sent_distance == 1 else 0
    features["is_nearby"] = 1 if sent_distance <= 2 else 0

    # === Discourse features ===
    role1 = sent1.get("discourse_role", "NONE")
    role2 = sent2.get("discourse_role", "NONE")
    features["src_discourse_weight"] = DISCOURSE_WEIGHTS.get(role1, 0.5)
    features["tgt_discourse_weight"] = DISCOURSE_WEIGHTS.get(role2, 0.5)
    features["discourse_diff"] = abs(
        features["src_discourse_weight"] - features["tgt_discourse_weight"]
    )
    features["is_cause_to_main"] = (
        1 if role1 in ["Cause_General", "Cause_Specific"] and role2 == "Main" else 0
    )
    features["is_main_to_consequence"] = (
        1 if role1 == "Main" and role2 == "Main_Consequence" else 0
    )
    features["both_main"] = 1 if role1 == "Main" and role2 == "Main" else 0

    # === Entity overlap ===
    entities1 = {e1.get("actor"), e1.get("object")} - {None}
    entities2 = {e2.get("actor"), e2.get("object")} - {None}
    shared = entities1 & entities2
    features["has_shared_entity"] = 1 if shared else 0
    features["num_shared_entities"] = len(shared)
    features["same_actor"] = (
        1 if e1.get("actor") and e1.get("actor") == e2.get("actor") else 0
    )
    features["same_object"] = (
        1 if e1.get("object") and e1.get("object") == e2.get("object") else 0
    )

    # === Trigger and semantic features ===
    features["same_trigger"] = 1 if e1.get("trigger") == e2.get("trigger") else 0
    e1_trigger, e2_trigger = e1.get("trigger", ""), e2.get("trigger", "")
    try:
        e1_emb = EMBED_MODEL.encode(e1_trigger, normalize_embeddings=True)
        e2_emb = EMBED_MODEL.encode(e2_trigger, normalize_embeddings=True)
        features["trigger_similarity"] = float(np.dot(e1_emb, e2_emb))
    except Exception:
        features["trigger_similarity"] = 0.0

    # === Sentiment features ===
    s1_val, s2_val = e1.get("sentiment", 0), e2.get("sentiment", 0)
    features["sentiment_diff"] = abs(s1_val - s2_val)
    features["sentiment_product"] = s1_val * s2_val
    features["both_negative"] = 1 if s1_val < -0.1 and s2_val < -0.1 else 0
    features["both_positive"] = 1 if s1_val > 0.1 and s2_val > 0.1 else 0

    # === Bias features ===
    features["src_has_bias"] = 1 if sent1.get("has_bias", False) else 0
    features["tgt_has_bias"] = 1 if sent2.get("has_bias", False) else 0
    features["both_have_bias"] = (
        1 if sent1.get("has_bias") and sent2.get("has_bias") else 0
    )

    # === Lexical features ===
    text_pair = (sent1.get("text", "") + " " + sent2.get("text", "")).lower()
    features["has_causal_marker"] = 1 if re.search(
        r"\b(because|due to|led to|caused|resulted|therefore|hence)\b", text_pair
    ) else 0
    features["has_temporal_marker"] = 1 if re.search(
        r"\b(before|after|then|when|since|while|subsequently)\b", text_pair
    ) else 0
    features["has_conditional_marker"] = 1 if re.search(
        r"\b(if|unless|provided|assuming)\b", text_pair
    ) else 0
    features["has_contrast_marker"] = 1 if re.search(
        r"\b(but|however|although|yet|nevertheless)\b", text_pair
    ) else 0

    return features


# -------------------------------
# Heuristic weak labels
# -------------------------------
def get_heuristic_label(e1, e2, sent1, sent2, sent_distance, f):
    """Weak labeling function for training data."""
    # strong causal
    if f["is_cause_to_main"] and f["has_causal_marker"]:
        return 1, "causal"
    # temporal
    if f["is_main_to_consequence"] and sent_distance <= 2:
        return 1, "temporal"
    # coreference
    if f["same_trigger"] and f["same_actor"] and sent_distance <= 2:
        return 1, "coreference"
    # same actor, adjacent
    if f["same_actor"] and f["is_adjacent"] and not f["same_trigger"]:
        return 1, "temporal"
    # both main and shared entity
    if f["both_main"] and f["has_shared_entity"] and sent_distance <= 2:
        return 1, "continuation"
    # fallback strong cause
    if f["is_cause_to_main"] and sent_distance <= 3:
        return 1, "causal"

    # negatives
    if sent_distance > 4:
        return 0, "none"
    if not f["has_shared_entity"] and sent_distance > 1 and np.random.random() < 0.3:
        return 0, "none"
    return 0, "none"


# -------------------------------
# Create weakly labeled dataset
# -------------------------------
def create_training_data(data, max_samples_per_article=60):
    """Generate event pairs and weak labels."""
    articles = defaultdict(list)
    for idx, sent in enumerate(data):
        sent["sentence_idx"] = idx
        articles[sent["article_id"]].append(sent)

    X, y, edge_types = [], [], []
    print("🧩 Creating training pairs...")

    for article_id, sents in tqdm(list(articles.items())[:100]):
        sents = sorted(sents, key=lambda x: x["sentence_idx"])
        events = []
        for s in sents:
            for ev in s.get("events", []):
                events.append({"event": ev, "sentence": s})
        sampled = 0
        for i, e1_ctx in enumerate(events):
            if sampled >= max_samples_per_article:
                break
            for j in range(i + 1, min(i + 8, len(events))):
                e1, e2 = e1_ctx["event"], events[j]["event"]
                s1, s2 = e1_ctx["sentence"], events[j]["sentence"]
                dist = abs(s1["sentence_idx"] - s2["sentence_idx"])
                feats = extract_pair_features(e1, e2, s1, s2, dist)
                label, etype = get_heuristic_label(e1, e2, s1, s2, dist, feats)
                X.append(list(feats.values()))
                y.append(label)
                edge_types.append(etype)
                sampled += 1

    feat_names = list(
        extract_pair_features(
            {"trigger": "a"}, {"trigger": "b"},
            {"discourse_role": "NONE"}, {"discourse_role": "NONE"}, 1
        ).keys()
    )
    return np.array(X), np.array(y), feat_names, edge_types


# -------------------------------
# Train edge classifier
# -------------------------------
def train_edge_classifier(data):
    print("🎓 Training Random Forest classifier...")
    X, y, feat_names, edge_types = create_training_data(data)

    print(f"Samples: {len(X)}  Positives: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_split=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(Xtr, ytr)
    print(f"Train acc: {clf.score(Xtr, ytr):.3f}  Test acc: {clf.score(Xte, yte):.3f}")

    print("\n🔝 Top features:")
    for f, imp in sorted(zip(feat_names, clf.feature_importances_), key=lambda x: -x[1])[:10]:
        print(f"  {f:25s}: {imp:.3f}")
    return clf, feat_names


# -------------------------------
# Apply classifier to build ERG
# -------------------------------
def infer_relation_type(f):
    if f["same_trigger"] and f["same_actor"]:
        return "coreference"
    elif f["is_cause_to_main"] or f["has_causal_marker"]:
        return "causal"
    elif f["is_main_to_consequence"] or f["has_temporal_marker"]:
        return "temporal"
    elif f["has_contrast_marker"]:
        return "contrast"
    elif f["same_actor"]:
        return "continuation"
    else:
        return "related"


def build_event_relations_with_classifier(data, clf, feat_names, threshold=0.6):
    articles = defaultdict(list)
    for idx, s in enumerate(data):
        s["sentence_idx"] = idx
        articles[s["article_id"]].append(s)

    edges = []
    print(f"⚙️ Building ERG edges (thr={threshold})...")
    for aid, sents in tqdm(articles.items()):
        sents = sorted(sents, key=lambda x: x["sentence_idx"])
        evs = []
        for s in sents:
            for ev in s.get("events", []):
                evs.append({"event": ev, "sentence": s})
        for i, e1_ctx in enumerate(evs):
            for j in range(i + 1, min(i + 8, len(evs))):
                e1, e2 = e1_ctx["event"], evs[j]["event"]
                s1, s2 = e1_ctx["sentence"], evs[j]["sentence"]
                dist = abs(s1["sentence_idx"] - s2["sentence_idx"])
                if dist > 5:
                    continue
                feats = extract_pair_features(e1, e2, s1, s2, dist)
                fv = np.array([list(feats.values())])
                prob = clf.predict_proba(fv)[0][1]
                if prob >= threshold:
                    rel = infer_relation_type(feats)
                    e1_id = f"{aid}_{s1['sentence_idx']}_{e1['trigger']}_{i}"
                    e2_id = f"{aid}_{s2['sentence_idx']}_{e2['trigger']}_{j}"
                    edges.append({
                        "article_id": aid,
                        "src_event_id": e1_id,
                        "src_trigger": e1["trigger"],
                        "tgt_event_id": e2_id,
                        "tgt_trigger": e2["trigger"],
                        "relation": rel,
                        "confidence": round(float(prob), 3),
                        "sentence_distance": dist
                    })
    return edges


# -------------------------------
# Main pipeline
# -------------------------------
def main():
    input_path = "/Users/nainapanjwani/444/UnbiasedNet/sentence-level/relation-graph/discourse_with_events.json"
    model_path = "/Users/nainapanjwani/444/UnbiasedNet/sentence-level/relation-graph/edge_classifier.pkl"
    output_path = "/Users/nainapanjwani/444/UnbiasedNet/sentence-level/relation-graph/erg_edges_learned.json"

    print("📘 Loading data...")
    with open(input_path) as f:
        data = json.load(f)

    try:
        with open(model_path, "rb") as f:
            clf, feat_names = pickle.load(f)
        print("✅ Loaded existing classifier")
    except FileNotFoundError:
        print("⚙️ Training new classifier...")
        clf, feat_names = train_edge_classifier(data)
        with open(model_path, "wb") as f:
            pickle.dump((clf, feat_names), f)
        print(f"💾 Saved model → {model_path}")

    for thr in [0.5, 0.6, 0.7]:
        print(f"\n{'='*60}\n Building with threshold={thr}\n{'='*60}")
        edges = build_event_relations_with_classifier(data, clf, feat_names, thr)
        print(f"Edges: {len(edges)}")
        rel_counts = Counter(e["relation"] for e in edges)
        for r, c in rel_counts.most_common():
            print(f"  {r:15s}: {c:5d}")

    optimal = 0.6
    edges = build_event_relations_with_classifier(data, clf, feat_names, optimal)
    with open(output_path, "w") as f:
        json.dump(edges, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved {len(edges)} edges → {output_path}")


if __name__ == "__main__":
    main()