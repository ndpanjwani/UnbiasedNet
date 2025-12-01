import json, re, random, numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EvalPrediction
)
import torch
from sklearn.metrics import f1_score, precision_recall_curve

# -------------------------------
# Helpers to align with SRL step-2
# -------------------------------
def get_actor(ev):
    return ev.get("actor") or ev.get("ARG0") or ev.get("args", {}).get("ARG0")

def get_object(ev):
    return ev.get("object") or ev.get("ARG1") or ev.get("args", {}).get("ARG1")

def get_trigger(ev):
    return ev.get("trigger") or ev.get("lemma") or ev.get("predicate")


# -------------------------------
# Detect weak relations (with SRL support)
# -------------------------------
def detect_relation_type(e1, e2, s1, s2, dist):
    txt_pair = (s1.get("text","") + " " + s2.get("text","")).lower()
    role1 = s1.get("discourse_role", "NONE")
    role2 = s2.get("discourse_role", "NONE")

    e1_trigger, e2_trigger = get_trigger(e1), get_trigger(e2)
    e1_actor,   e2_actor   = get_actor(e1),  get_actor(e2)
    e1_obj,     e2_obj     = get_object(e1), get_object(e2)

    # ---- Causal
    has_causal = bool(re.search(r"\b(because|due to|caused|led to|therefore|hence|resulted|consequently)\b", txt_pair))
    is_cause_to_main = role1 in ["Cause_General", "Cause_Specific"] and role2 == "Main"

    if has_causal and is_cause_to_main:
        return "causal", 0.95
    if has_causal and dist <= 2:
        return "causal", 0.85
    if is_cause_to_main:
        return "causal", 0.75

    # ---- Temporal
    has_temp = bool(re.search(r"\b(before|after|then|since|while|subsequently|meanwhile|following)\b", txt_pair))
    if role1 == "Main" and role2 == "Main_Consequence":
        return "temporal", 0.9
    if has_temp and dist <= 2:
        return "temporal", 0.8

    # ---- Coreference (SRL-friendly)
    if (
        e1_trigger == e2_trigger and
        e1_actor is not None and e1_actor == e2_actor and
        dist <= 2
    ):
        return "coreference", 0.95

    # ---- Continuation (same actor)
    if e1_actor and e1_actor == e2_actor and e1_trigger != e2_trigger and dist <= 2:
        return "continuation", 0.8

    # ---- Weak continuation via shared entities
    ents1 = {e1_actor, e1_obj} - {None}
    ents2 = {e2_actor, e2_obj} - {None}
    if (ents1 & ents2) and dist <= 3:
        return "continuation", 0.65

    # ---- Default none
    if dist > 4:
        return "none", 0.9
    return "none", 0.6


# -------------------------------
# Create training pairs (with negative sampling)
# -------------------------------
def create_training_data_text(data, max_samples_per_article=60, min_confidence=0.6):
    articles = defaultdict(list)
    for i, s in enumerate(data):
        s["sentence_idx"] = i
        articles[s["article_id"]].append(s)

    pairs, stats = [], Counter()

    for aid, sents in tqdm(list(articles.items())[:200], desc="Pairing events"):
        sents = sorted(sents, key=lambda x: x["sentence_idx"])
        events = [{"event": ev, "sentence": s} for s in sents for ev in s.get("events", [])]

        sampled = 0
        for i, e1_ctx in enumerate(events):
            if sampled >= max_samples_per_article:
                break

            for j in range(i+1, min(i+8, len(events))):
                e1, e2 = e1_ctx["event"], events[j]["event"]
                s1, s2 = e1_ctx["sentence"], events[j]["sentence"]
                dist = abs(s1["sentence_idx"] - s2["sentence_idx"])
                if dist > 5: 
                    continue

                label, conf = detect_relation_type(e1, e2, s1, s2, dist)
                if conf < min_confidence:
                    continue

                # Build improved text representation
                text1 = (
                    f"Sentence: {s1['text']} "
                    f"Trigger: {get_trigger(e1)} "
                    f"ARG0: {get_actor(e1)} ARG1: {get_object(e1)} "
                    f"Discourse: {s1.get('discourse_role','NONE')}"
                )
                text2 = (
                    f"Sentence: {s2['text']} "
                    f"Trigger: {get_trigger(e2)} "
                    f"ARG0: {get_actor(e2)} ARG1: {get_object(e2)} "
                    f"Discourse: {s2.get('discourse_role','NONE')}"
                )

                pairs.append({
                    "text1": text1,
                    "text2": text2,
                    "label": LABEL2ID[label],
                    "confidence": conf
                })

                stats[label] += 1
                sampled += 1

                # --- 🔥 Negative sampling (strengthens "none")
                if random.random() < 0.25:
                    pairs.append({
                        "text1": f"Sentence: {s1['text']}",
                        "text2": f"Sentence: {s2['text']}",
                        "label": LABEL2ID["none"],
                        "confidence": 0.95
                    })
                    stats["none"] += 1

    print("\n📊 Weak-label distribution (after negative sampling):")
    for l, c in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {l:15s}: {c:5d}")
    return pairs


# --------------------------------------------------------
# Everything else (training, threshold search, build graph)
# remains unchanged
# --------------------------------------------------------

# -------------------------------
# Metrics
# -------------------------------
def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": (preds == labels).mean(),
        "macro_f1": f1_score(labels, preds, average="macro")
    }


# -------------------------------
# Train the transformer model
# -------------------------------
def train_transformer_relation_model(pairs, model_name="roberta-base", output_dir="./relation_model"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(RELATION_LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )

    ds = Dataset.from_list(pairs).shuffle(seed=42)
    split = ds.train_test_split(test_size=0.15, seed=42)
    tokenized = split.map(lambda e: tokenizer(e["text1"], e["text2"], truncation=True, padding="max_length", max_length=256), batched=True)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_dir="./logs",
        report_to="none",
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("\n🚀 Training transformer...")
    trainer.train()
    results = trainer.evaluate()
    print("\n📊 Eval results:", results)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Saved model → {output_dir}")
    return trainer, tokenized


# -------------------------------
# Threshold selection
# -------------------------------
def find_optimal_threshold(trainer, tokenizer, model, val_dataset):
    model.eval()
    probs_all, labels_all = [], []
    dl = trainer.get_eval_dataloader(val_dataset)
    device = next(model.parameters()).device

    for batch in dl:
        inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        probs_all.append(probs)
        labels_all.extend(batch["labels"].cpu().numpy())

    probs_all = np.concatenate(probs_all, axis=0)
    labels_all = np.array(labels_all)
    edge_conf = 1.0 - probs_all[:, LABEL2ID["none"]]
    y_true = (labels_all != LABEL2ID["none"]).astype(int)

    precision, recall, thr = precision_recall_curve(y_true, edge_conf)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    best_thr = thr[np.argmax(f1)]

    print(f"\n🔍 Optimal confidence threshold (by F1): {best_thr:.3f}")
    return float(best_thr)


# -------------------------------
# Build ERG edges (unchanged except SRL fixes)
# -------------------------------
def build_graph_with_model(data, model_path="./relation_model", confidence_threshold=0.7):
    print(f"\n📊 Building graph with model (threshold={confidence_threshold})...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()

    articles = defaultdict(list)
    for i, s in enumerate(data):
        s["sentence_idx"] = i
        articles[s["article_id"]].append(s)

    edges, stats = [], Counter()
    for aid, sents in tqdm(articles.items(), desc="Building graph"):
        sents = sorted(sents, key=lambda x: x["sentence_idx"])
        events = [{"event": ev, "sentence": s} for s in sents for ev in s.get("events", [])]

        for i, e1_ctx in enumerate(events):
            for j in range(i+1, min(i+8, len(events))):
                e1, e2 = e1_ctx["event"], events[j]["event"]
                s1, s2 = e1_ctx["sentence"], events[j]["sentence"]
                dist = abs(s1["sentence_idx"] - s2["sentence_idx"])
                if dist > 5:
                    continue

                text1 = (
                    f"Sentence: {s1['text']} "
                    f"Trigger: {get_trigger(e1)} "
                    f"ARG0: {get_actor(e1)} ARG1: {get_object(e1)} "
                    f"Discourse: {s1.get('discourse_role','NONE')}"
                )
                text2 = (
                    f"Sentence: {s2['text']} "
                    f"Trigger: {get_trigger(e2)} "
                    f"ARG0: {get_actor(e2)} ARG1: {get_object(e2)} "
                    f"Discourse: {s2.get('discourse_role','NONE')}"
                )

                inputs = tokenizer(text1, text2, return_tensors="pt", truncation=True, max_length=256).to(model.device)

                with torch.no_grad():
                    probs = torch.softmax(model(**inputs).logits, dim=-1)[0].cpu().numpy()

                order = np.argsort(-probs)
                top1, top2 = order[0], order[1]

                pred_label = top1
                confidence = probs[top1]

                # Fallback: promote top2 if "none" dominates
                if ID2LABEL[top1] == "none" and probs[top2] > 0.3 and ID2LABEL[top2] != "none":
                    pred_label = top2
                    confidence = probs[top2]

                relation = ID2LABEL[pred_label]
                stats[relation] += 1

                if confidence >= confidence_threshold and relation != "none":
                    e1_id = f"{aid}_{s1['sentence_idx']}_{get_trigger(e1)}_{i}"
                    e2_id = f"{aid}_{s2['sentence_idx']}_{get_trigger(e2)}_{j}"
                    edges.append({
                        "article_id": aid,
                        "src_event_id": e1_id,
                        "tgt_event_id": e2_id,
                        "src_trigger": get_trigger(e1),
                        "tgt_trigger": get_trigger(e2),
                        "relation": relation,
                        "confidence": round(float(confidence), 3),
                        "sentence_distance": dist
                    })

    print(f"\n✅ Built graph with {len(edges)} edges")
    print("Relation counts:")
    for rel, c in stats.items():
        print(f"{rel:15s}: {c}")

    return edges


# -------------------------------
# Main
# -------------------------------
def main():
    input_path = "discourse_with_events.json"
    model_dir = "./relation_model"
    output_edges = "erg_edges_transformer.json"

    print("📘 Loading data...")
    with open(input_path) as f:
        data = json.load(f)

    pairs = create_training_data_text(data)
    trainer, tokenized = train_transformer_relation_model(pairs, output_dir=model_dir)
    best_thr = find_optimal_threshold(trainer, trainer.tokenizer, trainer.model, tokenized["test"])

    edges = build_graph_with_model(data, model_path=model_dir, confidence_threshold=best_thr)
    with open(output_edges, "w") as f:
        json.dump(edges, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Saved edges → {output_edges}")


if __name__ == "__main__":
    main()