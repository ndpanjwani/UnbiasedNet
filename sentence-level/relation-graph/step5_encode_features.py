import os
import json
import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ===============================================================
# CONFIG
# ===============================================================
GRAPH_DIR = "erg_graphs"
OUTPUT_DIR = "erg_features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    "Speech": 0.7,
    "NONE": 0.3
}

# ===============================================================
# HELPER FUNCTIONS (SRL-compatible)
# ===============================================================

def get_actor(ev):
    return ev.get("actor") or ev.get("ARG0") or ev.get("args", {}).get("ARG0")

def get_object(ev):
    return ev.get("object") or ev.get("ARG1") or ev.get("args", {}).get("ARG1")

def get_trigger(ev):
    return ev.get("trigger") or ev.get("predicate") or ev.get("lemma") or ""

def as_bool(x):
    return str(x).lower() == "true"

def normalize(v):
    v = np.array(v)
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


# ===============================================================
# FEATURE ENCODING
# ===============================================================

def encode_graph_features(graph_path):
    """
    Encode node features for heterogeneous ERGs.
    Output:
    - 396-dim sentence features
    - 396-dim event features
    - metadata with bias labels
    """
    G = nx.read_graphml(graph_path)
    article_id = os.path.basename(graph_path).replace(".graphml", "")

    sentence_features = {}
    event_features = {}
    node_metadata = {}

    for node, data in G.nodes(data=True):

        node_type = data.get("node_type", "unknown")

        # ===============================================================
        # SENTENCE NODE FEATURES (396-dim)
        # ===============================================================
        if node_type == "sentence":

            text = data.get("text", "")

            # 384-dim normalized embedding
            text_emb = EMBED_MODEL.encode(text)
            text_emb = normalize(text_emb)

            # discourse roles
            discourse_role = data.get("discourse_role", "NONE")
            prev_discourse_role = data.get("prev_discourse_role", "NONE")
            next_discourse_role = data.get("next_discourse_role", "NONE")

            discourse_weight = DISCOURSE_WEIGHTS.get(discourse_role, 0.5)
            prev_discourse_weight = DISCOURSE_WEIGHTS.get(prev_discourse_role, 0.5)
            next_discourse_weight = DISCOURSE_WEIGHTS.get(next_discourse_role, 0.5)

            # Connected events (via predecessors)
            connected_events = [
                ev for ev in G.predecessors(node)
                if G.nodes[ev].get("node_type") == "event"
            ]

            if connected_events:
                sentiments = [float(G.nodes[e].get("sentiment", 0)) for e in connected_events]
                has_bias_flags = [
                    1.0 if as_bool(G.nodes[e].get("has_bias", "False")) else 0.0
                    for e in connected_events
                ]

                mean_sentiment = float(np.mean(sentiments))
                std_sentiment = float(np.std(sentiments))
                min_sentiment = float(np.min(sentiments))
                max_sentiment = float(np.max(sentiments))
                event_count = len(connected_events)
                bias_ratio = float(np.mean(has_bias_flags))
            else:
                mean_sentiment = std_sentiment = min_sentiment = max_sentiment = 0.0
                event_count = 0
                bias_ratio = 0.0

            in_degree = float(G.in_degree(node))
            out_degree = float(G.out_degree(node))

            sentence_feat = np.concatenate([
                text_emb,
                np.array([
                    discourse_weight,
                    prev_discourse_weight,
                    next_discourse_weight,
                    mean_sentiment,
                    std_sentiment,
                    min_sentiment,
                    max_sentiment,
                    event_count,
                    bias_ratio,
                    in_degree,
                    out_degree,
                    0.0
                ])
            ])

            sentence_features[node] = sentence_feat.tolist()

            node_metadata[node] = {
                "type": "sentence",
                "text": text[:120],
                "discourse_role": discourse_role,
                "prev_discourse_role": prev_discourse_role,
                "next_discourse_role": next_discourse_role,
                "has_bias": as_bool(data.get("has_bias", "False")),
                "event_count": event_count,
                "bias_ratio": bias_ratio
            }

        # ===============================================================
        # EVENT NODE FEATURES (396-dim)
        # ===============================================================
        elif node_type == "event":

            trigger = get_trigger(data)
            actor = get_actor(data) or ""
            obj = get_object(data) or ""
            sentiment = float(data.get("sentiment", 0))

            # 384-dim normalized embedding
            trigger_emb = EMBED_MODEL.encode(trigger)
            trigger_emb = normalize(trigger_emb)

            has_bias = 1.0 if as_bool(data.get("has_bias", "False")) else 0.0

            has_actor = 1.0 if actor != "" else 0.0
            has_object = 1.0 if obj != "" else 0.0

            in_degree = float(G.in_degree(node))
            out_degree = float(G.out_degree(node))

            # Parent sentence (successor)
            parent_sents = [
                s for s in G.successors(node)
                if G.nodes[s].get("node_type") == "sentence"
            ]

            if parent_sents:
                parent = G.nodes[parent_sents[0]]
                pd = parent.get("discourse_role", "NONE")
                pp = parent.get("prev_discourse_role", "NONE")
                pn = parent.get("next_discourse_role", "NONE")

                parent_discourse_weight = DISCOURSE_WEIGHTS.get(pd, 0.5)
                parent_prev_weight = DISCOURSE_WEIGHTS.get(pp, 0.5)
                parent_next_weight = DISCOURSE_WEIGHTS.get(pn, 0.5)
            else:
                parent_discourse_weight = parent_prev_weight = parent_next_weight = 0.5

            neighbor_events = [
                e for e in G.neighbors(node)
                if G.nodes[e].get("node_type") == "event"
            ]
            num_neighbor_events = float(len(neighbor_events))

            event_feat = np.concatenate([
                trigger_emb,
                np.array([
                    sentiment,
                    has_bias,
                    has_actor,
                    has_object,
                    in_degree,
                    out_degree,
                    parent_discourse_weight,
                    parent_prev_weight,
                    parent_next_weight,
                    num_neighbor_events,
                    0.0,
                    0.0
                ])
            ])

            event_features[node] = event_feat.tolist()

            node_metadata[node] = {
                "type": "event",
                "trigger": trigger,
                "actor": actor[:40],
                "object": obj[:40],
                "sentiment": sentiment,
                "has_bias": bool(has_bias),
                "degree": in_degree + out_degree
            }

    return article_id, sentence_features, event_features, node_metadata


# ===============================================================
# MAIN
# ===============================================================
print(f"🧩 Encoding features for graphs in {GRAPH_DIR}/ ...")

total_sent_nodes = 0
total_event_nodes = 0
total_biased_sents = 0
total_graphs = 0

for file in tqdm(os.listdir(GRAPH_DIR), desc="Encoding features"):
    if not file.endswith(".graphml"):
        continue

    gpath = os.path.join(GRAPH_DIR, file)

    try:
        aid, sent_feats, event_feats, meta = encode_graph_features(gpath)
    except Exception as e:
        print(f"❌ Error processing {file}: {e}")
        continue

    base = os.path.join(OUTPUT_DIR, aid)

    # Save sentence features
    with open(base + "_sentence_features.json", "w") as f:
        json.dump(sent_feats, f, indent=2)
    if sent_feats:
        torch.save(
            torch.tensor(list(sent_feats.values()), dtype=torch.float32),
            base + "_sentence_features.pt"
        )

    # Save event features
    with open(base + "_event_features.json", "w") as f:
        json.dump(event_feats, f, indent=2)
    if event_feats:
        torch.save(
            torch.tensor(list(event_feats.values()), dtype=torch.float32),
            base + "_event_features.pt"
        )

    # Save metadata
    with open(base + "_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    total_sent_nodes += len(sent_feats)
    total_event_nodes += len(event_feats)
    total_biased_sents += sum(1 for m in meta.values()
                              if m.get("type") == "sentence" and m.get("has_bias"))
    total_graphs += 1

print(f"\n{'='*70}")
print(f"✅ Done! Encoded feature files saved → {OUTPUT_DIR}/")
print(f"{'='*70}")
print(f"📊 Summary:")
print(f"  Total graphs processed: {total_graphs}")
print(f"  Total sentence nodes: {total_sent_nodes}")
print(f"  Total event nodes: {total_event_nodes}")
print(f"  Sentences with bias: {total_biased_sents}/{total_sent_nodes} ({total_biased_sents/total_sent_nodes*100:.1f}%)")
print(f"\n💾 Output files per article:")
print(f"  - <id>_sentence_features.json / .pt")
print(f"  - <id>_event_features.json / .pt")
print(f"  - <id>_metadata.json")