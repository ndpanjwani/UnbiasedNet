import os
import json
import torch
import torch.nn.functional as F
import networkx as nx

from typing import List, Dict

# 🔹 Adjust this import to match your Step 6 file name
# e.g., from step6_rgat import RGAT, RELATION_TYPES, NUM_RELATIONS, HIDDEN_DIM, NUM_HEADS, NUM_LAYERS, DROPOUT, DEVICE
from step6_rgat import RGAT, RELATION_TYPES, NUM_RELATIONS, HIDDEN_DIM, NUM_HEADS, NUM_LAYERS, DROPOUT, DEVICE

GRAPH_DIR = "erg_graphs"
FEATURES_DIR = "erg_features"
MODEL_DIR = "rgat_model"


def load_article_graph_and_features(article_id: str):
    """Load graph, node features, edge index/types, labels, and sentence texts for a single article."""
    graph_path = os.path.join(GRAPH_DIR, f"{article_id}.graphml")
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph file not found: {graph_path}")

    G = nx.read_graphml(graph_path)

    sent_feat_json = os.path.join(FEATURES_DIR, f"{article_id}_sentence_features.json")
    event_feat_json = os.path.join(FEATURES_DIR, f"{article_id}_event_features.json")
    metadata_path = os.path.join(FEATURES_DIR, f"{article_id}_metadata.json")

    with open(sent_feat_json) as f:
        sent_features = json.load(f)
    with open(event_feat_json) as f:
        event_features = json.load(f)
    with open(metadata_path) as f:
        metadata = json.load(f)

    # Sentence and event nodes, in graph order
    sent_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "sentence"]
    event_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "event"]

    if len(sent_nodes) == 0:
        raise ValueError(f"No sentence nodes found for article {article_id}")

    node_to_idx = {}
    idx_to_node = {}
    features_list = []
    sentence_texts = []

    idx = 0
    # Sentences first (consistent with training)
    for node in sent_nodes:
        node_to_idx[node] = idx
        idx_to_node[idx] = node
        features_list.append(torch.tensor(sent_features[node], dtype=torch.float32))
        sentence_texts.append(G.nodes[node].get("text", ""))
        idx += 1

    # Events next
    for node in event_nodes:
        node_to_idx[node] = idx
        idx_to_node[idx] = node
        features_list.append(torch.tensor(event_features[node], dtype=torch.float32))
        idx += 1

    node_features = torch.stack(features_list, dim=0)

    # Build edge index / types (only 4 relation types)
    edge_index = []
    edge_types = []

    for u, v, data in G.edges(data=True):
        if u in node_to_idx and v in node_to_idx:
            rel = data.get("relation", "")
            if rel in RELATION_TYPES:
                edge_index.append([node_to_idx[u], node_to_idx[v]])
                edge_types.append(RELATION_TYPES[rel])

    if len(edge_index) == 0:
        raise ValueError(f"No usable edges for article {article_id}")

    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    edge_types = torch.tensor(edge_types, dtype=torch.long)

    # Sentence mask (first |sent_nodes| are sentences)
    num_nodes = len(node_to_idx)
    sentence_mask = torch.zeros(num_nodes, dtype=torch.bool)
    for i in range(len(sent_nodes)):
        sentence_mask[i] = True

    # Labels (if needed; can be dummy at inference)
    labels = torch.zeros(num_nodes, dtype=torch.long)
    for i, node in idx_to_node.items():
        meta = metadata.get(node, {})
        if meta.get("type") == "sentence":
            labels[i] = 1 if meta.get("has_bias", False) else 0

    return {
        "graph": G,
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_types": edge_types,
        "sentence_mask": sentence_mask,
        "labels": labels,
        "sentence_texts": sentence_texts,
    }


def load_trained_model(in_dim: int) -> torch.nn.Module:
    """Load RGAT model with the trained weights."""
    model = RGAT(
        in_dim=in_dim,
        hidden_dim=HIDDEN_DIM,
        num_relations=NUM_RELATIONS,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    ckpt_path = os.path.join(MODEL_DIR, "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_article_bias(article_id: str) -> List[Dict]:
    """
    Run full inference for a single article_id:
    - loads graph + features
    - runs RGAT
    - returns list of {sentence_idx, sentence, prob_biased, prob_unbiased, predicted_label}
    """
    batch = load_article_graph_and_features(article_id)

    node_features = batch["node_features"].to(DEVICE)
    edge_index = batch["edge_index"].to(DEVICE)
    edge_types = batch["edge_types"].to(DEVICE)
    sentence_mask = batch["sentence_mask"].to(DEVICE)
    sentence_texts = batch["sentence_texts"]

    in_dim = node_features.size(1)
    model = load_trained_model(in_dim)

    with torch.no_grad():
        logits = model(node_features, edge_index, edge_types, sentence_mask)
        probs = F.softmax(logits, dim=-1)  # [num_sentences, 2]

    probs = probs.cpu().numpy()
    outputs = []

    for idx, (sent, p_vec) in enumerate(zip(sentence_texts, probs)):
        prob_unbiased = float(p_vec[0])
        prob_biased = float(p_vec[1])
        pred_label = int(p_vec.argmax())

        outputs.append({
            "sentence_idx": idx,
            "sentence": sent,
            "prob_unbiased": prob_unbiased,
            "prob_biased": prob_biased,
            "predicted_label": pred_label,  # 0 = unbiased, 1 = biased
        })

    return outputs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="UnbiasedNet Step 8 — Single Article Inference")
    parser.add_argument("--article_id", type=str, required=True, help="Article ID (must match graph/feature files)")
    args = parser.parse_args()

    preds = predict_article_bias(args.article_id)

    print(f"\nBias predictions for article_id={args.article_id}:\n")
    for p in preds:
        label_str = "Biased" if p["predicted_label"] == 1 else "Unbiased"
        print(f"[{p['sentence_idx']:02d}] ({label_str}, p_bias={p['prob_biased']:.3f}) {p['sentence']}")