import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import networkx as nx
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support

# ===============================================================
# CONFIG
# ===============================================================
GRAPH_DIR = "erg_graphs"
FEATURES_DIR = "erg_features"
OUTPUT_DIR = "rgat_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Hyperparameters
HIDDEN_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
PATIENCE = 10

# Only 4 relation types
RELATION_TYPES = {
    "causal": 0,
    "temporal": 1,
    "coreference": 2,
    "continuation": 3,
}
NUM_RELATIONS = len(RELATION_TYPES)


# ===============================================================
# DATASET (patched ordering + SRL-safe metadata)
# ===============================================================
class ERGDataset(Dataset):

    def __init__(self, article_ids):
        self.data = []
        print("Loading graphs and features...")

        for aid in tqdm(article_ids):

            try:
                # Load graph
                graph_path = os.path.join(GRAPH_DIR, f"{aid}.graphml")
                if not os.path.exists(graph_path):
                    continue
                G = nx.read_graphml(graph_path)

                # Load features
                sent_feat_path = os.path.join(FEATURES_DIR, f"{aid}_sentence_features.pt")
                event_feat_path = os.path.join(FEATURES_DIR, f"{aid}_event_features.pt")
                metadata_path = os.path.join(FEATURES_DIR, f"{aid}_metadata.json")
                if not (os.path.exists(sent_feat_path) and os.path.exists(event_feat_path)):
                    continue

                sent_feats_dict = json.load(open(os.path.join(FEATURES_DIR, f"{aid}_sentence_features.json")))
                event_feats_dict = json.load(open(os.path.join(FEATURES_DIR, f"{aid}_event_features.json")))

                with open(metadata_path) as f:
                    metadata = json.load(f)

                # Extract node lists from graph (sentence first, then event)
                sent_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "sentence"]
                event_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "event"]

                if len(sent_nodes) == 0:
                    continue

                # KEEP ORDER: EXACTLY AS IN GRAPHML
                node_to_idx = {}
                idx_to_node = {}
                idx = 0

                ordered_features = []

                # Add sentences in graph order
                for node in sent_nodes:
                    node_to_idx[node] = idx
                    idx_to_node[idx] = node
                    ordered_features.append(torch.tensor(sent_feats_dict[node], dtype=torch.float32))
                    idx += 1

                # Add events in graph order
                for node in event_nodes:
                    node_to_idx[node] = idx
                    idx_to_node[idx] = node
                    ordered_features.append(torch.tensor(event_feats_dict[node], dtype=torch.float32))
                    idx += 1

                node_features = torch.stack(ordered_features, dim=0)

                # Build edges (filtering to 4 relation types)
                edge_index = []
                edge_types = []

                for u, v, data in G.edges(data=True):
                    if u in node_to_idx and v in node_to_idx:
                        rel = data.get("relation", "")
                        if rel in RELATION_TYPES:
                            edge_index.append([node_to_idx[u], node_to_idx[v]])
                            edge_types.append(RELATION_TYPES[rel])

                if len(edge_index) == 0:
                    continue

                edge_index = torch.tensor(edge_index, dtype=torch.long).t()
                edge_types = torch.tensor(edge_types, dtype=torch.long)

                # Labels for sentence nodes
                labels = torch.zeros(len(node_to_idx), dtype=torch.long)
                sentence_mask = torch.zeros(len(node_to_idx), dtype=torch.bool)

                for i, node in idx_to_node.items():
                    meta = metadata.get(node, {})
                    if meta.get("type") == "sentence":
                        sentence_mask[i] = True
                        labels[i] = 1 if meta.get("has_bias", False) else 0

                if sentence_mask.sum() == 0:
                    continue

                self.data.append({
                    "article_id": aid,
                    "node_features": node_features,
                    "edge_index": edge_index,
                    "edge_types": edge_types,
                    "labels": labels,
                    "sentence_mask": sentence_mask,
                })

            except Exception as e:
                continue

        print(f"Loaded {len(self.data)} graphs")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ===============================================================
# RELATIONAL GAT LAYER (fixed message passing)
# ===============================================================
class RelationalGATLayer(nn.Module):

    def __init__(self, in_dim, out_dim, num_relations, num_heads=4, dropout=0.3):
        super().__init__()
        assert out_dim % num_heads == 0

        self.num_heads = num_heads
        self.num_relations = num_relations
        self.out_dim = out_dim
        self.head_dim = out_dim // num_heads

        self.W_rel = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False)
            for _ in range(num_relations)
        ])

        self.attn = nn.Parameter(torch.Tensor(num_relations, num_heads, 2 * self.head_dim))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

        nn.init.xavier_uniform_(self.attn)

    def forward(self, x, edge_index, edge_types):
        num_nodes = x.size(0)
        out = torch.zeros_like(x)

        for rel in range(self.num_relations):
            mask = (edge_types == rel)
            if mask.sum() == 0:
                continue

            edges = edge_index[:, mask]
            src, dst = edges[0], edges[1]

            x_rel = self.W_rel[rel](x).view(num_nodes, self.num_heads, self.head_dim)

            src_feat = x_rel[src]
            dst_feat = x_rel[dst]

            cat = torch.cat([src_feat, dst_feat], dim=-1)
            alpha = (cat * self.attn[rel]).sum(-1)
            alpha = self.leaky_relu(alpha)

            # softmax over edges grouped by dst node
            alpha_softmax = torch.zeros_like(alpha)
            for n in range(num_nodes):
                idx = (dst == n)
                if idx.sum() > 0:
                    alpha_softmax[idx] = F.softmax(alpha[idx], dim=0)

            alpha_softmax = self.dropout(alpha_softmax)

            # weighted message passing
            msg = src_feat * alpha_softmax.unsqueeze(-1)
            msg = msg.view(-1, self.out_dim)

            out.index_add_(0, dst, msg)

        return out


# ===============================================================
# R-GAT MODEL
# ===============================================================
class RGAT(nn.Module):

    def __init__(self, in_dim, hidden_dim, num_relations, num_heads=4, num_layers=2, dropout=0.3):
        super().__init__()

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.gat_layers = nn.ModuleList([
            RelationalGATLayer(hidden_dim, hidden_dim, num_relations, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, node_features, edge_index, edge_types, sentence_mask):

        x = F.relu(self.input_proj(node_features))

        for i, layer in enumerate(self.gat_layers):
            x_new = layer(x, edge_index, edge_types)
            x_new = self.layer_norms[i](x_new)
            x_new = F.relu(x_new)
            x = x + x_new  # residual

        sentence_embeddings = x[sentence_mask]
        logits = self.classifier(sentence_embeddings)

        return logits


# ===============================================================
# TRAINING
# ===============================================================
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    preds, labels_col = [], []

    for batch in loader:

        nf = batch["node_features"].to(device)
        ei = batch["edge_index"].to(device)
        et = batch["edge_types"].to(device)
        labels = batch["labels"].to(device)
        smask = batch["sentence_mask"].to(device)

        optimizer.zero_grad()

        logits = model(nf, ei, et, smask)
        sent_labels = labels[smask]

        loss = criterion(logits, sent_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        preds.extend(torch.argmax(logits, -1).cpu().numpy())
        labels_col.extend(sent_labels.cpu().numpy())

    f1 = f1_score(labels_col, preds, average="macro")
    return total_loss / len(loader), f1


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    preds, labels_col = [], []

    with torch.no_grad():
        for batch in loader:
            nf = batch["node_features"].to(device)
            ei = batch["edge_index"].to(device)
            et = batch["edge_types"].to(device)
            labels = batch["labels"].to(device)
            smask = batch["sentence_mask"].to(device)

            logits = model(nf, ei, et, smask)
            sent_labels = labels[smask]

            loss = criterion(logits, sent_labels)
            total_loss += loss.item()

            preds.extend(torch.argmax(logits, -1).cpu().numpy())
            labels_col.extend(sent_labels.cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_col, preds, average="macro"
    )
    return total_loss / len(loader), precision, recall, f1, preds, labels_col


# ===============================================================
# MAIN
# ===============================================================
def main():

    # article ids
    article_ids = [f.replace(".graphml", "") for f in os.listdir(GRAPH_DIR) if f.endswith(".graphml")]
    print(f"Loaded {len(article_ids)} articles")

    # split
    train_ids, temp_ids = train_test_split(article_ids, test_size=0.3, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

    train_dataset = ERGDataset(train_ids)
    val_dataset = ERGDataset(val_ids)
    test_dataset = ERGDataset(test_ids)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)
    test_loader = DataLoader(test_dataset, batch_size=1)

    sample = train_dataset[0]
    in_dim = sample["node_features"].size(1)

    # model
    model = RGAT(
        in_dim=in_dim,
        hidden_dim=HIDDEN_DIM,
        num_relations=NUM_RELATIONS,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0
    patience = 0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_precision, val_recall, val_f1, _, _ = evaluate(model, val_loader, criterion, DEVICE)

        print(f"Train Loss: {train_loss:.4f}, F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))
            patience = 0
            print("Saved new best model")
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    # testing
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pt")))
    test_loss, p, r, f1, preds, labels = evaluate(model, test_loader, criterion, DEVICE)

    print(f"\nTest F1: {f1:.4f}")
    print(classification_report(labels, preds, target_names=["Unbiased", "Biased"]))


if __name__ == "__main__":
    main()