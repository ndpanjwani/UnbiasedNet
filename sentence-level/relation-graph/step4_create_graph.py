import json
import networkx as nx
from tqdm import tqdm
import os

DISCOURSE_WITH_EVENTS = "discourse_with_events.json"
ERG_EDGES = "erg_edges_transformer.json"
OUTPUT_DIR = "erg_graphs"

# -------------------------------------------------
# SRL-compatible event helpers
# -------------------------------------------------
def get_actor(ev):
    return ev.get("actor") or ev.get("ARG0") or ev.get("args", {}).get("ARG0")

def get_object(ev):
    return ev.get("object") or ev.get("ARG1") or ev.get("args", {}).get("ARG1")

def get_trigger(ev):
    return ev.get("trigger") or ev.get("lemma") or ev.get("predicate") or "UNK"


# -------------------------------------------------
# Build heterogeneous graph
# -------------------------------------------------
def build_article_graph(article_id, discourse_data, erg_edges):
    G = nx.DiGraph()

    # Extract sentences for this article, sorted by sentence_idx
    sents = [s for s in discourse_data if s["article_id"] == article_id]
    sents = sorted(sents, key=lambda s: s["sentence_idx"])

    # ERG edges
    edges = [e for e in erg_edges if e["article_id"] == article_id]

    sent_id_map = {}

    # ----------------------------
    # STEP 1: Add sentence nodes
    # ----------------------------
    for sent_idx, sent in enumerate(sents):
        sent_id = f"{article_id}_sent_{sent_idx}"
        sent_id_map[sent_idx] = sent_id

        G.add_node(
            sent_id,
            node_type="sentence",
            text=sent["text"],
            discourse_role=sent.get("discourse_role", "NONE"),
            has_bias=bool(sent.get("has_bias", False)),
            prev_discourse_role=sent.get("prev_discourse_role", "NONE"),
            next_discourse_role=sent.get("next_discourse_role", "NONE"),
        )

    # ----------------------------
    # STEP 2: Add event nodes
    # ----------------------------
    for sent_idx, sent in enumerate(sents):
        sent_id = sent_id_map[sent_idx]
        for ev_idx, ev in enumerate(sent.get("events", [])):

            trigger = get_trigger(ev)
            actor = get_actor(ev) or ""
            object_ = get_object(ev) or ""

            event_id = f"{article_id}_{sent_idx}_{trigger}_{ev_idx}"

            G.add_node(
                event_id,
                node_type="event",
                trigger=trigger,
                actor=actor,
                object=object_,
                sentiment=float(ev.get("sentiment", 0.0)),
                has_bias=bool(ev.get("has_bias", False)),
            )

            G.add_edge(event_id, sent_id, relation="belongs_to", edge_type="event_to_sentence", weight=1.0)

    # ----------------------------
    # STEP 3: Sentence-sentence edges
    # ----------------------------
    for i in range(len(sents) - 1):
        s1 = sent_id_map[i]
        s2 = sent_id_map[i + 1]

        G.add_edge(s1, s2, relation="sequential", edge_type="sentence_to_sentence", weight=1.0)

        role1 = sents[i].get("discourse_role", "NONE")
        role2 = sents[i + 1].get("discourse_role", "NONE")

        if role1 in ["Cause_General", "Cause_Specific"] and role2 == "Main":
            G.add_edge(s1, s2, relation="causal_discourse", edge_type="sentence_to_sentence", weight=1.5)
        elif role1 == "Main" and role2 == "Main_Consequence":
            G.add_edge(s1, s2, relation="consequence_discourse", edge_type="sentence_to_sentence", weight=1.5)

    # ----------------------------
    # STEP 4: Event-event edges from Step 3
    # ----------------------------
    missing_src = 0
    missing_tgt = 0

    for e in edges:
        src = e["src_event_id"]
        tgt = e["tgt_event_id"]
        rel = e["relation"]

        # Fallback matching (SRL-friendly)
        if src not in G:
            src_trigger = e.get("src_trigger", "").lower()
            matches = [n for n in G.nodes() if G.nodes[n].get("node_type") == "event" and G.nodes[n].get("trigger","").lower() == src_trigger]
            if matches:
                src = matches[0]
            else:
                missing_src += 1
                continue

        if tgt not in G:
            tgt_trigger = e.get("tgt_trigger", "").lower()
            matches = [n for n in G.nodes() if G.nodes[n].get("node_type") == "event" and G.nodes[n].get("trigger","").lower() == tgt_trigger]
            if matches:
                tgt = matches[0]
            else:
                missing_tgt += 1
                continue

        G.add_edge(
            src,
            tgt,
            relation=rel,
            edge_type="event_to_event",
            weight=float(e.get("confidence", 1.0)),
            sentence_distance=int(e.get("sentence_distance", 0)),
        )

    if missing_src or missing_tgt:
        print(f"⚠️ {article_id}: missing {missing_src} src, {missing_tgt} tgt")

    return G