## **Step 1. Data Setup — Sentence-Level Structure**
Each record in the dataset represents one sentence with fields such as:
- `text`: the sentence text  
- `discourse_role`: the rhetorical function of the sentence  
- `annotations`: lexical or polarity cues  
- `has_bias`: sentence-level bias label  
- `article_id`: groups sentences by article  
- `triplet_uuid`: groups related articles covering the same story  

This flattened format provides the foundation for discourse-aware and event-based modeling.

---

## **Step 2. Event Extraction — Create Event Nodes**
For each sentence, extract the main events and their participants using a semantic role labeling (SRL) or event trigger model.  
Each event contains attributes such as `trigger`, `actor`, `object`, and `sentiment`.  
These extracted events become the event nodes of the graph.

---

## **Step 3. Event–Event Relation Extraction — Create ERG Edges**
Within each article, identify relationships between events.  
Relations include **causal**, **temporal**, **coreference**, and **subevent** links.  
Each relation is stored as a directed edge connecting two events with a relation type and confidence score.  
These edges define the structure of the Event Relation Graph (ERG).

---

## **Step 4. Graph Construction — Combine Discourse + Events**
Build a heterogeneous graph that integrates discourse and event information.  
Nodes include **sentence nodes** (containing text and discourse data) and **event nodes** (from Step 2).  
Edges connect sentences to their events, events to related events, and adjacent sentences.  
This structure links rhetorical framing with factual context.

**TEST: VISUALIZE A GRAPH**
---

## **Step 5. Feature Encoding**
Assign interpretable features to each node:
- **Sentence nodes:** sentence embeddings, discourse role embeddings, and aggregate event statistics (e.g., event count, mean sentiment).  
- **Event nodes:** trigger embeddings, actor names, sentiment polarity, and relation degree.  
These features provide both linguistic and contextual signals for bias detection.

**TEST: Train a simple classifier (ex logistic regression) on the node features to predict has_bias.**

---

## **Step 6. Model Architecture**
Use a language model encoder (e.g., Longformer or DeBERTa) to embed sentence text.  
Combine these embeddings with discourse and event features.  
Apply a Relational Graph Attention Network (R-GAT) to propagate contextual information across the graph.  
A classification head predicts sentence-level bias using a binary cross-entropy loss.

---

## **Step 7. Triplet-Based Training**
Train the model using article triplets covering the same story.  
Align canonical events across the three articles and apply auxiliary training objectives:
- **Contrastive loss:** encourages same-event sentences across outlets to be close in content space.  
- **Framing divergence loss:** separates sentences with differing sentiment or discourse framing.  
- **Omission loss:** predicts how many outlets mention a given event.  
These losses teach the model to differentiate factual consistency from outlet-specific framing.

---

## **Step 8. Inference — Single Article Input**
During inference, input a single article.  
Segment it into sentences, assign discourse roles, extract events, and build the article’s ERG.  
Run the encoder and graph network to predict a bias probability for each sentence.  
The output is a list of sentences with associated bias scores.

---

## **Step 9. Evaluation**
Evaluate using precision, recall, and F1 metrics on biased sentences.  

---