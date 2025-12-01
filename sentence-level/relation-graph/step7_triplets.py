def train_step7(model, rgat_loader, triplet_loader, optimizer, criterion, lambdas):
    model.train()
    total_loss = 0

    lambda_trip, lambda_frame, lambda_omit = lambdas

    # Iterator for triplets
    trip_iter = iter(triplet_loader)

    for batch in rgat_loader:

        optimizer.zero_grad()

        # --- Sentence-level classification loss ---
        logits = model(
            batch["node_features"].to(DEVICE),
            batch["edge_index"].to(DEVICE),
            batch["edge_types"].to(DEVICE),
            batch["sentence_mask"].to(DEVICE)
        )
        labels = batch["labels"][batch["sentence_mask"]].to(DEVICE)
        loss_sentence = criterion(logits, labels)

        # --- Triplet losses ---
        try:
            trip = next(trip_iter)
        except StopIteration:
            trip_iter = iter(triplet_loader)
            trip = next(trip_iter)

        # Compute embeddings for 3 outlets
        H = model(trip["HPO"].to(DEVICE), None, None, None, return_article_emb=True)[1]
        N = model(trip["NYT"].to(DEVICE), None, None, None, return_article_emb=True)[1]
        F = model(trip["FOX"].to(DEVICE), None, None, None, return_article_emb=True)[1]

        loss_trip = triplet_alignment_loss(H, N, F)

        # Framing divergence (using article embeddings too)
        loss_frame = framing_divergence_loss(H.unsqueeze(0), N.unsqueeze(0), F.unsqueeze(0))

        # Omission loss (use event counts)
        event_counts = {
            "HPO": trip["HPO"].size(0),
            "NYT": trip["NYT"].size(0),
            "FOX": trip["FOX"].size(0),
        }
        loss_omit = omission_loss(event_counts)

        # Combined loss
        loss = (
            loss_sentence +
            lambda_trip * loss_trip +
            lambda_frame * loss_frame +
            lambda_omit * loss_omit
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(rgat_loader)