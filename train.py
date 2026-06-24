from __future__ import annotations

import os
import time
import pickle
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datetime import datetime
from pathlib import Path

from data_loader import FashionDataLoader, DataConfig, FrozenMultimodalEmbedder, SiamesePairDataset, append_experiment_result
from models import SiameseNetwork, SiameseConfig, ContrastiveLoss


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_cached_embeddings(
    products_df: pd.DataFrame,
    loader: FashionDataLoader,
    embedder: FrozenMultimodalEmbedder,
    device: torch.device,
    cache_path: str = "raw_embeddings.pkl"
) -> dict[str, torch.Tensor]:
    """Helper to cache 2816-dimensional raw concatenated ResNet50 and DistilBERT features."""
    if os.path.exists(cache_path):
        print(f"Loading raw features from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("Pre-computing raw features using ResNet50 and DistilBERT (this may take a minute)...")
    embedding_index = loader.build_embedding_index(products_df, embedder, device)
    with open(cache_path, "wb") as f:
        pickle.dump(embedding_index, f)
    print(f"Saved raw features to cache: {cache_path}")
    return embedding_index


def evaluate_outfit(
    model: nn.Module,
    val_outfit_df: pd.DataFrame,
    embedding_index: dict[str, torch.Tensor],
    loader: FashionDataLoader,
    product_ids: list[str],
    full_compat: dict[str, set[str]],
    device: torch.device
) -> tuple[float, float]:
    """Calculate HR@5 and NDCG@5 for a single validation outfit."""
    positive_df = loader.build_positive_pairs(val_outfit_df)
    if len(positive_df) == 0:
        return 0.0, 0.0

    hits = []
    ndcgs = []

    model.eval()
    with torch.no_grad():
        for _, row in positive_df.iterrows():
            id_a = str(row["id_a"])
            id_b = str(row["id_b"])

            # Bidirectional evaluation:
            # Anchor is id_a, target is id_b; and anchor is id_b, target is id_a.
            for anchor, target in [(id_a, id_b), (id_b, id_a)]:
                if anchor not in embedding_index or target not in embedding_index:
                    continue

                # Build candidate pool: target item + all items not compatible with anchor
                non_compat_candidates = [
                    pid for pid in product_ids
                    if pid != anchor and pid != target and pid not in full_compat.get(anchor, set())
                ]
                candidates = [target] + non_compat_candidates

                # Prepare tensors
                emb_anchor = embedding_index[anchor].to(device).unsqueeze(0)  # (1, 2816)
                emb_candidates = torch.stack([embedding_index[pid] for pid in candidates]).to(device)  # (N, 2816)

                # Project to Siamese embedding space
                z_anchor = model.encoder(emb_anchor)  # (1, 256)
                z_candidates = model.encoder(emb_candidates)  # (N, 256)

                # Compute Euclidean distances
                distances = F.pairwise_distance(z_anchor, z_candidates, keepdim=False)  # (N,)

                # Rank candidates (smaller distance is better)
                sorted_indices = torch.argsort(distances).tolist()
                rank = sorted_indices.index(0) + 1  # Find target (index 0) position, 1-indexed

                # Calculate metrics @5
                if rank <= 5:
                    hits.append(1.0)
                    ndcgs.append(1.0 / np.log2(rank + 1))
                else:
                    hits.append(0.0)
                    ndcgs.append(0.0)

    if not hits:
        return 0.0, 0.0
    return float(np.mean(hits)), float(np.mean(ndcgs))


def train_one_fold(
    fold_idx: int,
    train_pairs_df: pd.DataFrame,
    embedding_index: dict[str, torch.Tensor],
    config: DataConfig,
    device: torch.device,
    epochs: int = 15,
    batch_size: int = 64,
    val_pairs_df: pd.DataFrame | None = None
) -> tuple[nn.Module, list[float], list[float]]:
    """Train Siamese Network and return train (and optionally val) losses."""
    dataset = SiamesePairDataset(train_pairs_df, embedding_index)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    val_dataloader = None
    if val_pairs_df is not None and len(val_pairs_df) > 0:
        val_dataset = SiamesePairDataset(val_pairs_df, embedding_index)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    siamese_cfg = SiameseConfig(
        input_dim=2816,
        hidden_dim=512,
        embedding_dim=config.embedding_dim,
        dropout_p=config.dropout_p,
        margin=1.0
    )
    model = SiameseNetwork(siamese_cfg).to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)  # weight_decay adds L2 Regularization

    epoch_losses = []
    epoch_val_losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for emb_a, emb_b, label in dataloader:
            emb_a = emb_a.to(device)
            emb_b = emb_b.to(device)
            label = label.to(device)

            z1, z2, dist = model(emb_a, emb_b)
            loss = criterion(dist, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(label)
        epoch_losses.append(total_loss / len(dataset))

        if val_dataloader is not None:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for emb_a, emb_b, label in val_dataloader:
                    emb_a = emb_a.to(device)
                    emb_b = emb_b.to(device)
                    label = label.to(device)

                    z1, z2, dist = model(emb_a, emb_b)
                    loss = criterion(dist, label)
                    total_val_loss += loss.item() * len(label)
            epoch_val_losses.append(total_val_loss / len(val_dataset))

    return model, epoch_losses, epoch_val_losses


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    config = DataConfig()
    loader = FashionDataLoader(config)
    products_df, outfits_df = loader.load_dataframes()
    product_ids = products_df["id"].astype(str).tolist()
    full_compat = loader._compatibility_map(outfits_df)

    # 2. Extract and Cache Raw Embeddings
    embedder = FrozenMultimodalEmbedder(embedding_dim=config.embedding_dim, dropout_p=config.dropout_p)
    embedding_index = get_cached_embeddings(products_df, loader, embedder, device)

    # 3. Leave-One-Out Cross-Validation (LOOCV)
    outfits_list = outfits_df["outfit_id"].unique().tolist()
    num_folds = len(outfits_list)
    print(f"Starting Leave-One-Out Cross-Validation on {num_folds} outfits...")

    loocv_hrs = []
    loocv_ndcgs = []
    loocv_train_losses = []
    fold_train_history = []
    fold_val_history = []

    start_time = time.time()
    for i, val_outfit_id in enumerate(outfits_list):
        train_outfits_df = outfits_df[outfits_df["outfit_id"] != val_outfit_id]
        val_outfit_df = outfits_df[outfits_df["outfit_id"] == val_outfit_id]

        # Build training pairs (excluding validation outfit items' specific combinations)
        train_pairs_df = loader.build_pair_dataframe(products_df, train_outfits_df)
        val_pairs_df = loader.build_pair_dataframe(products_df, val_outfit_df)

        # Train model
        model, train_losses, val_losses = train_one_fold(
            fold_idx=i,
            train_pairs_df=train_pairs_df,
            embedding_index=embedding_index,
            config=config,
            device=device,
            epochs=15,
            batch_size=64,
            val_pairs_df=val_pairs_df
        )

        # Evaluate on left-out outfit
        hr, ndcg = evaluate_outfit(model, val_outfit_df, embedding_index, loader, product_ids, full_compat, device)
        loocv_hrs.append(hr)
        loocv_ndcgs.append(ndcg)
        loocv_train_losses.append(train_losses[-1])
        fold_train_history.append(train_losses)
        fold_val_history.append(val_losses)

        print(f"Fold {i+1:02d}/{num_folds:02d} | Val Outfit: {val_outfit_id} | Final Train Loss: {train_losses[-1]:.4f} | HR@5: {hr:.4f} | NDCG@5: {ndcg:.4f}")

    mean_hr = float(np.mean(loocv_hrs))
    mean_ndcg = float(np.mean(loocv_ndcgs))
    mean_loss = float(np.mean(loocv_train_losses))
    elapsed = time.time() - start_time

    # Average loss history over all folds
    avg_train_losses = np.mean(fold_train_history, axis=0).tolist()
    avg_val_losses = np.mean(fold_val_history, axis=0).tolist()

    print("\n--- LOOCV Evaluation Summary ---")
    print(f"Mean Training Loss: {mean_loss:.4f}")
    print(f"LOOCV Mean Hit Ratio (HR@5): {mean_hr:.4f}")
    print(f"LOOCV Mean NDCG@5: {mean_ndcg:.4f}")
    print(f"Total CV time: {elapsed:.2f} seconds")

    # 4. Train Final Model on All Data
    print("\nTraining final model on full dataset...")
    full_pairs_df = loader.build_pair_dataframe(products_df, outfits_df)
    final_model, final_losses, _ = train_one_fold(
        fold_idx=-1,
        train_pairs_df=full_pairs_df,
        embedding_index=embedding_index,
        config=config,
        device=device,
        epochs=15,
        batch_size=64
    )

    # Save model checkpoint
    model_path = "best_model.pth"
    torch.save(final_model.state_dict(), model_path)
    print(f"Saved final trained model weights to {model_path}")

    # Save loss history for visualization
    history_path = "train_history.pkl"
    with open(history_path, "wb") as f:
        pickle.dump((avg_train_losses, avg_val_losses), f)
    print(f"Saved final model training history to {history_path}")

    # 5. Automatically Log Experiment Result
    run_date = datetime.now().strftime("%Y-%m-%d")
    append_experiment_result(
        log_path="experiment_log.md",
        run_id="baseline_001",
        run_date=run_date,
        method_applied="Initial Baseline (ResNet50 + DistilBERT + Siamese projection)",
        training_loss=mean_loss,
        validation_ndcg5=mean_ndcg,
        validation_hr5=mean_hr,
        conclusion="Successful baseline with LOOCV. Frozen ResNet50/DistilBERT with trainable projection head. Highly efficient training using feature caching."
    )
    print("Experiment metrics successfully appended to experiment_log.md!")


if __name__ == "__main__":
    main()
