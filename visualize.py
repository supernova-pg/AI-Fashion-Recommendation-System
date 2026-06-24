from __future__ import annotations

import os
import pickle
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from data_loader import FashionDataLoader, DataConfig
from models import SiameseNetwork, SiameseConfig


# High-level category mapping for cleaner t-SNE visualization
HIGH_LEVEL_CATEGORIES = {
    # Topwear
    'formal-shirts': 'Topwear', 'sweatshirts': 'Topwear', 'linen-shirts': 'Topwear', 
    'casual-shirts': 'Topwear', 'party-shirts': 'Topwear', 'tshirts': 'Topwear', 
    'polo-tshirts': 'Topwear', 'tops': 'Topwear', 'sweaters': 'Topwear', 
    'Formal Shirts': 'Topwear', 'Linen Shirts': 'Topwear', 'Casual Shirts': 'Topwear',
    'Party Shirts': 'Topwear', 'Tshirts': 'Topwear', 'Polo Tshirts': 'Topwear', 
    'Tops': 'Topwear', 'Sweaters': 'Topwear', 'Sweatshirts': 'Topwear',

    # Bottomwear
    'track-pants': 'Bottomwear', 'trousers': 'Bottomwear', 'jeans': 'Bottomwear', 
    'chinos': 'Bottomwear', 'shorts': 'Bottomwear', 'skirts': 'Bottomwear', 
    'leggings': 'Bottomwear', 'Track Pants': 'Bottomwear', 'Trousers': 'Bottomwear',
    'Jeans': 'Bottomwear', 'Chinos': 'Bottomwear', 'Shorts': 'Bottomwear', 
    'Skirts': 'Bottomwear', 'Leggings': 'Bottomwear',

    # Footwear
    'running-shoes': 'Footwear', 'sneakers': 'Footwear', 'ethnic-footwear': 'Footwear', 
    'heels': 'Footwear', 'boots': 'Footwear', 'flats': 'Footwear', 
    'formal-shoes': 'Footwear', 'loafers': 'Footwear', 'sandals': 'Footwear',
    'Running Shoes': 'Footwear', 'Sneakers': 'Footwear', 'Ethnic Footwear': 'Footwear', 
    'Heels': 'Footwear', 'Boots': 'Footwear', 'Flats': 'Footwear', 
    'Formal Shoes': 'Footwear', 'Loafers': 'Footwear', 'Sandals': 'Footwear',

    # Accessories
    'necklaces': 'Accessories', 'clutches': 'Accessories', 'handbags': 'Accessories', 
    'earrings': 'Accessories', 'sunglasses': 'Accessories', 'watches': 'Accessories', 
    'caps': 'Accessories', 'Necklaces': 'Accessories', 'Clutches': 'Accessories', 
    'Handbags': 'Accessories', 'Earrings': 'Accessories', 'Sunglasses': 'Accessories', 
    'Watches': 'Accessories', 'Caps': 'Accessories',

    # One-Piece / Sets
    'suits': 'One-Piece/Sets', 'sherwanis': 'One-Piece/Sets', 'party-dresses': 'One-Piece/Sets', 
    'wedding-sarees': 'One-Piece/Sets', 'sharara-sets': 'One-Piece/Sets', 'casual-dresses': 'One-Piece/Sets', 
    'kurta-sets': 'One-Piece/Sets', 'co-ord-sets': 'One-Piece/Sets', 'salwar-suits': 'One-Piece/Sets', 
    'maxi-dresses': 'One-Piece/Sets', 'Suits': 'One-Piece/Sets', 'Sherwanis': 'One-Piece/Sets', 
    'Party Dresses': 'One-Piece/Sets', 'Wedding Sarees': 'One-Piece/Sets', 'Sharara Sets': 'One-Piece/Sets', 
    'Casual Dresses': 'One-Piece/Sets', 'Kurta Sets': 'One-Piece/Sets', 'Co Ord Sets': 'One-Piece/Sets', 
    'Salwar Suits': 'One-Piece/Sets', 'Maxi Dresses': 'One-Piece/Sets',

    # Layering
    'nehru-jackets': 'Layering', 'denim-jackets': 'Layering', 'long-coats': 'Layering', 
    'blazers': 'Layering', 'Nehru Jackets': 'Layering', 'Denim Jackets': 'Layering', 
    'Long Coats': 'Layering', 'Blazers': 'Layering'
}


def plot_loss_curves(history_path: str = "train_history.pkl", save_path: str = "loss_curves.png"):
    """Plot training vs validation loss curves over epochs."""
    if not os.path.exists(history_path):
        print(f"Error: History file {history_path} not found. Please run training first.")
        return

    with open(history_path, "rb") as f:
        history = pickle.load(f)

    # In our train.py, history is a tuple of (mean_train_losses_across_folds, mean_val_losses_across_folds)
    # or just final_losses. Let's make sure it handles both.
    if isinstance(history, tuple) and len(history) == 2:
        train_losses, val_losses = history
    else:
        # Fallback if only train_losses are stored
        train_losses = history
        val_losses = None

    epochs = range(1, len(train_losses) + 1)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    plt.plot(epochs, train_losses, label="Training Loss", color="#1f77b4", linewidth=2.5, marker="o")
    if val_losses is not None:
        plt.plot(epochs, val_losses, label="Validation Loss", color="#ff7f0e", linewidth=2.5, marker="s")
    
    plt.title("A) Siamese Network Loss curves over Epochs", fontsize=16, pad=15)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Contrastive Loss", fontsize=12)
    plt.xticks(epochs)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Loss curves saved successfully to {save_path}")


def plot_embeddings_tsne(
    model_path: str = "best_model.pth",
    cache_path: str = "raw_embeddings.pkl",
    save_path: str = "tsne_embeddings.png"
):
    """Generate a t-SNE scatter plot of product embeddings color-coded by high-level category."""
    if not os.path.exists(model_path) or not os.path.exists(cache_path):
        print("Error: Trained model weights or raw features cache not found. Run training first.")
        return

    # Load data
    config = DataConfig()
    loader = FashionDataLoader(config)
    products_df, _ = loader.load_dataframes()

    # Load raw features and model
    with open(cache_path, "rb") as f:
        embedding_index = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    siamese_cfg = SiameseConfig(input_dim=2816, hidden_dim=512, embedding_dim=config.embedding_dim)
    model = SiameseNetwork(siamese_cfg).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Project raw features into the learned 256-dim Siamese space
    product_ids = []
    category_labels = []
    embeddings_list = []

    with torch.no_grad():
        for _, row in products_df.iterrows():
            pid = str(row["id"])
            if pid not in embedding_index:
                continue

            raw_feat = embedding_index[pid].to(device).unsqueeze(0)
            proj_emb = model.encoder(raw_feat).squeeze(0).cpu().numpy()

            product_ids.append(pid)
            embeddings_list.append(proj_emb)
            
            # Map category
            cat = row.get("category", "")
            high_cat = HIGH_LEVEL_CATEGORIES.get(cat, HIGH_LEVEL_CATEGORIES.get(row.get("category_label", ""), "Other"))
            category_labels.append(high_cat)

    embeddings = np.array(embeddings_list) # (N, 256)

    # Perform t-SNE dimension reduction to 2D
    # Since N is small (68), perplexity should be small (e.g. 15 or 20)
    perplexity = min(15, len(embeddings) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create Scatter Plot
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="white")
    
    unique_cats = sorted(list(set(category_labels)))
    palette = sns.color_palette("muted", len(unique_cats))
    
    df_plot = pd.DataFrame({
        "t-SNE Dim 1": embeddings_2d[:, 0],
        "t-SNE Dim 2": embeddings_2d[:, 1],
        "Category": category_labels
    })

    sns.scatterplot(
        data=df_plot,
        x="t-SNE Dim 1",
        y="t-SNE Dim 2",
        hue="Category",
        palette=palette,
        s=120,
        alpha=0.85,
        edgecolor="w",
        linewidth=1.2
    )

    plt.title("B) t-SNE 2D Projection of Fashion Item Embeddings", fontsize=16, pad=15)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.legend(title="Product Category", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11, title_fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"t-SNE projection scatter plot saved successfully to {save_path}")


def main():
    print("Generating training curves and embedding visualizations...")
    plot_loss_curves()
    plot_embeddings_tsne()


if __name__ == "__main__":
    main()
