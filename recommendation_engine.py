from __future__ import annotations

import os
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

from data_loader import FashionDataLoader, DataConfig
from models import SiameseNetwork, SiameseConfig
from visualize import HIGH_LEVEL_CATEGORIES


class OutfitRecommender:
    def __init__(
        self,
        model_path: str = "best_model.pth",
        cache_path: str = "raw_embeddings.pkl",
        config_path: str = "products.csv"
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load configuration and loader
        self.config = DataConfig()
        self.loader = FashionDataLoader(self.config)
        self.products_df, self.outfits_df = self.loader.load_dataframes()
        self.products_df["id"] = self.products_df["id"].astype(str)
        self.products_df["gender"] = self.products_df["gender"].str.lower()
        self.products_df["occasion"] = self.products_df["occasion"].str.lower()
        self.full_compat = self.loader._compatibility_map(self.outfits_df)

        # Load raw features index
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Raw features cache not found at {cache_path}. Please run train.py first.")
        with open(cache_path, "rb") as f:
            self.raw_embeddings = pickle.load(f)

        # Load trained Siamese Network
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}. Please run train.py first.")
        siamese_cfg = SiameseConfig(input_dim=2816, hidden_dim=512, embedding_dim=self.config.embedding_dim)
        self.model = SiameseNetwork(siamese_cfg).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Pre-compute projected 256-dimensional embeddings for all products
        self.projected_embeddings = self._precompute_projected_embeddings()

    @torch.no_grad()
    def _precompute_projected_embeddings(self) -> dict[str, torch.Tensor]:
        projected = {}
        for pid, raw_feat in self.raw_embeddings.items():
            raw_feat_t = raw_feat.to(self.device).unsqueeze(0)
            proj_emb = self.model.encoder(raw_feat_t).squeeze(0).cpu()
            projected[pid] = proj_emb
        return projected

    def get_high_level_category(self, row: pd.Series) -> str:
        cat = row.get("category", "")
        cat_lbl = row.get("category_label", "")
        return HIGH_LEVEL_CATEGORIES.get(cat, HIGH_LEVEL_CATEGORIES.get(cat_lbl, "Other"))

    def get_compatible_items(
        self,
        anchor_id: str,
        target_high_level_cat: str,
        gender_filter: str | None = None,
        occasion_filter: str | None = None,
        limit: int = 5
    ) -> list[dict]:
        """Query the Siamese network to find items of target_high_level_cat compatible with anchor_id."""
        anchor_id = str(anchor_id)
        if anchor_id not in self.projected_embeddings:
            return []

        z_anchor = self.projected_embeddings[anchor_id].to(self.device).unsqueeze(0)

        # Retrieve candidates matching the target category and filters
        candidates_rows = []
        for _, row in self.products_df.iterrows():
            pid = str(row["id"])
            if pid == anchor_id:
                continue

            # Check category match
            if self.get_high_level_category(row) != target_high_level_cat:
                continue

            # Check gender filter (unisex/men/women match)
            if gender_filter:
                p_gender = str(row.get("gender", "")).lower()
                req_gender = gender_filter.lower()
                if p_gender != "unisex" and p_gender != req_gender:
                    continue

            candidates_rows.append(row)

        if not candidates_rows:
            return []

        # Candidate IDs
        candidate_ids = [str(r["id"]) for r in candidates_rows]
        z_candidates = torch.stack([self.projected_embeddings[cid] for cid in candidate_ids]).to(self.device)

        # Compute Euclidean distance
        distances = F.pairwise_distance(z_anchor, z_candidates, keepdim=False).cpu().numpy()

        # Build list with distances
        results = []
        for idx, row in enumerate(candidates_rows):
            dist = float(distances[idx])
            results.append({
                "product": row.to_dict(),
                "high_level_category": target_high_level_cat,
                "distance": dist
            })

        # Sort by distance (smaller distance = more compatible)
        # Apply occasion filter as a soft scoring booster: subtract a small value from distance if occasion matches
        if occasion_filter:
            occ = occasion_filter.lower().strip()
            for r in results:
                p_occ = str(r["product"].get("occasion", "")).lower()
                p_tags = str(r["product"].get("tags", "")).lower()
                if occ in p_occ or occ in p_tags:
                    r["distance"] -= 0.15  # compatibility boost for matching occasion

        results = sorted(results, key=lambda x: x["distance"])
        return results[:limit]

    def recommend_full_outfit(
        self,
        anchor_id: str,
        gender_filter: str | None = None,
        occasion_filter: str | None = None
    ) -> dict:
        """Compile a complete compatible outfit starting from an anchor product."""
        anchor_id = str(anchor_id)
        anchor_row = self.products_df[self.products_df["id"] == anchor_id]
        if anchor_row.empty:
            raise ValueError(f"Product ID {anchor_id} not found in products.csv")
        
        anchor_item = anchor_row.iloc[0]
        anchor_high_cat = self.get_high_level_category(anchor_item)
        gender = gender_filter or str(anchor_item.get("gender", "men"))

        # Determine target categories based on starting item
        if anchor_high_cat == "One-Piece/Sets":
            target_categories = ["Footwear", "Accessories"]
        elif anchor_high_cat in ["Topwear", "Bottomwear", "Layering"]:
            target_categories = ["Topwear", "Bottomwear", "Footwear", "Accessories"]
            # Remove the anchor's own category from targets
            if anchor_high_cat in target_categories:
                target_categories.remove(anchor_high_cat)
        else:
            # Accessory or Footwear: compile a standard Topwear + Bottomwear + Footwear/Accessory
            target_categories = ["Topwear", "Bottomwear"]
            if anchor_high_cat != "Footwear":
                target_categories.append("Footwear")
            if anchor_high_cat != "Accessories":
                target_categories.append("Accessories")

        outfit_items = []
        total_price = int(anchor_item.get("price_inr", 0))

        for target_cat in target_categories:
            compat = self.get_compatible_items(
                anchor_id=anchor_id,
                target_high_level_cat=target_cat,
                gender_filter=gender,
                occasion_filter=occasion_filter,
                limit=1
            )
            if compat:
                outfit_items.append(compat[0]["product"])
                total_price += int(compat[0]["product"].get("price_inr", 0))

        # Stylist rationale lookup fallback
        # Check if this exact anchor belongs to a ground truth outfit, extract the stylist explanation
        ground_truth_rationale = ""
        for _, row in self.outfits_df.iterrows():
            item_ids = self.loader._extract_outfit_item_ids(row)
            if anchor_id in item_ids:
                ground_truth_rationale = str(row.get("stylist_rationale", ""))
                break

        return {
            "anchor": anchor_item.to_dict(),
            "anchor_category": anchor_high_cat,
            "outfit_items": outfit_items,
            "total_price_inr": total_price,
            "ground_truth_rationale": ground_truth_rationale
        }


# Quick diagnostic verification
if __name__ == "__main__":
    recommender = OutfitRecommender()
    print("Outfit Recommender instantiated successfully!")
    # Test on a known item from Ajio
    test_id = "myntra_28569210"  # Arrow white shirt
    outfit = recommender.recommend_full_outfit(test_id, gender_filter="men", occasion_filter="office")
    print(f"\nAnchor: {outfit['anchor']['name']} ({outfit['anchor_category']})")
    print("Compatible recommendations:")
    for item in outfit["outfit_items"]:
        print(f"- {item['name']} | Category: {HIGH_LEVEL_CATEGORIES.get(item['category'], 'Other')} | Price: INR {item['price_inr']}")
    print(f"Total Outfit Price: INR {outfit['total_price_inr']}")
