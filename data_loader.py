from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import ResNet50_Weights, resnet50
from transformers import DistilBertModel, DistilBertTokenizerFast


ID_COLUMNS = [
    "hero_id",
    "second_id",
    "layer_id",
    "footwear_id",
    "accessory_1_id",
    "accessory_2_id",
]


@dataclass
class DataConfig:
    data_root: str = "."
    products_csv: str = "products.csv"
    outfits_csv: str = "outfits.csv"
    image_col: str = "image"
    text_col: str = "description"
    category_col: str = "category_label"
    gender_col: str = "gender"
    embedding_dim: int = 256
    dropout_p: float = 0.2
    negatives_per_positive: int = 2
    random_seed: int = 42


class FrozenMultimodalEmbedder(nn.Module):
    """Frozen ResNet50 + DistilBERT feature extractors with trainable projections."""

    def __init__(self, embedding_dim: int = 256, dropout_p: float = 0.2) -> None:
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V2
        backbone = resnet50(weights=weights)
        self.image_encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.image_transform = weights.transforms()

        self.text_tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        self.text_encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")

        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.image_proj = nn.Sequential(
            nn.Linear(2048, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(768, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    @torch.no_grad()
    def encode_image(self, image_path: Path, device: torch.device) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.image_transform(image).unsqueeze(0).to(device)
        features = self.image_encoder(pixel_values).flatten(1)
        return features

    @torch.no_grad()
    def encode_text(self, text: str, device: torch.device) -> torch.Tensor:
        text = text or ""
        tokenized = self.text_tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        outputs = self.text_encoder(**tokenized)
        cls_like = outputs.last_hidden_state[:, 0, :]
        return cls_like

    def forward(self, image_feats: torch.Tensor, text_feats: torch.Tensor) -> torch.Tensor:
        img = self.image_proj(image_feats)
        txt = self.text_proj(text_feats)
        fused = torch.cat([img, txt], dim=1)
        return self.fusion(fused)


class SiamesePairDataset(Dataset):
    """Dataset returning (embedding_a, embedding_b, label) for contrastive learning."""

    def __init__(
        self,
        pairs_df: pd.DataFrame,
        embedding_index: Dict[str, torch.Tensor],
        id_a_col: str = "id_a",
        id_b_col: str = "id_b",
        label_col: str = "label",
    ) -> None:
        self.pairs_df = pairs_df.reset_index(drop=True)
        self.embedding_index = embedding_index
        self.id_a_col = id_a_col
        self.id_b_col = id_b_col
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.pairs_df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.pairs_df.iloc[idx]
        emb_a = self.embedding_index[row[self.id_a_col]]
        emb_b = self.embedding_index[row[self.id_b_col]]
        label = torch.tensor(float(row[self.label_col]), dtype=torch.float32)
        return emb_a, emb_b, label


class FashionDataLoader:
    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.data_root = Path(config.data_root)
        self.products_path = self.data_root / config.products_csv
        self.outfits_path = self.data_root / config.outfits_csv

        if not self.products_path.exists() or not self.outfits_path.exists():
            raise FileNotFoundError(
                "Could not find products/outfits CSV files. "
                "Set DataConfig.data_root to your dataset folder."
            )

    def load_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        products = pd.read_csv(self.products_path)
        outfits = pd.read_csv(self.outfits_path)

        required_product_cols = {"id", self.config.image_col, self.config.text_col, self.config.gender_col}
        required_outfit_cols = set(ID_COLUMNS)

        missing_p = required_product_cols - set(products.columns)
        missing_o = required_outfit_cols - set(outfits.columns)

        if missing_p:
            raise ValueError(f"products.csv missing columns: {sorted(missing_p)}")
        if missing_o:
            raise ValueError(f"outfits.csv missing columns: {sorted(missing_o)}")

        products = products.drop_duplicates(subset=["id"]).copy()
        products = products[products["id"].notna()].copy()

        return products, outfits

    @staticmethod
    def _extract_outfit_item_ids(outfit_row: pd.Series) -> List[str]:
        ids: List[str] = []
        for col in ID_COLUMNS:
            val = outfit_row.get(col)
            if pd.notna(val) and str(val).strip():
                ids.append(str(val).strip())
        return list(dict.fromkeys(ids))

    def build_positive_pairs(self, outfits_df: pd.DataFrame) -> pd.DataFrame:
        rows: List[Tuple[str, str, int, str]] = []
        for _, outfit in outfits_df.iterrows():
            outfit_id = str(outfit.get("outfit_id", "unknown_outfit"))
            item_ids = self._extract_outfit_item_ids(outfit)
            for id_a, id_b in combinations(item_ids, 2):
                rows.append((id_a, id_b, 1, outfit_id))

        positive_df = pd.DataFrame(rows, columns=["id_a", "id_b", "label", "outfit_id"])
        positive_df = positive_df.drop_duplicates(subset=["id_a", "id_b", "label"])
        return positive_df

    def _compatibility_map(self, outfits_df: pd.DataFrame) -> Dict[str, Set[str]]:
        compat: Dict[str, Set[str]] = {}
        for _, outfit in outfits_df.iterrows():
            ids = self._extract_outfit_item_ids(outfit)
            for x in ids:
                compat.setdefault(x, set())
                for y in ids:
                    if x != y:
                        compat[x].add(y)
        return compat

    def build_negative_pairs(
        self,
        products_df: pd.DataFrame,
        outfits_df: pd.DataFrame,
        positive_df: pd.DataFrame,
    ) -> pd.DataFrame:
        rng = torch.Generator().manual_seed(self.config.random_seed)
        compat = self._compatibility_map(outfits_df)

        gender_by_id = products_df.set_index("id")[self.config.gender_col].to_dict()
        product_ids = products_df["id"].astype(str).tolist()

        rows: List[Tuple[str, str, int]] = []

        for _, pos in positive_df.iterrows():
            anchor = str(pos["id_a"])
            partner = str(pos["id_b"])
            anchor_gender = gender_by_id.get(anchor)

            candidates = [
                pid
                for pid in product_ids
                if pid not in {anchor, partner}
                and pid not in compat.get(anchor, set())
                and gender_by_id.get(pid) == anchor_gender
            ]

            if not candidates:
                candidates = [
                    pid
                    for pid in product_ids
                    if pid not in {anchor, partner} and pid not in compat.get(anchor, set())
                ]

            if not candidates:
                continue

            perm = torch.randperm(len(candidates), generator=rng).tolist()
            take = min(self.config.negatives_per_positive, len(candidates))
            for idx in perm[:take]:
                rows.append((anchor, candidates[idx], 0))

        negative_df = pd.DataFrame(rows, columns=["id_a", "id_b", "label"])
        negative_df = negative_df.drop_duplicates(subset=["id_a", "id_b", "label"])
        return negative_df

    def build_pair_dataframe(self, products_df: pd.DataFrame, outfits_df: pd.DataFrame) -> pd.DataFrame:
        positive_df = self.build_positive_pairs(outfits_df)
        negative_df = self.build_negative_pairs(products_df, outfits_df, positive_df)

        pairs_df = pd.concat([positive_df[["id_a", "id_b", "label"]], negative_df], ignore_index=True)
        pairs_df = pairs_df.sample(frac=1.0, random_state=self.config.random_seed).reset_index(drop=True)
        return pairs_df

    def _resolve_image_path(self, image_ref: str) -> Path:
        image_ref = str(image_ref)
        candidate_paths = [
            self.data_root / image_ref,
            Path(image_ref),
            self.data_root / "images" / Path(image_ref).name,
        ]

        for path in candidate_paths:
            if path.exists():
                return path

        raise FileNotFoundError(f"Image file not found for reference: {image_ref}")

    def build_embedding_index(
        self,
        products_df: pd.DataFrame,
        embedder: FrozenMultimodalEmbedder,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        embedder = embedder.to(device)
        embedder.eval()

        embedding_index: Dict[str, torch.Tensor] = {}

        for _, row in products_df.iterrows():
            pid = str(row["id"])
            image_path = self._resolve_image_path(row[self.config.image_col])
            text = str(row.get(self.config.text_col, ""))

            with torch.no_grad():
                image_feats = embedder.encode_image(image_path, device)
                text_feats = embedder.encode_text(text, device)
                raw_feats = torch.cat([image_feats, text_feats], dim=1).squeeze(0).cpu()

            embedding_index[pid] = raw_feats

        return embedding_index


def build_dataset_bundle(config: Optional[DataConfig] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience function to quickly load data and produce Siamese pairs."""
    cfg = config or DataConfig()
    loader = FashionDataLoader(cfg)
    products_df, outfits_df = loader.load_dataframes()
    pairs_df = loader.build_pair_dataframe(products_df, outfits_df)
    return products_df, outfits_df, pairs_df


def append_experiment_result(
    log_path: str,
    run_id: str,
    run_date: str,
    method_applied: str,
    training_loss: float,
    validation_ndcg5: float,
    validation_hr5: float,
    conclusion: str,
) -> None:
    """Insert a result row into the Run History table and append detailed notes at the end."""
    import os
    if not os.path.exists(log_path):
        # Create standard headers if file doesn't exist
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("# Experiment Log\n\n## Run History\n\n| Run ID | Date | Method Applied | Training Loss | Validation NDCG@5 | Validation HR@5 | Conclusion / Next Steps |\n|---|---|---|---:|---:|---:|---|\n")

    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Find the Run History table header separator
    table_sep_idx = -1
    for idx, line in enumerate(lines):
        if "|---|---|---|---:|---:|---:|---|" in line.replace(" ", ""):
            table_sep_idx = idx
            break

    row_line = (
        f"| {run_id} | {run_date} | {method_applied} | {training_loss:.4f} | "
        f"{validation_ndcg5:.4f} | {validation_hr5:.4f} | {conclusion} |\n"
    )

    if table_sep_idx != -1:
        # Insert right after the table separator
        lines.insert(table_sep_idx + 1, row_line)
    else:
        lines.append(row_line)

    # Append a detailed run block at the end of the file
    detail_block = (
        f"\n### Run: {run_id}\n"
        f"- Date: {run_date}\n"
        f"- Method Applied: {method_applied}\n"
        f"- Training Loss: {training_loss:.4f}\n"
        f"- Validation NDCG@5: {validation_ndcg5:.4f}\n"
        f"- Validation HR@5: {validation_hr5:.4f}\n"
        f"- Conclusion / Next Steps: {conclusion}\n"
    )
    lines.append(detail_block)

    with open(log_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
