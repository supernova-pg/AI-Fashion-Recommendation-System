# 🏗️ Architecture Documentation — Dare XAI Fashion Recommendation System

## System Overview

This system is a **hybrid multi-modal AI recommendation engine** that combines:
- **Deep Learning** (Siamese Networks) for learning fashion compatibility
- **Pre-trained Foundation Models** (ResNet50, DistilBERT) for feature extraction
- **LLM Integration** (Gemini API) for natural language understanding and explainability
- **Streamlit** for an interactive web dashboard

---

## Component Architecture

### 1. Data Layer (`data_loader.py`)

**Responsibilities:**
- Load and normalize `products.csv` (68 products) and `outfits.csv` (25 outfits)
- Construct positive pairs: item pairs that appear in the same ground-truth outfit
- Construct negative pairs: randomly sampled item pairs from different outfits
- Extract raw feature embeddings per product (once, then cached to `raw_embeddings.pkl`)

**Feature Extraction:**
```
Product Image (JPEG/PNG)
    └── torchvision transforms (resize 224×224, normalize)
    └── ResNet50 (pretrained ImageNet, avgpool output)
    └── 2048-dim image feature vector

Product Metadata (name + description + tags + category + occasion)
    └── DistilBERT tokenizer (max_length=128)
    └── DistilBERT encoder (frozen, [CLS] token output)
    └── 768-dim text feature vector

Concatenate [2048 || 768] → 2816-dim raw embedding
```

Both ResNet50 and DistilBERT are **completely frozen** (no gradient updates). They act as fixed feature extractors. This is intentional to prevent overfitting on the 68-item dataset.

---

### 2. Model Layer (`models.py`)

**Siamese Network Architecture:**
```
Input: (anchor_embedding, candidate_embedding) — each 2816-dim

Shared Encoder (applied to both):
  Linear(2816 → 512) + BatchNorm + ReLU + Dropout(0.3)
  Linear(512 → 256) + L2-Normalize
  → 256-dim unit-sphere embedding

Distance Computation:
  Euclidean Distance = ||z_anchor - z_candidate||₂

Loss Function:
  Contrastive Loss:
    L = (1-y) * d²  +  y * max(margin - d, 0)²
  where y=0 for compatible pairs, y=1 for incompatible pairs
```

The shared encoder maps high-dimensional raw embeddings into a compact 256-dim space where **compatible items are pulled together** and **incompatible items are pushed apart**.

---

### 3. Training Pipeline (`train.py`)

**Strategy: 25-Fold Leave-One-Out Cross-Validation (LOOCV)**

Since the dataset has only 25 outfits, standard train/val splits would result in insufficient data. LOOCV ensures:
- Every outfit is used exactly once as a validation set
- The model is validated on genuinely unseen outfit combinations
- The best checkpoint (lowest validation loss across all 25 folds) is saved

**Anti-Overfitting Measures:**
- Frozen backbone encoders (no risk of encoder overfitting)
- Dropout(0.3) in the projection head
- Weight Decay (L2 regularization, 1e-4)
- BatchNormalization after each linear layer
- LOOCV prevents data leakage

---

### 4. Recommendation Engine (`recommendation_engine.py`)

**Pre-computation at startup:**
```
For each of 68 products:
  raw_emb (2816-dim) → encoder → projected_emb (256-dim)
  Stored in memory dict: { product_id → tensor }
```

**Outfit recommendation flow:**
```
1. Receive anchor_id (e.g., "myntra_28569210" — white formal shirt)
2. Determine anchor high-level category (e.g., "Topwear")
3. Define target slots: Bottomwear, Footwear, Accessories
4. For each slot:
   a. Filter candidates by: category slot + gender + has embedding
   b. Compute Euclidean distance: z_anchor vs. all candidates
   c. Apply occasion soft-boost: -0.15 to distance if occasion matches
   d. Sort by distance → pick top-1
5. Return: anchor + list of slot winners = complete outfit
```

**High-Level Category Mapping:**
| Raw Category | High-Level |
|---|---|
| formal-shirts, tshirts, polo-tshirts, tops, kurta-sets, sweatshirts | Topwear |
| trousers, jeans, chinos, shorts, track-pants, leggings, skirts | Bottomwear |
| formal-shoes, sneakers, heels, boots, ethnic-footwear, sandals, flats, loafers, running-shoes | Footwear |
| watches, necklaces, earrings, handbags, clutches, sunglasses, caps | Accessories |
| blazers, denim-jackets, long-coats, nehru-jackets, sweaters | Layering |
| dresses, maxi-dresses, co-ord-sets, salwar-suits, sharara-sets, wedding-sarees, suits, sherwanis | One-Piece/Sets |

---

### 5. LLM Integration — Gemini API (`app.py`)

**Two separate Gemini calls:**

**Call 1 — Query Parsing (Conversational Tab only):**
```
Input: User's raw natural language query
Output (JSON):
  {
    "detected_gender": "men" | "women",
    "detected_occasion": "office" | "wedding" | ...,
    "anchor_product_id": "<product_id from database>"
  }
```
This replaces fragile regex-based parsing with semantic understanding.

**Call 2 — Stylist Rationale Generation (both tabs):**
```
Input: 
  - Anchor product name + brand
  - List of recommended items (name + brand)
  - User's age, gender, occasion, style preference
Output: A professional paragraph explaining:
  - Color coordination rationale
  - Occasion suitability
  - Stylistic cohesion
```

**Important design decision:** The rationale is generated **after** the Siamese Network retrieves the items. This ensures the text description matches the visual output exactly (fixes the grey pants vs. navy pants discrepancy bug).

**Fallback Mode (no API key):**
- Query parsing uses rule-based keyword matching
- Rationale uses templated string generation from the outfit's ground-truth stylist notes

---

### 6. Web Application (`app.py`)

**Two-Tab Streamlit Interface:**

**Tab 1 — Outfit Builder Showcase:**
```
Sidebar (User Profile) → Product Selector → Anchor Display
→ [Generate Compatible Outfit Button]
→ Outfit Grid (anchor + 3-4 recommended items)
→ Stylist Commentary box
```

**Tab 2 — Conversational Assistant:**
```
Chat Input → Gemini LLM parsing → Outfit Recommendation
→ Response Text (context + rationale)
→ Visual Outfit Grid inline in chat
→ Chat history persistence (session state)
```

---

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                        OFFLINE (Pre-computation)                      │
│                                                                      │
│  products.csv + images/  →  data_loader.py  →  raw_embeddings.pkl   │
│                                ↕                                     │
│                           train.py                                   │
│                       (25-fold LOOCV)                                │
│                              ↓                                       │
│                        best_model.pth                                │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                         ONLINE (Inference)                            │
│                                                                      │
│  app.py loads:                                                        │
│    raw_embeddings.pkl → encoder → projected_embeddings (in memory)  │
│    best_model.pth → SiameseNetwork.encoder (frozen for inference)   │
│                                                                      │
│  User Query                                                          │
│      ↓                                                               │
│  Gemini API (parse intent) → anchor_id, gender, occasion            │
│      ↓                                                               │
│  recommendation_engine.recommend_full_outfit()                       │
│      ↓                                                               │
│  Euclidean Distance Search → outfit_items list                       │
│      ↓                                                               │
│  Gemini API (generate rationale from exact outfit_items)            │
│      ↓                                                               │
│  Streamlit renders product cards + commentary                        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Design Decisions & Tradeoffs

| Decision | Rationale |
|---|---|
| Frozen ResNet50 + DistilBERT | Dataset too small (68 items) to fine-tune without severe overfitting |
| 256-dim projection head | Compact embedding space improves distance-based ranking quality |
| LOOCV instead of random split | Maximizes use of 25-outfit ground truth while ensuring valid evaluation |
| Occasion as soft boost (not hard filter) | Prevents empty results when no items exactly match the occasion |
| Gemini rationale after retrieval | Prevents LLM from hallucinating item names that differ from visually shown items |
| Streamlit for UI | Rapid prototyping; enables both visual grid and chat interface in one framework |
