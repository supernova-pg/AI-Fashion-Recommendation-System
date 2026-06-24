# 📊 Dataset Analysis — Dare XAI Fashion Dataset

## Overview

The provided dataset (`NEWDATASET/`) contains expert-curated fashion items sourced from three major Indian e-commerce platforms: **Ajio**, **Myntra**, and **Nykaa**.

| Metric | Value |
|---|---|
| Total Products | 68 unique items |
| Total Outfits | 25 expert-curated complete outfits |
| Sources | Ajio (images/ajio/), Myntra (images/myntra/), Nykaa (images/nykaa/) |
| Data Files | `products.csv`, `outfits.csv`, `curated25.xlsx`, `images/` |

---

## 1. Products Dataset (`products.csv`)

### Schema

| Column | Type | Description |
|---|---|---|
| `id` | string | Unique product ID (e.g., `ajio_703182002`) |
| `name` | string | Product title |
| `brand` | string | Brand name |
| `price_inr` | float | Retail price in INR |
| `rating` | float | Customer rating (0–5) |
| `rating_count` | int | Number of ratings |
| `gender` | categorical | `men` / `women` |
| `wear_type` | categorical | `western` / `ethnic` |
| `category` | string | Fine-grained category (e.g., `formal-shirts`) |
| `category_label` | string | Human-readable category label |
| `occasion` | categorical | `casual` / `party` / `office` / `festive` / `wedding` / `sports` / `vacation` / `winter` |
| `tags` | string | Semicolon-separated descriptive tags |
| `description` | text | Full product description |
| `image` | filepath | Relative path to product image |

### Gender Distribution

| Gender | Count | Percentage |
|---|---|---|
| Women | 41 | 60.3% |
| Men | 27 | 39.7% |

> ⚠️ **Imbalance Note**: Women's items outnumber men's items by ~1.5x. This doesn't significantly affect training since pairs are constructed per-outfit, but may lead to fewer men's recommendations in edge cases.

### Occasion Distribution

| Occasion | Products |
|---|---|
| Casual | 15 |
| Party | 13 |
| Office | 12 |
| Festive | 9 |
| Wedding | 6 |
| Sports | 5 |
| Winter | 4 |
| Vacation | 4 |

> ⚠️ **Long-Tail Note**: `vacation` and `winter` have very few items (4 each), limiting variety for these occasion types.

### Price Distribution

| Statistic | INR |
|---|---|
| Min | ₹270 |
| 25th Percentile | ₹732 |
| Median | ₹1,082 |
| Mean | ₹1,518 |
| 75th Percentile | ₹1,596 |
| Max | ₹7,799 |

The dataset spans budget to premium price points, providing realistic diversity for a styling assistant.

### Category Distribution (Fine-Grained)

| Category | Count |
|---|---|
| ethnic-footwear | 4 |
| heels | 4 |
| clutches | 4 |
| formal-shirts | 3 |
| party-dresses | 3 |
| necklaces | 2 |
| trousers | 2 |
| jeans | 2 |
| kurta-sets | 2 |
| boots | 2 |
| handbags | 2 |
| formal-shoes | 2 |
| watches | 2 |
| *...and 34 more single-item categories* | 1 each |

> ⚠️ **Sparsity Challenge**: 34 out of 47 categories have only 1 item. This means the recommendation engine has no alternative to pick if the single item in a category is already the anchor. This directly motivates using **high-level categories** (Topwear, Bottomwear, Footwear, Accessories) instead of fine-grained categories for retrieval.

### Missing Values

| Column | Missing | Impact |
|---|---|---|
| rating | 25 (37%) | Not used in model. Low impact. |
| rating_count | 42 (62%) | Not used in model. Low impact. |
| All other columns | 0 | No impact. |

---

## 2. Outfits Dataset (`outfits.csv`)

### Schema

| Column | Description |
|---|---|
| `outfit_id` | Unique outfit ID (e.g., `W1`, `M3`) |
| `gender` | Target gender |
| `wear_type` | `western` / `ethnic` |
| `occasion` | Occasion context |
| `theme` | Styled theme name |
| `hero` / `hero_id` | Main anchor item |
| `second` / `second_id` | Complementary item (usually bottoms) |
| `layer` / `layer_id` | Optional layering piece |
| `footwear` / `footwear_id` | Footwear item |
| `accessory_1`, `accessory_2` | Optional accessories |
| `palette` | Color combination description |
| `stylist_rationale` | Expert explanation of the outfit's compatibility |

### Outfit Distribution

| Occasion | Outfits | Gender |
|---|---|---|
| Party | 6 | Women |
| Casual | 5 | Mixed |
| Festive | 4 | Women |
| Wedding | 3 | Mixed |
| Office | 2 | Men |
| Vacation | 2 | Women |
| Sports | 2 | Men |
| Winter | 1 | Women |

> ⚠️ **Small Dataset Challenge**: 25 outfits is extremely small for training a deep learning model. Standard 80/20 splits leave only 5 validation outfits. This is why we use **25-fold Leave-One-Out Cross-Validation (LOOCV)** to maximize utility of every data point.

---

## 3. Image Analysis

- All images are in `.jpg` / `.jpeg` format
- Resolution varies (original e-commerce images, typically 400–800px)
- Images are resized to **224×224** pixels during feature extraction (ResNet50 standard input)
- Each product has exactly **1 image** — no multi-angle views

> ⚠️ **Limitation**: Single-angle product images may not capture full garment detail (texture, fabric drape). Multi-angle images would improve visual embedding quality.

---

## 4. Key Challenges

| Challenge | Impact | Mitigation |
|---|---|---|
| **Very small dataset** (68 items, 25 outfits) | High risk of overfitting | Frozen encoders + LOOCV + Dropout + Weight Decay |
| **Category sparsity** (34 single-item categories) | Limited retrieval diversity | Grouped into 6 high-level categories for retrieval |
| **Gender imbalance** (41W vs 27M) | Fewer men's recommendations | Strict gender filtering in recommendation engine |
| **Occasion imbalance** (vacation=4 items) | Poor vacation recommendations | Soft occasion boost (not hard filter) |
| **No negative style labels** | Cannot learn what NOT to pair | Negative pairs constructed as random cross-outfit items |
| **Single image per product** | Limited visual signal | Compensated by rich text (DistilBERT) fusion |
| **Missing ratings** (62% empty) | Cannot use popularity as signal | Ratings excluded from model |

---

## 5. How the Dataset is Used

### For Training
1. **Positive Pairs**: Any two items that co-occur in the same outfit → label = 0 (compatible)
2. **Negative Pairs**: Randomly sampled items from *different* outfits → label = 1 (incompatible)
3. **Pair ratio**: 1:1 positive to negative (balanced sampling)

With 25 outfits averaging ~4 items each, we get:
- ~150 positive pairs (combinations within outfits)
- ~150 randomly sampled negative pairs
- **Total: ~300 training pairs** — extremely small, reinforcing the need for frozen encoders.

### For Inference (Recommendation)
- All 68 product embeddings are pre-computed offline
- At query time, Euclidean distance in 256-dim space identifies closest compatible items per category slot

---

## 6. Summary & Observations

> ✅ **Strengths**: Rich metadata (tags, descriptions, occasion, palette), expert-curated outfit ground truth, multi-source diversity (Ajio + Myntra + Nykaa), clear stylist rationale for explainability.

> ⚠️ **Weaknesses**: Extremely small scale (68 items), heavy category imbalance, no multi-angle images, no explicit incompatibility annotations.

> 💡 **Recommendations for production**: Expand to 1000+ products per category, add negative style annotations from fashion experts, use FashionCLIP for better domain-specific embeddings, integrate a vector database (FAISS/Qdrant) for scalable retrieval.
