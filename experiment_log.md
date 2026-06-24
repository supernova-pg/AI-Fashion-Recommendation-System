# Experiment Log - Hybrid AI Fashion Recommendation

Use this file to track every training run and compare improvements.

## Initial Baseline
- Date:
- Run Name:
- Dataset Split / CV Strategy:
- Backbone Setup:
  - Image Encoder: ResNet50 (frozen)
  - Text Encoder: DistilBERT (frozen)
- Siamese Margin:
- Embedding Dim:
- Batch Size:
- Epochs:
- Optimizer / LR / Weight Decay:
- Notes:

## Run History

| Run ID | Date | Method Applied | Training Loss | Validation NDCG@5 | Validation HR@5 | Conclusion / Next Steps |
|---|---|---|---:|---:|---:|---|
| baseline_001 | 2026-06-23 | Initial Baseline (ResNet50 + DistilBERT + Siamese projection) | 0.0715 | 0.1342 | 0.1807 | Successful baseline with LOOCV. Frozen ResNet50/DistilBERT with trainable projection head. Highly efficient training using feature caching. |

## Method Applied
Examples:
- Image Augmentation
- Hard Negative Mining
- Dropout Tuning
- Margin Tuning
- Weight Decay Tuning
- Text Cleaning Improvements

## Detailed Run Template

### Run: <run_id>
- Date:
- Initial Baseline Reference:
- Method Applied:
- Training Loss:
- Validation NDCG@5:
- Validation HR@5:
- Conclusion / Next Steps:

## Notes
- Keep all metrics on the same evaluation protocol (LOOCV, @5).
- Track only one major change per run when possible.

### Run: baseline_001
- Date: 2026-06-23
- Method Applied: Initial Baseline (ResNet50 + DistilBERT + Siamese projection)
- Training Loss: 0.0715
- Validation NDCG@5: 0.1342
- Validation HR@5: 0.1807
- Conclusion / Next Steps: Successful baseline with LOOCV. Frozen ResNet50/DistilBERT with trainable projection head. Highly efficient training using feature caching.
