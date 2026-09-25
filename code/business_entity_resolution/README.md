# Business Entity Resolution Pipeline

End-to-end ML pipeline for the ML Challenge 2026. Maps each Source 1 entity to all matching records in Source 2 and Source 3, optimized for macro F0.5.

## Quick Start

Install dependencies:
```bash
pip install -r requirements.txt
```

### Train + Evaluate (validation split)
```bash
cd code/business_entity_resolution
python pipeline.py --mode train
```
Trains on 80% of Source 1, evaluates on 20%, prints blocking recall and macro F0.5, saves model + threshold artifacts.

### Full Train + Test Inference
```bash
python pipeline.py --mode full
```
Trains on all labeled data using the threshold found in `--mode train`, then runs inference on the test set and writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

### Test Inference Only (requires saved model)
```bash
python pipeline.py --mode test
```

## Architecture

```
Data Loading & Validation
  ↓
Multi-View Normalization  (Unicode, script detection, name/address, transliteration)
  ↓
Multi-Route Blocking      (exact core name, rare tokens, char n-grams,
                           locality, postal, transliteration)
  ↓
Candidate Union           (per S1, capped at 300 candidates)
  ↓
IDF-Weighted Pair Features (56 features: name, address, transliteration,
                             script, rarity, context, contradiction)
  ↓
LightGBM Binary Classifier
  ↓
Threshold Optimization    (grid search on macro F0.5)
  ↓
Output: matching_results.tsv + candidate_pairs.tsv
```

## Project Structure

```
src/
  data/
    loader.py         - TSV loading, ground truth parsing
    validation.py     - ID, column, and integrity checks
  normalization/
    unicode.py        - NFC/NFKC, zero-width char removal, punctuation normalization
    scripts.py        - Unicode script block detection (Latin, Devanagari, Tamil, …)
    names.py          - Core name extraction, legal suffix detection, DBA/FKA markers
    addresses.py      - Address cleaning, postal/PIN extraction, component parsing
    transliteration.py- Character-level Indian-script to Latin transliteration
  blocking/
    candidate_generator.py - Multi-route inverted-index candidate generation
  features/
    pair_features.py  - 56 numeric pair features (name, addr, translit, script, ...)
  model/
    train.py          - IDF computation, pair construction, LightGBM training, scoring
  evaluation/
    blocking_eval.py  - Blocking recall measurement
    metrics.py        - F0.5, macro F0.5, threshold optimization
    error_analysis.py - FP/FN classification by category
pipeline.py           - Main entry point
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Multiple blocking routes | No single route survives all noise; union maximizes recall |
| Transliteration (char map) | Connects native-script and Latin names without external APIs |
| Legal suffix extraction | Suffix is not part of identity; removing it improves core-name similarity |
| IDF-weighted token overlap | Down-weights generic words like "Services", "Trading" |
| Separate name + address similarity | One field can be clean while the other is corrupted |
| Missing = unknown (not conflict) | A blank address contributes no evidence either way |
| LightGBM on tabular features | Fast, accurate on structured pair features; no LLM needed |
| Macro F0.5 threshold sweep | Direct optimization of the challenge metric |

## Validate Submission

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Reproducing Results

1. Place raw data in `dataset/train/` and `dataset/test/`.
2. `pip install -r requirements.txt`
3. `python pipeline.py --mode train` — validate blocking recall and F0.5.
4. `python pipeline.py --mode full` — write final outputs.
5. Validate with the official validator above.
