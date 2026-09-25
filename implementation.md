# Business Entity Resolution — Implementation Plan

## 1. Objective

Build an end-to-end business entity resolution system that maps each Source 1 entity to **all** matching records in Source 2 and Source 3, while allowing zero, one, or many matches.

The system is optimized for the challenge's **macro F0.5** objective, where precision is weighted more heavily than recall. It must produce the required `candidate_pairs.tsv` and `matching_results.tsv`.

The training audit confirms that the data is a genuinely noisy multilingual problem involving native and mixed scripts, transliteration, Unicode variation, typos, token joining/splitting, legal suffixes, DBA/aliases, numeric and partial addresses, shared names/addresses, source-specific noise, and zero/singleton/multi-match relationships.

---

## 2. Design Principles

1. Do not use brute-force all-pairs matching.
2. Do not rely on one blocking key.
3. Blocking should maximize recall; final matching should maximize precision.
4. Keep raw text and create multiple normalized representations.
5. Treat names and addresses as separate evidence sources.
6. Treat missing information as unknown, not contradiction.
7. Preserve numeric address information.
8. Handle multilingual and mixed-script records explicitly.
9. Use transliteration as an additional representation, not a replacement.
10. Treat common/short names and shared addresses as high-risk false-positive cases.
11. Do not force one-to-one matching; one S1 entity may map to multiple S2/S3 records.
12. Do not force every S1 entity to have a match.
13. Tune the final decision threshold against macro F0.5.
14. Evaluate blocking recall before evaluating the final classifier.
15. Use hard negatives from the dataset.
16. Keep the pipeline reproducible.
17. Do not use external entity-resolution APIs, registries, geocoding, web lookup, or external augmentation.

---

## 3. End-to-End Architecture

```text
TRAIN S1 / S2 / S3 + GROUND TRUTH
                |
                v
        DATA VALIDATION
                |
                v
          DATA AUDIT / EDA
                |
                v
      MULTI-VIEW NORMALIZATION
                |
                v
       MULTI-ROUTE BLOCKING
                |
                v
       CANDIDATE UNION + DEDUP
                |
                v
       CANDIDATE PRE-FILTER
                |
                v
       PAIR FEATURE ENGINEERING
                |
                v
        MATCHING CLASSIFIER
                |
                v
        DECISION / THRESHOLD
                |
        +-------+--------+
        |                |
        v                v
candidate_pairs.tsv   matching_results.tsv
        |
        v
   VALIDATION + F0.5
        |
        v
   ERROR ANALYSIS
        |
        v
  ITERATIVE IMPROVEMENT
```

---

## 4. Repository Structure

```text
business_entity_resolution/
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   ├── validation.py
│   │   └── audit.py
│   ├── normalization/
│   │   ├── unicode.py
│   │   ├── scripts.py
│   │   ├── names.py
│   │   ├── addresses.py
│   │   ├── aliases.py
│   │   └── transliteration.py
│   ├── blocking/
│   │   ├── exact_name.py
│   │   ├── rare_token.py
│   │   ├── char_ngram.py
│   │   ├── address.py
│   │   ├── numeric.py
│   │   ├── locality.py
│   │   ├── transliteration.py
│   │   ├── phonetic.py
│   │   └── candidate_generator.py
│   ├── features/
│   │   ├── name_features.py
│   │   ├── address_features.py
│   │   ├── script_features.py
│   │   ├── numeric_features.py
│   │   ├── rarity_features.py
│   │   ├── quality_features.py
│   │   ├── contradiction_features.py
│   │   └── pair_features.py
│   ├── model/
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── threshold.py
│   ├── evaluation/
│   │   ├── blocking_eval.py
│   │   ├── metrics.py
│   │   ├── validation.py
│   │   └── error_analysis.py
│   └── pipeline.py
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── README.md
└── requirements.txt
```

---

## 5. Phase 0 — Input Validation

Training files:

```text
train_source1.tsv
train_source2.tsv
train_source3.tsv
train_ground_truth.tsv
```

Test files:

```text
test_source1.tsv
test_source2.tsv
test_source3.tsv
```

Read with:

```python
pd.read_csv(path, sep="\t")
```

Validate:

- expected columns
- tab-separated parsing
- source ID prefixes
- duplicate entity IDs
- malformed rows
- ground-truth target IDs
- every ground-truth target exists in Source 2 or Source 3
- every Source 1 training ID appears in ground truth
- country remains an open-set string field

---

## 6. Phase 1 — Full Dataset Audit

For every source and field calculate:

### Text

- raw length
- normalized length
- token count
- digit count
- punctuation/symbol count
- whitespace anomalies
- zero-width/control characters
- Unicode normalization changes

### Script

Detect:

- Latin
- Devanagari
- Tamil
- Malayalam
- Telugu
- Gujarati
- Kannada
- Bengali
- other scripts

Also calculate mixed-script flags and script proportions.

### Names

Detect:

- legal suffixes
- abbreviations
- DBA/d/b/a/FKA
- domain-like names
- repeated tokens
- very short names
- generic tokens
- acronym-like names
- punctuation-heavy names
- token joining/splitting

### Addresses

Detect:

- postal/PIN patterns
- house/building numbers
- unit/floor numbers
- landmarks
- abbreviations
- partial/missing components
- reordered components
- repeated components

### Frequency tables

Build frequencies for:

- normalized names
- core names
- name tokens
- address tokens
- postal codes
- cities/localities
- addresses

These become later rarity/IDF-style features.

---

## 7. Phase 2 — Ground Truth Analysis

Parse the comma-separated `matched_entity_ids` into individual labeled pairs.

For each S1 entity calculate:

- number of matches
- zero-match flag
- singleton flag
- multi-match flag
- S2-only matches
- S3-only matches
- mixed S2/S3 matches
- maximum targets

Also measure:

- repeated names across different S1 entities
- repeated addresses across different S1 entities
- same-name/different-address groups
- same-address/different-name groups
- duplicate/near-duplicate source records

This tells us how the dataset defines identity in practice.

---

## 8. Phase 3 — Multi-View Normalization

Never overwrite original columns.

Create:

```text
name_raw
name_unicode
name_clean
name_tokens
name_core
name_translit
name_char_ngrams
```

and corresponding address representations.

Name normalization:

1. Unicode NFC/NFKC as appropriate.
2. Normalize whitespace.
3. Remove unsafe invisible formatting characters.
4. Case normalization where applicable.
5. Normalize punctuation in a separate representation.
6. Normalize common separators.
7. Keep `&`/`and` handling as a separate representation.
8. Detect legal suffixes rather than blindly deleting them.
9. Extract DBA/FKA structures.
10. Preserve original representation.
11. Preserve script information.

Do not blindly expand ambiguous abbreviations such as `St`, `No`, and `Co`.

---

## 9. Phase 4 — Transliteration and Script Handling

For native-script records create an additional transliteration-aware representation.

Keep:

```text
original
+
transliterated
```

Calculate:

- script distribution
- mixed-script flag
- transliteration availability
- original/transliterated similarity

Use transliteration for:

- blocking
- candidate recall
- pair features

Do not treat transliteration alone as proof of identity.

---

## 10. Phase 5 — Address Normalization

Create:

```text
address_raw
address_unicode
address_clean
address_tokens
address_numbers
postal_code
city
state
locality
street
unit
```

Preserve numeric information.

Extract:

- house/building numbers
- unit/flat numbers
- floor numbers
- postal/PIN codes
- numeric ranges
- alphanumeric identifiers

Normalize common address abbreviations carefully:

```text
Rd -> Road
St -> Street
Ave -> Avenue
Nr -> Near
Opp -> Opposite
C/O -> Care Of
H.No -> House Number
```

Landmarks should be weaker evidence than specific building/unit/postal information.

---

## 11. Phase 6 — Blocking

Blocking must be high-recall. Use multiple routes and take their union.

### Block A — Exact normalized core name

Use keys such as:

```text
country + core_name
```

when country exists.

### Block B — Rare name tokens

Build:

```text
token -> record IDs
```

Use token frequency/IDF to prioritize rare informative tokens. Do not rely on generic words alone.

### Block C — Character n-grams

Use character n-gram retrieval for:

- typos
- token joins
- token splits
- punctuation changes
- spelling changes

### Block D — Address/locality

Use combinations such as:

```text
country + state + city
country + locality
city + street
```

where available.

### Block E — Postal/numeric

Use:

```text
postal
postal + house number
postal + unit
house number + locality
```

Do not require postal agreement for final matching.

### Block F — Transliteration

Use transliterated core names and n-grams.

### Block G — Selective phonetic retrieval

Experiment with Soundex/Metaphone/Double Metaphone for Latin-heavy records only if validation proves useful.

### Block H — Source-specific blocking

Evaluate S1-S2 and S1-S3 separately because their corruption patterns differ.

---

## 12. Blocking Evaluation

For each blocking route and its union calculate:

```text
candidate_count
average_candidates_per_S1
reduction_ratio
true_match_recall
recall_by_source
recall_by_country
recall_by_ambiguity_category
```

Critical metric:

```text
blocking recall =
true matches present in candidate set
/
all true matches
```

Select the smallest candidate system that achieves sufficiently high recall.

---

## 13. Candidate Pre-Filtering

After high-recall blocking, remove candidates with virtually no evidence.

Keep a candidate when at least one meaningful signal exists:

- name similarity
- address similarity
- postal agreement
- numeric agreement
- rare-token overlap
- transliteration similarity
- locality agreement

Keep this stage more permissive than final matching.

The final `candidate_pairs.tsv` must represent the candidates actually passed to the final matching model.

---

## 14. Phase 7 — Pair Features

### Name

- raw/normalized/core exact equality
- Levenshtein
- Jaro-Winkler
- character n-gram similarity
- token Jaccard
- weighted token overlap
- TF-IDF cosine
- token-order similarity
- transliteration similarity
- legal-suffix agreement
- alias/DBA indicators

### Address

- raw/normalized equality
- character similarity
- token overlap
- component overlap
- postal agreement
- city/state/locality/street agreement
- house/unit agreement
- numeric overlap
- landmark overlap

### Quality

- missingness
- name/address length
- token count
- script count
- mixed-script flag
- placeholder flag

### Rarity

- name frequency
- core-name frequency
- rare-token count
- address frequency
- postal frequency

### Context

- country equality
- source pair
- source-specific quality
- region indicators

### Contradictions

Create features such as:

```text
strong_name_weak_address
strong_name_conflicting_address
strong_address_weak_name
both_strong
both_weak
```

Distinguish missing from conflicting evidence.

---

## 15. Phase 8 — Hard Negatives

Construct genuine nonmatches such as:

- same normalized name, different entity
- same core name, different address
- same address, different name
- same postal code, different entity
- same city/locality, different entity
- high name similarity + low address similarity
- high address similarity + low name similarity
- common/generic names
- short names
- shared landmarks
- related corporate names

Use ground truth to verify that they are nonmatches.

---

## 16. Phase 9 — Validation Split

Split by Source 1 entity, not candidate pair.

This prevents information leakage from the same S1 entity appearing in both training and validation.

Run the complete pipeline separately inside train and validation.

---

## 17. Phase 10 — Matching Model

Start with a tabular gradient-boosting classifier such as:

- LightGBM
- XGBoost
- CatBoost

Input:

```text
candidate pair + engineered features
```

Output:

```text
P(match)
```

Do not start with a huge LLM/neural model. If embeddings are later useful, add their similarities as features.

---

## 18. Phase 11 — Threshold Optimization

Do not assume threshold 0.5.

Evaluate a threshold grid, for example:

```text
0.30, 0.35, 0.40, ... 0.95
```

For each threshold:

1. Generate entity-level match lists.
2. Calculate precision/recall.
3. Calculate F0.5 per S1.
4. Macro-average across S1.
5. Include zero-match and multi-match entities.

Choose the configuration with the best validation macro F0.5.

---

## 19. Phase 12 — Decision Layer

Add only validated safeguards for:

- short names
- common names
- shared addresses
- strong address conflicts
- alias/DBA cases
- missing fields
- source-specific behavior

Rules must distinguish:

```text
missing != contradiction
```

and:

```text
high name similarity != automatic match
```

Do not force a match for every S1. Allow multiple matches when multiple candidates pass the validated criteria.

---

## 20. Phase 13 — Error Analysis

### False positives

Classify into:

- common-name collision
- short-name collision
- shared address
- same name/different location
- landmark collision
- legal-suffix confusion
- alias/acronym confusion
- corporate-family confusion
- normalization collision
- missing-field false evidence

### False negatives

Classify into:

- transliteration
- native-script mismatch
- mixed-script record
- typo
- joined/split token
- DBA/alias
- missing address
- partial address
- reordered address
- source-specific corruption
- blocking failure

Fix the layer responsible for the error.

---

## 21. Phase 14 — Test Inference

Run:

```text
test S1/S2/S3
 -> normalization
 -> blocking
 -> candidate union
 -> pre-filter
 -> features
 -> trained model
 -> decision
 -> outputs
```

Train the final model on all labeled training data before test inference.

---

## 22. Output Requirements

### matching_results.tsv

Columns:

```text
source1_entity_id
matched_entity_ids
```

Requirements:

- every test S1 entity appears exactly once
- empty match list allowed
- no duplicate target IDs
- only valid S2/S3 IDs
- multiple targets allowed

### candidate_pairs.tsv

Contains the final candidate set immediately before the final matching model.

Every final predicted match must occur in this file.

---

## 23. Validation Command

```bash
python3 utils/validate_submission.py     --matching output/matching_results.tsv     --candidate output/candidate_pairs.tsv     --test-dir dataset/test
```

Validator success proves structural validity, not model quality.

---

## 24. Optimization Loop

```text
Baseline
   |
Measure blocking recall
   |
Measure macro F0.5
   |
Error analysis
   |
   +--> blocking failure --> improve blocking
   |
   +--> normalization failure --> improve normalization
   |
   +--> feature/model failure --> improve features/model
   |
   +--> false positives --> improve decision
   |
   +--> false negatives --> improve recall
   |
Re-evaluate
```

Track every experiment with its blocking configuration, features, model, threshold, blocking recall, macro F0.5 and observations.

---

## 25. Recommended Build Order

```text
1. loader.py
2. validation.py
3. audit.py
4. unicode.py
5. scripts.py
6. names.py
7. addresses.py
8. transliteration.py
9. aliases.py
10. exact_name.py
11. rare_token.py
12. char_ngram.py
13. address.py
14. numeric.py
15. transliteration.py
16. candidate_generator.py
17. blocking_eval.py
18. feature modules
19. validation.py
20. train.py
21. threshold.py
22. predict.py
23. error_analysis.py
24. pipeline.py
```

---

## 26. Final Acceptance Checklist

- [ ] TSV loading works.
- [ ] Ground truth integrity verified.
- [ ] Raw values preserved.
- [ ] Unicode normalization works.
- [ ] Mixed scripts detected.
- [ ] Transliteration representation works.
- [ ] Legal suffixes handled separately.
- [ ] DBA/FKA/aliases detected.
- [ ] Numeric address information preserved.
- [ ] Missing values distinguished from contradictions.
- [ ] Multiple blockers implemented.
- [ ] Blocking recall measured.
- [ ] Candidate count measured.
- [ ] Hard negatives included.
- [ ] Entity-level train/validation split used.
- [ ] Macro F0.5 threshold optimized.
- [ ] Zero-match entities supported.
- [ ] Multi-match entities supported.
- [ ] Final matches are a subset of final candidates.
- [ ] Every test S1 appears exactly once.
- [ ] No duplicate targets.
- [ ] Only valid S2/S3 IDs output.
- [ ] Official validator passes.
- [ ] README and requirements are complete.
