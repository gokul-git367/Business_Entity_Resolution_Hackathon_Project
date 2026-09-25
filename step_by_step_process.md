# Business Entity Resolution — Step-by-Step Process Explanation

## 1. What We Are Solving

We have three independent business data sources.

- Source 1 is the reference set.
- Source 2 and Source 3 contain noisy records.
- There is no shared real-world business ID.
- One Source 1 entity can have zero, one, or many matches.

Our task is:

> For every Source 1 business, identify the complete set of Source 2 and Source 3 records that refer to the same real-world business.

This is not simple string matching.

---

## 2. Why the Dataset Is Difficult

The training audit confirms:

### Name problems

- typos
- punctuation variation
- abbreviations
- legal suffixes
- DBA/trade names
- aliases
- word reordering
- token joining/splitting
- repeated tokens
- acronyms
- different scripts
- transliteration

### Address problems

- missing components
- abbreviations
- reordered components
- landmarks
- missing postal codes
- transliteration
- numeric variation
- partial addresses

Therefore one similarity function cannot solve the problem.

---

## 3. The Five Questions Behind the Architecture

### Question 1: Can the records be made comparable?

**Normalization.**

### Question 2: Which records are worth comparing?

**Blocking.**

### Question 3: How similar are the records?

**Feature engineering.**

### Question 4: Does the evidence indicate the same business?

**Matching model.**

### Question 5: Is the evidence strong enough to output?

**Decision layer / threshold.**

---

# PART A — UNDERSTAND THE DATA

## 4. Step 1 — Load and Validate

Load:

```text
train_source1.tsv
train_source2.tsv
train_source3.tsv
train_ground_truth.tsv
```

and later:

```text
test_source1.tsv
test_source2.tsv
test_source3.tsv
```

Use:

```python
pd.read_csv(file, sep="\t")
```

Validate columns, IDs, duplicates, malformed rows and ground-truth references.

### Why?

A parsing or ID error contaminates every later stage.

---

## 5. Step 2 — Perform Dataset Audit

Measure:

- row counts
- missing names
- missing addresses
- scripts
- mixed scripts
- Unicode issues
- typos
- aliases
- legal suffixes
- address patterns
- duplicate names
- duplicate addresses
- zero/one/many match counts

### Why?

We design the solution from the real data rather than assumptions.

The ambiguity audit already confirms multilingual scripts, mixed scripts, Unicode complexity, typos, joined tokens, aliases, partial addresses, shared names/addresses and source-specific corruption.

---

## 6. Step 3 — Analyze Ground Truth

Turn:

```text
S1-001 -> S2-10,S2-11,S3-20
```

into three labeled pairs.

Calculate:

- zero-match S1 entities
- singleton S1 entities
- multi-match S1 entities
- S2-only matches
- S3-only matches
- mixed-source matches

### Why?

This determines the entity structure the model must support.

---

# PART B — REPRESENT THE RECORDS PROPERLY

## 7. Step 4 — Preserve Raw Data

Never overwrite:

```text
business_name
business_address
```

Create additional fields.

For name:

```text
raw
unicode
clean
tokens
core
transliterated
n-grams
```

For address:

```text
raw
unicode
clean
tokens
numbers
postal
city
state
locality
street
unit
```

### Why?

Different representations solve different corruption types.

---

## 8. Step 5 — Normalize Unicode

Normalize Unicode and whitespace.

Handle:

- zero-width characters
- unusual spaces
- combining characters
- special punctuation

Keep the original value.

### Why?

Two visually identical strings can have different internal representations.

---

## 9. Step 6 — Detect Scripts

Calculate script proportions for each field.

Example:

```text
ABC ट्रेडर्स
```

contains both Latin and Devanagari.

### Why?

Normal string similarity can fail across scripts, and one record may contain multiple scripts.

Do not assume one record has one language.

---

## 10. Step 7 — Transliteration

Keep:

```text
native/original
+
transliterated representation
```

Compare both.

### Why?

This helps connect native-script and Latin representations.

But transliteration is only supporting evidence.

---

## 11. Step 8 — Normalize Names

Separate the identity-bearing core from legal structure.

Example:

```text
ABC Traders Pvt Ltd
```

becomes conceptually:

```text
core_name = ABC Traders
legal_suffix = Pvt Ltd
```

Also detect:

```text
dba
d/b/a
fka
```

### Why?

Legal suffixes and alias markers should not be treated like ordinary business-name tokens.

---

## 12. Step 9 — Normalize Addresses

Extract address components.

Preserve numbers such as:

```text
12-5-35/C/47
Flat 402
37/4
560001
```

### Why?

Numeric components can strongly distinguish businesses sharing the same road/city.

---

## 13. Step 10 — Treat Missing Values Correctly

If a postal code is missing:

```text
unknown
```

not:

```text
different
```

### Why?

Missing information should not create artificial negative evidence.

---

# PART C — FIND CANDIDATES

## 14. Step 11 — Understand Blocking

Suppose millions of possible pairs exist.

We do not compare everything.

For each S1 record, blocking finds a smaller set of plausible S2/S3 records.

### Why?

It makes the problem computationally manageable.

More importantly:

> A true match missed by blocking can never be recovered by the matcher.

Therefore blocking must be recall-oriented.

---

## 15. Step 12 — Exact Core-Name Blocking

Example:

```text
ABC Traders Pvt Ltd
```

becomes:

```text
abc traders
```

Retrieve records with the same core name.

### Good for

Clean records.

### Bad for

Typos, aliases, transliteration and spelling changes.

---

## 16. Step 13 — Rare-Token Blocking

Build:

```text
token -> record IDs
```

Give more importance to rare tokens.

Example:

```text
services -> huge number of records
xyzmedical -> very few
```

### Why?

Rare tokens identify records more precisely.

---

## 17. Step 14 — Character N-Gram Blocking

Use character n-grams to retrieve similar names.

### Good for

- typos
- joined words
- split words
- punctuation changes
- small spelling changes

### Why?

It is less dependent on correct word boundaries.

---

## 18. Step 15 — Address Blocking

Use:

```text
city
state
locality
street
postal
```

and combinations.

### Why?

A corrupted name can sometimes be rescued by address evidence.

Do not make every address component mandatory.

---

## 19. Step 16 — Postal/Numeric Blocking

Use:

```text
postal
postal + house number
house number + locality
unit + postal
```

### Why?

These combinations can be highly selective.

But they should not be hard requirements for final matching because address information can be missing or noisy.

---

## 20. Step 17 — Transliteration Blocking

Use transliterated name representations.

### Why?

It can recover native-script versus Latin-script matches that ordinary character blocking misses.

---

## 21. Step 18 — Optional Phonetic Blocking

Test phonetic methods on Latin-heavy data.

Keep them only if experiments show real benefit.

### Why?

English-oriented phonetic methods are not guaranteed to work well across all scripts and languages.

---

## 22. Step 19 — Union the Blocking Results

Suppose:

```text
Exact name -> A,B
Rare token -> B,C,D
N-gram -> A,C,E
Address -> D,E
```

Union:

```text
A,B,C,D,E
```

### Why?

Different blockers fail on different corruption types.

---

## 23. Step 20 — Measure Blocking Recall

For every true training match:

> Was this pair generated by blocking?

Calculate recall for:

- each blocker
- blocker combinations
- final union

Also measure recall by:

- source
- country
- script
- ambiguity category

### Why?

This tells us which blockers actually protect true matches.

---

## 24. Step 21 — Make Blocking More Precise

If candidate counts become too large:

1. Prefer rare tokens.
2. Use top-k n-gram retrieval.
3. Combine weak keys with stronger keys.
4. Use postal/numeric information where available.
5. Use source/country context as soft retrieval information.
6. Add a permissive candidate pre-filter.

Do not tighten blocking until true-match recall is measured.

---

# PART D — DECIDE WHETHER CANDIDATES MATCH

## 25. Step 22 — Build Pair Features

For every S1/candidate pair calculate:

### Name evidence

- exact equality
- normalized equality
- core-name equality
- edit similarity
- Jaro-Winkler
- n-gram similarity
- token Jaccard
- weighted token overlap
- TF-IDF cosine
- transliteration similarity

### Address evidence

- normalized similarity
- token overlap
- postal match
- city/state/locality match
- street match
- house/unit match
- numeric overlap

### Other evidence

- script information
- missingness
- rarity
- source
- country
- alias indicators
- contradiction indicators

---

## 26. Step 23 — Why We Need Multiple Features

Consider:

```text
ABC Traders
```

and:

```text
ABC Traders
```

Name similarity is 1.0.

But if the addresses are:

```text
12 MG Road
```

versus:

```text
77 MG Road
```

the name alone is not enough.

Another case:

```text
native-script name
```

versus:

```text
Latin transliteration
```

may have low raw similarity but high transliteration similarity.

Different features solve different problems.

---

## 27. Step 24 — Add Rarity

Common names should be treated differently from rare names.

For example:

```text
ABC Services
```

is less informative than:

```text
RareTech Medical Systems
```

### Why?

A common token matching another common token is weak evidence.

A rare token matching the same rare token is stronger evidence.

---

## 28. Step 25 — Add Missingness and Contradiction

Distinguish:

```text
address missing
```

from:

```text
address conflicts
```

For example:

```text
name similarity = 0.98
address missing
```

is not the same as:

```text
name similarity = 0.98
house number conflicts
postal conflicts
city conflicts
```

This helps prevent false merges.

---

## 29. Step 26 — Build Hard Negatives

Find real nonmatches that look similar:

```text
same name
same address
same postal
same city
high name similarity
high address similarity
```

### Why?

These are the examples that teach the model what a dangerous false match looks like.

---

# PART E — TRAIN AND DECIDE

## 30. Step 27 — Split by S1 Entity

Do not randomly split candidate pairs.

Split Source 1 entities into train and validation.

### Why?

Otherwise candidate pairs belonging to the same entity can leak across the split.

---

## 31. Step 28 — Train a Matching Model

Start with a gradient-boosting model:

```text
LightGBM / XGBoost / CatBoost
```

Input:

```text
pair features
```

Output:

```text
P(match)
```

### Why?

The input is structured/tabular evidence.

A large language model is not the natural first choice for millions of structured candidate pairs.

---

## 32. Step 29 — Tune the Threshold

Do not automatically use:

```text
P(match) > 0.5
```

Try multiple thresholds.

For every threshold:

```text
generate entity-level predictions
calculate macro F0.5
```

Choose the best validation configuration.

---

## 33. Step 30 — Handle Zero Matches

If no candidate is sufficiently convincing:

```text
S1-100
```

can legitimately have an empty prediction.

### Why?

The challenge allows zero matches.

Forcing the highest-scoring candidate creates false positives.

---

## 34. Step 31 — Handle Multiple Matches

If several candidates are independently strong:

```text
S2-10
S2-11
S3-20
```

can all be output.

### Why?

The task is not one-to-one entity matching.

---

# PART F — IMPROVE SYSTEMATICALLY

## 35. Step 32 — Analyze False Positives

Ask:

```text
Why did we merge these?
```

Possible reasons:

- common name
- short name
- shared address
- shared landmark
- corporate-family confusion
- legal suffix confusion
- normalization collision
- missing information treated as evidence

Then fix the relevant feature/decision layer.

---

## 36. Step 33 — Analyze False Negatives

Ask:

```text
Why did we miss these?
```

Possible reasons:

- transliteration
- native script
- typo
- token joining
- DBA
- missing address
- partial address
- reordered address
- source-specific corruption
- blocking failure

Then fix the relevant layer.

---

## 37. Step 34 — Know Which Layer to Fix

If the true match never entered the candidate set:

```text
BLOCKING problem
```

If it entered but the model scored it incorrectly:

```text
FEATURE/MODEL problem
```

If the score was correct but threshold rejected it:

```text
DECISION problem
```

If normalization destroyed useful information:

```text
NORMALIZATION problem
```

This prevents random changes to the entire system.

---

# PART G — FINAL PIPELINE

## 38. Step 35 — Train Final Model

After validation:

1. Freeze preprocessing.
2. Freeze blocking.
3. Freeze features.
4. Freeze model configuration.
5. Freeze threshold/decision rules.
6. Train using all labeled training data.

---

## 39. Step 36 — Run Test Inference

```text
test data
  ↓
normalization
  ↓
blocking
  ↓
candidate union
  ↓
candidate pre-filter
  ↓
pair features
  ↓
trained matcher
  ↓
decision layer
  ↓
outputs
```

---

## 40. Step 37 — Produce `candidate_pairs.tsv`

This must be the final candidate set immediately before the final matching model.

Every predicted match must exist in it.

---

## 41. Step 38 — Produce `matching_results.tsv`

Every test S1 entity must appear exactly once.

Example:

```text
source1_entity_id    matched_entity_ids
S1-001               S2-145,S3-892
S1-002
S1-003               S2-90
```

Empty lists are valid.

---

## 42. Step 39 — Run the Official Validator

```bash
python3 utils/validate_submission.py     --matching output/matching_results.tsv     --candidate output/candidate_pairs.tsv     --test-dir dataset/test
```

Remember:

> Passing the validator means the files satisfy structural requirements. It does not prove high matching accuracy.

---

# 43. Final Development Strategy

Build in this order:

```text
DATA
 ↓
AUDIT
 ↓
NORMALIZATION
 ↓
BLOCKING V1
 ↓
BLOCKING RECALL
 ↓
BLOCKING V2/V3
 ↓
PAIR FEATURES
 ↓
HARD NEGATIVES
 ↓
MATCHER
 ↓
F0.5 THRESHOLD
 ↓
ERROR ANALYSIS
 ↓
IMPROVEMENT
 ↓
FINAL TEST
 ↓
VALIDATION
```

Do not build everything at once.

---

# 44. The Most Important Rule

The objective is not:

> Find the most similar string.

The objective is:

> **Recover the complete set of records representing the same real-world business while avoiding false merges.**

That is why the final system must combine:

```text
Name
+
Address
+
Script
+
Transliteration
+
Numbers
+
Rarity
+
Context
+
Entity multiplicity
+
Ambiguity handling
```

---

# 45. Final Mental Model

```text
              RAW RECORDS
                   |
                   v
             UNDERSTAND DATA
                   |
                   v
        KEEP RAW + NORMALIZED VIEWS
                   |
        +----------+----------+
        |          |          |
       NAME     ADDRESS     SCRIPT
        |          |          |
        +----------+----------+
                   |
                   v
           MULTIPLE BLOCKERS
                   |
                   v
             CANDIDATES
                   |
                   v
            PAIR FEATURES
                   |
                   v
          MATCHING CLASSIFIER
                   |
                   v
           DECISION / THRESHOLD
                   |
          +--------+--------+
          |                 |
       NO MATCH           MATCHES
          |                 |
          +--------+--------+
                   |
                   v
          FINAL S1 OUTPUT
```
