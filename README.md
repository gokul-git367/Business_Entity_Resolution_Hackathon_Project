# Business Entity Resolution Pipeline

A resume-friendly project for resolving duplicate or fragmented business records across multiple noisy data sources. The system connects records that refer to the same real-world company even when names, addresses, and spelling conventions differ across datasets.

## Overview

This project tackles a real-world entity resolution problem in which business records from different sources must be matched to the same underlying organization. In practice, this is difficult because the same company may appear with:

- Name variations such as abbreviations, legal suffix differences, punctuation changes, transliterations, and spelling inconsistencies
- Address differences such as missing postal codes, shortened street names, city/state formatting differences, or landmark-based references
- Multi-source noise where the same business appears in different scripts or with different formatting conventions

The solution combines cleaning, normalization, blocking, feature engineering, and a supervised learning model to identify true matches while minimizing false positives.

## What We Achieved

This project is designed to go beyond simple rule-based matching and build a practical ML pipeline for real-world entity resolution.

- Built an end-to-end entity matching system for three noisy business datasets
- Implemented robust preprocessing for business names and addresses to normalize inconsistent records
- Added transliteration-aware and script-aware matching to handle names from different writing systems
- Designed a multi-route blocking strategy to keep candidate generation high-recall without exploding the search space
- Engineered domain-specific pairwise features covering name similarity, address similarity, rarity, context, and contradiction signals
- Trained a LightGBM classifier to classify whether two records refer to the same business
- Optimized the decision threshold for the challenge metric (macro F0.5), balancing precision and recall for real-world matching
- Delivered a reproducible pipeline with validation and output generation for submission-ready results

## Project Impact

The model is built for the kinds of problems commonly found in data quality, customer deduplication, and enterprise identity matching:

- Deduplicating customer or business records across internal and external sources
- Matching vendor or supplier records across fragmented systems
- Improving master data quality and reducing duplicate entities in operations teams
- Enabling scalable resolution workflows in noisy, real-world datasets

## Solution Architecture

The system follows a standard but high-performing entity resolution workflow:

1. Data loading and validation
2. Name/address normalization and noise reduction
3. Candidate generation using blocking strategies
4. Pairwise feature construction
5. Train/validate a binary matching model
6. Tune threshold and generate final outputs

### Core components

- Data loading and validation for Source 1, Source 2, and Source 3 records
- Unicode and script normalization
- Name cleaning and legal suffix handling
- Address normalization and postal/component extraction
- Transliteration support for multilingual name matching
- Candidate generation using exact and approximate blocking routes
- Pair-feature extraction and LightGBM-based classification

## Tech Stack

- Python
- Pandas
- NumPy
- LightGBM
- Scikit-learn style workflow patterns for feature engineering and evaluation

## Repository Structure

```text
.
├── README.md
├── Documentation_template.md
├── ambiguity_audit.md
├── implementation.md
├── step_by_step_process.md
├── Abiguity.txt
├── code/
│   └── business_entity_resolution/
│       ├── pipeline.py
│       ├── requirements.txt
│       ├── README.md
│       └── src/
│           ├── data/
│           ├── normalization/
│           ├── blocking/
│           ├── features/
│           ├── model/
│           └── evaluation/
└── utils/
```

## How the Pipeline Works

The project uses an ML-first approach rather than a hardcoded rule set.

- Candidate generation finds likely matches using multiple blocking signals
- Feature engineering compares names and addresses in a structured way
- Model scoring classifies each candidate pair as either a match or non-match
- Final thresholding decides which candidates are accepted as valid matches
- Output files are produced in the format required for final prediction submission

This makes the project suitable for both research and practical deployment scenarios where entity matching quality matters.

## Run the Project

From the project root:

```bash
cd code/business_entity_resolution
pip install -r requirements.txt
python pipeline.py --mode train
```

To run full training and generate final predictions:

```bash
python pipeline.py --mode full
```

To run inference using saved artifacts:

```bash
python pipeline.py --mode test
```

## Summary

This repository showcases a complete, end-to-end business entity resolution pipeline built to handle noisy, fragmented, multi-source records. Instead of just satisfying a challenge specification, the project focuses on the actual business value: identifying the same company across imperfect datasets using machine learning, normalization, and careful candidate evaluation.

