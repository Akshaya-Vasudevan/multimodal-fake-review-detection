# Multimodal Fake Review Detection

This project implements a modular multimodal machine learning pipeline for detecting fake reviews using both textual content and metadata features.

---

## Problem Statement

Fake reviews significantly impact consumer trust in e-commerce platforms.  
This project explores whether combining review text and metadata improves detection performance compared to text-only models.

---

## Dataset

Amazon Handmade Products Review Dataset [Click the link to download]: https://amazon-reviews-2023.github.io/

Due to size constraints, the dataset is not included in this repository.

After downloading, place the files inside:

```
data/
    Handmade_Products.jsonl
    meta_Handmade_Products.jsonl
```

---

## Project Structure

```
src/
    text_pipeline.py
    metadata_pipeline.py
    fusion_model.py

notebooks/
    demo.ipynb
```

- `text_pipeline.py` → Text preprocessing & TF-IDF feature extraction  
- `metadata_pipeline.py` → Metadata feature engineering & scaling  
- `fusion_model.py` → Multimodal feature fusion and classification  
- `demo.ipynb` → End-to-end training and evaluation workflow  

---

## Methodology

1. Text features extracted using TF-IDF.
2. Metadata features include:
   - Rating
   - Review length
   - Helpful votes
   - Verified purchase flag
3. Features are fused using sparse matrix concatenation.
4. Classification performed using SGD-based logistic regression.

---

## Results

### Text-Only Model
- Poor recall for fake reviews
- Struggled under class imbalance

### Multimodal Model
- Significant improvement in recall and F1-score
- Demonstrated effectiveness of combining metadata with textual signals

---

## Installation

Install dependencies:

```
pip install -r requirements.txt
```

---

## Running the Project

Open:

```
notebooks/demo.ipynb
```

Run cells sequentially to:
- Load data
- Generate features
- Train multimodal model
- Evaluate performance

---

## Contributors

- Fusion & Integration: Akshaya Vasudevan [CB.AI.U4AID23064]
- Text Pipeline Module: Gowri J S [CB.AI.U4AID23055]
- Metadata Pipeline Module: Karishini S [CB.AI.U4AID23013]
