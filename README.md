# Aspect_Based_Sentiment_Analysis


The data is from SemEval 2014


run train_crf
```bash
 python -m src.train_crf --domain restaurants
```

baseline results for train_crf

Precision = 0.51

When the CRF predicts an aspect-sentiment span, it’s correct about 51% of the time.

Recall = 0.372

It only finds 37.2% of all the true aspect-sentiment spans in the test set.
So it’s missing many aspects (low recall).

F1 = 0.43

Overall balance between precision & recall is 0.43 — a reasonable baseline for a simple CRF with only local features and no POS/embeddings.

This is exactly the kind of “baseline” result you want before building the BiLSTM-CRF and before doing feature engineering.


Created test_crf_synthetic.py 
Synthetic Experiment for Step 3a (Probabilistic Model)
-------------------------------------------------------

This script creates a tiny synthetic dataset for ABSA-style
BIO + sentiment labeling and trains a CRF model on it.

Purpose:
- Demonstrate CRF behavior in controlled settings
- Show where CRF performs well and where it struggles
- Required for Step 3a ("probabilistic/generative model on synthetic data")

run test_crf_synthetic.py
```bash
python tests/test_crf_synthetic.py
```