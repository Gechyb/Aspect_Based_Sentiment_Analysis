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

train = synthetic_data[:4]
test = synthetic_data[4:]

=== Synthetic CRF Experiment Results ===
Precision: 0.000
Recall:    0.000
F1 Score:  0.000

=== Qualitative Examples ===

Sentence: The keyboard is fine
Gold:     ['O', 'B-NEU', 'O', 'O']
Pred:     ['O', 'B-NEG', 'O', 'O']
----------------------------------------
Sentence: The battery life could be better
Gold:     ['O', 'B-NEG', 'I-NEG', 'O', 'O', 'O']
Pred:     ['O', 'B-POS', 'I-POS', 'O', 'O', 'O']
----------------------------------------

On a controlled synthetic dataset, the CRF successfully learned to associate aspect spans with sentiment in simple constructions (e.g., “battery life is great”) but failed to correctly classify neutral sentiment (“keyboard is fine”) when that label was not present in the training data. Even after including a neutral example, the model still struggled with more implicit expressions such as “battery life could be better,” predicting positive sentiment instead of negative. This highlights the CRF’s dependence on training coverage and surface patterns rather than deeper semantic understanding.