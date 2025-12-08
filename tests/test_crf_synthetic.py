"""
Synthetic Experiment for CRF Model (Probabilistic Baseline)
"""

import sys
from pathlib import Path

# Ensure project imports work when running script standalone
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models.crf_model import CRFTagger
from src.features_crf import sent2features
from src.metrics import span_f1
from src.synthetic_data import (
    generate_challenging_train_set,
    generate_challenging_test_set,
)

# =========================================================
# 1. Load Synthetic Dataset
# =========================================================

train_data = generate_challenging_train_set(num_examples=80)
test_data = generate_challenging_test_set()


def prepare(dataset):
    X, y = [], []
    for tokens, labels in dataset:

        # ---- Safety: Fix empty or missing labels ----
        if len(labels) != len(tokens):
            print(f"⚠️ Fixing label length mismatch:\nTokens={tokens}\nLabels={labels}")
            labels = labels + ["O"] * (len(tokens) - len(labels))

        labels = ["O" if (lbl is None or lbl.strip() == "") else lbl for lbl in labels]

        X.append(sent2features(tokens))
        y.append(labels)

    return X, y


# =========================================================
# 2. Prepare CRF training features
# =========================================================

X_train, y_train = prepare(train_data)
X_test, y_test = prepare(test_data)

# =========================================================
# 3. Train CRF Model
# =========================================================

print("\n Training CRF on synthetic dataset...")

crf = CRFTagger(c1=0.1, c2=0.1, max_iterations=300)
crf.fit(X_train, y_train)

# =========================================================
# 4. Evaluate the Model
# =========================================================

predictions = crf.predict(X_test)
y_pred = [list(seq) for seq in predictions]

precision, recall, f1 = span_f1(y_test, y_pred)

print("\n===== Synthetic CRF Evaluation Results =====")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")
print("============================================")

# =========================================================
# 5. Qualitative Inspect — Required
# =========================================================

print("\n Sample Predictions:\n")

for (tokens, gold), pred in zip(test_data[:10], y_pred[:10]):
    print("Sentence:", " ".join(tokens))
    print("Gold:    ", gold)
    print("Pred:    ", pred)
    print("-" * 50)
