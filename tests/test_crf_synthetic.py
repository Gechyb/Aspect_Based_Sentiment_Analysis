"""
Synthetic Experiment for Probabilistic Model
"""

import sys
from pathlib import Path

# Add project root so `src` package is importable when running this file directly
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.crf_model import CRFTagger
from src.features_crf import sent2features
from src.metrics import span_f1

# 1. Synthetic dataset (clean, controlled)

# Each example: (tokens, labels)
synthetic_data = [
    # Positive sentiment examples
    (["The", "battery", "life", "is", "great"], ["O", "B-POS", "I-POS", "O", "O"]),
    (["Amazing", "battery", "quality"], ["O", "B-POS", "I-POS"]),
    # Negative sentiment examples
    (["The", "screen", "is", "terrible"], ["O", "B-NEG", "O", "O"]),
    (["Battery", "life", "is", "awful"], ["B-NEG", "I-NEG", "O", "O"]),
    # Neutral / mixed
    (["The", "keyboard", "is", "fine"], ["O", "B-NEU", "O", "O"]),
    # Hard example (hidden sentiment)
    (
        ["The", "battery", "life", "could", "be", "better"],
        ["O", "B-NEG", "I-NEG", "O", "O", "O"],
    ),
]

# Split into train/test manually (small dataset)
train = synthetic_data[:4]
test = synthetic_data[4:]


def prepare(dataset):
    X, y = [], []
    for tokens, labels in dataset:
        X.append(sent2features(tokens))
        y.append(labels)
    return X, y


# 2. Prepare features/labels
X_train, y_train = prepare(train)
X_test, y_test = prepare(test)

# 3. Train CRF
crf = CRFTagger(c1=0.1, c2=0.1, max_iterations=100)
crf.fit(X_train, y_train)

# 4. Predict & Evaluate
y_pred = crf.predict(X_test)
p, r, f = span_f1(y_test, y_pred)

print("\n=== Synthetic CRF Experiment Results ===")
print(f"Precision: {p:.3f}")
print(f"Recall:    {r:.3f}")
print(f"F1 Score:  {f:.3f}")

# 5. Qualitative inspection (required)
print("\n=== Qualitative Examples ===\n")

for (tokens, gold), pred in zip(test, y_pred):
    print("Sentence:", " ".join(tokens))
    print("Gold:    ", gold)
    print("Pred:    ", pred)
    print("-" * 40)
