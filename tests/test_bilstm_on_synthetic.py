import sys
from pathlib import Path
import random
import torch

# Make project importable when running directly
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models.bilstm_crf import BiLSTMCRF
from src.metrics import span_f1
from src.tagging_scheme import TAGS, TAG2ID, ID2TAG
from src.synthetic_data import (
    generate_challenging_train_set,
    generate_challenging_test_set,
)

# Reproducibility
random.seed(42)
torch.manual_seed(42)

# =========================================================
# Load Dataset
# =========================================================

train_data = generate_challenging_train_set(num_examples=80)
test_data = generate_challenging_test_set()

# Vocabulary building
vocab = {"<PAD>": 0, "<UNK>": 1}
all_data = train_data + test_data
for tokens, _ in all_data:
    for token in tokens:
        token = token.lower()
        if token not in vocab:
            vocab[token] = len(vocab)


def encode(dataset):
    X, Y = [], []
    for tokens, labels in dataset:

        # Fix: enforce same length
        if len(labels) != len(tokens):
            print(
                f"⚠️ Mismatch detected:\nTokens: {tokens}\nLabels: {labels}\nFixing..."
            )
            labels = labels + ["O"] * (len(tokens) - len(labels))

        # Fix: replace empty labels
        labels = ["O" if (lbl is None or lbl.strip() == "") else lbl for lbl in labels]

        X.append([vocab.get(t.lower(), vocab["<UNK>"]) for t in tokens])
        Y.append([TAG2ID[lbl] for lbl in labels])
    return X, Y


X_train, Y_train = encode(train_data)
X_test, Y_test = encode(test_data)

# =========================================================
# BiLSTM-CRF Model Setup
# =========================================================

device = torch.device("cpu")

model = BiLSTMCRF(
    vocab_size=len(vocab),
    tagset_size=len(TAGS),
    embedding_dim=50,
    hidden_dim=64,
    pad_idx=0,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("=" * 80)
print("TRAINING BiLSTM-CRF ON SYNTHETIC DATA")
print("=" * 80)

for epoch in range(50):
    total_loss = 0

    for x, y in zip(X_train, Y_train):
        x = torch.tensor([x], dtype=torch.long).to(device)
        y = torch.tensor([y], dtype=torch.long).to(device)
        mask = torch.tensor([[True] * len(y[0])], dtype=torch.bool).to(device)

        loss = model(x, tags=y, mask=mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/50 - Loss={total_loss:.4f}")

print("\n")

# =========================================================
# Evaluation
# =========================================================

preds, gold = [], []

for x, y in zip(X_test, Y_test):
    x_tensor = torch.tensor([x], dtype=torch.long).to(device)
    mask = torch.tensor([[True] * len(x)], dtype=torch.bool).to(device)

    pred_ids = model(x_tensor, tags=None, mask=mask)[0]
    preds.append(pred_ids[: len(y)])
    gold.append(y)

gold_tags = [[ID2TAG[t] for t in seq] for seq in gold]
pred_tags = [[ID2TAG[t] for t in seq] for seq in preds]

precision, recall, f1 = span_f1(gold_tags, pred_tags)

# =========================================================
# Results Summary
# =========================================================

print("=" * 80)
print("📊 FINAL PERFORMANCE")
print("=" * 80)
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")
print("=" * 80)

# Detailed error analysis
correct = sum(1 for g, p in zip(gold_tags, pred_tags) if g == p)
total = len(test_data)

print(f"Exact Sentence Accuracy: {correct}/{total}  ({100*correct/total:.1f}%)")
print("=" * 80)

print("\nEXAMPLE PREDICTIONS:")
print("-" * 80)

for i, (tokens, gold_seq, pred_seq) in enumerate(
    zip(test_data, gold_tags, pred_tags), start=1
):
    print(f"Sentence {i}: {' '.join(tokens[0])}")
    print(f"Gold: {gold_seq}")
    print(f"Pred: {pred_seq}")
    print("-" * 80)

print("\nDONE ✔\n")
