# src/train_bilstm_crf.py

import argparse
from pathlib import Path
from collections import Counter
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from tabulate import tabulate
from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from .config import Config
from .data_utils import load_sentences, train_val_test_split
from .models.bilstm_crf import BiLSTMCRF
from .tagging_scheme import TAGS, TAG2ID, ID2TAG
from .glove_utils import load_embeddings_for_vocab
from .metrics import normalize_predictions

# from .results_logger import log_results
from TorchCRF import CRF


class ABSASeqDataset(Dataset):
    """PyTorch Dataset wrapping token/label sequences for ABSA tagging."""

    def __init__(self, data, vocab):
        """
        data: list of dicts like {"tokens": [...], "labels": [...]}
        vocab: dict mapping token(lowercased) -> int
        """
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        x = [self.vocab.get(t.lower(), self.vocab["<unk>"]) for t in ex["tokens"]]
        y = [TAG2ID[t] for t in ex["labels"]]
        return torch.tensor(x), torch.tensor(y)


def build_vocab(data, min_freq=1):
    """Build a simple token -> id vocab from training sentences."""
    cnt = Counter()
    for s in data:
        for t in s["tokens"]:
            cnt[t.lower()] += 1

    # Reserve 0 for <pad>, 1 for <unk>
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, c in cnt.items():
        if c >= min_freq:
            vocab[w] = len(vocab)
    return vocab


def collate(batch):
    """
    Collate function to:
    - pad token sequences with <pad> id (0)
    - pad label sequences with PAD label id
    - construct mask indicating real tokens (True) vs padding (False)
    """
    xs, ys = zip(*batch)
    maxlen = max(len(x) for x in xs)

    pad_x = torch.zeros(len(xs), maxlen, dtype=torch.long)  # <pad> id = 0
    pad_y = torch.full(
        (len(xs), maxlen), TAG2ID["PAD"], dtype=torch.long
    )  # labels padded with PAD
    mask = torch.zeros(len(xs), maxlen, dtype=torch.bool)

    for i, (x, y) in enumerate(zip(xs, ys)):
        L = len(x)
        pad_x[i, :L] = x
        pad_y[i, :L] = y
        mask[i, :L] = True

    return pad_x, pad_y, mask


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, dataloader, device):
    """
    Span-level evaluation using seqeval.

    This computes precision/recall/F1 over BIO entity spans,
    not per-token accuracy, and excludes 'O' dominance by using
    the standard CoNLL-style entity matching.
    """
    model.eval()
    all_true, all_pred = [], []

    with torch.no_grad():
        for x, y, mask in dataloader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)

            # CRF decode: list of paths, already trimmed by mask internally
            preds = model(x, tags=None, mask=mask)

            for pred_ids, gold_ids, mask_row in zip(preds, y, mask):
                L = int(mask_row.sum().item())  # real length

                gold_tags = [ID2TAG[int(t)] for t in gold_ids[:L].tolist()]

                fixed_pred_ids = normalize_predictions(pred_ids, L)
                pred_tags = [ID2TAG[int(p)] for p in fixed_pred_ids]

                all_true.append(gold_tags)
                all_pred.append(pred_tags)

    p = precision_score(all_true, all_pred, zero_division=0)
    r = recall_score(all_true, all_pred, zero_division=0)
    f = f1_score(all_true, all_pred, zero_division=0)
    report = classification_report(all_true, all_pred, zero_division=0)
    return p, r, f, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain", default="restaurants", choices=["restaurants", "laptops"]
    )
    parser.add_argument(
        "--use_glove", action="store_true", help="Use pre-trained GloVe embeddings"
    )
    args = parser.parse_args()

    cfg = Config(domain=args.domain)

    # Seed everything
    set_seed(cfg.seed)
    print(f"Random seed set to: {cfg.seed}")

    # Load data
    path = Path(cfg.data_dir) / f"{cfg.domain}.jsonl"
    data = load_sentences(path)
    train, val, test = train_val_test_split(data, seed=cfg.seed)

    # Vocab from train only
    vocab = build_vocab(train, min_freq=1)
    print(f"Vocabulary size: {len(vocab)}")

    # Load GloVe embeddings if requested
    pretrained_embeddings = None
    if args.use_glove:
        try:
            pretrained_embeddings = load_embeddings_for_vocab(
                vocab, cfg.glove_dir, embedding_dim=cfg.embedding_dim
            )
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            print("Continuing with random embeddings...")
            pretrained_embeddings = None

    train_ds = ABSASeqDataset(train, vocab)
    val_ds = ABSASeqDataset(val, vocab)
    test_ds = ABSASeqDataset(test, vocab)

    train_dl = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate
    )
    val_dl = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate
    )
    test_dl = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BiLSTMCRF(
        vocab_size=len(vocab),
        tagset_size=len(TAGS),
        embedding_dim=cfg.embedding_dim,
        hidden_dim=cfg.hidden_dim,
        pad_idx=0,
        dropout=cfg.dropout,
        pretrained_embeddings=pretrained_embeddings,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-5)

    best_val_f1 = 0.0
    best_model_state = None
    best_epoch = 0

    for epoch in range(cfg.epochs):
        model.train()
        total_train_loss = 0.0

        for x, y, mask in train_dl:
            x, y, mask = x.to(device), y.to(device), mask.to(device)

            loss = model(x, tags=y, mask=mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dl)
        p_val, r_val, f_val, _ = evaluate(model, val_dl, device)

        if f_val > best_val_f1:
            best_val_f1 = f_val
            best_epoch = epoch + 1
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

        print(
            f"Epoch {epoch+1}/{cfg.epochs} "
            f"- train_loss={avg_train_loss:.4f} "
            f"- val_F1={f_val:.3f} (P={p_val:.3f}, R={r_val:.3f}) "
            f"{'*' if f_val == best_val_f1 else ''}"
        )

    print(f"\nLoading best model from epoch {best_epoch} (val_F1={best_val_f1:.3f})")
    model.load_state_dict(best_model_state)
    model.to(device)

    p_test, r_test, f_test, report = evaluate(model, test_dl, device)

    print()
    print(
        tabulate(
            [
                [
                    "BiLSTM-CRF" + (" + GloVe" if args.use_glove else ""),
                    f"{p_test:.3f}",
                    f"{r_test:.3f}",
                    f"{f_test:.3f}",
                ]
            ],
            headers=["Model", "Precision", "Recall", "F1"],
        )
    )
    print("\nFinal Test Report:\n", report)

    print(f"\nBest validation F1: {best_val_f1:.3f}")


if __name__ == "__main__":
    main()
