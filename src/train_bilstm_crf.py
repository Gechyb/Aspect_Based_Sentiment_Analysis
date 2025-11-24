import argparse
from pathlib import Path
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from tabulate import tabulate
import random
import numpy as np
from .config import Config
from .data_utils import load_sentences, train_val_test_split
from .models.bilstm_crf import BiLSTMCRF
from .metrics import span_f1
from .tagging_scheme import TAGS, TAG2ID, ID2TAG
from .glove_utils import load_embeddings_for_vocab


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
        tokens = ex["tokens"]
        labels = ex["labels"]

        # Map tokens to ids (lowercased), unknown -> <unk>
        x = [self.vocab.get(t.lower(), self.vocab["<unk>"]) for t in tokens]
        # Map BIO-Sentiment tags to ids
        y = [TAG2ID[la] for la in labels]

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


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
    )  # pad labels with PAD
    mask = torch.zeros(len(xs), maxlen, dtype=torch.bool)

    for i, (x, y) in enumerate(zip(xs, ys)):
        L = len(x)
        pad_x[i, :L] = x
        pad_y[i, :L] = y
        mask[i, :L] = True

    return pad_x, pad_y, mask


def set_seed(seed):
    """
    Set all random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, dataloader, device):
    model.eval()
    all_true = []
    all_pred = []

    with torch.no_grad():
        for x, y, m in dataloader:
            x = x.to(device)
            y = y.to(device)
            m = m.to(device)

            # CRF decode (list of lists, already trimmed by mask)
            preds = model(x, tags=None, mask=m)

            # Loop over batch samples one by one
            for pred_ids, gold_ids, mask_row in zip(preds, y, m):
                L = int(mask_row.sum().item())  # REAL length

                gold_tags = [ID2TAG[int(t)] for t in gold_ids[:L].tolist()]

                # Handle case where viterbi_decode returns fewer predictions
                pred_tags = [ID2TAG[int(p)] for p in pred_ids]

                # Ensure both sequences have the same length
                if len(pred_tags) < L:
                    pred_tags = pred_tags + ["O"] * (L - len(pred_tags))
                elif len(pred_tags) > L:
                    pred_tags = pred_tags[:L]

                all_true.append(gold_tags)
                all_pred.append(pred_tags)

    return span_f1(all_true, all_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain", default="restaurants", choices=["restaurants", "laptops"]
    )
    parser.add_argument(
        "--use_glove", action="store_true", help="Use pre-trained GloVe embeddings"
    )
    parser.add_argument(
        "--glove_dir",
        type=str,
        default="./data/glove",
        help="Directory containing GloVe files",
    )
    args = parser.parse_args()

    cfg = Config(domain=args.domain)

    # Set random seeds for reproducibility
    set_seed(cfg.seed)
    print(f"Random seed set to: {cfg.seed}")

    path = Path(cfg.data_dir) / f"{cfg.domain}.jsonl"
    data = load_sentences(path)

    train, val, test = train_val_test_split(data, seed=cfg.seed)

    # Build vocab from training set only
    vocab = build_vocab(train, min_freq=1)
    print(f"Vocabulary size: {len(vocab)}")

    # Load GloVe embeddings if requested
    pretrained_embeddings = None
    if args.use_glove:
        try:
            pretrained_embeddings = load_embeddings_for_vocab(
                vocab, args.glove_dir, embedding_dim=cfg.embedding_dim
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

    # Model & optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BiLSTMCRF(
        vocab_size=len(vocab),
        tagset_size=len(TAGS),
        embedding_dim=cfg.embedding_dim,
        hidden_dim=cfg.hidden_dim,
        pad_idx=0,
        dropout=cfg.dropout,
        pretrained_embeddings=pretrained_embeddings,  # NEW
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-5)

    # Training loop with model checkpointing
    best_val_f1 = 0.0
    best_model_state = None

    for epoch in range(cfg.epochs):
        model.train()
        total_train_loss = 0.0

        for x, y, m in train_dl:
            x, y, m = x.to(device), y.to(device), m.to(device)

            loss = model(x, tags=y, mask=m)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dl)

        # Evaluate on validation set each epoch
        p_val, r_val, f_val = evaluate(model, val_dl, device)

        # Saving best model checkpoint
        if f_val > best_val_f1:
            best_val_f1 = f_val
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            best_epoch = epoch + 1

        print(
            f"Epoch {epoch+1}/{cfg.epochs} "
            f"- train_loss={avg_train_loss:.4f} "
            f"- val_F1={f_val:.3f} (P={p_val:.3f}, R={r_val:.3f}) "
            f"{'*' if f_val == best_val_f1 else ''}"
        )

    # 9. Load best model and evaluate on test set
    print(f"\nLoading best model from epoch {best_epoch} (val_F1={best_val_f1:.3f})")
    model.load_state_dict(best_model_state)
    model.to(device)

    p_test, r_test, f_test = evaluate(model, test_dl, device)
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
    print(f"\nBest validation F1: {best_val_f1:.3f}")


if __name__ == "__main__":
    main()
