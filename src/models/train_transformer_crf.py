import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW
from tabulate import tabulate

from src.config import Config
from src.metrics import span_f1
from src.tagging_scheme import TAGS, TAG2ID, ID2TAG
from src.data_utils import load_sentences, train_val_test_split
from src.models.bert_crf import BERT_CRF


# ------------------------------
# Dataset for BERT + CRF
# ------------------------------
class TransformerABSA(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        tokens = ex["tokens"]
        labels = ex["labels"]

        # Tokenize with mapping
        enc = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        word_ids = enc.word_ids()
        aligned_labels = []

        for wi in word_ids:
            if wi is None:
                aligned_labels.append(-100)
            else:
                aligned_labels.append(TAG2ID[labels[wi]])

        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(aligned_labels),
        }


# ------------------------------
# Training Loop
# ------------------------------
def train_epoch(model, dataloader, optim, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbl = batch["labels"].to(device)

        lbl = torch.where(lbl == -100, torch.tensor(TAG2ID["O"], device=device), lbl)

        optim.zero_grad()
        loss = model(input_ids=ids, attention_mask=mask, labels=lbl)

        # FIX: ensure scalar loss
        if loss.dim() > 0:
            loss = loss.mean()

        loss.backward()
        optim.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# ------------------------------
# Evaluation Loop
# ------------------------------
def evaluate(model, dataloader, device):
    model.eval()
    all_pred, all_true = [], []

    with torch.no_grad():
        for batch in dataloader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbl = batch["labels"].to(device)

            lbl = torch.where(
                lbl == -100, torch.tensor(TAG2ID["O"], device=device), lbl
            )

            preds = model(input_ids=ids, attention_mask=mask)

            for p, g, m in zip(preds, lbl, mask):
                L = int(m.sum().item())
                gold_tags = [ID2TAG[int(tag)] for tag in g[:L].tolist()]
                pred_tags = [ID2TAG[int(x)] for x in p][:L]

                all_true.append(gold_tags)
                all_pred.append(pred_tags)

    return span_f1(all_true, all_pred)


# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain", default="restaurants", choices=["restaurants", "laptops"]
    )
    parser.add_argument("--model", default="bert", choices=["bert"])
    args = parser.parse_args()

    cfg = Config(domain=args.domain)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load data
    path = Path(cfg.data_dir) / f"{cfg.domain}.jsonl"
    data = load_sentences(path)
    train, val, test = train_val_test_split(data, seed=cfg.seed)

    # Tokenizer + dataset
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    train_ds = TransformerABSA(train, tokenizer)
    val_ds = TransformerABSA(val, tokenizer)
    test_ds = TransformerABSA(test, tokenizer)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size)
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size)

    # Model
    model = BERT_CRF("bert-base-uncased", num_tags=len(TAGS)).to(device)
    optim = AdamW(model.parameters(), lr=2e-5)

    best_f1 = 0
    best_state = None

    print("\n🚀 Training transformer model...\n")

    for epoch in range(5):  # can increase later
        train_loss = train_epoch(model, train_dl, optim, device)
        p, r, f = evaluate(model, val_dl, device)

        print(f"Epoch {epoch+1} → Loss={train_loss:.4f}, Val F1={f:.3f}")

        if f > best_f1:
            best_f1 = f
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    print(f"\n📌 Loading best model (Val F1={best_f1:.3f})...")
    model.load_state_dict(best_state)
    model.to(device)

    # Test results
    p, r, f = evaluate(model, test_dl, device)

    print("\n===== FINAL TEST RESULTS =====\n")
    print(
        tabulate(
            [["BERT-CRF", f"{p:.3f}", f"{r:.3f}", f"{f:.3f}"]],
            headers=["Model", "Precision", "Recall", "F1"],
        )
    )


if __name__ == "__main__":
    main()
