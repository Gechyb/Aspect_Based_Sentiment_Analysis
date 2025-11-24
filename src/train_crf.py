import argparse
from pathlib import Path
from tabulate import tabulate
from .config import Config
from .data_utils import load_sentences, train_val_test_split
from .features_crf import sent2features
from .models.crf_model import CRFTagger
from .metrics import span_f1


def prepare(data):
    X, y = [], []
    for s in data:
        X.append(sent2features(s["tokens"]))
        y.append(s["labels"])
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain", default="restaurants", choices=["restaurants", "laptops"]
    )
    args = parser.parse_args()
    cfg = Config(domain=args.domain)

    path = Path(cfg.data_dir) / f"{cfg.domain}.jsonl"  # pre-converted JSONL
    data = load_sentences(path)

    # Train/val/test split
    train, val, test = train_val_test_split(data, seed=cfg.seed)

    X_tr, y_tr = prepare(train)
    X_va, y_va = prepare(val)
    X_te, y_te = prepare(test)

    # Train CRF on training set
    crf = CRFTagger(c1=cfg.crf_c1, c2=cfg.crf_c2, max_iterations=cfg.crf_max_iter)
    crf.fit(X_tr, y_tr)

    # Evaluate on validation set
    y_val_pred = crf.predict(X_va)
    p_va, r_va, f_va = span_f1(y_va, y_val_pred)

    print("\nValidation Performance (for model selection / monitoring):")
    print(
        tabulate(
            [["CRF", f"{p_va:.3f}", f"{r_va:.3f}", f_va]],
            headers=["Model", "Precision", "Recall", "F1"],
        )
    )

    # Evaluate on test set
    y_te_pred = crf.predict(X_te)
    p_te, r_te, f_te = span_f1(y_te, y_te_pred)

    print("\nTest Performance (held-out evaluation):")
    print(
        tabulate(
            [["CRF", f"{p_te:.3f}", f"{r_te:.3f}", f_te]],
            headers=["Model", "Precision", "Recall", "F1"],
        )
    )


if __name__ == "__main__":
    main()
