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

    path = (
        Path(cfg.data_dir) / f"{cfg.domain}.jsonl"
    )  # pre-converted JSONL (tokens + labels)
    data = load_sentences(path)
    train, val, test = train_val_test_split(data, seed=cfg.seed)

    X_tr, y_tr = prepare(train)
    X_va, y_va = prepare(val)
    X_te, y_te = prepare(test)

    crf = CRFTagger(c1=cfg.crf_c1, c2=cfg.crf_c2, max_iterations=cfg.crf_max_iter)
    crf.fit(X_tr, y_tr)

    y_pred = crf.predict(X_te)
    p, r, f = span_f1(y_te, y_pred)

    print(
        tabulate(
            [["CRF", f"{p:.3f}", f"{r:.3f}", f"{f:.3f}"]],
            headers=["Model", "Precision", "Recall", "F1"],
        )
    )


if __name__ == "__main__":
    main()
