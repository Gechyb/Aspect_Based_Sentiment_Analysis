import argparse
from pathlib import Path
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import (
    flat_f1_score,
    flat_precision_score,
    flat_recall_score,
)

from .config import Config
from .data_utils import load_sentences, train_val_test_split
from .features_crf import sent2features
from .glove_utils import load_embeddings_for_vocab
from .metrics import span_f1
import numpy as np
from src.glove_utils import load_glove_embeddings


def prepare(sentences, use_glove=False, glove_embeddings=None):
    """
    Returns feature sequences and label sequences formatted for CRF.
    """
    X, y = [], []

    for s in sentences:
        tokens = s["tokens"]
        labels = s["labels"]

        # Extract per-token handcrafted features
        feats = sent2features(tokens, use_glove)

        # OPTIONAL: add numeric GloVe-bucket feature (kept lightweight)
        if use_glove and glove_embeddings is not None:
            for i, token in enumerate(tokens):
                vec = glove_embeddings.get(token.lower())
                if vec is not None:
                    bucket = int(np.mean(vec) * 10)
                    feats[i]["glove_bucket"] = bucket

        X.append(feats)
        y.append(labels)

    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain", default="restaurants", choices=["restaurants", "laptops"]
    )
    parser.add_argument(
        "--use_glove", action="store_true", help="Use GloVe bucketed feature embeddings"
    )
    args = parser.parse_args()

    cfg = Config(domain=args.domain)

    print(f"\n📌 Dataset: {args.domain}")
    print(f"🔧 Using GloVe features: {args.use_glove}")

    path = Path(cfg.data_dir) / f"{cfg.domain}.jsonl"
    data = load_sentences(path)

    train, val, test = train_val_test_split(data, seed=cfg.seed)

    glove_embeddings = None
    if args.use_glove:
        print("\n📥 Loading GloVe embeddings...")
        vocab = {t.lower() for s in data for t in s["tokens"]}
        glove_embeddings = load_glove_embeddings(
            f"data/glove/glove.6B.{cfg.embedding_dim}d.txt", cfg.embedding_dim
        )
        print(f"✓ Loaded GloVe coverage: {len(glove_embeddings)}/{len(vocab)} tokens\n")

    X_tr, y_tr = prepare(train, args.use_glove, glove_embeddings)
    X_va, y_va = prepare(val, args.use_glove, glove_embeddings)
    X_te, y_te = prepare(test, args.use_glove, glove_embeddings)

    crf = CRF(
        algorithm="lbfgs",
        max_iterations=cfg.crf_max_iter,
        c1=cfg.crf_c1,
        c2=cfg.crf_c2,
        all_possible_transitions=True,
    )

    print("\n🚀 Training CRF...\n")
    crf.fit(X_tr, y_tr)

    # Validation
    y_val_pred = crf.predict(X_va)
    p_va, r_va, f_va = span_f1(y_va, y_val_pred)

    print("\n📊 Validation Results:")
    print(
        tabulate(
            [
                [
                    ("CRF + GloVe" if args.use_glove else "CRF Base"),
                    f"{p_va:.3f}",
                    f"{r_va:.3f}",
                    f"{f_va:.3f}",
                ]
            ],
            headers=["Model", "Precision", "Recall", "F1"],
        )
    )

    # Test
    y_te_pred = crf.predict(X_te)
    p_te, r_te, f_te = span_f1(y_te, y_te_pred)

    print("\n📌 Test Results:")
    print(
        tabulate(
            [
                [
                    ("CRF + GloVe" if args.use_glove else "CRF Base"),
                    f"{p_te:.3f}",
                    f"{r_te:.3f}",
                    f"{f_te:.3f}",
                ]
            ],
            headers=["Model", "Precision", "Recall", "F1"],
        )
    )


if __name__ == "__main__":
    main()
