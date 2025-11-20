from typing import List, Dict, Any
from pathlib import Path
import json
import random


def load_sentences(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL: one sentence per line with 'tokens' and 'labels'."""
    if not path.exists():
        raise FileNotFoundError(f"Expected SemEval data at {path}")
    sents = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            sents.append(json.loads(line))
    return sents


def train_val_test_split(data, seed=42, ratios=(0.7, 0.15, 0.15)):
    random.Random(seed).shuffle(data)
    n = len(data)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    train = data[:n_train]
    val = data[n_train : n_train + n_val]
    test = data[n_train + n_val :]
    return train, val, test
