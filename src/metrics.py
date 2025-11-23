from typing import List, Tuple
from seqeval.metrics import precision_score, recall_score, f1_score


def span_f1(
    y_true: List[List[str]], y_pred: List[List[str]]
) -> Tuple[float, float, float]:
    """
    Compute span-level precision, recall, and F1 using seqeval.

    Uses zero_division=0 to suppress warnings when no predictions are made
    (common in early epochs with random initialization).
    """
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    return p, r, f
