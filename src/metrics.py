from typing import List, Tuple
from seqeval.metrics import precision_score, recall_score, f1_score


def span_f1(
    y_true: List[List[str]], y_pred: List[List[str]]
) -> Tuple[float, float, float]:
    # Uses seqeval token-level tags consistent with BIO sentiment labels
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f = f1_score(y_true, y_pred)
    return p, r, f
