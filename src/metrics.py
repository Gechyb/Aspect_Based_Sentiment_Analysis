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


def normalize_predictions(pred_ids, target_len):
    """
    Ensures predictions match true sequence length.
    If predictions are shorter (rare CRF decode bug), pad with 'O'.
    If longer, trim to match length.
    """

    # Convert tensors to Python ints if needed
    pred_ids = [int(p) for p in pred_ids]

    if len(pred_ids) < target_len:
        pred_ids = pred_ids + [0] * (target_len - len(pred_ids))  # ID for "O"
    elif len(pred_ids) > target_len:
        pred_ids = pred_ids[:target_len]

    return pred_ids
