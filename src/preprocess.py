from typing import List, Dict, Any
import re


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def to_bio_sentiment(annotated_tokens: List[Dict[str, Any]]) -> List[str]:
    """Map token-level annotations to unified B/I/O-Sentiment tags.
    Expect tokens like: {"token": "great", "aspect": True/False, "sent": "POS|NEG|NEU", "begin": bool}
    """
    tags = []
    for tok in annotated_tokens:
        if tok.get("aspect"):
            pref = "B" if tok.get("begin", False) else "I"
            tags.append(f"{pref}-{tok['sent']}")
        else:
            tags.append("O")
    return tags
