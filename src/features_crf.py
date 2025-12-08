from typing import List, Dict
import json
import os

# Optional glove clustering file
CLUSTER_PATH = "data/glove/glove_clusters.json"
if os.path.exists(CLUSTER_PATH):
    with open(CLUSTER_PATH, "r") as f:
        GLOVE_CLUSTERS = json.load(f)
else:
    GLOVE_CLUSTERS = {}


def token2features(sent_tokens: List[str], i: int, use_glove=False) -> Dict[str, str]:
    """Returns handcrafted CRF features with optional GloVe cluster feature."""
    word = sent_tokens[i]
    word_lower = word.lower()

    features = {
        "bias": 1.0,
        "word.lower": word_lower,
        "word.isupper": str(word.isupper()),
        "word.istitle": str(word.istitle()),
        "word.isdigit": str(word.isdigit()),
    }

    # Previous token features
    if i > 0:
        prev = sent_tokens[i - 1]
        features.update(
            {
                "-1.word.lower": prev.lower(),
                "-1.isupper": str(prev.isupper()),
                "-1.istitle": str(prev.istitle()),
            }
        )
    else:
        features["BOS"] = True

    # Next token features
    if i < len(sent_tokens) - 1:
        nxt = sent_tokens[i + 1]
        features.update(
            {
                "+1.word.lower": nxt.lower(),
                "+1.isupper": str(nxt.isupper()),
                "+1.istitle": str(nxt.istitle()),
            }
        )
    else:
        features["EOS"] = True

    # Optional: Add glove embedding bucket
    if use_glove:
        cluster = GLOVE_CLUSTERS.get(word_lower, "UNK")
        features["glove_cluster"] = str(cluster)

    return features


def sent2features(sent_tokens: List[str], use_glove=False) -> List[Dict[str, str]]:
    """Applies token2features across an entire sentence."""
    return [token2features(sent_tokens, i, use_glove) for i in range(len(sent_tokens))]
