from typing import List, Dict


def token2features(sent_tokens: List[str], i: int) -> Dict[str, str]:
    word = sent_tokens[i]
    features = {
        "bias": 1.0,
        "word.lower": word.lower(),
        "word.isupper": str(word.isupper()),
        "word.istitle": str(word.istitle()),
        "word.isdigit": str(word.isdigit()),
    }
    if i > 0:
        prev = sent_tokens[i - 1]
        features.update(
            {
                "-1:word.lower": prev.lower(),
                "-1:word.istitle": str(prev.istitle()),
            }
        )
    else:
        features["BOS"] = True
    if i < len(sent_tokens) - 1:
        nxt = sent_tokens[i + 1]
        features.update(
            {
                "+1:word.lower": nxt.lower(),
                "+1:word.istitle": str(nxt.istitle()),
            }
        )
    else:
        features["EOS"] = True
    return features


def sent2features(sent_tokens: List[str]) -> List[Dict[str, str]]:
    return [token2features(sent_tokens, i) for i in range(len(sent_tokens))]
