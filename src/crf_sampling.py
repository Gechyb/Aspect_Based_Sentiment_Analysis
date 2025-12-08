"""
Generate synthetic data from a trained CRF model using probabilistic sampling.
Reviewer-required: (train CRF → sample synthetic → retrain + evaluate)
"""

import random
import numpy as np


def sample_labels_from_distribution(distribution: dict):
    """Sample one label based on CRF-provided probability distribution."""
    labels = list(distribution.keys())
    probs = list(distribution.values())
    return np.random.choice(labels, p=probs)


def generate_crf_synthetic_data(crf, vocab, n_samples=500, min_len=5, max_len=25):
    """
    Generate synthetic token-label pairs from a trained CRF model.

    Args:
        crf: Trained sklearn_crfsuite.CRF model
        vocab: vocab list/dict used for sampling word surface forms
        n_samples: number of synthetic sentences
    """
    synthetic_data = []

    vocab_words = [w for w in vocab.keys() if w not in ("<pad>", "<unk>")]

    for _ in range(n_samples):

        length = random.randint(min_len, max_len)
        tokens = random.choices(vocab_words, k=length)

        marginals = crf.predict_marginals_single(tokens)
        labels = [sample_labels_from_distribution(dist) for dist in marginals]

        synthetic_data.append({"tokens": tokens, "labels": labels})

    return synthetic_data
