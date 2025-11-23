"""
Utility functions for loading and using pre-trained GloVe embeddings.

Download GloVe embeddings from:
https://nlp.stanford.edu/projects/glove/

Recommended: glove.6B.zip (Common Crawl, 6B tokens)
Contains: 50d, 100d, 200d, 300d embeddings
"""

import numpy as np
from pathlib import Path


def load_glove_embeddings(glove_path, embedding_dim=100):
    """
    Load GloVe embeddings from a text file.

    Args:
        glove_path: Path to GloVe file (e.g., 'glove.6B.100d.txt')
        embedding_dim: Dimension of embeddings (50, 100, 200, or 300)

    Returns:
        word2vec: dict mapping word -> numpy array
    """
    print(f"Loading GloVe embeddings from {glove_path}...")
    word2vec = {}

    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")

            # Verify dimension
            if len(vector) == embedding_dim:
                word2vec[word] = vector

    print(f"Loaded {len(word2vec)} word vectors of dimension {embedding_dim}")
    return word2vec


def create_embedding_matrix(vocab, word2vec, embedding_dim=100):
    """
    Create an embedding matrix for the vocabulary using pre-trained GloVe.

    Args:
        vocab: dict mapping token -> id (includes <pad> and <unk>)
        word2vec: dict mapping word -> vector (from load_glove_embeddings)
        embedding_dim: dimension of embeddings

    Returns:
        embedding_matrix: numpy array of shape (vocab_size, embedding_dim)
        num_found: number of vocab words found in GloVe
    """
    vocab_size = len(vocab)
    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)

    num_found = 0
    for word, idx in vocab.items():
        if word in word2vec:
            embedding_matrix[idx] = word2vec[word]
            num_found += 1
        elif word.lower() in word2vec:
            # Try lowercase version
            embedding_matrix[idx] = word2vec[word.lower()]
            num_found += 1
        else:
            # Initialize OOV words with small random values
            # Note: <pad> at index 0 will remain zeros
            if word != "<pad>":
                embedding_matrix[idx] = np.random.normal(
                    scale=0.6, size=(embedding_dim,)
                ).astype(np.float32)

    coverage = num_found / vocab_size * 100
    print(f"GloVe coverage: {num_found}/{vocab_size} ({coverage:.1f}%)")

    return embedding_matrix, num_found


def load_embeddings_for_vocab(vocab, glove_dir, embedding_dim=100):
    """
    Convenience function to load GloVe and create embedding matrix.

    Args:
        vocab: vocabulary dict
        glove_dir: directory containing GloVe files
        embedding_dim: 50, 100, 200, or 300

    Returns:
        embedding_matrix: numpy array ready to load into nn.Embedding
    """
    glove_path = Path(glove_dir) / f"glove.6B.{embedding_dim}d.txt"

    if not glove_path.exists():
        raise FileNotFoundError(
            f"GloVe file not found: {glove_path}\n"
            f"Download from: https://nlp.stanford.edu/projects/glove/\n"
            f"Extract glove.6B.zip to {glove_dir}"
        )

    word2vec = load_glove_embeddings(glove_path, embedding_dim)
    embedding_matrix, _ = create_embedding_matrix(vocab, word2vec, embedding_dim)

    return embedding_matrix
