from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    seed: int = 42
    data_dir: Path = Path("data/intermediate")
    results_dir: Path = Path("results")
    domain: str = "restaurants"  # or "laptops"
    # CRF
    crf_c1: float = 0.1
    crf_c2: float = 0.1
    crf_max_iter: int = 200
    # BiLSTM
    embedding_dim: int = 100
    hidden_dim: int = 128
    dropout: float = 0.3
    lr: float = 1e-3
    batch_size: int = 32
    epochs: int = 10
    glove_path: Path = Path("data/glove.6B.100d.txt")  # will adjust if needed
