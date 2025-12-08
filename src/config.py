from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    def __init__(self, domain="restaurants"):
        self.domain = domain

        # Paths
        self.data_dir = "./data/intermediate"
        self.glove_dir = "./data/glove"  # <── ADDED FOR YOUR ERROR FIX

        # Model parameters
        self.embedding_dim = 200
        self.hidden_dim = 128
        self.dropout = 0.25

        # Training hyperparams
        self.batch_size = 16
        self.lr = 1e-3
        self.epochs = 20

        # CRF regularization
        self.crf_c1 = 0.1
        self.crf_c2 = 0.1
        self.crf_max_iter = 100

        # Reproducibility
        self.seed = 42
