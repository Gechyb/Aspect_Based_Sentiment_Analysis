import torch.nn as nn
from TorchCRF import CRF


class BiLSTMCRF(nn.Module):
    def __init__(
        self,
        vocab_size,
        tagset_size,
        embedding_dim=100,
        hidden_dim=128,
        pad_idx=0,
        dropout=0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, inputs, tags=None, mask=None):
        emb = self.embedding(inputs)
        outputs, _ = self.lstm(emb)
        outputs = self.dropout(outputs)
        emissions = self.hidden2tag(outputs)
        if tags is not None:
            loss = -self.crf(emissions, tags, mask=mask, reduction="mean")
            return loss
        else:
            pred = self.crf.decode(emissions, mask=mask)
            return pred
