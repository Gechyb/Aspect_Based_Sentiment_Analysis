import torch
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
        pretrained_embeddings=None,
    ):
        super().__init__()

        # Embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        if pretrained_embeddings is not None:
            print("Loading pre-trained embeddings...")
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

        # BiLSTM encoder
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        # tagset_size−1 removes PAD so CRF sees only valid labels.
        self.crf = CRF(tagset_size - 1)

        self.pad_idx = pad_idx

    def forward(self, inputs, tags=None, mask=None):
        emb = self.embedding(inputs)
        outputs, _ = self.lstm(emb)
        outputs = self.dropout(outputs)

        emissions = self.hidden2tag(outputs)

        emissions = emissions[:, :, 1:]  # (batch, seq_len, num_tags_without_PAD)

        if tags is not None:
            # Shift label IDs to match reduced emission space
            tags_shifted = tags - 1

            # TorchCRF uses: negative log-likelihood (batch,) → we average manually
            log_likelihood = self.crf(emissions, tags_shifted, mask=mask)
            loss = -log_likelihood.mean()
            return loss

        best_paths = self.crf.viterbi_decode(emissions, mask=mask)

        decoded = []
        for path, mask_row in zip(best_paths, mask):
            real_len = int(mask_row.sum().item())
            shifted = [p + 1 for p in path[:real_len]]
            decoded.append(shifted)

        return decoded
