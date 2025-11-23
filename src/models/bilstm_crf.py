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
        pretrained_embeddings=None,  # NEW: accepts pre-trained embeddings
    ):
        super().__init__()

        # Create embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # NEW: Load pre-trained embeddings if provided
        if pretrained_embeddings is not None:
            print("Loading pre-trained embeddings...")
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            # Optionally freeze embeddings (uncomment to prevent updates during training)
            # self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        # TorchCRF - use only actual tags (exclude PAD at index 0)
        self.crf = CRF(tagset_size - 1)  # Exclude PAD tag!
        self.pad_idx = pad_idx

    def forward(self, inputs, tags=None, mask=None):
        """
        inputs: (batch, seq_len)
        tags:   (batch, seq_len) or None
        mask:   (batch, seq_len)
        """

        emb = self.embedding(inputs)  # (batch, seq_len, emb_dim)
        outputs, _ = self.lstm(emb)  # (batch, seq_len, hidden_dim)
        outputs = self.dropout(outputs)

        emissions = self.hidden2tag(outputs)  # (batch, seq_len, num_tags)

        # Remove PAD tag dimension from emissions (it's at index 0)
        emissions = emissions[:, :, 1:]  # Remove index 0 (PAD)

        # TorchCRF expects seq_len first
        emissions = emissions.transpose(0, 1)  # (seq_len, batch, num_tags-1)

        if mask is not None:
            mask = mask.transpose(0, 1)  # (seq_len, batch)

        if tags is not None:
            # Shift tags down by 1 to account for removed PAD
            tags_shifted = tags - 1
            tags_shifted = tags_shifted.transpose(0, 1)  # (seq_len, batch)

            log_likelihood = self.crf(emissions, tags_shifted, mask=mask)
            loss = -log_likelihood.mean()
            return loss

        else:
            # Decode
            best_paths = self.crf.decode(emissions, mask=mask)

            # Shift predictions back up by 1 to match original tag indices
            result = []
            for i, path in enumerate(best_paths):
                # Add 1 back to predictions
                shifted_path = [p + 1 for p in path]

                # Ensure correct length
                if mask is not None:
                    actual_len = int(mask[:, i].sum().item())
                else:
                    actual_len = inputs.size(1)

                if len(shifted_path) < actual_len:
                    # Pad with O tag (index 1)
                    shifted_path = shifted_path + [1] * (actual_len - len(shifted_path))
                elif len(shifted_path) > actual_len:
                    shifted_path = shifted_path[:actual_len]

                result.append(shifted_path)

            return result
