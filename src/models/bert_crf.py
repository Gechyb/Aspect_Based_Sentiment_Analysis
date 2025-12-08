import torch
from torch import nn
from transformers import AutoModel
from TorchCRF import CRF
from src.tagging_scheme import TAG2ID, ID2TAG


class BERT_CRF(nn.Module):
    def __init__(self, model_name, num_tags):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size

        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden, num_tags)

        # TorchCRF expects num_tags only — no batch_first arg
        self.crf = CRF(num_tags)

        self.ignore_index = -100  # transformer padding label

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        If labels=None → decode
        If labels provided → training mode
        """

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        emissions = self.fc(self.dropout(outputs.last_hidden_state))
        mask = attention_mask.bool()

        # ---------- PREDICTION ----------
        if labels is None:
            return self.crf.viterbi_decode(emissions, mask=mask)

        # ---------- TRAINING ----------
        labels = labels.clone()
        labels[labels == self.ignore_index] = TAG2ID["O"]

        log_likelihood = self.crf(emissions, labels, mask=mask)

        # return negative mean likelihood for optimization
        return -log_likelihood.mean()
