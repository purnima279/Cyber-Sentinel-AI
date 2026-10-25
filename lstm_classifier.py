import torch
import torch.nn as nn
from typing import Dict, Any


class CyberSentinelLSTM(nn.Module):
    """
    Feature-Augmented Bidirectional LSTM Classifier for Cybersecurity Incident Analysis.

    Combines:
    1. Word Embedding Layer (dense continuous representations)
    2. Bidirectional LSTM Layer (captures sequence context and word order)
    3. Global Average + Max Pooling + Final Hidden State (captures salient incident signals)
    4. Lexical Domain Feature Fusion (combines learned sequence embeddings with cybersecurity domain feature counts)
    5. Dual Softmax Classification Heads:
       - Incident Category Head (Authentication, Malware, Network, Phishing, etc.)
       - Sentiment Classification Head (Negative, Neutral, Positive)
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 48,
        hidden_dim: int = 64,
        num_categories: int = 7,
        num_sentiments: int = 3,
        num_layers: int = 1,
        lexical_dim: int = 7,
        bidirectional: bool = True,
        dropout: float = 0.2,
        padding_idx: int = 0
    ):
        super(CyberSentinelLSTM, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.lexical_dim = lexical_dim

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        self.emb_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        lstm_dim = hidden_dim * self.num_directions
        fc_in_dim = (lstm_dim * 3) + lexical_dim

        self.fc_shared = nn.Sequential(
            nn.Linear(fc_in_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.fc_category = nn.Linear(64, num_categories)
        self.fc_sentiment = nn.Linear(64, num_sentiments)

    def forward(self, x: torch.Tensor, lexical_feats: torch.Tensor = None, return_internals: bool = False):
        # x: (batch_size, seq_len)
        embedded = self.emb_dropout(self.embedding(x))

        lstm_out, (hn, cn) = self.lstm(embedded)

        if self.bidirectional:
            last_hidden = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        else:
            last_hidden = hn[-1, :, :]

        # Mask padding for average pooling
        mask = (x != 0).unsqueeze(-1).float()
        sum_masked = (lstm_out * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1.0)
        avg_pool = sum_masked / lengths

        # Max pooling
        max_pool, _ = torch.max(lstm_out, dim=1)

        feature_list = [last_hidden, avg_pool, max_pool]

        if lexical_feats is not None:
            feature_list.append(lexical_feats)
        else:
            batch_size = x.size(0)
            dummy_lex = torch.zeros(batch_size, self.lexical_dim, device=x.device)
            feature_list.append(dummy_lex)

        pooled = torch.cat(feature_list, dim=1)
        shared_feat = self.fc_shared(pooled)

        cat_logits = self.fc_category(shared_feat)
        sent_logits = self.fc_sentiment(shared_feat)

        if return_internals:
            return {
                "cat_logits": cat_logits,
                "sent_logits": sent_logits,
                "lstm_out": lstm_out,
                "last_hidden": pooled,
                "embedded": embedded
            }

        return cat_logits, sent_logits
