import torch
import torch.nn as nn
import random
from typing import List, Dict, Any, Tuple


class IncidentEncoder(nn.Module):
    """
    Encoder Component:
    Processes the cybersecurity incident description token by token and compresses
    it into a learned context vector (final hidden and cell states).
    """
    def __init__(self, input_dim: int, emb_dim: int = 64, hidden_dim: int = 128, n_layers: int = 1, dropout: float = 0.2):
        super(IncidentEncoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor):
        # src: (batch_size, src_len)
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs: (batch_size, src_len, hidden_dim)
        # hidden, cell: (n_layers, batch_size, hidden_dim) -> The Context Vector
        return outputs, (hidden, cell)


class IncidentDecoder(nn.Module):
    """
    Decoder Component:
    Conditioned on the encoder's context vector, generates an incident remediation action
    or summary sequence token-by-token using autoregressive decoding.
    """
    def __init__(self, output_dim: int, emb_dim: int = 64, hidden_dim: int = 128, n_layers: int = 1, dropout: float = 0.2):
        super(IncidentDecoder, self).__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor):
        # input_token: (batch_size, 1)
        embedded = self.dropout(self.embedding(input_token))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        # output: (batch_size, 1, hidden_dim)
        prediction = self.fc_out(output.squeeze(1))  # (batch_size, output_dim)
        return prediction, hidden, cell


class IncidentSeq2Seq(nn.Module):
    """
    Full Sequence-to-Sequence (Encoder-Decoder) Architecture.
    Separates the encoder representation from decoder sequence generation.
    """
    def __init__(self, encoder: IncidentEncoder, decoder: IncidentDecoder):
        super(IncidentSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src: torch.Tensor, trg: torch.Tensor, teacher_forcing_ratio: float = 0.5):
        # src: (batch_size, src_len)
        # trg: (batch_size, trg_len)
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(src.device)

        _, (hidden, cell) = self.encoder(src)

        # First input to decoder is <SOS> token (trg[:, 0])
        input_token = trg[:, 0].unsqueeze(1)

        for t in range(1, trg_len):
            prediction, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[:, t, :] = prediction

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(1).unsqueeze(1)
            input_token = trg[:, t].unsqueeze(1) if teacher_force else top1

        return outputs

    def generate(self, src: torch.Tensor, max_len: int = 24, sos_idx: int = 2, eos_idx: int = 3) -> Dict[str, Any]:
        """
        Step-by-step autoregressive inference for visual inspection.
        Returns generated tokens, per-step probabilities, and encoder context vector statistics.
        """
        self.eval()
        with torch.no_grad():
            _, (hidden, cell) = self.encoder(src)

            # Record Context Vector summary
            context_vector_summary = {
                "hidden_norm": float(torch.norm(hidden).item()),
                "cell_norm": float(torch.norm(cell).item()),
                "hidden_mean": float(hidden.mean().item()),
                "hidden_std": float(hidden.std().item()),
                "dimensions": list(hidden.shape)
            }

            input_token = torch.tensor([[sos_idx]], dtype=torch.long, device=src.device)

            generated_indices = []
            steps = []

            for step in range(max_len):
                prediction, hidden, cell = self.decoder(input_token, hidden, cell)
                probs = torch.softmax(prediction, dim=1)
                conf, pred_token = torch.max(probs, dim=1)

                idx = pred_token.item()
                confidence = float(conf.item())

                steps.append({
                    "step": step + 1,
                    "token_idx": idx,
                    "confidence": round(confidence, 4)
                })

                if idx == eos_idx:
                    break

                generated_indices.append(idx)
                input_token = pred_token.unsqueeze(1)

            return {
                "generated_indices": generated_indices,
                "steps": steps,
                "context_vector": context_vector_summary
            }
