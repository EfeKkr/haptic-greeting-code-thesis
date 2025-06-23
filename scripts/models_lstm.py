# LSTMClassifier with attention mechanism, supporting bidirectional LSTM and dropout.
# Can process sequences of extracted features from CNNs (input_size=512) or ViTs (input_size=768),
# depending on which feature type is used during training. Handles variable-length input via packing and masking.

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output, mask):
        scores = self.attn(lstm_output).squeeze(-1)  # [B, T]
        scores = scores.masked_fill(mask == 0, float('-inf'))  # Mask out padding
        weights = torch.softmax(scores, dim=1)  # [B, T]
        attended = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1)  # [B, hidden_dim]
        return attended

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=512, hidden_size=128, num_layers=1, num_classes=9, bidirectional=True, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )

        lstm_out_dim = hidden_size * 2 if bidirectional else hidden_size
        self.attention = Attention(lstm_out_dim)
        self.norm = nn.LayerNorm(lstm_out_dim)
        self.fc = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, x, lengths):
        # x: [B, T, 512], lengths: [B]
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)  # [B, T, H]

        # Create mask
        max_len = lstm_out.size(1)
        mask = torch.arange(max_len).unsqueeze(0).to(lengths.device) < lengths.unsqueeze(1)

        attended = self.attention(lstm_out, mask)
        return self.fc(self.norm(attended))