import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=81):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe

class BaselineTransformer(nn.Module):
    def __init__(self, vocab_size=11, d_model=128, nhead=4, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output logic: for each of the 81 cells, predict the digit (1-9)
        # Output vocab size is same as input usually, but we really only care about digits 1-9 (indices 2-10)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x shape: [batch, 81]
        x = self.embedding(x) # [batch, 81, d_model]
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x)
        logits = self.fc_out(out) # [batch, 81, vocab_size]
        return logits