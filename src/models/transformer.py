import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]
    
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, nhead=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.pe = PositionalEncoding(hidden_size)

    def forward(self, x):
        emb = self.embedding(x)
        emb = self.pe(emb)
        emb = emb.transpose(0, 1)  # Transformer expects seq, batch, dim
        # causal mask
        mask = torch.triu(torch.ones(emb.size(0), emb.size(0)) * float('-inf'), diagonal=1).to(emb.device)
        out = self.transformer(emb, mask=mask, is_causal=True)
        out = out.transpose(0, 1)
        logits = self.fc(out)
        return logits
