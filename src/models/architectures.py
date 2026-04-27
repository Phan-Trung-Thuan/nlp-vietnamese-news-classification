import torch
import torch.nn as nn
import torch.nn.functional as F
from .kan_linear import KANLinear

class Block1(nn.Module):
    def __init__(self, in_channel, out_channel, seq_len):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, 5, stride=1, dilation=2, padding=4),
            nn.LayerNorm([out_channel, seq_len]),
            nn.SiLU(),
            nn.Dropout(0.5)
        )
        
    def forward(self, x):
        return self.backbone(x)

class CNN1D(nn.Module):
    def __init__(self, num_classes=12, vocab_size=64000, embedding_dim=32):
        super().__init__()
        st = embedding_dim
        
        self.input_embedding = nn.Embedding(vocab_size, st)
        
        self.backbone = nn.Sequential(
            Block1(st, 2*st, 8000),
            nn.AvgPool1d(2),
            Block1(2*st, 4*st, 4000),
            nn.AvgPool1d(2),
            Block1(4*st, 8*st, 2000),
            nn.AvgPool1d(2),
            Block1(8*st, 8*st, 1000),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = KANLinear(8*st, num_classes)
        
    def forward(self, x):
        x = self.input_embedding(x)
        if (x.shape[0] != 1):
            x = x.squeeze()
        x = x.transpose(1, 2)
        x = self.backbone(x).squeeze()
        x = self.classifier(x)
        return x

class BiLSTM(nn.Module):
    def __init__(self, num_classes=12, vocab_size=64000, embedding_dim=32):
        super().__init__()
        st = embedding_dim
        
        self.input_embedding = nn.Embedding(vocab_size, st)
        
        # Reduce sequence length from 8000 to 1000
        self.conv = nn.Sequential(
            nn.Conv1d(st, st*2, 5, stride=2, padding=2),
            nn.LayerNorm([st*2, 4000]),
            nn.SiLU(),
            nn.AvgPool1d(8, stride=4, padding=2),
            nn.Dropout(0.5)
        )
        
        self.hidden_size = st*4
        self.num_layers = 1
        self.backbone = nn.LSTM(st*2, self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=True)
        
        self.classifier = KANLinear(self.hidden_size*2, num_classes)
        
    def forward(self, x):        
        x = self.input_embedding(x)
        if (x.shape[0] != 1):
            x = x.squeeze()
        
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        
        h0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        output, _ = self.backbone(x, (h0, c0))
            
        output = self.classifier(output[:, -1, :])
        return output

class BiGRU(nn.Module):
    def __init__(self, num_classes=12, vocab_size=64000, embedding_dim=32):
        super().__init__()
        st = embedding_dim
        
        self.input_embedding = nn.Embedding(vocab_size, st)
        
        # Reduce sequence length from 8000 to 1000
        self.conv = nn.Sequential(
            nn.Conv1d(st, st*2, 5, stride=2, padding=2),
            nn.LayerNorm([st*2, 4000]),
            nn.SiLU(),
            nn.AvgPool1d(8, stride=4, padding=2),
            nn.Dropout(0.5)
        )
        
        self.hidden_size = st*4
        self.num_layers = 1
        self.backbone = nn.GRU(st*2, self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=True)
        
        self.classifier = KANLinear(self.hidden_size*2, num_classes)
        
    def forward(self, x):        
        x = self.input_embedding(x)
        if (x.shape[0] != 1):
            x = x.squeeze()
        
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        
        h0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        output, _ = self.backbone(x, h0)
            
        output = self.classifier(output[:, -1, :])
        return output

# Kernel function for MiniBERT
class Phi(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return F.elu(x) + 1

class AttentionLayer(nn.Module): # Linear Attention using Kernel
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_query = nn.Linear(d_model, d_model)
        self.W_key = nn.Linear(d_model, d_model)
        self.W_value = nn.Linear(d_model, d_model)
        
        self.phi = Phi()
        
        self.out_proj = KANLinear(d_model, d_model)

    def forward(self, x):
        bs, seq_len, d_model = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Unroll last dim: (bs, seq_len, d_model) -> (bs, num_tokens, seq_len, num_head, head_dim)
        keys = keys.view(bs, seq_len, self.num_heads, self.head_dim)
        values = values.view(bs, seq_len, self.num_heads, self.head_dim)
        queries = queries.view(bs, seq_len, self.num_heads, self.head_dim)

        # Transpose: (bs, seq_len, num_heads, head_dim) -> (bs, num_heads, seq_len, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_weights = self.phi(keys.transpose(-1, -2)) @ values
        attn_weights = self.phi(queries) @ attn_weights

        # Combine heads, where self.d_model = self.num_heads * self.head_dim
        attn_weights = attn_weights.reshape(bs, seq_len, self.d_model)
        out = self.out_proj(attn_weights)

        return out

class EncoderLayer(nn.Module): # Pre-layernorm
    def __init__(self, d_model=512, num_heads=8, dropout=0.2):
        super().__init__()
        
        self.attention = AttentionLayer(d_model, num_heads)
        self.ff = KANLinear(d_model, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        tmp = self.ln1(x)
        tmp = self.attention(tmp)
        tmp = self.dropout(tmp)
        x = x + tmp
        
        tmp = self.ln2(x)
        tmp = self.ff(tmp)
        tmp = self.dropout(tmp)
        x = x + tmp
        return x

class MiniBERT(nn.Module):
    def __init__(self, num_classes=12, vocab_size=64000, embedding_dim=32):
        super().__init__()
        st = embedding_dim
        
        self.input_embedding = nn.Embedding(vocab_size, st)
        
        # Reduce sequence length from 8000 to 1000
        self.conv = nn.Sequential(
            nn.Conv1d(st, st*2, 5, stride=2, padding=2),
            nn.LayerNorm([st*2, 4000]),
            nn.SiLU(),
            nn.AvgPool1d(8, stride=4, padding=2)
        )
        
        self.upsampling = nn.Sequential(
            KANLinear(st*2, st*4),
            nn.Dropout(0.4)
        )
        
        self.pos_emb = nn.Embedding(1000, st*4)
        position = torch.arange(0, 1000).unsqueeze(0)
        self.register_buffer('position', position)
        
        self.backbone = EncoderLayer(d_model=st*4, num_heads=4, dropout=0.4)
        self.lastnorm = nn.LayerNorm(st*4)
        
        self.classifier = KANLinear(st*4, num_classes)
        
    def forward(self, x):        
        x = self.input_embedding(x)
        if (x.shape[0] != 1):
            x = x.squeeze()
        
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.upsampling(x)
        
        x = x + self.pos_emb(self.position)
        
        output = self.backbone(x)
        output = self.lastnorm(output)
            
        output = self.classifier(output[:, 0, :])
        return output
