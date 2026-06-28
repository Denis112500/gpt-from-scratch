import torch
import torch.nn as nn
import math

# class SelfAttention(nn.Module):
#     def __init__(self, d_model):
#         super().__init__()
#         self.d_model = d_model
#         self.query = nn.Linear(d_model, d_model)
#         self.key = nn.Linear(d_model, d_model)
#         self.value = nn.Linear(d_model, d_model)
    
#     def forward(self, x, mask=None):       # forward pass
#         Q = self.query(x)
#         K = self.key(x)
#         V = self.value(x)
#         scores = (Q @ K.transpose (-2, -1)) / math.sqrt(self.d_model)
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, float('-inf'))
#         weights = torch.softmax(scores, dim=-1)
#         return weights @ V
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        batch, seq_len, _ = x.shape
        Q = Q.view(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        x = weights @ V
        x = x.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        x = self.out(x)
        return x

class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))
    
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads) # attention layer
        self.mlp = MLP(d_model)   # feedforward layer   
        self.ln1 = nn.LayerNorm(d_model) # norm before attention
        self.ln2 = nn.LayerNorm(d_model) # norm before MLP
    
    def forward(self, x, mask=None):
        x = x + self.attention(self.ln1(x), mask) # residual around attention
        x = x + self.mlp(self.ln2(x)) # residual around MLP
        return x
    
class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model) # token → vector
        self.position_embedding = nn.Embedding(max_seq_len, d_model) # position → vector
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, num_heads) for _ in range(num_layers)]
        )                                                             # N decoder blocks
        self.ln_final = nn.LayerNorm(d_model) # final layer norm
        self.head = nn.Linear(d_model, vocab_size, bias=False) # vector → vocab scores

    def forward(self, x, mask=None):
        positions = torch.arange(x.shape[1], device=x.device) # [0, 1, 2, ..., seq_len]
        x = self.token_embedding(x) + self.position_embedding(positions) # token + position vectors
        for block in self.blocks:
            x = block(x, mask) # pass through each decoder block
        x = self.ln_final(x) # final norm
        return self.head(x) # output vocab scores
