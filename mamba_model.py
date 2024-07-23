import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(x + self.mamba(x))

class MambaLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_layer, d_state, d_conv, expand):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layer)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)

def create_mamba_model(vocab_size, d_model=256, n_layer=4, d_state=16, d_conv=4, expand=2):
    return MambaLM(vocab_size, d_model, n_layer, d_state, d_conv, expand)