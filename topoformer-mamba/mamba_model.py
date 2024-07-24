import torch
import torch.nn as nn
from typing import Optional
from mamba_ssm import Mamba

class TopoformerMambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, dropout: float = 0.1):
        super().__init__()
        self.prenorm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.ones(1))

        # Topoformer-specific layers
        self.spatial_query = nn.Linear(d_model, d_model)
        self.spatial_reweight = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.prenorm(x)

        # Apply spatial querying
        query = self.spatial_query(x)
        x = x + query

        # Process through Mamba
        x = self.mamba(x)

        # Apply spatial reweighting
        reweight = self.spatial_reweight(x)
        x = x * reweight

        x = self.dropout(x)
        return residual + self.residual_scale * x

class MambaLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layer: int, d_state: int, d_conv: int, expand: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TopoformerMambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layer)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding(x)
        
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.final_norm(x)
        return self.lm_head(x)

def create_mamba_model(vocab_size: int, d_model: int = 512, n_layer: int = 8, d_state: int = 64, d_conv: int = 4, expand: int = 2, dropout: float = 0.1) -> MambaLM:
    return MambaLM(vocab_size, d_model, n_layer, d_state, d_conv, expand, dropout)