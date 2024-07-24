import torch
import torch.nn as nn
from typing import Optional

# Import Mamba components
from mamba_ssm import Mamba, Block

class TopoformerMambaBlock(Block):
    def __init__(self, d_model, **kwargs):
        super().__init__(d_model, **kwargs)
        
        # Add Topoformer-specific layers
        self.spatial_query = nn.Linear(d_model, kwargs.get('d_state', 16))
        self.spatial_reweight = nn.Linear(d_model, kwargs.get('d_state', 16))

    def forward(self, x):
        # Apply spatial querying before Mamba processing
        query = self.spatial_query(x)
        
        # Process through original Mamba layer
        x = super().forward(x)
        
        # Apply spatial reweighting after Mamba processing
        reweight = self.spatial_reweight(x)
        x = x * reweight.unsqueeze(-1)
        
        return x

class MambaLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layer: int, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TopoformerMambaBlock(d_model, **kwargs)
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

def create_mamba_model(vocab_size: int, d_model: int = 512, n_layer: int = 8, **kwargs) -> MambaLM:
    return MambaLM(vocab_size, d_model, n_layer, **kwargs)