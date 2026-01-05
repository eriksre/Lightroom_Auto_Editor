#!/usr/bin/env python3
"""
Shared neural network models for the Lightroom auto-editor.
"""

import torch
import torch.nn as nn


class LightroomMLP(nn.Module):
    """Multi-layer perceptron for predicting Lightroom edit parameters."""
    
    def __init__(
        self,
        input_dim: int = 574,  # 384 DINOv2 + 190 traditional features
        hidden_dims: list = None,
        output_dim: int = 57,
        dropout: float = 0.2
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        # Output layer (no activation for regression)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

