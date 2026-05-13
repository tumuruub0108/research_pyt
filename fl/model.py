import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
import numpy as np


class Expert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class MoELayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int,
        top_k: int = 2,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList(
            [Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_logits = self.gate(x)  # (B, num_experts)
        gate_weights = F.softmax(gate_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(gate_weights, self.top_k, dim=-1)

        output = torch.zeros(
            x.size(0), self.experts[0].net[-1].out_features, device=x.device
        )
        for i, expert in enumerate(self.experts):
            mask = (topk_indices == i).any(dim=-1)  # (B,)
            if mask.any():
                expert_out = expert(x[mask])
                w = gate_weights[mask, i].unsqueeze(-1)
                output[mask] += w * expert_out

        return output, gate_weights  # also return gate weights for selection

    def get_expert_importance(self, gate_weights: torch.Tensor) -> torch.Tensor:
        """Mean gate activation across batch → importance per expert."""
        return gate_weights.mean(dim=0)  # (num_experts,)


class FedMoEModel(nn.Module):
    def __init__(
        self, input_dim=784, hidden_dim=256, num_classes=10, num_experts=8, top_k=2
    ):
        super().__init__()
        self.encoder = nn.Linear(input_dim, 128)
        self.moe = MoELayer(128, hidden_dim, 128, num_experts, top_k)
        self.classifier = nn.Linear(128, num_classes)
        self.num_experts = num_experts

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.encoder(x))
        x, gate_weights = self.moe(x)
        return self.classifier(x), gate_weights
