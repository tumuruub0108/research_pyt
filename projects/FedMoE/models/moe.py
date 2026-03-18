import torch
from torch import nn

from models.cnn_feature import CNNFeatureExtracor
from models.expert import Expert
from models.gating import GatingNetwork


class MoE(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape, num_experts: int):
        super().__init__()

        # CNN backbone
        self.feature_extractor = CNNFeatureExtracor(
            input_shape=input_shape, hidden_units=hidden_units
        )

        self.feature_dim = hidden_units * 7 * 7

        # Experts
        self.experts = nn.ModuleList(
            [
                Expert(self.feature_dim, output_shape=output_shape)
                for _ in range(num_experts)
            ]
        )
        print(f"number of experts: {len(self.experts)}")

        # Gating
        self.gating = GatingNetwork(self.feature_dim, num_experts=num_experts)

    def forward(self, x):
        features = self.feature_extractor(x)

        gate_weights = self.gating(features)

        expert_outputs = torch.stack(
            [expert(features) for expert in self.experts], dim=1
        )

        output = torch.sum(gate_weights.unsqueeze(-1) * expert_outputs, dim=1)

        return output
