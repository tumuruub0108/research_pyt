import torch
from torch import nn

from models.cnn_feature import CNNFeatureExtracor
from models.expert import Expert
from models.gating import GatingNetwork
from conf.config import NUM_EXPERTS

self, input_shape, hidden_units, output_shape, num_experts = 4


class MoE(nn.Module):
    def __init__(
        self, input_shape, hidden_units, output_shape, num_experts=NUM_EXPERTS
    ):
        super().__init__()

        # CNN backbone
        self.feature_extractor = CNNFeatureExtracor(
            input_shape=input_shape, hidden_units=hidden_units
        )

        self.feature.dim
