from torch import nn
import torch.nn.functional as F


class GatingNetwork(nn.Module):
    def __init__(self, input_shape, num_experts):
        super().__init__()
        # fc = fully connected
        self.fc = nn.Linear(input_shape, num_experts)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=1)
