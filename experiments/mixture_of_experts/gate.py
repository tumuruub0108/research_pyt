import torch
import torch.nn as nn
import torch.nn.functional as F


class Gate(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.linear = nn.Line2ar(input_dim, num_experts)

    def forward(self, x):
        logits = self.linear(x)  # Wx + b
        probs = F.softmax(logits, dim=1)  # softmax
        return probs


x = torch.tensor([[2.0, 1.0]])
print(x.shape)
gate = Gate(2, 3)

probs = gate(x)
print(probs)
