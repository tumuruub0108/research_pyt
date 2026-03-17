from torch import nn


class Expert(nn.Module):
    def __init__(self, input_shape: int, output_shape: int):
        super().__init__()
        # fc = fully connected
        self.fc = nn.Linear(in_features=input_shape, out_features=output_shape)

    def forward(self, x):
        return self.fc(x)
