from torch import nn
from conf.config import KERNEL_SIZE, PADDING, STRIDE


class CNNFeatureExtracor(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int):
        super().__init__()
        self.conv_block_1 = (
            nn.Sequential(
                nn.Conv2d(
                    in_channels=input_shape,
                    out_channels=hidden_units,
                    kernel_size=KERNEL_SIZE,
                    padding=PADDING,
                    stride=STRIDE,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=KERNEL_SIZE,
                    padding=PADDING,
                    stride=STRIDE,
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
            ),
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=KERNEL_SIZE,
                padding=PADDING,
                stride=STRIDE,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=KERNEL_SIZE,
                padding=PADDING,
                stride=STRIDE,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.flatten(x)

        return x
