import torch
from torch import nn

from data_loader.dataset import load_fashion_mnist
from models.cnn import CNN


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_loader, test_loader = load_fashion_mnist()
    print(f"Lenght of train_loader: {len(train_loader)}.")
    print(f"Lenght of test_loader: {len(test_loader)}.")

    #  loss_fn = nn.CrossEntropyLoss()
    #  optimizer = torch.optim.SGD(params="model parameter here...", lr=0.1)


if __name__ == "__main__":
    main()
