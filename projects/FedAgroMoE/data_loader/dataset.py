from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


def load_fashion_mnist(batch_size=32):

    train_data = datasets.FashionMNIST(
        root="../../../datasets",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = datasets.FashionMNIST(
        root="../../../datasets",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loader, test_loader
