from torchvision.datasets import FashionMNIST, CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from conf.config import BATCH_SIZE, DATA_PATH


def load_fashion_mnist(batch_size=BATCH_SIZE):
    train_data = FashionMNIST(
        root=DATA_PATH,
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = FashionMNIST(
        root=DATA_PATH,
        train=False,
        download=True,
        transform=ToTensor(),
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loader, test_loader, train_data.classes


def load_cifar10(batch_size=BATCH_SIZE):
    train_data = CIFAR10(
        root=DATA_PATH,
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = CIFAR10(
        root=DATA_PATH,
        train=False,
        download=True,
        transform=ToTensor(),
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loader, test_loader, train_data.classes
