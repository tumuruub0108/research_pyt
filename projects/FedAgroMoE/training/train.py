import torch


def train_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    accuracy_fn,
    device: torch.device,
):
    print("implement train step...")
    train_loss = 0
    train_accuracy = 0

    model.to(device)

    for X, y in enumerate(data_loader):
        # send data to gpu
        X, y = X.to(device), y.to(device)
        print("for loop.")


def test_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    accurcay_fn: torch.nn.Module,
    device: torch.device,
):
    print("implement test step...")
