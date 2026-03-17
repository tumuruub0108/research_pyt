import torch


def train_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    accuracy_fn,
    device: torch.device,
):
    train_loss = 0
    train_accuracy = 0

    model.to(device)

    for X, y in data_loader:
        # send data to gpu
        X, y = X.to(device), y.to(device)

        # 1. forward pass
        y_pred = model(X)

        # 2.calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_accuracy += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        # 3. optimizer zero grad
        optimizer.zero_grad()

        # 4. loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    train_loss /= len(data_loader)
    train_accuracy /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_accuracy:.2f}%")


def test_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    accuracy_fn: torch.nn.Module,
    device: torch.device,
):
    test_loss = 0
    test_accuracy = 0

    model.to(device)
    model.eval()

    with torch.inference_mode():
        for X, y in data_loader:
            # send data to gpu
            X, y = X.to(device), y.to(device)

            # 1. forward pass
            test_pred = model(X)

            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_accuracy += accuracy_fn(
                y_true=y,
                y_pred=test_pred.argmax(dim=1),
            )

        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_accuracy /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_accuracy:.2f}%\n")
