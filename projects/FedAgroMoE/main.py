import torch
from torch import nn
from tqdm.auto import tqdm

from data_loader.dataset import load_fashion_mnist
from training.train import train_step, test_step
from models.cnn import CNN
from utils.config import LEARNING_RATE, EPOCHS
from utils.helper_functions import accuracy_fn, eval_model


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_loader, test_loader, class_names = load_fashion_mnist()
    print(f"Lenght of train_loader: {len(train_loader)}.")
    print(f"Lenght of test_loader: {len(test_loader)}.")
    print(f"class names: {class_names}")

    cnn_model = CNN(input_shape=1, hidden_units=10, output_shape=len(class_names))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=cnn_model.parameters(), lr=LEARNING_RATE)

    for epoch in tqdm(range(EPOCHS)):
        print(f" Epoch: {epoch} ")

        train_step(
            model=cnn_model,
            data_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            accuracy_fn=accuracy_fn,
            device=device,
        )

        test_step(
            model=cnn_model,
            data_loader=test_loader,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            device=device,
        )

    cnn_results = eval_model(
        model=cnn_model,
        data_loader=test_loader,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
    )

    print(f"cnn_results: {cnn_results}")


if __name__ == "__main__":
    main()
