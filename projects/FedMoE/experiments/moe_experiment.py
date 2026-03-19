import torch
from torch import nn
from tqdm.auto import tqdm
from conf.config import EPOCHS, LEARNING_RATE
from training.train import train_step, test_step
from utils.helper_functions import accuracy_fn, eval_model


def train_moe_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names,
):

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE)

    for epoch in tqdm(range(EPOCHS)):
        print(f" Epoch: {epoch} ")

        train_step(
            model=model,
            data_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            accuracy_fn=accuracy_fn,
            device=device,
        )

        test_step(
            model=model,
            data_loader=test_loader,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            device=device,
        )

    moe_results = eval_model(
        model=model,
        data_loader=test_loader,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
    )

    return moe_results
