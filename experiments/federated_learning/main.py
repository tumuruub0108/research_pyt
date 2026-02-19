from collections import OrderedDict
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10

import flwr as fl
from flwr.common import Metrics

DEVICE = torch.device("cpu")
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)

CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

NUM_CLIENTS = 10
BATCH_SIZE = 32


def load_datasets():
    # Download and transform CIFAR-10(train and test)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = CIFAR10("../dataset", train=True, download=True, transform=transform)
    testset = CIFAR10("../dataset", train=False, download=True, transform=transform)

    # Split training set into 10 partitions to simulate the individual dataset
    partition_size = len(trainset) // NUM_CLIENTS
    lengths = [partition_size] * NUM_CLIENTS
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

    # Split each partition into train, validation, and create DataLoader
    trainloaders = []
    valloaders = []

    for ds in datasets:
        len_val = len(ds) // 10  # 10% validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]

        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))

        trainloaders.append(DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=BATCH_SIZE))

    testloader = DataLoader(testset, batch_size=BATCH_SIZE)

    return trainloaders, valloaders, testloader


trainloaders, valloaders, testloader = load_datasets()

images, labels = next(iter(trainloaders[0]))

# Reshape and convert images to a NumPy array
# Matplotlib requires images with the shape(height, width, 3)
images = images.permute(0, 2, 3, 1).numpy()

# Denormalize
images = images / 2 + 0.5

# create a figure and a grid of subplots
fig, axs = plt.subplots(4, 8, figsize=(12, 6))

# loop over the images and plot them
for i, ax in enumerate(axs.flat):
    ax.imshow(images[i])
    ax.set_title(CLASSES[labels[i]])
    ax.axis("off")

# show the plot
fig.tight_layout()
plt.show()


# model
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def train(net, trainloader, epochs: int, verbose=False):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net.train()

    for epoch in range(epochs):
        correct, total = 0, 0
        running_loss = 0.0

        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total += labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()

        epoch_loss = running_loss / len(trainloader)
        epoch_acc = correct / total

        if verbose:
            print(
                f"Epoch {epoch + 1}: train loss={epoch_loss:.4f}, acc={epoch_acc:.4f}"
            )


def test(net, testloader):
    """Evaluate the network on the entire test set"""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss = loss + criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total = total + labels.size(0)

    loss = loss / len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


trainloader = trainloaders[0]
valloader = valloaders[0]

net = Net().to(DEVICE)

for epoch in range(5):
    train(net, trainloader, 1)
    loss, accuracy = test(net, valloader)
    print(f"Epoch {epoch + 1}: validation loss: {loss}, accuracy: {accuracy}")

loss, accuracy = test(net, testloader)
print(f"Final test set performance:\n\tloss: {loss}\n\taccuracy:{accuracy}")


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


class FLowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return (loss), len(self.valloader), {"accuracy": float(accuracy)}


def client_fn(cid: str) -> FLowerClient:
    """Create a Flower client representing a single organization"""

    # load Model
    net = Net().to(DEVICE)

    # Load data
    # Note: each client gets a different trainloader/valloader, so each client will train and evaluate on their own unique data

    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]

    # create a single flower client representing a single organization
    return FLowerClient(net, trainloader, valloader)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # aggregate and return custom metric(weighed average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# create Fedavg strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=0.5,
    min_fit_clients=10,
    min_evaluate_clients=5,
    min_available_clients=10,
    evaluate_metrics_aggregation_fn=weighted_average,
)

# specify client resource if you need GPU (defaults to 1 CPU and 0 GPU)
client_resources = None
if DEVICE.type == "cuda":
    client_resources = {"num_cpus": 2, "num_gpus": 1}

# start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
    client_resources=client_resources,
)
