import flwr as fl
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np

from model import FedMoEModel
from client import FedMoEClient
from strategy import QuantizedExpertAggregation

# ── Config ───────────────────────────────────────────────────────────────────

NUM_CLIENTS = 10
NUM_ROUNDS = 5
NUM_EXPERTS = 8
EXPERTS_TO_SEND = 4  # k: how many experts each client transmits
QUANT_BITS = 8
BATCH_SIZE = 32

# Non-expert param count: encoder(weight+bias) + gate(weight+bias) + classifier(weight+bias) = 6
NUM_NON_EXPERT_PARAMS = 6
# Layers per expert: Linear(weight+bias) × 2 = 4
LAYERS_PER_EXPERT = 4

# ── Data ─────────────────────────────────────────────────────────────────────


def load_datasets(num_clients: int):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_full = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_full = datasets.MNIST(
        "./data", train=False, download=True, transform=transform
    )

    # Non-IID partition: each client gets 2 dominant classes
    partition_size = len(train_full) // num_clients
    trainloaders, testloaders = [], []

    for i in range(num_clients):
        indices = list(range(i * partition_size, (i + 1) * partition_size))
        trainloaders.append(
            DataLoader(Subset(train_full, indices), batch_size=BATCH_SIZE, shuffle=True)
        )
        testloaders.append(DataLoader(test_full, batch_size=BATCH_SIZE))

    return trainloaders, testloaders


# ── Client factory ────────────────────────────────────────────────────────────


def client_fn(cid: str):
    trainloaders, testloaders = load_datasets(NUM_CLIENTS)
    model = FedMoEModel(num_experts=NUM_EXPERTS)
    return FedMoEClient(
        cid=cid,
        model=model,
        trainloader=trainloaders[int(cid)],
        testloader=testloaders[int(cid)],
        num_experts_to_send=EXPERTS_TO_SEND,
        quant_bits=QUANT_BITS,
    ).to_client()


# ── Strategy ──────────────────────────────────────────────────────────────────

strategy = QuantizedExpertAggregation(
    num_experts=NUM_EXPERTS,
    num_non_expert_params=NUM_NON_EXPERT_PARAMS,
    layers_per_expert=LAYERS_PER_EXPERT,
    fraction_fit=1.0,
    min_fit_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
)

# ── Simulation ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )
