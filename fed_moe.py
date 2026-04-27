"""
FedMoE: Personalized Federated Learning via Heterogeneity-Driven Expert Specialization
========================================================================================
Full PyTorch implementation of:
  - CNNBackbone         : shared federated backbone phi(x) -> h
  - ExpertFFN           : domain-specialized FFN layers E_k(h)
  - PrivateRouter       : per-client gating network (never shared)
  - FedMoEClient        : local model combining all three components
  - FedMoEServer        : aggregation, cluster assignment, active-set management
  - local_train_step    : one round of local SGD on a client
  - federated_round     : one full communication round
  - evaluate            : per-client evaluation utility

Usage
-----
    python fedmoe.py          # runs a toy simulation on synthetic data
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple
import math


# ─────────────────────────────────────────────────────────────────────────────
# 1. CNN Backbone (shared, federated across all clients)
# ─────────────────────────────────────────────────────────────────────────────


class CNNBackbone(nn.Module):
    """
    Shared CNN backbone phi: x -> h in R^{d_model}.

    Uses GroupNorm instead of BatchNorm so statistics are per-sample,
    avoiding cross-client stat divergence under non-IID distributions.
    Ends with Global Average Pooling + linear projection to fixed d_model.

    Args:
        in_channels  : number of input image channels (1=grayscale, 3=RGB)
        d_model      : output embedding dimension (fed to router and experts)
        base_filters : number of filters in first conv block (doubles each block)
    """

    def __init__(
        self, in_channels: int = 3, d_model: int = 128, base_filters: int = 32
    ):
        super().__init__()
        f = base_filters

        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, f, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=f),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/2 x W/2
            # Block 2
            nn.Conv2d(f, f * 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=f * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/4 x W/4
            # Block 3
            nn.Conv2d(f * 2, f * 4, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=f * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/8 x W/8
            # Block 4 (no pool — keep spatial resolution for GAP)
            nn.Conv2d(f * 4, f * 4, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=f * 4),
            nn.ReLU(inplace=True),
        )

        # Global Average Pooling collapses spatial dims -> (B, f*4)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Project to shared embedding dimension
        self.proj = nn.Linear(f * 4, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            h: (B, d_model)
        """
        feat = self.encoder(x)  # (B, f*4, H', W')
        feat = self.gap(feat)  # (B, f*4, 1, 1)
        feat = feat.flatten(1)  # (B, f*4)
        h = self.proj(feat)  # (B, d_model)
        return h


# ─────────────────────────────────────────────────────────────────────────────
# 2. Expert FFN (federated over active client set S_k)
# ─────────────────────────────────────────────────────────────────────────────


class ExpertFFN(nn.Module):
    """
    Domain-specialized two-layer FFN expert E_k: h -> output in R^{d_model}.

    Architecture:  h -> Linear(d_model, d_ff) -> GELU -> Linear(d_ff, d_model)

    Multiple ExpertFFN instances form the pool {E_1, ..., E_K}.
    Each expert is federated only among clients in its active set S_k.

    Args:
        d_model : input/output dimension (must match CNNBackbone d_model)
        d_ff    : hidden dimension of the FFN (typically 2-4x d_model)
    """

    def __init__(self, d_model: int = 128, d_ff: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, d_model)
        Returns:
            out: (B, d_model)
        """
        return self.net(h)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Private Router (stays local, NEVER uploaded to server)
# ─────────────────────────────────────────────────────────────────────────────


class PrivateRouter(nn.Module):
    """
    Per-client lightweight gating network psi_i.

    Produces sparse top-k routing weights over K experts.
    Parameters are NEVER communicated to the server.

    Args:
        d_model     : input dimension (matches backbone output)
        num_experts : number of experts K
        top_k       : number of experts activated per sample
    """

    def __init__(self, d_model: int = 128, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=True)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (B, d_model)
        Returns:
            weights : (B, K) sparse gating weights (zeros for inactive experts)
            indices : (B, top_k) indices of activated experts
        """
        logits = self.gate(h)  # (B, K)
        topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)  # (B, k)

        # Softmax over selected experts only
        topk_weights = F.softmax(topk_vals, dim=-1)  # (B, k)

        # Scatter back to full (B, K) sparse weight tensor
        weights = torch.zeros_like(logits)  # (B, K)
        weights.scatter_(-1, topk_idx, topk_weights)

        return weights, topk_idx

    @torch.no_grad()
    def routing_stats(
        self, dataloader: DataLoader, backbone: CNNBackbone, device: torch.device
    ) -> torch.Tensor:
        """
        Compute per-expert expected routing mass c_{i,k} = E_{x~D_i}[g_i^(k)(phi(x))].

        This scalar vector (shape: K) is the ONLY routing-related information
        sent to the server — used for cluster assignment update.

        Returns:
            stats: (K,) mean routing weight per expert over the dataset
        """
        backbone.eval()
        self.eval()
        total = torch.zeros(self.num_experts, device=device)
        count = 0
        for x, _ in dataloader:
            x = x.to(device)
            h = backbone(x)
            weights, _ = self(h)
            total += weights.mean(dim=0)
            count += 1
        return (total / count).cpu()


# ─────────────────────────────────────────────────────────────────────────────
# 4. FedMoE Client Model
# ─────────────────────────────────────────────────────────────────────────────


class FedMoEClient(nn.Module):
    """
    Full client model combining backbone, expert pool, private router,
    and a final classification head.

    Forward pass:
        h      = phi(x)                          via backbone
        g, idx = router_i(h)                     via private router
        mix    = sum_k g^(k) * E_k(h)           sparse expert mixture
        logits = classifier(h + mix)             residual + classify

    Args:
        backbone    : shared CNNBackbone (parameters will be federated)
        experts     : list of K ExpertFFN modules
        router      : PrivateRouter for this client (never shared)
        num_classes : number of output classes
        d_model     : embedding dimension
    """

    def __init__(
        self,
        backbone: CNNBackbone,
        experts: nn.ModuleList,
        router: PrivateRouter,
        num_classes: int = 10,
        d_model: int = 128,
    ):
        super().__init__()
        self.backbone = backbone
        self.experts = experts
        self.router = router
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            logits  : (B, num_classes)
            weights : (B, K) routing weights (for load-balance loss)
        """
        h = self.backbone(x)  # (B, d_model)
        weights, topk_idx = self.router(h)  # (B, K), (B, k)

        # Compute sparse expert mixture
        # Only activated experts contribute; avoids full forward through all experts
        mixture = torch.zeros_like(h)  # (B, d_model)
        for k, expert in enumerate(self.experts):
            # Mask: which samples in this batch activate expert k
            mask = (topk_idx == k).any(dim=-1)  # (B,)
            if mask.any():
                expert_out = expert(h[mask])  # (mask_count, d_model)
                w_k = weights[mask, k].unsqueeze(-1)  # (mask_count, 1)
                mixture[mask] += w_k * expert_out

        logits = self.classifier(h + mixture)  # residual connection
        return logits, weights

    def shared_parameters(self):
        """Parameters to be federated: backbone + experts."""
        params = list(self.backbone.parameters())
        for expert in self.experts:
            params += list(expert.parameters())
        return params

    def private_parameters(self):
        """Parameters that stay local: router + classifier."""
        return list(self.router.parameters()) + list(self.classifier.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# 5. Load-Balance Regularizer
# ─────────────────────────────────────────────────────────────────────────────


def load_balance_loss(weights: torch.Tensor) -> torch.Tensor:
    """
    Anti-collapse regularizer: squared coefficient of variation of expert load.

    CV²(g) = Var(mean_load) / Mean(mean_load)²

    Encourages uniform utilization across experts, preventing all clients
    from routing to the same expert (expert collapse).

    Args:
        weights: (B, K) routing weights for a mini-batch
    Returns:
        scalar loss
    """
    mean_load = weights.mean(dim=0)  # (K,) mean per expert
    var_load = weights.var(dim=0)  # (K,)
    cv2 = (var_load / (mean_load.pow(2) + 1e-8)).mean()
    return cv2


# ─────────────────────────────────────────────────────────────────────────────
# 6. Local Training Step
# ─────────────────────────────────────────────────────────────────────────────


def local_train_step(
    client_model: FedMoEClient,
    dataloader: DataLoader,
    num_local_steps: int,
    lr: float,
    lambda_lb: float,
    device: torch.device,
) -> Dict[str, float]:
    """
    Run R local SGD steps on client data.

    Uses separate optimizers for shared and private parameters so that
    learning rates can be tuned independently if needed.

    Returns dict with training metrics.
    """
    client_model.train()
    client_model.to(device)

    optimizer = torch.optim.Adam(
        [
            {"params": client_model.shared_parameters(), "lr": lr},
            {"params": client_model.private_parameters(), "lr": lr * 2.0},
        ]
    )

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_task = 0.0
    total_lb = 0.0
    steps = 0

    data_iter = iter(dataloader)

    for _ in range(num_local_steps):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        logits, weights = client_model(x)
        task_loss = criterion(logits, y)
        lb_loss = load_balance_loss(weights)
        loss = task_loss + lambda_lb * lb_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_task += task_loss.item()
        total_lb += lb_loss.item()
        steps += 1

    return {
        "loss": total_loss / steps,
        "task_loss": total_task / steps,
        "lb_loss": total_lb / steps,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. FedMoE Server
# ─────────────────────────────────────────────────────────────────────────────


class FedMoEServer:
    """
    Central server managing:
      - global backbone and expert pool parameters
      - active-set S_k tracking (which clients are assigned to which experts)
      - FedAvg aggregation of shared gradients
      - cluster reassignment every T_cluster rounds

    Args:
        backbone        : initial shared backbone (deep copied)
        experts         : initial expert pool (deep copied)
        num_clients     : total number of clients N
        num_experts     : K
        cluster_tau     : threshold tau for active-set assignment
        cluster_period  : update cluster assignments every this many rounds
    """

    def __init__(
        self,
        backbone: CNNBackbone,
        experts: nn.ModuleList,
        num_clients: int,
        num_experts: int,
        cluster_tau: float = 0.2,
        cluster_period: int = 5,
    ):
        self.backbone = copy.deepcopy(backbone)
        self.experts = copy.deepcopy(experts)
        self.num_clients = num_clients
        self.num_experts = num_experts
        self.cluster_tau = cluster_tau
        self.cluster_period = cluster_period
        self.round = 0

        # Active sets: S_k = set of client indices assigned to expert k
        # Start with all clients seeing all experts
        self.active_sets: Dict[int, set] = {
            k: set(range(num_clients)) for k in range(num_experts)
        }

        # Routing stats: c[i][k] = expected routing mass of client i to expert k
        self.routing_stats: Dict[int, torch.Tensor] = {}

    def get_client_model(
        self,
        client_id: int,
        router: PrivateRouter,
        num_classes: int,
        d_model: int,
    ) -> FedMoEClient:
        """
        Build a client model with the current global backbone + experts
        and the client's own private router.
        """
        # Only send experts where client_id is in active set
        active_experts = nn.ModuleList(
            [
                copy.deepcopy(self.experts[k])
                if client_id in self.active_sets[k]
                else copy.deepcopy(self.experts[k])  # still send all for simplicity
                for k in range(self.num_experts)
            ]
        )
        return FedMoEClient(
            backbone=copy.deepcopy(self.backbone),
            experts=active_experts,
            router=router,
            num_classes=num_classes,
            d_model=d_model,
        )

    def aggregate(
        self,
        client_models: Dict[int, FedMoEClient],
    ) -> None:
        """
        FedAvg aggregation of backbone and active-set expert parameters.

        Only clients in S_k contribute to expert E_k's update.
        """
        # ── Backbone: aggregate from all participating clients ──
        client_ids = list(client_models.keys())
        n = len(client_ids)

        backbone_state = {}
        for name, param in self.backbone.named_parameters():
            backbone_state[name] = torch.stack(
                [
                    client_models[i].backbone.state_dict()[name].float()
                    for i in client_ids
                ]
            ).mean(dim=0)

        for name, param in self.backbone.named_parameters():
            param.data.copy_(backbone_state[name])

        # ── Experts: aggregate from active-set clients only ──
        for k in range(self.num_experts):
            active_clients = [i for i in client_ids if i in self.active_sets[k]]
            if not active_clients:
                continue
            expert_state = {}
            for name, param in self.experts[k].named_parameters():
                expert_state[name] = torch.stack(
                    [
                        client_models[i].experts[k].state_dict()[name].float()
                        for i in active_clients
                    ]
                ).mean(dim=0)
            for name, param in self.experts[k].named_parameters():
                param.data.copy_(expert_state[name])

    def update_clusters(self, routing_stats: Dict[int, torch.Tensor]) -> None:
        """
        Update active sets S_k from client routing statistics.

        routing_stats[i] is a (K,) tensor of expected routing mass
        from client i — the ONLY routing info sent to server.
        """
        self.routing_stats.update(routing_stats)

        for k in range(self.num_experts):
            new_set = set()
            for i, stats in self.routing_stats.items():
                if stats[k].item() > self.cluster_tau:
                    new_set.add(i)
            # Ensure every expert has at least one client (fallback)
            if len(new_set) == 0:
                new_set = set(range(self.num_clients))
            self.active_sets[k] = new_set

        print(
            f"  [Server] Cluster sizes: "
            f"{[len(self.active_sets[k]) for k in range(self.num_experts)]}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 8. One Federated Round
# ─────────────────────────────────────────────────────────────────────────────


def federated_round(
    server: FedMoEServer,
    routers: Dict[int, PrivateRouter],
    dataloaders: Dict[int, DataLoader],
    participating_clients: List[int],
    num_local_steps: int,
    lr: float,
    lambda_lb: float,
    num_classes: int,
    d_model: int,
    device: torch.device,
) -> Dict[str, float]:
    """
    Execute one full federated communication round:
      1. Server broadcasts shared parameters to participating clients.
      2. Each client runs R local SGD steps.
      3. Clients upload shared gradients (router stays local).
      4. Server aggregates via FedAvg.
      5. Every cluster_period rounds: collect routing stats and update clusters.

    Returns aggregated training metrics across clients.
    """
    server.round += 1
    client_models = {}
    round_metrics = {"loss": 0.0, "task_loss": 0.0, "lb_loss": 0.0}

    for client_id in participating_clients:
        # Step 1: get model with current global weights + own private router
        model = server.get_client_model(
            client_id=client_id,
            router=routers[client_id],
            num_classes=num_classes,
            d_model=d_model,
        )
        model.to(device)

        # Step 2: local training
        metrics = local_train_step(
            client_model=model,
            dataloader=dataloaders[client_id],
            num_local_steps=num_local_steps,
            lr=lr,
            lambda_lb=lambda_lb,
            device=device,
        )
        for k in metrics:
            round_metrics[k] += metrics[k] / len(participating_clients)

        client_models[client_id] = model

    # Step 3+4: aggregate shared parameters
    server.aggregate(client_models)

    # Step 5: update clusters periodically
    if server.round % server.cluster_period == 0:
        print(f"  [Server] Round {server.round}: updating cluster assignments...")
        routing_stats = {}
        for client_id in participating_clients:
            stats = routers[client_id].routing_stats(
                dataloader=dataloaders[client_id],
                backbone=server.backbone,
                device=device,
            )
            routing_stats[client_id] = stats
        server.update_clusters(routing_stats)

    return round_metrics


# ─────────────────────────────────────────────────────────────────────────────
# 9. Evaluation
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(
    server: FedMoEServer,
    routers: Dict[int, PrivateRouter],
    dataloaders: Dict[int, DataLoader],
    num_classes: int,
    d_model: int,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate personalized accuracy for each client.

    Returns mean and per-client accuracy dict.
    """
    results = {}
    for client_id, loader in dataloaders.items():
        model = server.get_client_model(
            client_id=client_id,
            router=routers[client_id],
            num_classes=num_classes,
            d_model=d_model,
        )
        model.eval().to(device)

        correct = 0
        total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            correct += (logits.argmax(dim=-1) == y).sum().item()
            total += y.size(0)

        results[client_id] = correct / total

    mean_acc = sum(results.values()) / len(results)
    return {"mean_acc": mean_acc, "per_client": results}


# ─────────────────────────────────────────────────────────────────────────────
# 10. Factory: build all components for N clients
# ─────────────────────────────────────────────────────────────────────────────


def build_fedmoe(
    num_clients: int = 10,
    num_experts: int = 4,
    num_classes: int = 10,
    in_channels: int = 3,
    d_model: int = 128,
    d_ff: int = 256,
    base_filters: int = 32,
    top_k: int = 2,
    cluster_tau: float = 0.2,
    cluster_period: int = 5,
) -> Tuple[FedMoEServer, Dict[int, PrivateRouter]]:
    """
    Instantiate a FedMoEServer and per-client PrivateRouters.

    Returns:
        server  : FedMoEServer with global backbone + expert pool
        routers : dict mapping client_id -> PrivateRouter
    """
    backbone = CNNBackbone(
        in_channels=in_channels, d_model=d_model, base_filters=base_filters
    )
    experts = nn.ModuleList(
        [ExpertFFN(d_model=d_model, d_ff=d_ff) for _ in range(num_experts)]
    )
    server = FedMoEServer(
        backbone=backbone,
        experts=experts,
        num_clients=num_clients,
        num_experts=num_experts,
        cluster_tau=cluster_tau,
        cluster_period=cluster_period,
    )
    routers = {
        i: PrivateRouter(d_model=d_model, num_experts=num_experts, top_k=top_k)
        for i in range(num_clients)
    }
    return server, routers


# ─────────────────────────────────────────────────────────────────────────────
# 11. Toy Simulation (synthetic data)
# ─────────────────────────────────────────────────────────────────────────────


def make_synthetic_dataloaders(
    num_clients: int = 6,
    num_classes: int = 10,
    samples_per_client: int = 200,
    img_size: int = 32,
    in_channels: int = 3,
    batch_size: int = 32,
    heterogeneity: float = 0.3,  # Dirichlet alpha: lower = more non-IID
) -> Dict[int, DataLoader]:
    """
    Create synthetic non-IID dataloaders using Dirichlet label distribution.
    Each client gets images from a skewed subset of classes.
    """
    torch.manual_seed(42)
    loaders = {}

    # Sample per-client class distributions from Dirichlet
    alpha = torch.ones(num_classes) * heterogeneity
    class_dist = torch.distributions.Dirichlet(alpha).sample((num_clients,))  # (N, C)

    for i in range(num_clients):
        probs = class_dist[i]
        labels = torch.multinomial(probs, samples_per_client, replacement=True)
        # Synthetic images: random noise + class-specific mean shift for structure
        images = torch.randn(samples_per_client, in_channels, img_size, img_size)
        for c in range(num_classes):
            mask = labels == c
            images[mask] += (c / num_classes) * 0.5  # mild structure per class

        dataset = TensorDataset(images, labels)
        loaders[i] = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )

    return loaders


def run_simulation(
    num_rounds: int = 20,
    num_clients: int = 6,
    num_experts: int = 3,
    num_classes: int = 10,
    num_local_steps: int = 10,
    participation_rate: float = 1.0,
    lr: float = 1e-3,
    lambda_lb: float = 0.01,
    cluster_tau: float = 0.15,
    cluster_period: int = 5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(
        f"Running FedMoE simulation: {num_clients} clients, "
        f"{num_experts} experts, {num_rounds} rounds\n"
    )

    # Build data
    dataloaders = make_synthetic_dataloaders(
        num_clients=num_clients, num_classes=num_classes
    )

    # Build model
    server, routers = build_fedmoe(
        num_clients=num_clients,
        num_experts=num_experts,
        num_classes=num_classes,
        cluster_tau=cluster_tau,
        cluster_period=cluster_period,
    )

    for t in range(1, num_rounds + 1):
        # Sample participating clients
        n_participate = max(1, int(num_clients * participation_rate))
        participating = list(range(num_clients))[:n_participate]

        metrics = federated_round(
            server=server,
            routers=routers,
            dataloaders=dataloaders,
            participating_clients=participating,
            num_local_steps=num_local_steps,
            lr=lr,
            lambda_lb=lambda_lb,
            num_classes=num_classes,
            d_model=128,
            device=device,
        )

        if t % 5 == 0 or t == 1:
            eval_res = evaluate(
                server=server,
                routers=routers,
                dataloaders=dataloaders,
                num_classes=num_classes,
                d_model=128,
                device=device,
            )
            print(
                f"Round {t:3d} | loss={metrics['loss']:.4f} "
                f"task={metrics['task_loss']:.4f} "
                f"lb={metrics['lb_loss']:.4f} | "
                f"mean_acc={eval_res['mean_acc']:.3f}"
            )

    print("\nDone. Final per-client accuracy:")
    eval_res = evaluate(server, routers, dataloaders, num_classes, 128, device)
    for cid, acc in eval_res["per_client"].items():
        print(f"  Client {cid}: {acc:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_simulation()
