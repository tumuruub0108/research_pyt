import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import (
    Parameters,
    FitRes,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from collections import defaultdict

from quantization import dequantize_expert_params


class QuantizedExpertAggregation(FedAvg):
    """
    Custom Flower strategy implementing Quantized Expert Aggregation.

    Non-expert params   → standard FedAvg
    Expert params       → sparse weighted average by participation count
                          after dequantization
    """

    def __init__(
        self,
        num_experts: int,
        num_non_expert_params: int,
        layers_per_expert: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.num_non_expert_params = num_non_expert_params
        self.layers_per_expert = layers_per_expert  # weights + biases per expert

        # Running server-side expert store (float32 full precision)
        self.server_expert_store: Dict[int, Optional[List[np.ndarray]]] = {
            i: None for i in range(num_experts)
        }
        self.expert_participation: Dict[int, int] = defaultdict(int)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple],
        failures: List,
    ) -> Tuple[Optional[Parameters], Dict]:

        if not results:
            return None, {}

        # ── 1. Separate non-expert and expert payloads ──────────────────────
        all_non_expert: List[Tuple[List[np.ndarray], int]] = []
        expert_updates: Dict[int, List[Tuple[List[np.ndarray], int]]] = defaultdict(
            list
        )

        for client_proxy, fit_res in results:
            params = parameters_to_ndarrays(fit_res.parameters)
            num_samples = fit_res.num_examples

            non_expert = params[: self.num_non_expert_params]
            expert_payload = params[self.num_non_expert_params :]

            all_non_expert.append((non_expert, num_samples))

            # Unpack expert payload
            selected_experts, dequantized = self._unpack_expert_payload(expert_payload)
            for idx, deq_params in zip(selected_experts, dequantized):
                expert_updates[idx].append((deq_params, num_samples))

        # ── 2. FedAvg on non-expert params ──────────────────────────────────
        total_samples = sum(n for _, n in all_non_expert)
        agg_non_expert = [
            sum(p[i] * n / total_samples for p, n in all_non_expert)
            for i in range(self.num_non_expert_params)
        ]

        # ── 3. Weighted average on expert params (sparse) ───────────────────
        for expert_idx, updates in expert_updates.items():
            total = sum(n for _, n in updates)
            if self.server_expert_store[expert_idx] is None:
                # First time: initialise from first update
                self.server_expert_store[expert_idx] = [
                    sum(p[i] * n / total for p, n in updates)
                    for i in range(len(updates[0][0]))
                ]
            else:
                # Weighted blend: existing store + new updates
                existing = self.server_expert_store[expert_idx]
                new_avg = [
                    sum(p[i] * n / total for p, n in updates)
                    for i in range(len(existing))
                ]
                # Momentum-style blend (gives weight to past knowledge)
                alpha = total / (total + self.expert_participation[expert_idx] + 1e-8)
                self.server_expert_store[expert_idx] = [
                    (1 - alpha) * e + alpha * n for e, n in zip(existing, new_avg)
                ]
            self.expert_participation[expert_idx] += total

        # ── 4. Pack aggregated params back for broadcast ─────────────────────
        # Only non-expert params are broadcast; experts pulled on demand
        aggregated = ndarrays_to_parameters(agg_non_expert)

        metrics = {
            "round": server_round,
            "experts_updated": str(list(expert_updates.keys())),
            "experts_coverage": f"{len(expert_updates)}/{self.num_experts}",
        }
        print(
            f"[Round {server_round}] Expert coverage: {metrics['experts_coverage']} "
            f"| Updated: {metrics['experts_updated']}"
        )

        return aggregated, metrics

    # ── Payload unpacking ────────────────────────────────────────────────────

    def _unpack_expert_payload(
        self, payload: List[np.ndarray]
    ) -> Tuple[List[int], List[List[np.ndarray]]]:
        """
        Inverse of client._pack_expert_payload.
        Returns selected expert indices and their dequantized params.
        """
        if len(payload) == 0:
            return [], []

        selected = payload[0].astype(int).tolist()
        cursor = 1
        dequantized_all = []

        for _ in selected:
            expert_layers = []
            for _ in range(self.layers_per_expert):
                if cursor + 2 >= len(payload):
                    break
                q_flat = payload[cursor]
                scale, zero_point = (
                    float(payload[cursor + 1][0]),
                    float(payload[cursor + 1][1]),
                )
                shape = tuple(payload[cursor + 2].astype(int).tolist())
                cursor += 3

                # Dequantize
                reconstructed = (q_flat * scale + zero_point).reshape(shape)
                expert_layers.append(reconstructed)

            dequantized_all.append(expert_layers)

        return selected, dequantized_all
