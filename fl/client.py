import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import OrderedDict
from typing import List, Tuple, Dict

from model import FedMoEModel
from quantization import (
    quantize_expert_params,
    dequantize_expert_params,
    compute_compression_ratio,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_expert_params(model: FedMoEModel, expert_idx: int) -> List[np.ndarray]:
    """Extract all parameter tensors for one expert as numpy arrays."""
    expert = model.moe.experts[expert_idx]
    return [p.detach().cpu().numpy() for p in expert.parameters()]


def set_expert_params(model: FedMoEModel, expert_idx: int, params: List[np.ndarray]):
    """Load numpy arrays back into a specific expert."""
    expert = model.moe.experts[expert_idx]
    for p_model, p_new in zip(expert.parameters(), params):
        p_model.data = torch.tensor(p_new, dtype=p_model.dtype).to(DEVICE)


def select_top_experts(importance: np.ndarray, top_k: int) -> List[int]:
    """Return indices of top-k experts by importance score."""
    return np.argsort(importance)[-top_k:].tolist()


class FedMoEClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: str,
        model: FedMoEModel,
        trainloader,
        testloader,
        num_experts_to_send: int = 4,
        quant_bits: int = 8,
    ):
        self.cid = cid
        self.model = model.to(DEVICE)
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_experts_to_send = num_experts_to_send
        self.quant_bits = quant_bits

    # ── Flower required methods ─────────────────────────────────────────────

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return NON-expert params (encoder + gate + classifier) as full float32.
        Expert params are handled via quantized sparse payload in fit()."""
        return self._get_non_expert_params()

    def fit(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # 1. Load non-expert params from server
        self._set_non_expert_params(parameters)

        # 2. Local training
        importance = self._train_and_get_importance()

        # 3. Select top experts
        selected = select_top_experts(importance, self.num_experts_to_send)

        # 4. Quantize selected experts
        quantized_payload = {}
        for idx in selected:
            raw_params = get_expert_params(self.model, idx)
            quantized_payload[idx] = quantize_expert_params(
                raw_params, bits=self.quant_bits
            )

        # 5. Pack everything into a flat numpy list for Flower
        #    Format: [non_expert_params..., packed_expert_metadata]
        non_expert = self._get_non_expert_params()
        packed = self._pack_expert_payload(quantized_payload, selected)

        metrics = {
            "selected_experts": str(selected),
            "compression_ratio": float(
                compute_compression_ratio(
                    get_expert_params(self.model, selected[0]),
                    quantized_payload[selected[0]],
                    self.quant_bits,
                )
            ),
            "client_id": self.cid,
        }

        return non_expert + packed, len(self.trainloader.dataset), metrics

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[float, int, Dict]:
        self._set_non_expert_params(parameters)
        loss, accuracy = self._test()
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}

    # ── Training helpers ────────────────────────────────────────────────────

    def _train_and_get_importance(self, epochs: int = 1) -> np.ndarray:
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        accumulated_importance = np.zeros(self.model.num_experts)

        for _ in range(epochs):
            for X, y in self.trainloader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                logits, gate_weights = self.model(X)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                # Accumulate expert importance from gate activations
                importance_batch = self.model.moe.get_expert_importance(
                    gate_weights.detach().cpu()
                ).numpy()
                accumulated_importance += importance_batch

        return accumulated_importance / len(self.trainloader)

    def _test(self) -> Tuple[float, float]:
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for X, y in self.testloader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                logits, _ = self.model(X)
                total_loss += criterion(logits, y).item()
                correct += (logits.argmax(dim=1) == y).sum().item()
                total += y.size(0)

        return total_loss / len(self.testloader), correct / total

    # ── Parameter packing helpers ───────────────────────────────────────────

    def _get_non_expert_params(self) -> List[np.ndarray]:
        params = []
        for name, p in self.model.named_parameters():
            if "experts" not in name:
                params.append(p.detach().cpu().numpy())
        return params

    def _set_non_expert_params(self, params: List[np.ndarray]):
        idx = 0
        for name, p in self.model.named_parameters():
            if "experts" not in name:
                p.data = torch.tensor(params[idx], dtype=p.dtype).to(DEVICE)
                idx += 1

    def _pack_expert_payload(
        self, quantized_payload: Dict, selected: List[int]
    ) -> List[np.ndarray]:
        """
        Pack quantized expert data into flat numpy arrays for Flower transport.
        Layout per expert: [index_array, q_weights_flat, scales, zero_points, shapes_flat]
        """
        packed = []
        # Metadata: which experts are included
        packed.append(np.array(selected, dtype=np.int32))

        for idx in selected:
            quant_layers = quantized_payload[idx]
            for layer in quant_layers:
                packed.append(
                    layer["q_weights"].flatten().astype(np.float32)
                )  # cast for Flower compat
                packed.append(
                    np.array([layer["scale"], layer["zero_point"]], dtype=np.float32)
                )
                packed.append(np.array(layer["shape"], dtype=np.int32))

        return packed
