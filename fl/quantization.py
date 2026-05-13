import numpy as np
from typing import List, Tuple, Dict


def quantize_weights(
    weights: np.ndarray, bits: int = 8
) -> Tuple[np.ndarray, float, float]:
    """
    Symmetric min-max quantization to `bits` bits.

    Returns:
        q_weights : quantized int array
        scale     : float scale factor
        zero_point: float zero point
    """
    w_min, w_max = weights.min(), weights.max()
    q_max = (2 ** (bits - 1)) - 1  # e.g. 127 for INT8

    scale = (
        max(abs(w_min), abs(w_max)) / q_max if max(abs(w_min), abs(w_max)) > 0 else 1.0
    )
    zero_point = 0.0  # symmetric → zero_point = 0

    q_weights = np.clip(np.round(weights / scale), -q_max, q_max).astype(
        np.int8 if bits == 8 else np.int16
    )
    return q_weights, scale, zero_point


def dequantize_weights(
    q_weights: np.ndarray, scale: float, zero_point: float = 0.0
) -> np.ndarray:
    """Reconstruct float weights from quantized representation."""
    return q_weights.astype(np.float32) * scale + zero_point


# ── Expert-level helpers ────────────────────────────────────────────────────


def quantize_expert_params(params: List[np.ndarray], bits: int = 8) -> List[Dict]:
    """
    Quantize all weight tensors of a single expert.

    Returns a list of dicts: {q_weights, scale, zero_point, shape, dtype}
    """
    quantized = []
    for p in params:
        original_shape = p.shape
        flat = p.flatten()
        q, scale, zp = quantize_weights(flat, bits=bits)
        quantized.append(
            {
                "q_weights": q,
                "scale": scale,
                "zero_point": zp,
                "shape": original_shape,
            }
        )
    return quantized


def dequantize_expert_params(quantized: List[Dict]) -> List[np.ndarray]:
    """Reconstruct float params from quantized expert representation."""
    return [
        dequantize_weights(d["q_weights"], d["scale"], d["zero_point"]).reshape(
            d["shape"]
        )
        for d in quantized
    ]


def compute_compression_ratio(
    original_params: List[np.ndarray], quantized_params: List[Dict], bits: int = 8
) -> float:
    """Estimate bytes saved vs. full float32 transmission."""
    original_bytes = sum(p.nbytes for p in original_params)
    quantized_bytes = sum(
        d["q_weights"].nbytes + 8 for d in quantized_params
    )  # +8 for scale/zp
    return original_bytes / quantized_bytes if quantized_bytes > 0 else 1.0
