"""Block-level quantization and dequantization algorithms for GGML formats (Q8_0, Q4_0)."""

from __future__ import annotations

import numpy as np

BLOCK_SIZE = 32


def _quantize_q8_0_chunk(flat_blocks: np.ndarray) -> bytes:
    n_blocks = flat_blocks.shape[0]
    max_val = np.max(np.abs(flat_blocks), axis=1)
    scale = np.where(max_val > 0, max_val / 127.0, 0.0)
    scale_fp16 = scale.astype(np.float16)

    safe_scale = np.where(scale != 0, scale, 1.0)[:, None]
    inv_scale = (1.0 / safe_scale).astype(np.float32)
    scaled = flat_blocks * inv_scale
    np.round(scaled, out=scaled)
    np.clip(scaled, -128, 127, out=scaled)
    qs = scaled.astype(np.int8)

    scale_bytes = scale_fp16.view(np.uint8).reshape(n_blocks, 2)
    qs_bytes = qs.view(np.uint8).reshape(n_blocks, BLOCK_SIZE)
    return np.hstack([scale_bytes, qs_bytes]).tobytes()


def quantize_q8_0(data: np.ndarray) -> bytes:
    """Quantizes float32 array into standard GGML Q8_0 format.
    Processes in cache-friendly chunks to minimize peak memory consumption."""
    flat = np.ascontiguousarray(data, dtype=np.float32).ravel()
    n_elements = flat.size
    pad_len = (BLOCK_SIZE - (n_elements % BLOCK_SIZE)) % BLOCK_SIZE
    if pad_len > 0:
        flat = np.pad(flat, (0, pad_len))

    n_blocks = flat.size // BLOCK_SIZE
    CHUNK_BLOCKS = 65536
    if n_blocks <= CHUNK_BLOCKS:
        return _quantize_q8_0_chunk(flat.reshape(n_blocks, BLOCK_SIZE))

    chunks = []
    for i in range(0, n_blocks, CHUNK_BLOCKS):
        end = min(i + CHUNK_BLOCKS, n_blocks)
        chunk_flat = flat[i * BLOCK_SIZE : end * BLOCK_SIZE].reshape(end - i, BLOCK_SIZE).copy()
        chunks.append(_quantize_q8_0_chunk(chunk_flat))
    return b"".join(chunks)


def dequantize_q8_0(raw_bytes: bytes, shape: tuple[int, ...]) -> np.ndarray:
    """Dequantizes standard GGML Q8_0 bytes back into float32 array."""
    block_bytes = 34
    total_blocks = len(raw_bytes) // block_bytes
    total_elements = np.prod(shape)

    raw_arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(total_blocks, block_bytes)
    scale_fp16 = raw_arr[:, :2].copy().view(np.float16).reshape(total_blocks, 1)
    scale = scale_fp16.astype(np.float32)

    qs = raw_arr[:, 2:].view(np.int8).astype(np.float32)
    out = (qs * scale).flatten()
    return out[:total_elements].reshape(shape)


def quantize_q4_0(data: np.ndarray) -> bytes:
    """Quantizes a float32 array into standard GGML Q4_0 format.

    Each 32-element block contains a 16-bit half-precision float scale
    and 16 bytes packing 32 4-bit signed nibbles (18 bytes per 32 float values).

    Args:
        data: NumPy array of float32 values.

    Returns:
        Bytes containing Q4_0 packed binary payload.
    """
    flat = np.ascontiguousarray(data, dtype=np.float32).flatten()
    n_elements = flat.size
    if n_elements % BLOCK_SIZE != 0:
        pad_len = BLOCK_SIZE - (n_elements % BLOCK_SIZE)
        flat = np.pad(flat, (0, pad_len))

    n_blocks = flat.size // BLOCK_SIZE
    flat_blocks = flat.reshape(n_blocks, BLOCK_SIZE)

    max_val = np.max(np.abs(flat_blocks), axis=1)
    scale = np.where(max_val > 0, max_val / -8.0, 0.0)
    scale_fp16 = scale.astype(np.float16)

    safe_scale = np.where(scale != 0, scale, 1.0)[:, None]
    q_vals = np.where(
        scale[:, None] != 0,
        np.clip(np.round(flat_blocks / safe_scale) + 8, 0, 15),
        8,
    ).astype(np.uint8)

    low_nibbles = q_vals[:, :16] & 0x0F
    high_nibbles = (q_vals[:, 16:] & 0x0F) << 4
    packed = (low_nibbles | high_nibbles).astype(np.uint8)

    scale_bytes = scale_fp16.view(np.uint8).reshape(n_blocks, 2)
    block_data = np.hstack([scale_bytes, packed])
    return block_data.tobytes()


def dequantize_q4_0(raw_bytes: bytes, shape: tuple[int, ...]) -> np.ndarray:
    """Dequantizes standard GGML Q4_0 bytes back into float32 array."""
    block_bytes = 18
    total_blocks = len(raw_bytes) // block_bytes
    total_elements = np.prod(shape)

    out = np.empty(total_blocks * BLOCK_SIZE, dtype=np.float32)
    for b in range(total_blocks):
        offset = b * block_bytes
        scale_fp16 = np.frombuffer(raw_bytes[offset : offset + 2], dtype=np.float16)[0]
        scale = float(scale_fp16)
        packed = np.frombuffer(raw_bytes[offset + 2 : offset + 18], dtype=np.uint8)

        low = (packed & 0x0F).astype(np.int8) - 8
        high = ((packed >> 4) & 0x0F).astype(np.int8) - 8

        out[b * BLOCK_SIZE : b * BLOCK_SIZE + 16] = low.astype(np.float32) * scale
        out[b * BLOCK_SIZE + 16 : (b + 1) * BLOCK_SIZE] = high.astype(np.float32) * scale

    return out[:total_elements].reshape(shape)
