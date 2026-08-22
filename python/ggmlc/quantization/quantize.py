"""Block-level quantization and dequantization algorithms for GGML formats (Q8_0, Q4_0)."""

from __future__ import annotations

import numpy as np

BLOCK_SIZE = 32


def quantize_q8_0(data: np.ndarray) -> bytes:
    """Quantizes a float32 array into standard GGML Q8_0 format.

    Each 32-element block contains a 16-bit half-precision float scale
    and 32 signed 8-bit integers (34 bytes per 32 float values).

    Args:
        data: NumPy array of float32 values.

    Returns:
        Bytes containing Q8_0 packed binary payload.
    """
    flat = np.ascontiguousarray(data, dtype=np.float32).flatten()
    n_elements = flat.size
    if n_elements % BLOCK_SIZE != 0:
        pad_len = BLOCK_SIZE - (n_elements % BLOCK_SIZE)
        flat = np.pad(flat, (0, pad_len))

    n_blocks = flat.size // BLOCK_SIZE
    flat_blocks = flat.reshape(n_blocks, BLOCK_SIZE)

    out = bytearray()
    for block in flat_blocks:
        max_val = float(np.max(np.abs(block)))
        scale = max_val / 127.0 if max_val > 0 else 0.0
        # Convert scale to float16 bytes
        scale_fp16 = np.float16(scale)
        scale_bytes = scale_fp16.tobytes()

        # Quantize integers
        if scale > 0:
            id_scale = 1.0 / scale
            qs = np.clip(np.round(block * id_scale), -128, 127).astype(np.int8)
        else:
            qs = np.zeros(BLOCK_SIZE, dtype=np.int8)

        out.extend(scale_bytes)
        out.extend(qs.tobytes())

    return bytes(out)


def dequantize_q8_0(raw_bytes: bytes, shape: tuple[int, ...]) -> np.ndarray:
    """Dequantizes standard GGML Q8_0 bytes back into float32 array."""
    block_bytes = 34
    total_blocks = len(raw_bytes) // block_bytes
    total_elements = np.prod(shape)

    out = np.empty(total_blocks * BLOCK_SIZE, dtype=np.float32)
    for b in range(total_blocks):
        offset = b * block_bytes
        scale_fp16 = np.frombuffer(raw_bytes[offset : offset + 2], dtype=np.float16)[0]
        scale = float(scale_fp16)
        qs = np.frombuffer(raw_bytes[offset + 2 : offset + 34], dtype=np.int8)
        out[b * BLOCK_SIZE : (b + 1) * BLOCK_SIZE] = qs.astype(np.float32) * scale

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

    out = bytearray()
    for block in flat_blocks:
        max_val = float(np.max(np.abs(block)))
        scale = max_val / -8.0 if max_val > 0 else 0.0
        scale_fp16 = np.float16(scale)
        scale_bytes = scale_fp16.tobytes()

        # Quantize nibbles
        if scale != 0:
            id_scale = 1.0 / scale
            # Quantize to [-8, 7] and offset by 8 to [0, 15] for unsigned nibble storage
            q_vals = np.clip(np.round(block * id_scale) + 8, 0, 15).astype(np.uint8)
        else:
            q_vals = np.full(BLOCK_SIZE, 8, dtype=np.uint8)

        # Pack 32 nibbles into 16 bytes: byte i contains (q[i] & 0x0F) | ((q[i+16] & 0x0F) << 4)
        low_nibbles = q_vals[:16] & 0x0F
        high_nibbles = (q_vals[16:] & 0x0F) << 4
        packed = (low_nibbles | high_nibbles).astype(np.uint8)

        out.extend(scale_bytes)
        out.extend(packed.tobytes())

    return bytes(out)


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
