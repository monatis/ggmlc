"""Quantization policies and mixed-precision recipes (Unsloth Dynamic, Q4_K_M, Q8_0, etc.)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ggmlc.ir.dtype import DType
from ggmlc.quantization.roles import TensorRole


@dataclass
class QuantizationPolicy:
    """Policy mapping tensor roles and specific name patterns to target data types."""

    name: str
    role_map: dict[TensorRole, DType]
    name_overrides: dict[str, DType] = field(default_factory=dict)
    fallback_dtype: DType = DType.Q4_0

    def resolve_dtype(
        self, name: str, role: TensorRole, dims: list[int] | tuple[int, ...] | None = None
    ) -> DType:
        """Determines the target precision for a tensor under this policy."""
        # 1. 1D tensors ALWAYS remain in F32 (Strict 1D F32 Rule)
        if role == TensorRole.NORMALIZATION_1D:
            return DType.F32
        if dims is not None:
            non_unit_dims = [d for d in dims if d > 1]
            if len(non_unit_dims) <= 1:
                return DType.F32

        # 2. Check explicit regex/name overrides
        for pattern, dtype in self.name_overrides.items():
            if re.search(pattern, name, re.IGNORECASE):
                return dtype

        # 3. Check role mapping
        if role in self.role_map:
            return self.role_map[role]

        # 4. Fallback
        return self.fallback_dtype


# ---------------------------------------------------------------------------
# Standard Built-In Policies
# ---------------------------------------------------------------------------

# 1. Unsloth Dynamic Quantization Recipe (UD_Q4_M)
# Protects embeddings and output in high precision, sensitive Value/Down in Q8_0,
# other weights in Q4_0, and 1D in F32.
UNSLOTH_DYNAMIC_POLICY = QuantizationPolicy(
    name="unsloth_dynamic",
    role_map={
        TensorRole.EMBEDDING: DType.F16,
        TensorRole.OUTPUT_HEAD: DType.Q8_0,
        TensorRole.ATTN_VALUE: DType.Q8_0,
        TensorRole.ATTN_OUT: DType.Q8_0,
        TensorRole.FFN_DOWN: DType.Q8_0,
        TensorRole.ATTN_QK: DType.Q4_0,
        TensorRole.FFN_GATE_UP: DType.Q4_0,
        TensorRole.GENERIC_WEIGHT: DType.Q4_0,
        TensorRole.NORMALIZATION_1D: DType.F32,
    },
    fallback_dtype=DType.Q4_0,
)

# 2. Standard llama.cpp Q4_K_M Mixed-Precision
Q4_K_M_POLICY = QuantizationPolicy(
    name="q4_k_m",
    role_map={
        TensorRole.EMBEDDING: DType.Q4_0,
        TensorRole.OUTPUT_HEAD: DType.Q8_0,
        TensorRole.ATTN_VALUE: DType.Q8_0,
        TensorRole.ATTN_OUT: DType.Q8_0,
        TensorRole.FFN_DOWN: DType.Q8_0,
        TensorRole.ATTN_QK: DType.Q4_0,
        TensorRole.FFN_GATE_UP: DType.Q4_0,
        TensorRole.GENERIC_WEIGHT: DType.Q4_0,
        TensorRole.NORMALIZATION_1D: DType.F32,
    },
    fallback_dtype=DType.Q4_0,
)

# 3. Uniform Q8_0
Q8_0_POLICY = QuantizationPolicy(
    name="q8_0",
    role_map={
        TensorRole.EMBEDDING: DType.Q8_0,
        TensorRole.OUTPUT_HEAD: DType.Q8_0,
        TensorRole.ATTN_VALUE: DType.Q8_0,
        TensorRole.ATTN_OUT: DType.Q8_0,
        TensorRole.ATTN_QK: DType.Q8_0,
        TensorRole.FFN_DOWN: DType.Q8_0,
        TensorRole.FFN_GATE_UP: DType.Q8_0,
        TensorRole.GENERIC_WEIGHT: DType.Q8_0,
        TensorRole.NORMALIZATION_1D: DType.F32,
    },
    fallback_dtype=DType.Q8_0,
)

# 4. Uniform Q4_0
Q4_0_POLICY = QuantizationPolicy(
    name="q4_0",
    role_map={
        TensorRole.EMBEDDING: DType.Q4_0,
        TensorRole.OUTPUT_HEAD: DType.Q4_0,
        TensorRole.ATTN_VALUE: DType.Q4_0,
        TensorRole.ATTN_OUT: DType.Q4_0,
        TensorRole.ATTN_QK: DType.Q4_0,
        TensorRole.FFN_DOWN: DType.Q4_0,
        TensorRole.FFN_GATE_UP: DType.Q4_0,
        TensorRole.GENERIC_WEIGHT: DType.Q4_0,
        TensorRole.NORMALIZATION_1D: DType.F32,
    },
    fallback_dtype=DType.Q4_0,
)

# 5. Uniform F16
F16_POLICY = QuantizationPolicy(
    name="f16",
    role_map={
        TensorRole.EMBEDDING: DType.F16,
        TensorRole.OUTPUT_HEAD: DType.F16,
        TensorRole.ATTN_VALUE: DType.F16,
        TensorRole.ATTN_OUT: DType.F16,
        TensorRole.ATTN_QK: DType.F16,
        TensorRole.FFN_DOWN: DType.F16,
        TensorRole.FFN_GATE_UP: DType.F16,
        TensorRole.GENERIC_WEIGHT: DType.F16,
        TensorRole.NORMALIZATION_1D: DType.F32,
    },
    fallback_dtype=DType.F16,
)

BUILTIN_POLICIES: dict[str, QuantizationPolicy] = {
    "unsloth_dynamic": UNSLOTH_DYNAMIC_POLICY,
    "ud_q4_m": UNSLOTH_DYNAMIC_POLICY,
    "ud_q4_k_m": UNSLOTH_DYNAMIC_POLICY,
    "q4_k_m": Q4_K_M_POLICY,
    "q8_0": Q8_0_POLICY,
    "q4_0": Q4_0_POLICY,
    "f16": F16_POLICY,
    "fp16": F16_POLICY,
}


def get_quantization_policy(policy_spec: str | QuantizationPolicy | DType) -> QuantizationPolicy:
    """Resolves a policy specification into a concrete QuantizationPolicy."""
    if isinstance(policy_spec, QuantizationPolicy):
        return policy_spec

    if isinstance(policy_spec, DType):
        if policy_spec == DType.Q4_0:
            return Q4_0_POLICY
        if policy_spec == DType.Q8_0:
            return Q8_0_POLICY
        if policy_spec in (DType.F16, DType.BF16):
            return F16_POLICY
        return QuantizationPolicy(
            name=f"uniform_{policy_spec.name.lower()}",
            role_map={
                role: policy_spec for role in TensorRole if role != TensorRole.NORMALIZATION_1D
            },
            fallback_dtype=policy_spec,
        )

    spec_key = str(policy_spec).strip().lower()
    if spec_key in BUILTIN_POLICIES:
        return BUILTIN_POLICIES[spec_key]

    raise ValueError(
        f"Unknown quantization policy: '{policy_spec}'. Available policies: {list(BUILTIN_POLICIES.keys())}"
    )
