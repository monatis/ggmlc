# Troubleshooting, Debugging & Gotchas Guide

This document captures architectural insights, subtle mathematical gotchas, known pitfalls, and debugging workflows discovered during the development and verification of `ggmlc`.

---

## 1. Top Gotchas & Things to Avoid

### Gotcha 1: The Row-Major vs. Column-Major Inversion
- **The Concept**: PyTorch / JAX use **row-major (C-order)** memory layouts $(D_0, D_1, \dots, D_{R-1})$, where the last dimension $D_{R-1}$ changes fastest in contiguous memory. GGML uses **column-major (Fortran-order)** tensor representation, where `ne[0]` changes fastest.
- **The Golden Rule**:
  $$\text{shape}_{GGML} = \text{reverse}(\text{shape}_{Canonical})$$
  $$\text{PyTorch } [B, S, H, D] \iff \text{GGML } ne = [D, H, S, B]$$
- **What to Avoid**: Never transpose underlying memory arrays during ingestion or serialization. Because PyTorch row-major contiguous memory is identical byte-for-byte to GGML column-major memory, reversing the dimension tuple preserves exact tensor stride memory layout without physical copies!

---

### Gotcha 2: Permutation & Transpose Indexing Formula
- **The Concept**: When PyTorch performs `torch.permute(x, dims)` or `x.transpose(d1, d2)`, mapping that permutation $P$ into GGML's `ggml_permute(ctx, a, axis0, axis1, axis2, axis3)` requires an inverted index formula.
- **The Formula**:
  $$\text{dest\_axis}_i = R - 1 - P.\text{index}(R - 1 - i) \quad \text{for } 0 \le i < R$$
- **What to Avoid**: Do not simply pass $P$ or reversed $P$ directly into `ggml_permute`. In GGML, `axis_i` specifies where source axis `ne[i]` lands in the target tensor (`ne_dst[axis_i] = ne_src[i]`), whereas PyTorch's $P[j]$ specifies which source axis goes into destination dimension $j$.

---

### Gotcha 3: General Activation-Activation Matrix Multiplication (`ggml_mul_mat`)
- **The Concept**: In GGML, `ggml_mul_mat(ctx, a, b)` computes:
  $$\text{Output} = b \times a^T$$
- **For Parameter Weights (`Linear`)**:
  - Weight $W(N, K) \to \text{GGML } ne = [K, N]$.
  - Activation $x(M, K) \to \text{GGML } ne = [K, M]$.
  - `ggml_mul_mat(W, x)` computes $x(M, K) \times W(N, K)^T = (M, N)$, yielding GGML shape $ne = [N, M]$. Direct and zero-overhead.
- **For Dynamic Activation-Activation MatMul (`Q @ K^T` and `Attn @ V`)**:
  - If $A$ is $(B, H, M, K)$ and $B$ is $(B, H, K, N)$, in GGML $A$ is $[K, M, H, B]$ and $B$ is $[N, K, H, B]$.
  - To compute $A \times B$, tensor $B$ **must be transposed in GGML** to $[K, N, H, B]$ prior to calling `ggml_mul_mat(B_transposed, A)`.
  - In `ggmlc`, tag the operation with `attrs["transpose_in0"] = 1` so the Generic C++ runtime automatically invokes `ggml_transpose(ctx, in0)` before `ggml_mul_mat`.

---

### Gotcha 4: GQA Head Expansion: `repeat_interleave` vs `repeat`
- **The Pitfall**:
  - In PyTorch, Grouped Query Attention (GQA) expands KV heads using `torch.repeat_interleave(k, repeats=num_groups, dim=1)`. This duplicates **each individual head** contiguously: $[h_0, h_0, h_1, h_1]$.
  - In GGML, `ggml_repeat(ctx, a, b)` tiles the **entire tensor periodically**: $[h_0, h_1, h_0, h_1]$.
  - Using naive `ggml_repeat` for GQA produces scrambled attention scores and garbage token generation!
- **The Solution in `ggmlc`**:
  - In the GGML lowering pass, unpack GQA KV projections by slicing individual heads or reshaping to a 4D view $(D, S, \text{num\_groups}, N_{kv\_heads})$, repeating along the group dimension, and concatenating along head axes.

---

### Gotcha 5: Pretrained Model Tokenizer & Position ID Offsets
- **XLM-RoBERTa / BGE-M3**:
  - XLM-RoBERTa uses `padding_idx = 1`. Valid token position IDs begin at `padding_idx + 1 = 2`. Passing 0-indexed position embeddings will result in positional misalignment.
- **Qwen2.5 / LLaMA RoPE Frequency**:
  - Rotary position embeddings in Qwen/LLaMA operate on interleaved even/odd or split halves depending on the exact rope type attribute (`GGML_ROPE_TYPE_NEOX` vs `GGML_ROPE_TYPE_NORM`). Always verify `attrs["rope_type"]` against model config `rope_theta` and `max_position_embeddings`.

---

### Gotcha 6: ATen Decompositions & Device/Type Cast Operators
- When capturing graphs with `torch.export.export()`, PyTorch emits device transfers (`aten.to.device`, `aten.to.dtype_device`) and factory constants (`aten.zeros`, `aten.ones`, `aten.new_zeros`).
- In `ggmlc/frontend/pytorch/importer.py`:
  - Treat `to.device` and `to.dtype_device` as identity no-ops if destination dtype matches current tensor dtype.
  - Constant-fold `zeros`, `ones`, `zeros_like` at compile time by allocating constant parameter buffers rather than attempting dynamic graph memory allocation.

---

## 2. Debugging Workflow & Diagnostic Strategies

### Strategy 1: Layer-by-Layer Subgraph Isolation
When compiling a multi-layer architecture (e.g. 24-layer Qwen or 8-layer BGE-M3) and output diverges from PyTorch:
1. Wrap intermediate submodules in isolated PyTorch modules:
   ```python
   # Test single attention block
   class SingleBlock(nn.Module):
       def __init__(self, block):
           super().__init__()
           self.block = block
       def forward(self, hidden_states, mask):
           return self.block(hidden_states, attention_mask=mask)[0]
   ```
2. Compile and run `check_numerical_accuracy(ref_out, actual_out)` on the single block.
3. Compare intermediate attention query/key/value projections, attention score logits before softmax, and post-LayerNorm activations.

---

### Strategy 2: Inspecting Serialized Binary Graphs
Use the Python inspection helper to inspect the `.ggmlc` binary contents:
```python
from ggmlc.serialization.graph import deserialize_ggml_graph

with open("model.ggmlc", "rb") as f:
    g = deserialize_ggml_graph(f.read())

print(f"Nodes: {len(g.nodes)}")
for node in g.nodes:
    print(f"  Op: {node.opcode.name:<20} in={node.inputs} out={node.outputs} attrs={node.attributes}")
```

---

### Strategy 3: Inspecting C++ Generic Runtime Execution
When executing via WSL or local binary:
1. Run `ggmlc-run` with verbose logging:
   ```bash
   ./build/runtime/ggmlc-run --model model.ggmlc --input x=input.bin --output out=output.bin --verbose
   ```
2. Check the tensor dimensions reported by `ggmlc-run`:
   ```
   [ggmlc-run] Tensor 0: name='input_ids' type=I32 ne=[16, 1, 1, 1] nb=[4, 64, 64, 64]
   [ggmlc-run] Tensor 12: name='wte.weight' type=F32 ne=[768, 50257, 1, 1]
   [ggmlc-run] Op 45: GGML_OP_MUL_MAT in0=12 in1=11 out=13
   ```

---

### Strategy 4: Tolerance Guide for Differential Numerical Testing
Floating-point rounding differences accumulate across deep networks. Use the following calibrated tolerances:

| Model Type | Recommended `atol` | Notes |
| :--- | :--- | :--- |
| **Elementwise / Linear MLP** | `1e-5` | Direct kernel match with reference PyTorch/JAX FP32 |
| **Attention / Single Transformer Layer** | `1e-4` | Softmax exponential precision |
| **ResNet-18 / ResNet-50** | `1e-2` | Accumulated Conv2D + BatchNorm running variance |
| **GPT-2 (12 Layers)** | `1e-3` | 12 transformer layers, layer norms, and causal attention |
| **Qwen2.5-0.5B (24 Layers)** | `1e-3` | 24 transformer layers, RMSNorm, SwiGLU, RoPE |
| **BGE-M3 (8 Layers, 1024 Dim)** | `0.15` | Large 1024 embedding width and multi-layer reductions |

---

## 3. Useful Development Rules
- **Rule 1: Always add a unit test for new operators in `tests/ops/` before compiling full models.**
- **Rule 2: Never modify C++ kernel pointers in-place without verifying GGML compute graph dependency ordering.**
- **Rule 3: Ensure all symbolic dimensions (`SymbolDim`) are registered in `graph.symbol_table` during lowering so the runtime can resolve them dynamically.**
