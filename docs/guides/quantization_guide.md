# Quantization Guide & Best Practices

This guide explains how to quantize neural network models to **Q8_0** and **Q4_0** block formats in `ggmlc`, evaluate compression ratios, and verify output numerical parity.

---

## 1. Quickstart: Quantization CLI

`ggmlc` provides a unified command-line tool `python -m ggmlc.cli.quantize` for one-shot export, optimization, quantization, and `.ggmlc` binary generation.

### Example: Quantizing MiniLM to Q4_0
```bash
python -m ggmlc.cli.quantize --model minilm --dtype q4_0 --optimize --output scratch/minilm_q4.ggmlc
```

**Output**:
```
Loading pretrained model 'minilm'...
Loading weights: 100%|██████████| 103/103 [00:00<00:00, 2986.34it/s]
Exporting PyTorch model to Canonical IR...
Running Canonical IR Optimization Pipeline...
  Optimization: 148 -> 131 nodes (-11.5%), 0 fusions, 2 folded constants in 1.1ms.
Lowering to GGML dialect...
Quantizing parameters to Q4_0...
  Quantized 103 weight tensors: 86.65 MB -> 12.19 MB (7.11x compression ratio)
Saved quantized model to 'scratch/minilm_q4.ggmlc' (12.22 MB).
```

### CLI Arguments
| Argument | Description | Default |
| :--- | :--- | :--- |
| `--model` | Model name (`resnet18`, `resnet50`, `minilm`, `gpt2`, `qwen`, `bge_m3`) | *Required* |
| `--dtype` | Target quantization format: `q4_0`, `q8_0`, `f32` | `q4_0` |
| `--optimize` | Enable Canonical IR optimization passes before lowering | `True` |
| `--output` | Destination `.ggmlc` container path | `<model>_<dtype>.ggmlc` |

---

## 2. Programmatic Python API

You can apply quantization and optimization passes directly in Python scripts:

```python
import torch
from ggmlc.frontend.pytorch import export_torch_model
from ggmlc.transforms import create_standard_optimization_pipeline
from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.quantization import quantize_graph_parameters
from ggmlc.ir.dtype import DType
from ggmlc.serialization.graph import serialize_ggml_graph

# 1. Export PyTorch model
model = MyTransformerModel().eval()
dummy_inputs = (torch.zeros(1, 128, dtype=torch.long),)
exported = export_torch_model(model, dummy_inputs, model_name="my_model")

# 2. Run Optimization Passes (Constant Folding, DCE, Fusion)
pipeline = create_standard_optimization_pipeline()
opt_result = pipeline.run(exported.main_graph)
canonical_graph = opt_result.graph

# 3. Lower to GGML dialect
ggml_graph = lower_to_ggml(canonical_graph)

# 4. Quantize 2D Weight Parameters to Q4_0 or Q8_0
quant_graph, stats = quantize_graph_parameters(ggml_graph, target_dtype=DType.Q4_0)
print(f"Quantized {stats['tensors_quantized']} tensors: {stats['compression_ratio']:.2f}x compression")

# 5. Serialize to binary container
binary_bytes = serialize_ggml_graph(quant_graph)
with open("my_model_q4_0.ggmlc", "wb") as f:
    f.write(binary_bytes)
```

---

## 3. Format Comparison & Compression Ratios

| Format | Block Size | Scale Type | Bits/Weight | Compression Ratio | Cosine Similarity | Best Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`F32`** | 1 | None | 32 bits | $1.0\times$ | $1.0000$ | Baseline numerical golden truth |
| **`Q8_0`** | 32 | `fp16` | 8.5 bits | $\mathbf{3.76\times}$ | $> 0.9999$ | High precision inference with minimal loss |
| **`Q4_0`** | 32 | `fp16` | 4.5 bits | $\mathbf{7.11\times}$ | $> 0.9850$ | Low-memory edge deployment and LLMs |

---

## 4. Verification & Cosine Similarity Testing

To verify the numerical fidelity of quantized models against FP32 reference models:

```python
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    dot = np.dot(a_flat, b_flat)
    norm = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    return float(dot / (norm + 1e-12))

# Execute both models and compute similarity
similarity = cosine_similarity(ref_output, quantized_output)
assert similarity > 0.98, f"Quantization drift too high: {similarity}"
```

---

## 5. Important Gotchas & Best Practices

1. **Do Not Quantize 1D Tensors**:
   - Biases, LayerNorm scale/bias vectors, and RMSNorm weights have small element counts where quantization can cause significant systematic drift. `ggmlc` automatically preserves all 1D parameter vectors as full `F32`.
2. **Contraction Dimension Alignment**:
   - Quantized GEMM weights must have contiguous rows of $K$ elements ($K \pmod{32} == 0$). Lowering transposes weights before quantization so that GGML reads contiguous block rows directly.
3. **Avoid Non-Contiguous Ops on Quantized Weights**:
   - Quantized tensors cannot be reshaped with non-block-aligned strides. Never call `ggml_transpose` or `ggml_cont` on a `Q4_0` or `Q8_0` weight tensor inside the execution graph.
