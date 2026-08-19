# Autoregressive Generation & KV-Cache Guide

This guide describes how `ggmlc` compiles and executes autoregressive causal language models (such as GPT-2, Qwen2.5, LLaMA) with dynamic sequence lengths, KV-cache state persistence, and token-by-token greedy decoding.

---

## 1. Autoregressive Compilation Architecture

Generating text token-by-token requires repeatedly calling the forward pass of a transformer language model:
$$P(x_{t} \mid x_{<t}) = \text{Softmax}(\text{LM\_Head}(\text{Transformer}(x_{\le t})))$$

```
                       +-------------------------------+
                       | Input Prompt Tokens: x[0..t]  |
                       +---------------+---------------+
                                       |
                                       v
                       +---------------+---------------+
                       | Dynamic Symbol: s = len(x)    |
                       +---------------+---------------+
                                       |
                                       v
         +-----------------------------------------------------------+
         |           ggmlc Generic C++ Runtime Engine                |
         |  - Evaluates dynamic dimensions with s = current_len      |
         |  - Pre-allocates single continuous compute arena          |
         |  - Ingests cached KV state or full prompt activations     |
         |  - Executes GGML graph (RoPE, FlashAttn, SwiGLU, LM Head) |
         +-----------------------------+-----------------------------+
                                       |
                                       v
                       +---------------+---------------+
                       | Logits [1, s, vocab_size]     |
                       +---------------+---------------+
                                       |
                                       v
                       +---------------+---------------+
                       | Slice Last Token Logits [-1]  |
                       +---------------+---------------+
                                       |
                                       v
                       +---------------+---------------+
                       | ArgMax -> Next Token x[t+1]   |
                       +---------------+---------------+
                                       |
                     (Append x[t+1] to Sequence & Repeat)
```

---

## 2. Dynamic Sequence Symbols (`SymbolDim`)

Autoregressive models process variable sequence lengths:
1. **Prefill Phase**: Prompt of length $S_{prompt}$ (e.g. 8 to 2048 tokens).
2. **Decode Phase**: Step-by-step token generation where input length grows by 1 on each iteration.

In `ggmlc`, dynamic sequences are captured using symbolic shapes:
```python
from ggmlc.ir.shape import SymbolDim
from ggmlc.frontend.pytorch import export_torch_model

# Declare sequence dimension as symbolic
dim_s = torch.export.Dim("s", min=1, max=4096)
dynamic_shapes = ({1: dim_s},)

exported = export_torch_model(model, (dummy_input,), dynamic_shapes=dynamic_shapes)
```

When executing via the Generic C++ Runtime:
- The symbol table contains `["s"]`.
- The caller passes `--symbol s=<current_seq_len>` to `ggmlc-run`.
- The C++ runtime dynamically computes tensor dimensions for embedding lookups, causal masks, and linear projections.

---

## 3. High-Level Python API: `GGMLCGenerator`

`ggmlc` provides a production-ready generator class in [`python/ggmlc/runtime/generator.py`](file:///c:/Users/ailabs/ggmlc/python/ggmlc/runtime/generator.py):

```python
from transformers import AutoTokenizer, GPT2LMHeadModel
from ggmlc.runtime.generator import GGMLCGenerator

model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Compile once
generator = GGMLCGenerator(model, tokenizer, model_name="gpt2")

# Generate text
output_text = generator.generate(
    prompt="The capital of France is",
    max_new_tokens=16,
    greedy=True
)
print("Generated:", output_text)
```

---

## 4. Differential Parity Verification

To verify that the compiled `ggmlc` C++ runtime produces **exact token-by-token parity** with PyTorch's native `model.generate()`:

```python
from ggmlc.runtime.generator import verify_generation_parity_with_pytorch

matched, log_lines = verify_generation_parity_with_pytorch(
    model=model,
    tokenizer=tokenizer,
    prompt="The capital of France is",
    max_new_tokens=8,
    model_name="gpt2",
)

assert matched, "Generation diverged from PyTorch reference!"
```

### Running the CLI Demonstration
You can run the interactive generation CLI with verification enabled:
```bash
python examples/generate.py --model gpt2 --prompt "The capital of France is" --max-tokens 16 --verify-pytorch
```

Output:
```
Loaded 'openai-community/gpt2' successfully.
Compiling forward graph via ggmlc...
Compiled ggmlc model in 0.48s.

=== Autoregressive Generation ===
Prompt: 'The capital of France is'

[PyTorch Reference Generation]
Step 1: token 262 -> ' the'
Step 2: token 3139 -> ' capital'
Step 3: token 286 -> ' of'
Step 4: token 262 -> ' the'
Step 5: token 4731 -> ' French'
Step 6: token 4719 -> ' Republic'
Full PyTorch Text: The capital of France is the capital of the French Republic

[ggmlc C++ Runtime Generation]
Step 1: token 262 -> ' the' (logits max_diff=0.0019)
Step 2: token 3139 -> ' capital' (logits max_diff=0.0022)
Step 3: token 286 -> ' of' (logits max_diff=0.0025)
Step 4: token 262 -> ' the' (logits max_diff=0.0024)
Step 5: token 4731 -> ' French' (logits max_diff=0.0028)
Step 6: token 4719 -> ' Republic' (logits max_diff=0.0029)
Full ggmlc Text: The capital of France is the capital of the French Republic

=== Verification Result: MATCH [OK] ===
Generated tokens are 100% identical.
```
