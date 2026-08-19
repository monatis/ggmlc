# C++ Generic Runtime Architecture

The `ggmlc` C++ runtime (`ggmlc_runtime`) is a high-performance, lightweight execution engine that evaluates serialized `.ggmlc` execution graphs using the underlying GGML library.

---

## 1. Design Overview

Rather than generating custom C++ files for each neural network architecture, `ggmlc` employs a **generic graph interpreter**:
- **Zero dynamic code compilation**: The runtime binary `ggmlc-run` is compiled once and can execute any `.ggmlc` model artifact.
- **Deterministic memory planning**: Context memory is calculated dynamically during `prepare()` and allocated in a single contiguous arena.
- **Dynamic shape binding**: Symbolic dimensions (e.g. batch size, sequence length) are supplied as command-line arguments or runtime environment dictionaries.
- **Persistent state buffers**: Stateful memory (`StorageClass.STATE`) persists across sequential inference invocations.
- **Hardware-accelerated quantized execution**: Direct SIMD kernel execution (AVX2, AVX-512, ARM NEON) for `Q4_0` and `Q8_0` quantized models.

---

## 2. Core Runtime Classes

### `ModelLoader` (`runtime/src/loader.cpp`)
Responsible for reading `.ggmlc` binary files into in-memory `SerializedModelGraph` structures:
- Reads file header, version, and global `symbol_table`.
- Deserializes tensor definitions, storage classes, and dimension expression trees (`DimExpr`).
- Deserializes the operation schedule with attributes.
- Loads raw parameter/constant weight byte buffers with 16-byte alignment.

### `ModelExecutor` (`runtime/src/executor.cpp`)
Main runtime orchestrator:
- `prepare(symbol_env)`:
  1. Recursively evaluates all `DimExpr` trees against the supplied `symbol_env` map to produce concrete 4D tensor shapes `concrete_shapes_[tensor_id]`.
  2. Computes total tensor and overhead memory requirements taking block quantization into account:
     $$\text{tensor\_bytes} = \frac{\text{numel}}{\text{ggml\_blck\_size}(\text{type})} \times \text{ggml\_type\_size}(\text{type})$$
  3. Initializes `ggml_context` and creates all `ggml_tensor` objects.
  4. Copies parameter and constant data into tensor buffers.
  5. Restores persistent states from `persistent_states_` or zeros them on first invocation.
  6. Builds the execution graph (`ggml_cgraph`).
- `set_input(tensor_id, data, size)` / `set_input_by_name(name, data, size)`: Copies user input buffers into graph inputs with shape/size verification.
- `set_state(tensor_id, data, size)` / `set_state_by_name(name, data, size)`: Explicitly sets or restores persistent state buffers.
- `run(n_threads)`: Computes forward evaluation across the GGML thread pool and captures final state buffers.
- `get_output_data(tensor_id)`: Accesses computed output memory pointers.
- `get_state_data(tensor_id)` / `get_state_data_by_name(name)`: Accesses persistent state buffers after inference.

---

## 3. CLI Runner (`ggmlc-run`)

```bash
ggmlc-run <model.ggmlc> [options]
```

### Supported CLI Options
- `--input <name:file.bin>`: Loads input tensor from binary file.
- `--output <id:file.bin>`: Writes output tensor to binary file.
- `--state-in <name:file.bin>`: Loads initial state tensor from binary file.
- `--state-out <name:file.bin>`: Writes final state tensor to binary file.
- `--symbol <name=value>`: Binds a dynamic symbolic dimension (e.g. `--symbol batch=4 --symbol seq=32`).
- `--threads <N>`: Sets the number of CPU execution threads.

---

## 4. C++ API Embedding Example

```cpp
#include "ggmlc/loader.h"
#include "ggmlc/executor.h"

// 1. Load compiled model graph (FP32, Q8_0, or Q4_0)
auto model_graph = ggmlc::ModelLoader::load_from_file("minilm_q4.ggmlc");

// 2. Instantiate executor and bind dynamic dimensions
ggmlc::ModelExecutor executor(model_graph);
executor.prepare({{"batch", 1}, {"seq", 16}});

// 3. Set input data
std::vector<float> input_tokens(1 * 16 * 384, 1.0f);
executor.set_input_by_name("input_ids", input_tokens.data(), input_tokens.size() * sizeof(float));

// 4. Run forward pass across 4 CPU threads
executor.run(/*n_threads=*/4);

// 5. Read output tensor
uint32_t out_id = model_graph.outputs[0];
const float* out_ptr = reinterpret_cast<const float*>(executor.get_output_data(out_id));
```
