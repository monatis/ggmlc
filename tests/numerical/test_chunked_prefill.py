from pathlib import Path

import numpy as np
import pytest
from ggmlc._runtime import ModelExecutor, ModelLoader, get_available_devices
from ggmlc.runtime.runner import ModelRunner


@pytest.fixture
def smollm2_model_path():
    p = Path("scratch/smollm2_chat.gguf")
    if not p.exists():
        pytest.skip("scratch/smollm2_chat.gguf not found")
    return str(p)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("chunk_size", [16, 32])
def test_chunked_prefill_differential(smollm2_model_path, device, chunk_size):
    devices = get_available_devices()
    if device.startswith("cuda") and not any(d.startswith("cuda") for d in devices):
        pytest.skip("CUDA device not available")

    model_graph = ModelLoader.load_from_file(smollm2_model_path)
    out_tid = model_graph.outputs[0]

    np.random.seed(42)
    prompt_len = 64
    tokens = list(np.random.randint(100, 3000, size=prompt_len))

    # 1. Full single-pass prefill baseline
    exec_full = ModelExecutor(model_graph, device)
    exec_full.init_kv_cache(256)
    env_full = {sym: prompt_len for sym in model_graph.symbol_table if sym.startswith("s")}
    env_full["s"] = prompt_len
    env_full["pos"] = 0
    exec_full.prepare(env_full)
    exec_full.set_input_by_name("input_ids", np.array(tokens, dtype=np.int32))
    exec_full.run()

    full_logits = np.frombuffer(exec_full.get_output_bytes(out_tid), dtype=np.float32).reshape(
        prompt_len, -1
    )
    last_full_logits = full_logits[-1]
    expected_next_token = int(np.argmax(last_full_logits))

    # Decode step 1 after full prefill
    env_dec = {sym: 1 for sym in model_graph.symbol_table if sym.startswith("s")}
    env_dec["s"] = 1
    env_dec["pos"] = prompt_len
    exec_full.prepare(env_dec)
    exec_full.set_input_by_name("input_ids", np.array([expected_next_token], dtype=np.int32))
    exec_full.run()
    full_dec1_logits = np.frombuffer(exec_full.get_output_bytes(out_tid), dtype=np.float32)

    # 2. Chunked prefill
    exec_chunk = ModelExecutor(model_graph, device)
    exec_chunk.init_kv_cache(256)
    n_chunks = (prompt_len + chunk_size - 1) // chunk_size

    last_chunk_logits = None
    for chunk_idx in range(n_chunks):
        c_start = chunk_idx * chunk_size
        c_len = min(chunk_size, prompt_len - c_start)
        c_tokens = np.array(tokens[c_start : c_start + c_len], dtype=np.int32)

        env_c = {sym: c_len for sym in model_graph.symbol_table if sym.startswith("s")}
        env_c["s"] = c_len
        env_c["pos"] = c_start
        exec_chunk.prepare(env_c)
        exec_chunk.set_input_by_name("input_ids", c_tokens)
        exec_chunk.run()

        chunk_logits = np.frombuffer(
            exec_chunk.get_output_bytes(out_tid), dtype=np.float32
        ).reshape(c_len, -1)
        last_chunk_logits = chunk_logits[-1]

    actual_next_token = int(np.argmax(last_chunk_logits))

    # Assert exact token equivalence and tight logit parity
    assert actual_next_token == expected_next_token
    diff_prefill = np.max(np.abs(last_full_logits - last_chunk_logits))
    assert diff_prefill < 5e-3, f"Prefill logit diff too high: {diff_prefill}"

    # Decode step 1 after chunked prefill
    exec_chunk.prepare(env_dec)
    exec_chunk.set_input_by_name("input_ids", np.array([actual_next_token], dtype=np.int32))
    exec_chunk.run()
    chunk_dec1_logits = np.frombuffer(exec_chunk.get_output_bytes(out_tid), dtype=np.float32)

    diff_dec1 = np.max(np.abs(full_dec1_logits - chunk_dec1_logits))
    assert diff_dec1 < 5e-3, f"Decode step 1 logit diff too high: {diff_dec1}"


def test_chunked_prefill_with_runner(smollm2_model_path):
    runner = ModelRunner(smollm2_model_path, device="cpu")
    if hasattr(runner, "init_kv_cache"):
        runner.init_kv_cache(256)

    tokens = list(range(100, 148))  # 48 tokens
    C = 16
    n_chunks = len(tokens) // C

    # Run chunks through runner
    for chunk_idx in range(n_chunks):
        c_start = chunk_idx * C
        c_len = C
        c_tokens = np.array([tokens[c_start : c_start + c_len]], dtype=np.int32)
        out = runner(c_tokens, symbols={"pos": c_start, "s": c_len})
        assert out is not None


def test_chunked_prefill_with_generator(smollm2_model_path):
    from ggmlc._runtime import NativeBPETokenizer
    from ggmlc.runtime.generator import GGMLCGenerator

    tokenizer = NativeBPETokenizer()
    assert tokenizer.init_from_gguf_file(smollm2_model_path)

    runner = ModelRunner(smollm2_model_path, device="cpu")
    gen = GGMLCGenerator(runner, tokenizer)

    prompt = "The quick brown fox jumps over the lazy dog."
    text_chunked = gen.generate(prompt, max_new_tokens=8, chunk_size=4)
    text_full = gen.generate(prompt, max_new_tokens=8, chunk_size=0)

    assert text_chunked == text_full
