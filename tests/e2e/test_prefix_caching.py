from pathlib import Path

import ggmlc._runtime as _rt
import pytest
from ggmlc.runtime.runner import ModelRunner, get_available_devices

cuda_available = "cuda" in get_available_devices() or "cuda:0" in get_available_devices()
pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not cuda_available, reason="Native CUDA device not available"),
]


def test_radix_tree_unit():
    """Unit test verifying PagedRadixTree trie prefix matching and metrics."""
    tree = _rt.PagedRadixTree(128)
    assert tree.tokens_per_page == 128
    assert tree.total_cached_pages == 0
    assert tree.total_nodes == 1  # Root node

    # A prompt smaller than tokens_per_page must yield 0 matched pages
    short_prompt = list(range(10, 50))
    m_short = tree.match_prefix(short_prompt)
    assert m_short.matched_pages == 0
    assert m_short.matched_tokens == 0

    # A prompt on empty tree yields 0 matches
    full_prompt = list(range(100, 100 + 128)) + [999]
    m_empty = tree.match_prefix(full_prompt)
    assert m_empty.matched_pages == 0
    assert m_empty.matched_tokens == 0


def test_vmm_warm_pool_recycling():
    """Unit test verifying VMMBlockManager warm pool allocation, recycling, and draining."""
    vmm = _rt.NativeVMMBlockManager()
    assert vmm.init(0)
    assert vmm.is_initialized

    # 1. Configure pool with upfront allocation
    vmm.configure_pool(max_warm_pages=4, prealloc_pages=2)
    assert vmm.max_warm_pages == 4
    assert vmm.free_pool_pages == 2
    assert vmm.total_allocated_pages == 2

    # 2. Pop the 2 pre-allocated pages
    h1 = vmm.alloc_physical_page()
    h2 = vmm.alloc_physical_page()
    assert h1 != 0 and h2 != 0
    assert vmm.free_pool_pages == 0

    # 3. Allocating 3rd page dynamically creates one
    h3 = vmm.alloc_physical_page()
    assert h3 != 0
    assert vmm.free_pool_pages == 0

    # 4. Releasing h3 recycles it into warm pool in O(1)
    vmm.release_physical_page(h3)
    assert vmm.free_pool_pages == 1

    # 5. Re-allocating immediately reuses h3
    h3_reused = vmm.alloc_physical_page()
    assert h3_reused == h3
    assert vmm.free_pool_pages == 0

    # 6. Release all handles and drain warm pool
    vmm.release_physical_page(h1)
    vmm.release_physical_page(h2)
    vmm.release_physical_page(h3)
    assert vmm.free_pool_pages == 3

    vmm.drain_warm_pool()
    assert vmm.free_pool_pages == 0


def test_continuous_batching_prefix_caching_e2e():
    """E2E test: 4 concurrent requests with shared system prompt using Radix Tree prefix caching."""
    model_path = Path("scratch/smollm2_batched_f16.gguf")
    if not model_path.exists():
        pytest.skip(f"Test model checkpoint {model_path} not found")

    runner = ModelRunner(str(model_path), device="cuda")
    runner.executor.set_tokens_per_page(128)
    tokens_per_page = runner.executor.tokens_per_page
    assert tokens_per_page == 128

    # Configure executor warm pool
    runner.executor.configure_vmm_pool(max_warm_pages=32, prealloc_pages=8)

    scheduler = _rt.ContinuousBatchScheduler(runner.executor, max_batch_size=8, eos_token_id=0)
    scheduler.enable_prefix_caching(True)
    assert scheduler.is_prefix_caching_enabled()

    # Shared system prompt fills 1 full page (128 tokens)
    shared_prefix = [100 + (i % 300) for i in range(128)]

    # Request 1: Shared prefix + unique tail
    p1 = shared_prefix + [1001, 1002]
    rid_1 = scheduler.add_request(p1, max_new_tokens=4, temperature=0.0)

    # Ingest Request 1 until it completes prefill and generates first token
    while scheduler.has_work():
        scheduler.step()
        r1 = scheduler.get_request(rid_1)
        if r1.current_pos >= len(p1):
            break

    assert scheduler.total_prefix_cache_hits == 0, "First request must be cold"

    # Submit Requests 2, 3, 4 with identical shared prefix but distinct user queries
    p2 = shared_prefix + [2001, 2002]
    p3 = shared_prefix + [3001, 3002]
    p4 = shared_prefix + [4001, 4002]

    rid_2 = scheduler.add_request(p2, max_new_tokens=4, temperature=0.0)
    rid_3 = scheduler.add_request(p3, max_new_tokens=4, temperature=0.0)
    rid_4 = scheduler.add_request(p4, max_new_tokens=4, temperature=0.0)

    # Step scheduler: Requests 2, 3, 4 must be admitted and match the prefix
    scheduler.step()

    req2 = scheduler.get_request(rid_2)
    req3 = scheduler.get_request(rid_3)
    req4 = scheduler.get_request(rid_4)

    assert scheduler.total_prefix_cache_hits >= 3
    assert scheduler.total_prefix_tokens_saved >= 3 * 128
    assert req2.prefix_tokens_matched == 128
    assert req3.prefix_tokens_matched == 128
    assert req4.prefix_tokens_matched == 128

    # Step all to completion
    while scheduler.has_work():
        scheduler.step()

    for rid in [rid_1, rid_2, rid_3, rid_4]:
        r = scheduler.get_request(rid)
        assert r.finished, f"Request {rid} did not finish"
        assert len(r.generated_tokens) == 4

    # Physical memory reclaimed to 0.0 MB
    assert runner.executor.get_paged_active_vram_bytes() == 0
