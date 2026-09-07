from pathlib import Path

import ggmlc._runtime as _rt
import pytest
from ggmlc.runtime.runner import ModelRunner, get_available_devices

cuda_available = "cuda" in get_available_devices() or "cuda:0" in get_available_devices()
pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not cuda_available, reason="Native CUDA device not available"),
]


def test_vmm_block_manager_unit():
    """Unit test verifying CUDA Driver Virtual Memory Management (cuMemMap)."""
    assert _rt.NativeVMMBlockManager.is_supported_on_device(0), "VMM must be supported on GPU 0"
    vmm = _rt.NativeVMMBlockManager()
    ok = vmm.init(0)
    assert ok, "Failed to initialize NativeVMMBlockManager on device 0"
    assert vmm.is_initialized

    page_sz = vmm.page_size
    assert page_sz > 0
    # Reserve 64 MB virtual address window (no physical VRAM consumed)
    va_window = vmm.reserve_virtual_window(64 * 1024 * 1024)
    assert va_window != 0
    assert vmm.total_reserved_va_bytes >= 64 * 1024 * 1024
    assert vmm.total_mapped_physical_bytes == 0

    # Allocate and map 2 physical pages
    h1 = vmm.alloc_physical_page()
    h2 = vmm.alloc_physical_page()
    assert h1 != 0 and h2 != 0

    assert vmm.map_page(va_window, h1)
    assert vmm.map_page(va_window + page_sz, h2)
    assert vmm.total_mapped_physical_bytes == 2 * page_sz

    # Unmap and release
    assert vmm.unmap_page(va_window)
    vmm.release_physical_page(h1)
    assert vmm.total_mapped_physical_bytes == 1 * page_sz

    assert vmm.unmap_page(va_window + page_sz)
    vmm.release_physical_page(h2)
    assert vmm.total_mapped_physical_bytes == 0

    vmm.free_virtual_window(va_window, 64 * 1024 * 1024)
    vmm.reset()


def test_continuous_batching_scheduler():
    """End-to-end continuous batching verification with iteration-level dynamic scheduling."""
    model_path = Path("scratch/smollm2_batched_f16.gguf")
    if not model_path.exists():
        pytest.skip(f"Model file {model_path} not found in scratch/")

    runner = ModelRunner(str(model_path), device="cuda")
    scheduler = _rt.ContinuousBatchScheduler(runner.executor, max_batch_size=8, eos_token_id=0)
    assert scheduler.max_batch_size() == 8

    # Add 3 initial requests with staggered lengths
    r1 = scheduler.add_request([101], max_new_tokens=4, temperature=0.0)
    r2 = scheduler.add_request([202], max_new_tokens=8, temperature=0.0)
    r3 = scheduler.add_request([303], max_new_tokens=6, temperature=0.0)

    assert scheduler.pending_count() == 3
    assert scheduler.active_count() == 0

    step_count = 0
    late_admitted = False

    while scheduler.has_work():
        step_count += 1
        if step_count == 3 and not late_admitted:
            # Dynamic mid-flight admission while earlier requests are running
            r4 = scheduler.add_request([404], max_new_tokens=5, temperature=0.0)
            late_admitted = True

        res = scheduler.step()
        assert len(res.new_tokens) >= 0

        if step_count > 30:
            raise RuntimeError("Continuous batching test exceeded 30 steps!")

    # Verify all requests completed
    for rid in [r1, r2, r3, r4]:
        req = scheduler.get_request(rid)
        assert req is not None
        assert req.finished, f"Request {rid} did not finish"
        assert len(req.generated_tokens) == req.max_new_tokens

    # Verify 0 VRAM leak after all completed
    assert runner.executor.get_paged_active_vram_bytes() == 0
