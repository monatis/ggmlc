import torch
from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.frontend.pytorch import export_torch_model
from ggmlc.runtime.runner import ModelRunner
from ggmlc.serialization.graph import serialize_ggml_graph
from ggmlc.validation.numerical import check_numerical_accuracy, run_compiled_model_wsl

from examples.models.hub_models import (
    load_mobilenet_v3_model,
    load_ssdlite320_mobilenet_v3_model,
)


def test_mobilenet_v3_small_e2e():
    """Validates full end-to-end compilation & numerical parity for MobileNetV3-Small."""
    torch.manual_seed(42)
    model, inputs, names = load_mobilenet_v3_model(variant="small", resolution=224)

    # 1. Reference PyTorch computation
    with torch.no_grad():
        ref_out = model(*inputs).numpy()

    # 2. Export & Lower
    exported = export_torch_model(model, inputs, model_name="mobilenet_v3_small")
    ggml_graph = lower_to_ggml(exported.main_graph)
    ser_bytes = serialize_ggml_graph(ggml_graph)

    # 3. Windows Native CPU execution
    runner = ModelRunner(ser_bytes, device="cpu")
    act_out = runner(inputs[0].numpy())
    if isinstance(act_out, dict):
        act_out = next(iter(act_out.values()))
    act_out = act_out.reshape(ref_out.shape)

    cmp = check_numerical_accuracy(ref_out, act_out, atol=1e-3)
    assert cmp.passed, f"MobileNetV3-Small CPU parity failed: {cmp.message}"

    # 4. WSL execution (CUDA / CPU)
    inputs_dict = {names[0]: inputs[0].numpy()}
    out_id = exported.main_graph.outputs[0]
    wsl_res = run_compiled_model_wsl(ser_bytes, inputs_dict, [out_id])
    wsl_out = wsl_res[out_id].reshape(ref_out.shape)
    cmp_wsl = check_numerical_accuracy(ref_out, wsl_out, atol=1e-3)
    assert cmp_wsl.passed, f"MobileNetV3-Small WSL parity failed: {cmp_wsl.message}"


def test_mobilenet_v3_large_e2e():
    """Validates full end-to-end compilation & numerical parity for MobileNetV3-Large."""
    torch.manual_seed(42)
    model, inputs, names = load_mobilenet_v3_model(variant="large", resolution=224)

    with torch.no_grad():
        ref_out = model(*inputs).numpy()

    exported = export_torch_model(model, inputs, model_name="mobilenet_v3_large")
    ggml_graph = lower_to_ggml(exported.main_graph)
    ser_bytes = serialize_ggml_graph(ggml_graph)

    runner = ModelRunner(ser_bytes, device="cpu")
    act_out = runner(inputs[0].numpy())
    if isinstance(act_out, dict):
        act_out = next(iter(act_out.values()))
    act_out = act_out.reshape(ref_out.shape)

    cmp = check_numerical_accuracy(ref_out, act_out, atol=1e-3)
    assert cmp.passed, f"MobileNetV3-Large CPU parity failed: {cmp.message}"

    inputs_dict = {names[0]: inputs[0].numpy()}
    out_id = exported.main_graph.outputs[0]
    wsl_res = run_compiled_model_wsl(ser_bytes, inputs_dict, [out_id])
    wsl_out = wsl_res[out_id].reshape(ref_out.shape)
    cmp_wsl = check_numerical_accuracy(ref_out, wsl_out, atol=1e-3)
    assert cmp_wsl.passed, f"MobileNetV3-Large WSL parity failed: {cmp_wsl.message}"


def test_ssdlite320_mobilenet_v3_e2e():
    """Validates object detection model SSDLite320-MobileNetV3-Large."""
    torch.manual_seed(42)
    model, inputs, names = load_ssdlite320_mobilenet_v3_model()

    with torch.no_grad():
        ref_bbox, ref_cls = model(*inputs)
        ref_bbox_np = ref_bbox.numpy()
        ref_cls_np = ref_cls.numpy()

    exported = export_torch_model(model, inputs, model_name="ssdlite320_mobilenet_v3")
    ggml_graph = lower_to_ggml(exported.main_graph)
    ser_bytes = serialize_ggml_graph(ggml_graph)

    # Windows Native CPU
    runner = ModelRunner(ser_bytes, device="cpu")
    act_out = runner(inputs[0].numpy())
    act_vals = list(act_out.values()) if isinstance(act_out, dict) else act_out

    # Match outputs
    for v in act_vals:
        if v.size == ref_bbox_np.size:
            cmp_bbox = check_numerical_accuracy(
                ref_bbox_np, v.reshape(ref_bbox_np.shape), atol=1e-3
            )
            assert cmp_bbox.passed, f"SSDLite bbox parity failed: {cmp_bbox.message}"
        elif v.size == ref_cls_np.size:
            cmp_cls = check_numerical_accuracy(ref_cls_np, v.reshape(ref_cls_np.shape), atol=1e-3)
            assert cmp_cls.passed, f"SSDLite cls parity failed: {cmp_cls.message}"

    # WSL Native CPU/CUDA
    inputs_dict = {names[0]: inputs[0].numpy()}
    out_ids = exported.main_graph.outputs
    wsl_res = run_compiled_model_wsl(ser_bytes, inputs_dict, out_ids)
    for oid in out_ids:
        val = wsl_res[oid]
        if val.size == ref_bbox_np.size:
            cmp = check_numerical_accuracy(ref_bbox_np, val.reshape(ref_bbox_np.shape), atol=1e-3)
            assert cmp.passed, f"SSDLite WSL bbox parity failed: {cmp.message}"
        elif val.size == ref_cls_np.size:
            cmp = check_numerical_accuracy(ref_cls_np, val.reshape(ref_cls_np.shape), atol=1e-3)
            assert cmp.passed, f"SSDLite WSL cls parity failed: {cmp.message}"
