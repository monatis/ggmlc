import pytest
from ggmlc.pipeline.huggingface import from_huggingface_tokenizer

try:
    from transformers import CLIPTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
def test_clip_tokenizer_parity():
    model_id = "openai/clip-vit-base-patch32"
    hf_tok = CLIPTokenizer.from_pretrained(model_id)
    py_tok = from_huggingface_tokenizer(hf_tok)

    sample_texts = [
        "a photo of a cat",
        "a cute fluffy dog playing in the grass",
        "futuristic robot in cyberpunk city",
    ]

    for text in sample_texts:
        hf_ids = hf_tok(text, padding="max_length", max_length=77, return_tensors="pt")[
            "input_ids"
        ][0].tolist()
        py_ids = py_tok.encode(text, max_length=77, add_special_tokens=True, pad_to_max=True)
        assert hf_ids == py_ids, (
            f"Mismatch for text '{text}':\nHF: {hf_ids[:10]}\nPY: {py_ids[:10]}"
        )
