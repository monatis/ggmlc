"""End-to-end tests for Audio models (Whisper Tiny Encoder & Decoder with Cross-Attention)."""

import pytest
import torch
from transformers import WhisperConfig, WhisperForConditionalGeneration

from ggmlc.frontend.pytorch import export_torch_model
from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.serialization.graph import serialize_ggml_graph
from ggmlc.runtime.runner import ModelRunner
from ggmlc.validation.numerical import check_numerical_accuracy


@pytest.fixture(scope="module")
def whisper_tiny_models():
    """Builds lightweight Whisper-Tiny architecture."""
    config = WhisperConfig(
        vocab_size=51865,
        num_mel_bins=80,
        d_model=256,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=512,
        decoder_ffn_dim=512,
        max_source_positions=1500,
        max_target_positions=448,
    )
    model = WhisperForConditionalGeneration(config).eval()
    return model


def test_whisper_encoder_cpu(whisper_tiny_models):
    """Verifies Whisper Encoder on CPU."""
    encoder = whisper_tiny_models.model.encoder
    mel = torch.randn(1, 80, 3000, dtype=torch.float32)
    with torch.no_grad():
        ref = encoder(mel).last_hidden_state.numpy()

    ep = export_torch_model(encoder, (mel,), model_name="whisper_tiny_encoder")
    assert len(ep.main_graph.nodes) > 0

    ggml_graph = lower_to_ggml(ep.main_graph)
    ser = serialize_ggml_graph(ggml_graph)

    runner = ModelRunner(ser, device="cpu")
    act = runner(mel.numpy())
    if isinstance(act, dict):
        act = next(iter(act.values()))
    act = act.reshape(ref.shape)

    result = check_numerical_accuracy(ref, act, atol=1e-3)
    assert result.passed, f"Whisper Encoder CPU accuracy check failed: {result}"


def test_whisper_decoder_cpu(whisper_tiny_models):
    """Verifies Whisper Decoder with Self-Attention and Cross-Attention on CPU."""
    decoder = whisper_tiny_models.model.decoder
    input_ids = torch.tensor([[50258, 50259, 50359]], dtype=torch.long)
    enc_hidden = torch.randn(1, 1500, 256, dtype=torch.float32)

    class DecoderPredictor(torch.nn.Module):
        def __init__(self, dec):
            super().__init__()
            self.dec = dec

        def forward(self, input_ids, encoder_hidden_states):
            # Evaluate single decoder step with cross-attention
            h = self.dec.embed_tokens(input_ids) + self.dec.embed_positions(input_ids)
            h = self.dec.layers[0](h, encoder_hidden_states=encoder_hidden_states)[0]
            return self.dec.layer_norm(h)

    m = DecoderPredictor(decoder).eval()
    with torch.no_grad():
        ref = m(input_ids, enc_hidden).numpy()

    ep = export_torch_model(m, (input_ids, enc_hidden), model_name="whisper_tiny_decoder")
    assert len(ep.main_graph.nodes) > 0

    ggml_graph = lower_to_ggml(ep.main_graph)
    ser = serialize_ggml_graph(ggml_graph)

    runner = ModelRunner(ser, device="cpu")
    act = runner(input_ids.numpy(), enc_hidden.numpy()).reshape(ref.shape)

    result = check_numerical_accuracy(ref, act, atol=2e-2)
    assert result.passed, f"Whisper Decoder Cross-Attention CPU check failed: {result}"
