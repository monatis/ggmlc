from ggmlc.pipeline.spec import TokenizerSpec
from ggmlc.pipeline.tokenizer import BPETokenizer, WordPieceTokenizer


def test_bpe_tokenizer_basic():
    vocab = [
        "<|unk|>",
        "<|startoftext|>",
        "<|endoftext|>",
        "h",
        "e",
        "l",
        "o",
        "w",
        "r",
        "d",
        "he",
        "ll",
        "lo",
        "world",
    ]
    merges = ["h e", "l l", "l o", "w o r l d"]
    spec = TokenizerSpec(
        model_type="bpe",
        vocab=vocab,
        merges=merges,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=2,
        context_length=10,
    )
    tok = BPETokenizer(spec)
    assert tok.vocab_size == len(vocab)
    assert tok.bos_token_id == 1
    assert tok.eos_token_id == 2


def test_wordpiece_tokenizer_basic():
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", "world", "##ing"]
    spec = TokenizerSpec(
        model_type="wordpiece",
        vocab=vocab,
        bos_token_id=2,
        eos_token_id=3,
        pad_token_id=0,
        unk_token_id=1,
        context_length=8,
    )
    tok = WordPieceTokenizer(spec)
    encoded = tok.encode("hello world", add_special_tokens=True, pad_to_max=True)
    assert encoded[0] == 2  # [CLS]
    assert encoded[1] == 4  # hello
    assert encoded[2] == 5  # world
    assert encoded[3] == 3  # [SEP]
    assert len(encoded) == 8
    assert encoded[4:] == [0, 0, 0, 0]  # [PAD]


def test_bpe_tokenizer_chat_template():
    vocab = {"<|im_start|>": 0, "<|im_end|>": 1, "hello": 2}
    tok = BPETokenizer(vocab=vocab, merges=[], pre_tokenizer="llama", chat_template="chatml")
    res = tok.apply_chat_template("What is 2+2?", system="Be concise.")
    assert "<|im_start|>system\nBe concise.<|im_end|>\n" in res
    assert "<|im_start|>user\nWhat is 2+2?<|im_end|>\n" in res
    assert "<|im_start|>assistant\n" in res

    meta = tok.to_gguf_metadata()
    assert meta["tokenizer.chat_template"] == "chatml"
    assert meta["tokenizer.ggml.pre"] == "llama"
