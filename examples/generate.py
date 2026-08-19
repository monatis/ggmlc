import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ggmlc.runtime.generator import GGMLCGenerator, verify_generation_parity_with_pytorch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer


def main():
    parser = argparse.ArgumentParser(description="Autoregressive text generation with ggmlc")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        choices=["gpt2", "qwen", "qwen2.5"],
        help="Autoregressive language model to load",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The capital of France is",
        help="Text prompt to generate from",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--verify-pytorch",
        action="store_true",
        help="Run differential parity test against PyTorch model.generate()",
    )
    args = parser.parse_args()

    print(f"=== [ggmlc Text Generation] Loading model '{args.model}' ===")
    if args.model == "gpt2":
        model_id = "openai-community/gpt2"
        tokenizer = GPT2Tokenizer.from_pretrained(model_id)
        from examples.models.hub_models import load_gpt2_model

        model, _, _ = load_gpt2_model()
        raw_model = GPT2LMHeadModel.from_pretrained(model_id).eval()
    else:
        model_id = "Qwen/Qwen2.5-0.5B"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        from examples.models.hub_models import load_qwen_model

        model, _, _ = load_qwen_model(model_id)
        raw_model = AutoModelForCausalLM.from_pretrained(model_id).eval()

    print(f"Loaded '{model_id}' successfully.")
    print("Compiling forward graph via ggmlc...")
    t0 = time.time()
    generator = GGMLCGenerator(model, tokenizer, model_name=args.model)
    t_compile = time.time() - t0
    print(f"Compiled successfully in {t_compile:.2f}s.")

    print(f'\nPrompt: "{args.prompt}"')
    print("Generating tokens with ggmlc generic C++ runtime...")
    t1 = time.time()
    output_text = generator.generate(args.prompt, max_new_tokens=args.max_tokens, greedy=True)
    t_gen = time.time() - t1
    print("\n--- Output Generation ---")
    print(output_text)
    print("-------------------------")
    print(f"Generation completed in {t_gen:.2f}s.")

    if args.verify_pytorch:
        print("\nVerifying parity against PyTorch model.generate()...")
        passed, ref_text, actual_text = verify_generation_parity_with_pytorch(
            raw_model, tokenizer, args.prompt, max_new_tokens=args.max_tokens
        )
        print(f"Status:        {'MATCH [OK]' if passed else 'MISMATCH [FAIL]'}")
        print(f"PyTorch text:  {ref_text}")
        print(f"ggmlc text:    {actual_text}")


if __name__ == "__main__":
    main()
