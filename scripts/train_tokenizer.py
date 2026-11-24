import argparse
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset
from src.tokenizer.bpe import ByteLevelBPETokenizer


def iter_corpus(tinystories_samples: int, gsm8k_samples: int):
    ts = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    for i, row in enumerate(ts):
        if i >= tinystories_samples:
            break
        yield row["text"]

    gsm = load_dataset("openai/gsm8k", "main", split="train")
    for i, row in enumerate(gsm):
        if i >= gsm8k_samples:
            break
        yield f"{row['question']}\n{row['answer']}\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--tinystories-samples", type=int, default=50000)
    parser.add_argument("--gsm8k-samples", type=int, default=7473)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tokenizer"))
    args = parser.parse_args()

    tok = ByteLevelBPETokenizer()
    tok.train(iter_corpus(args.tinystories_samples, args.gsm8k_samples), vocab_size=args.vocab_size)
    tok.save(args.output_dir)
    print(f"Saved tokenizer with vocab_size={tok.vocab_size} to {args.output_dir}")


if __name__ == "__main__":
    main()
