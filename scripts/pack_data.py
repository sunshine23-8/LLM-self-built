import argparse
import json
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset
from src.data.pretrain_dataset import pack_texts_to_bin
from src.tokenizer.bpe import ByteLevelBPETokenizer


def iter_jsonl(file_path: Path):
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            yield item.get("text", "")


def iter_hf_streaming(dataset_name: str, name: str | None, split: str, max_samples: int):
    kwargs = {"split": split, "streaming": True}
    if name:
        kwargs["name"] = name
    ds = load_dataset(dataset_name, **kwargs)
    for i, item in enumerate(ds):
        if i >= max_samples:
            break
        yield item.get("text", "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack text data into uint16 binary train.bin using custom tokenizer")
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("artifacts/tokenizer"))
    parser.add_argument("--input-jsonl", type=Path, default=None, help="Path to local .jsonl file")
    parser.add_argument("--hf-dataset", type=str, default=None, help="Hugging Face dataset name (e.g. HuggingFaceFW/fineweb-2)")
    parser.add_argument("--hf-subset", type=str, default=None, help="Subset (e.g. vie_Latn)")
    parser.add_argument("--max-samples", type=int, default=300_000)
    parser.add_argument("--output-bin", type=Path, default=Path("data/train.bin"))
    args = parser.parse_args()

    if not args.tokenizer_dir.exists():
        raise FileNotFoundError(f"Tokenizer directory {args.tokenizer_dir} does not exist. Train tokenizer first.")

    print(f"Loading Tokenizer from {args.tokenizer_dir}...")
    tokenizer = ByteLevelBPETokenizer.load(args.tokenizer_dir)
    print(f"Tokenizer loaded with vocab_size={tokenizer.vocab_size}")

    if args.input_jsonl and args.input_jsonl.exists():
        print(f"Packing from local JSONL: {args.input_jsonl}...")
        texts = iter_jsonl(args.input_jsonl)
    elif args.hf_dataset:
        print(f"Streaming from HF {args.hf_dataset} (subset={args.hf_subset}, max_samples={args.max_samples})...")
        texts = iter_hf_streaming(args.hf_dataset, args.hf_subset, "train", args.max_samples)
    else:
        raise ValueError("Must provide either --input-jsonl or --hf-dataset")

    total_tokens = pack_texts_to_bin(texts, tokenizer, args.output_bin)
    file_size_mb = args.output_bin.stat().st_size / (1024**2)
    print("=" * 60)
    print(f"Packed file  : {args.output_bin}")
    print(f"Total tokens : {total_tokens:,} tokens")
    print(f"File size    : {file_size_mb:.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
