from datasets import load_dataset
import json

dataset = load_dataset(
    "HuggingFaceFW/fineweb-2",
    name="vie_Latn",
    split="train",
    streaming=True
)

with open("vietnamese_data.jsonl", "w", encoding="utf-8") as f:
    for i, sample in enumerate(dataset):
        json.dump(sample, f, ensure_ascii=False)
        f.write("\n")

        if i == 99:
            break

print("Đã lưu 100 mẫu!")