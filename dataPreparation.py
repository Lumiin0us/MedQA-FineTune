from datasets import load_dataset
import json
import os

def is_valid(entry):
    # skip empty inputs or outputs
    if not entry['input'].strip() or not entry['output'].strip():
        return False
    # skip very short outputs (likely bad entries)
    if len(entry['output'].split()) < 20:
        return False
    return True

dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k")

subset = dataset['train'].shuffle(seed=42).select(range(5000))
split = subset.train_test_split(test_size=0.1, seed=42)
train = split['train']
test = split['test']

os.makedirs("data", exist_ok=True)

with open("data/train.jsonl", "w") as f:
    for data in train:
        entry = {"text": f"<|user|>\n{data['input']}\n<|assistant|>\n{data['output']}"}
        f.write(json.dumps(entry) + "\n")

with open("data/valid.jsonl", "w") as f:
    for data in test:
        entry = {"text": f"<|user|>\n{data['input']}\n<|assistant|>\n{data['output']}"}
        f.write(json.dumps(entry) + "\n")

print(f"Train: {len(train)} entries")
print(f"Valid: {len(test)} entries")
print("Saved to data/train.jsonl and data/valid.jsonl")