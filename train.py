from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset
import torch

# QLoRA quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

dataset = load_dataset("json", data_files={
    "train": "data/train.jsonl",
    "validation": "data/valid.jsonl"
})

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    peft_config=lora_config,
    args=TrainingArguments(
        output_dir="./adapters",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        logging_steps=50,
        save_steps=200,
        eval_strategy="steps",
        eval_steps=200,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        report_to="none",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine"
    )
)

trainer.train()
trainer.save_model("./adapters/final")
print("Training complete")