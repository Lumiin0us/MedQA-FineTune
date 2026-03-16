from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

# load base model
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float32
)

# load fine-tuned adapters on top
ft_model = PeftModel.from_pretrained(base_model, "adapters/final")

# test prompt
prompt = "<|user|>\nI have been having chest pain for 2 days. What could it be?\n<|assistant|>\n"

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = ft_model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)