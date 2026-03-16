from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    dtype=torch.float32
)

prompt = "<|user|>\nI have been having chest pain for 2 days. What could it be?\n<|assistant|>\n"
inputs = tokenizer(prompt, return_tensors="pt")

# base model response
with torch.no_grad():
    base_outputs = base_model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True)
base_response = tokenizer.decode(base_outputs[0], skip_special_tokens=True)


print("BASE MODEL:")
print(base_response)
print()
print("FINE-TUNED MODEL:")
print(ft_response)