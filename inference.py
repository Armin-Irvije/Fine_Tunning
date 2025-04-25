from transformers import AutoModelForCausalLM, GPT2Tokenizer
import torch

model_path = "models/drug-llm/final"  # path to your saved model

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

prompt = "Instruction: What is amoxicillin used for?\nResponse:"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
