import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Configuration
MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # or any other suitable base model
OUTPUT_DIR = "models/drug-llm"
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
MAX_LENGTH = 512

# Load datasets
dataset = load_dataset('json', data_files={
    'train': 'data/processed/train.jsonl',
    'validation': 'data/evaluation/eval.jsonl'
})

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Configure LoRA for parameter-efficient fine-tuning
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "v_proj"]  # Adjust based on model architecture
)

model = get_peft_model(model, peft_config)

# Tokenize the datasets
def preprocess_function(examples):
    """Preprocess the dataset by tokenizing and formatting."""
    # Format: instruction followed by output
    texts = [
        f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
        for instruction, response in zip(examples['input'], examples['output'])
    ]
    
    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    
    # Create labels (same as input_ids for causal LM)
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized

tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=['input', 'output']
)

# Define training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,
    eval_steps=100,
    save_steps=100,
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    report_to="tensorboard",
    fp16=True,
    gradient_checkpointing=True,
    warmup_ratio=0.1
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

# Train model
trainer.train()

# Save the fine-tuned model
model.save_pretrained(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

print("Fine-tuning completed and model saved!")