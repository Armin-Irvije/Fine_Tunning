import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    GPT2Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Configuration
MODEL_NAME = "openai-community/gpt2"
OUTPUT_DIR = "models/drug-llm"
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
BATCH_SIZE = 4  # Reduced batch size
GRADIENT_ACCUMULATION_STEPS = 8  # Increased gradient accumulation
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3
MAX_LENGTH = 512

# Load datasets
dataset = load_dataset('json', data_files={
    'train': 'data/processed/train.jsonl',
    'validation': 'data/evaluation/eval.jsonl'
})

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
# GPT-2 doesn't have a pad token by default
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    use_cache=False,  # Turn off past key values caching - fixes gradient checkpointing
    device_map="auto"
)
# Resize embeddings to account for added tokens
model.resize_token_embeddings(len(tokenizer))

# Configure LoRA - use correct target modules for GPT-2
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["c_attn"],  # This is the correct module for GPT-2
    fan_in_fan_out=False        # Set to False as the warning will handle it
)

model = get_peft_model(model, peft_config)

# Tokenize the datasets
def preprocess_function(examples):
    """Preprocess the dataset by tokenizing and formatting."""
    texts = [
        f"Instruction: {instruction}\nResponse: {response}"
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
    # Use "eval_strategy" instead of "evaluation_strategy" to avoid the deprecation warning
    eval_strategy="steps",  
    save_strategy="steps",
    eval_steps=100,
    save_steps=100,
    load_best_model_at_end=True,
    report_to="tensorboard",
    fp16=True,
    # Turn off gradient_checkpointing which is causing issues
    gradient_checkpointing=False,  
    warmup_ratio=0.1,
    weight_decay=0.01,
    remove_unused_columns=False  # Important for PeftModel
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