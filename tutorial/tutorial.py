from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, TrainingArguments, Trainer
import pandas as pd
import numpy as np
import evaluate
import torch
dataset = load_dataset("mteb/tweet_sentiment_extraction")
df = pd.DataFrame(dataset['train'])

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
def tokenize_function(examples):
   return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# to improve our processing requirements, we can create a smaller subset of the
# full dataset to fine-tune our model. The training set will be used to fine-tune our model
# while the testing set will be used to evaluate it.
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=3)
# device = torch.device('cpu')
# model.to(device)
# transformers provides a trainer class optimized for traiining,
# however, this method does not include how to evaluate the model.
# thus we need to pass an evaluation function to trainer
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
   logits, labels = eval_pred
   predictions = np.argmax(logits, axis =1)
   return metric.compute(predictions=predictions, references=labels)

# final step is to set up training arguments and start the training process
# transformers library contains the trainer class which supports a wide range of training options
training_args = TrainingArguments(
   output_dir = "test_trainer",
   per_device_train_batch_size = 1,
   per_device_eval_batch_size = 1,
   gradient_accumulation_steps = 4,
   # use_cpu = True
)

trainer = Trainer(
   model=model,
   args = training_args,
   train_dataset=small_train_dataset,
   eval_dataset=small_eval_dataset,
   compute_metrics=compute_metrics
)

trainer.train()
metrics = trainer.evaluate()
print("Evaluation Loss: ", metrics)
