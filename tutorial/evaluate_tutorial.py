from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, TrainingArguments, Trainer
import pandas as pd
import numpy as np
import evaluate
import torch

# Load the saved model after training 
model_path = "test_trainer/checkpoint-750"  # This should match your output_dir from training
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2ForSequenceClassification.from_pretrained(model_path)

# Set the model to evaluation mode
model.eval()

# Define a function to make predictions
def predict_sentiment(text):
    # Tokenize the input text
    encoded_input = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")
    
    # Make prediction
    with torch.no_grad():
        # Extract input tensors from the tokenizer output
        input_ids = encoded_input['input_ids']
        attention_mask = encoded_input['attention_mask']
        
        # Pass tensors to the model
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    
    # Map prediction to sentiment label (adjust these labels based on your dataset)
    sentiment_labels = {0: "negative", 1: "neutral", 2: "positive"}
    predicted_sentiment = sentiment_labels[prediction]
    
    # Get confidence scores using softmax
    probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
    confidence = probabilities[prediction].item()
    
    return {
        "text": text,
        "predicted_label": prediction,
        "predicted_sentiment": predicted_sentiment,
        "confidence": f"{confidence:.4f}"
    }

# Test with hardcoded examples
test_examples = [
    "I love this product! It's amazing and works perfectly.",
    "This is okay, nothing special but gets the job done.",
    "Terrible experience, would not recommend to anyone.",
    "Just got my new phone and I'm so excited to use it!",
    "The weather today is quite boring."
]

# Make predictions for each example
results = []
for example in test_examples:
    result = predict_sentiment(example)
    results.append(result)

# Display results
print("\n===== MODEL PREDICTIONS =====")
for i, result in enumerate(results):
    print(f"\nExample {i+1}:")
    print(f"Text: '{result['text']}'")
    print(f"Predicted sentiment: {result['predicted_sentiment']} (class {result['predicted_label']})")
    print(f"Confidence: {result['confidence']}")

# Optional: Evaluate on a subset of the test dataset with known labels
def evaluate_on_test_set(num_samples=50):
    dataset = load_dataset("mteb/tweet_sentiment_extraction")
    test_data = dataset["test"].shuffle(seed=42).select(range(num_samples))
    
    correct = 0
    predictions = []
    
    for item in test_data:
        try:
            text = item["text"]
            true_label = item["label"]
            result = predict_sentiment(text)
            pred_label = result["predicted_label"]
            
            if pred_label == true_label:
                correct += 1
            
            predictions.append({
                "text": text,
                "true_label": true_label, 
                "predicted_label": pred_label,
                "correct": pred_label == true_label
            })
        except Exception as e:
            print(f"Error processing item: {e}")
            continue
    
    # Recalculate accuracy based on processed examples
    processed_count = len(predictions)
    if processed_count == 0:
        print("No examples were successfully processed")
        return
        
    accuracy = correct / processed_count
    print(f"\n===== EVALUATION ON TEST SET =====")
    print(f"Tested on {num_samples} examples")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Print some examples (correct and incorrect)
    print("\nSample correct predictions:")
    correct_samples = [p for p in predictions if p["correct"]][:3]
    for i, sample in enumerate(correct_samples):
        print(f"{i+1}. '{sample['text']}' - Predicted: {sample['predicted_label']}")
    
    print("\nSample incorrect predictions:")
    incorrect_samples = [p for p in predictions if not p["correct"]][:3]
    for i, sample in enumerate(incorrect_samples):
        print(f"{i+1}. '{sample['text']}' - Predicted: {sample['predicted_label']}, True: {sample['true_label']}")

# Run evaluation on test set
evaluate_on_test_set(50)