import os
import torch
import json
from transformers import AutoModelForCausalLM, GPT2Tokenizer
from peft import PeftModel, PeftConfig
import numpy as np
from datasets import load_dataset

# Model paths
BASE_MODEL_NAME = "openai-community/gpt2"
FINETUNED_MODEL_PATH = "models/drug-llm/final"

# Load tokenizer and model
print("Loading tokenizer and model...")
tokenizer = GPT2Tokenizer.from_pretrained(FINETUNED_MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)
base_model.resize_token_embeddings(len(tokenizer))

# Load the PEFT model with LoRA adaptors
try:
    print("Loading PEFT model configuration...")
    config = PeftConfig.from_pretrained(FINETUNED_MODEL_PATH)
    print(f"PEFT Config: {config}")
    
    print("Loading fine-tuned model...")
    model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL_PATH)
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading PEFT model: {e}")
    print("Attempting to load model directly...")
    model = AutoModelForCausalLM.from_pretrained(FINETUNED_MODEL_PATH)
    model.eval()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")

def generate_response(instruction, max_length=150, temperature=0.7, top_p=0.9):
    """Generate a response to a given instruction."""
    # Format the input as in the training data
    input_text = f"Instruction: {instruction}\nResponse:"
    
    # Tokenize input
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    
    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            num_return_sequences=1
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # Extract only the response part
    try:
        response = generated_text.split("Response:")[1].strip()
    except IndexError:
        response = generated_text.strip()
        
    return response

def evaluate_on_test_set(filepath=None, num_samples=10):
    """Evaluate the model on examples from a test set or from an evaluation dataset."""
    responses = []
    
    if filepath and os.path.exists(filepath):
        # Load from file
        with open(filepath, 'r') as f:
            test_data = [json.loads(line) for line in f.readlines()]
        
        # Select a subset if needed
        if len(test_data) > num_samples:
            test_data = np.random.choice(test_data, num_samples, replace=False)
    else:
        # Try to use the evaluation dataset from the dataset
        try:
            dataset = load_dataset('json', data_files={
                'validation': 'data/evaluation/eval.jsonl'
            })
            test_data = dataset['validation'].select(range(min(num_samples, len(dataset['validation']))))
        except Exception as e:
            print(f"Error loading evaluation dataset: {e}")
            # Create some example prompts if we can't load the dataset
            test_data = [
                {"input": "What are the common side effects of Lisinopril?"},
                {"input": "How should I take Metformin?"},
                {"input": "What is Atorvastatin used for?"},
                {"input": "Can you explain how Sertraline works?"},
                {"input": "What should I know before taking Amlodipine?"},
                {"input": "Are there any drug interactions with Levothyroxine?"},
                {"input": "What's the typical dosage for Omeprazole?"},
                {"input": "What are the warnings for Ibuprofen?"},
                {"input": "Is it safe to take Acetaminophen during pregnancy?"},
                {"input": "How long does it take for Amoxicillin to work?"}
            ]
    
    print(f"\n===== EVALUATING MODEL ON {len(test_data)} EXAMPLES =====")
    
    for i, example in enumerate(test_data):
        instruction = example["input"]
        print(f"\nExample {i+1}:")
        print(f"Instruction: {instruction}")
        
        # If we have a ground truth response, show it
        if "output" in example:
            print(f"Ground Truth: {example['output']}")
        
        # Generate response
        response = generate_response(instruction)
        responses.append({
            "instruction": instruction,
            "response": response,
            "ground_truth": example.get("output", None)
        })
        
        print(f"Model Response: {response}")
        print("-" * 50)
    
    return responses

def interactive_mode():
    """Allow the user to interactively query the model."""
    print("\n===== INTERACTIVE MODE =====")
    print("Type 'exit' to quit.")
    
    while True:
        instruction = input("\nEnter your drug-related question: ")
        if instruction.lower() == 'exit':
            break
            
        response = generate_response(instruction)
        print(f"\nResponse: {response}")

def main():
    print("\n===== PRESCRIPTION DRUG MODEL EVALUATION =====")
    print("1. Test with predefined examples")
    print("2. Interactive mode (ask your own questions)")
    
    choice = input("\nSelect an option (1 or 2): ")
    
    if choice == '1':
        # Test with evaluation dataset
        eval_file = 'data/evaluation/eval.jsonl'
        responses = evaluate_on_test_set(eval_file)
        
        # Save responses
        output_file = 'model_evaluation_results.json'
        with open(output_file, 'w') as f:
            json.dump(responses, f, indent=2)
        print(f"\nEvaluation results saved to {output_file}")
    
    elif choice == '2':
        interactive_mode()
    
    else:
        print("Invalid choice. Exiting.")

# Run evaluation with some hardcoded examples if not able to load test data
print("\n===== HARDCODED EXAMPLES =====")
hardcoded_examples = [
    "What are the common side effects of Lisinopril?",
    "How should I take Metformin?",
    "What is Atorvastatin used for?",
    "Can you explain how Sertraline works?"
]

print("Testing model with a few examples:")
for example in hardcoded_examples:
    print(f"\nPrompt: {example}")
    response = generate_response(example)
    print(f"Response: {response}")

# Run the main function if executed as script
if __name__ == "__main__":
    main()