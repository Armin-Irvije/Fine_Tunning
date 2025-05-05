# Drug-LLM: Fine-tuned Language Model for Prescription Medication Information

This project fine-tunes a GPT-2 model on prescription medication data to create a specialized language model that can provide accurate responses to medication-related queries.

## Overview

Drug-LLM leverages the OpenAI GPT-2 base model and applies Parameter-Efficient Fine-Tuning (PEFT) using Low-Rank Adaptation (LoRA) to create a domain-specific model for medication information. The model is trained on instruction-following data formatted as input-output pairs for medication-related queries.

## Project Structure

```
.
├── data/
│   ├── processed/
│   │   └── train.jsonl
│   └── evaluation/
│       └── eval.jsonl
├── models/
│   └── drug-llm/
│       └── final/           # Fine-tuned model checkpoint
├── results/                 # Training metrics and evaluation results
└── train.py                # Training script
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- Datasets
- TensorBoard (for logging)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/drug-llm.git
cd drug-llm

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch transformers datasets peft tensorboard
```

## Data Format

The training and evaluation data are expected to be in JSONL format with each line containing a JSON object with the following fields:

- `input`: The instruction or query about medication
- `output`: The expected response

Example:
```json
{"input": "What are the common side effects of Lisinopril?", "output": "Common side effects of Lisinopril include dry cough, dizziness, headache, fatigue, and nausea. In some cases, it may cause more serious side effects such as swelling of the face, lips, tongue, or throat. Contact your healthcare provider if you experience severe side effects."}
```

## Training

The training script (`train.py`) configures and runs the fine-tuning process:

```bash
python train.py
```

### Training Configuration

The script uses the following hyperparameters:

- Base model: `openai-community/gpt2`
- LoRA rank: 8
- LoRA alpha: 16
- LoRA dropout: 0.05
- Batch size: 4
- Gradient accumulation steps: 8
- Learning rate: 1e-4
- Epochs: 3
- Max sequence length: 512

These parameters can be adjusted in the training script according to your requirements and hardware constraints.

## Model Architecture

The model uses Low-Rank Adaptation (LoRA) to efficiently fine-tune the base GPT-2 model:

- LoRA applies trainable rank decomposition matrices to the attention layers
- Target module: `c_attn` (attention layer in GPT-2)
- The approach reduces the number of trainable parameters significantly compared to full fine-tuning

## Evaluation

During training, the model is evaluated periodically on a validation dataset. After training completes, final metrics are reported:

- Loss
- Perplexity (lower is better)

## Acknowledgements

- OpenAI for the GPT-2 base model
- Hugging Face for the Transformers and PEFT libraries