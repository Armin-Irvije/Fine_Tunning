# Drug-LLM: Fine-tuned Language Model for Prescription Medication Information

This project fine-tunes a GPT-2 model on prescription medication data to create a specialized language model that can provide accurate responses to medication-related queries.

## Overview

Drug-LLM leverages the OpenAI GPT-2 base model and applies Parameter-Efficient Fine-Tuning using Low-Rank Adaptation to create a domain-specific model for medication information. The model is trained on instruction-following data formatted as input-output pairs for medication-related queries.

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
- PEFT 
- Datasets

## Data Format

The training and evaluation data are expected to be in JSONL format with each line containing a JSON object with the following fields:

- `input`: The instruction or query about medication
- `output`: The expected response

Example:
```json
{"input": "What are the common side effects of Lisinopril?", "output": "Common side effects of Lisinopril include dry cough, dizziness, headache, fatigue, and nausea. In some cases, it may cause more serious side effects such as swelling of the face, lips, tongue, or throat. Contact your healthcare provider if you experience severe side effects."}
```

## Model Architecture

The model uses Low-Rank Adaptation to efficiently fine-tune the base GPT-2 model:

- LoRA applies trainable rank decomposition matrices to the attention layers
- Target module: `c_attn` 
- The approach reduces the number of trainable parameters significantly compared to full fine-tuning

## Evaluation

During training, the model is evaluated periodically on a validation dataset. After training completes, final metrics are reported:

- Loss
- Perplexity (lower is better)

## Acknowledgements

- OpenAI for the GPT-2 base model
- Hugging Face for the Transformers and PEFT libraries
- SDSU Tide cluster for GPU usage 