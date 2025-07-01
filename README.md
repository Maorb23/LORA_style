# Multi-Author LoRA Training System

This system trains separate LoRA (Low-Rank Adaptation) adapters for different author writing styles using a hybrid approach for data collection from Hugging Face and Project Gutenberg.

## Features

- **Multi-Author Support**: Currently supports John Steinbeck, Kurt Vonnegut, and easily extensible to other authors
- **Hybrid Data Collection**: Combines Hugging Face datasets with Project Gutenberg scraping
- **Multiple Base Models**: Supports Llama-3-8B, Llama-2-7B, and Mistral-7B
- **Accelerate Integration**: Uses Hugging Face Accelerate for efficient training
- **W&B Integration**: Comprehensive logging with Weights & Biases
- **Quantization**: 4-bit quantization for memory efficiency
- **Comprehensive Evaluation**: Both quantitative and qualitative evaluation metrics

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Login to Hugging Face (Required for Llama models)

```bash
huggingface-cli login
```

### 3. Login to Weights & Biases (Optional but recommended)

```bash
wandb login
```

## Usage

### Training

#### Basic Training (Steinbeck and Vonnegut)

```bash
python scripts/multi_author_train.py
```

#### Custom Training Configuration

```bash
python scripts/multi_author_train.py \
    --authors steinbeck vonnegut hemingway \
    --model llama-3-8b \
    --output-dir ./my_adapters \
    --epochs 3 \
    --batch-size 4 \
    --max-samples 5000
```

#### Available Options

- `--authors`: List of authors to train (default: steinbeck vonnegut)
- `--model`: Base model to use (llama-3-8b, llama-2-7b, mistral-7b)
- `--output-dir`: Output directory for adapters (default: ./lora_adapters)
- `--epochs`: Number of training epochs (default: 2)
- `--batch-size`: Training batch size (default: 4)
- `--max-samples`: Maximum samples per author (default: 3000)
- `--no-wandb`: Disable W&B logging
- `--no-accelerate`: Disable accelerate

### Evaluation

#### Evaluate All Trained Adapters

```bash
python scripts/evaluation.py --adapters-dir ./lora_adapters
```

#### Evaluate Specific Authors

```bash
python scripts/evaluation.py \
    --adapters-dir ./lora_adapters \
    --authors steinbeck vonnegut \
    --base-model meta-llama/Meta-Llama-3-8B \
    --output-dir ./evaluation_results \
    --use-wandb
```

#### Evaluation Features

- **Style Consistency**: Measures how consistent the generated text is across different prompts
- **Comparative Analysis**: Compares different authors' styles quantitatively
- **Readability Metrics**: Flesch reading ease, Flesch-Kincaid grade level
- **Vocabulary Analysis**: Type-token ratio, vocabulary diversity
- **Authenticity Assessment**: Keyword presence and thematic alignment
- **Visualizations**: Automatic generation of comparison plots and heatmaps

## Output Structure

```
lora_adapters/
├── steinbeck/
│   ├── final_adapter/          # LoRA adapter files
│   ├── training_config.json    # Training configuration
│   ├── style_prompt.txt        # Style prompt template
│   └── checkpoint-*/           # Training checkpoints
├── vonnegut/
│   ├── final_adapter/
│   ├── training_config.json
│   ├── style_prompt.txt
│   └── checkpoint-*/
└── training_summary.json       # Overall training summary
```

## Using Trained Adapters

### Loading and Using an Adapter

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Load adapter
model = PeftModel.from_pretrained(base_model, "./lora_adapters/steinbeck/final_adapter")

# Load style prompt
with open("./lora_adapters/steinbeck/style_prompt.txt", "r") as f:
    style_prompt = f.read()

# Generate text
user_prompt = "The old man walked down the dusty road."
full_prompt = style_prompt.format(user_prompt)

inputs = tokenizer(full_prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.8)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Configuration

### Adding New Authors

To add a new author, modify the `AUTHOR_CONFIGS` dictionary in `scripts/multi_author_train.py`:

```python
AUTHOR_CONFIGS = {
    # ... existing authors ...
    "new_author": {
        "name": "New Author Name",
        "style_prompt": """### Instruction:
Adopt the writing style of New Author, focusing on [specific characteristics].

### Text:
{}""",
        "hf_datasets": ["gutenberg"],
        "gutenberg_ids": [1234, 5678],  # Project Gutenberg book IDs
        "gutenberg_search": ["New Author Name"],
        "lora_params": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
        }
    },
}
```

### Customizing LoRA Parameters

Each author can have different LoRA parameters optimized for their style:

- `r`: Rank of the adaptation (lower = fewer parameters)
- `alpha`: Scaling factor for the adaptation
- `dropout`: Dropout rate for regularization

### Model Configuration

Add new base models in the `MODEL_CONFIGS` dictionary:

```python
MODEL_CONFIGS = {
    "new-model": {
        "model_name": "path/to/new/model",
        "lora_target_modules": ["q_proj", "v_proj", ...],
        "quantization": True,
    },
}
```

## Performance Optimization

### Memory Usage

- Uses 4-bit quantization by default
- Gradient accumulation to simulate larger batch sizes
- Automatic cleanup between author training sessions

### Speed Optimization

- Reduced epochs (2 by default) for faster training
- Optimized data collection with caching
- Efficient tokenization and batching

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use gradient accumulation
2. **Slow Data Collection**: Project Gutenberg servers may be slow; consider using cached datasets
3. **Model Loading Issues**: Ensure you're logged in to Hugging Face for gated models

### Memory Requirements

- **Minimum**: 16GB RAM, 8GB VRAM
- **Recommended**: 32GB RAM, 16GB VRAM
- **Optimal**: 64GB RAM, 24GB VRAM

## Future Enhancements

- [ ] Support for more authors
- [ ] Multi-GPU training support
- [ ] Advanced evaluation metrics
- [ ] Integration with more data sources
- [ ] Fine-tuning hyperparameter optimization
- [ ] Deployment utilities for inference

## License

This project is for educational and research purposes. Please respect the licensing terms of the base models and datasets used.
