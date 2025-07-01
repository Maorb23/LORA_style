# Quick Start Guide - Multi-Author LoRA Training

## Prerequisites

1. **Python 3.8+** installed
2. **CUDA-capable GPU** (recommended, 8GB+ VRAM)
3. **16GB+ RAM** (32GB+ recommended)

## Step 1: Setup

### Windows:
```batch
# Run the setup script
setup.bat

# Or manually:
pip install -r requirements.txt
```

### Linux/Mac:
```bash
pip install -r requirements.txt
```

## Step 2: Authentication

### Hugging Face Login (Required for Llama models)
```bash
huggingface-cli login
```
Enter your Hugging Face token when prompted.

### W&B Login (Optional but recommended)
```bash
wandb login
```
Enter your W&B API key when prompted.

## Step 3: Test Installation

```bash
python test_setup.py
```

This will verify all dependencies are installed correctly.

## Step 4: Train Your First Adapters

### Quick Training (Steinbeck & Vonnegut, 2 epochs)
```bash
python scripts/multi_author_train.py
```

### Custom Training
```bash
python scripts/multi_author_train.py \
    --authors steinbeck vonnegut \
    --epochs 3 \
    --batch-size 4 \
    --max-samples 3000
```

**Training will take approximately:**
- 30-60 minutes per author (with GPU)
- 2-4 hours per author (CPU only)

## Step 5: Evaluate Results

```bash
python scripts/evaluation.py --adapters-dir ./lora_adapters
```

## Step 6: Generate Text

### Interactive Mode
```bash
python scripts/generate_example.py \
    --adapter ./lora_adapters/steinbeck/final_adapter \
    --interactive
```

### Single Generation
```bash
python scripts/generate_example.py \
    --adapter ./lora_adapters/steinbeck/final_adapter \
    --prompt "The old man walked down the dusty road"
```

## Expected Output Structure

After training, you'll have:

```
lora_adapters/
├── steinbeck/
│   ├── final_adapter/          # Use this for inference
│   ├── training_config.json
│   └── style_prompt.txt
├── vonnegut/
│   ├── final_adapter/
│   ├── training_config.json
│   └── style_prompt.txt
└── training_summary.json
```

## Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python scripts/multi_author_train.py --batch-size 2
   ```

2. **Slow Training**
   ```bash
   # Reduce samples and epochs
   python scripts/multi_author_train.py --max-samples 1000 --epochs 1
   ```

3. **Import Errors**
   ```bash
   # Reinstall dependencies
   pip install -r requirements.txt --force-reinstall
   ```

4. **Hugging Face Access Denied**
   - Make sure you're logged in: `huggingface-cli login`
   - Request access to Llama models on Hugging Face

## Next Steps

- Check `README.md` for detailed documentation
- Explore evaluation results in `./evaluation_results/`
- Modify `scripts/multi_author_train.py` to add new authors
- Experiment with different prompts and generation parameters

## Example Generations

After training, you should see outputs like:

**Steinbeck Style:**
> "The dust settled on the workers' faces as they gathered at dawn, their weathered hands already reaching for the tools that would define another day of struggle against the unforgiving land..."

**Vonnegut Style:**
> "So it goes, I thought, watching the absurd spectacle of human beings pretending their little machines could solve the fundamental problem of being alive on this peculiar planet..."

Happy training! 🚀
