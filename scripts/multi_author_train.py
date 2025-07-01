"""
Multi-Author Style LoRA Training Pipeline

This script trains separate LoRA adapters for different author writing styles.
It supports both Hugging Face datasets and Project Gutenberg scraping.
"""

import os
import json
import time
import torch
import wandb
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer
from accelerate import Accelerator
import numpy as np
from sklearn.model_selection import train_test_split
import copy

# Import data collection utilities
try:
    from data_collection import DatasetCollector
except ImportError:
    # Handle relative import
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from data_collection import DatasetCollector

#================================================================================#
# CONFIGURATION
#================================================================================#

# Author configurations with their specific prompts and data sources
AUTHOR_CONFIGS = {
    "steinbeck": {
        "name": "John Steinbeck",
        "style_prompt": """### Instruction:
Adopt the writing style of John Steinbeck, focusing on themes of social justice, working-class struggles, vivid descriptions of California landscapes, and realistic, empathetic dialogue. Use his characteristic blend of naturalism and symbolism.

### Text:
{}""",
        "hf_datasets": ["gutenberg"],  # Will check for specific author
        "gutenberg_ids": [132, 1023, 2168, 947, 4217],  # Of Mice and Men, Grapes of Wrath, etc.
        "gutenberg_search": ["John Steinbeck"],
        "lora_params": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
        }
    },
    
    "vonnegut": {
        "name": "Kurt Vonnegut",
        "style_prompt": """### Instruction:
Adopt the writing style of Kurt Vonnegut, incorporating dark humor, satirical commentary on war and society, fragmented narrative structure, and his characteristic phrase "So it goes." Use his blend of science fiction and social criticism.

### Text:
{}""",
        "hf_datasets": ["gutenberg"],
        "gutenberg_ids": [1413, 2446, 9363],  # Cat's Cradle, Player Piano, etc.
        "gutenberg_search": ["Kurt Vonnegut"],
        "lora_params": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
        }
    },
    
    # Easily extensible for more authors
    "hemingway": {
        "name": "Ernest Hemingway",
        "style_prompt": """### Instruction:
Adopt the writing style of Ernest Hemingway, using his iceberg theory with understated prose, sparse dialogue, and themes of war, death, and human dignity. Focus on his economical, direct style.

### Text:
{}""",
        "hf_datasets": ["gutenberg"],
        "gutenberg_ids": [4300, 61],  # The Sun Also Rises, etc.
        "gutenberg_search": ["Ernest Hemingway"],
        "lora_params": {
            "r": 12,
            "alpha": 24,
            "dropout": 0.1,
        }
    },
}

# Model configurations
MODEL_CONFIGS = {
    "llama-3-8b": {
        "model_name": "meta-llama/Meta-Llama-3-8B",
        "lora_target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "quantization": True,
    },
    "llama-2-7b": {
        "model_name": "meta-llama/Llama-2-7b-hf",
        "lora_target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "quantization": True,
    },
    "mistral-7b": {
        "model_name": "mistralai/Mistral-7B-v0.1",
        "lora_target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "quantization": True,
    },
}

class TrainingConfig:
    def __init__(
        self,
        authors: List[str] = ["steinbeck", "vonnegut"],
        model_config: str = "llama-3-8b",
        output_base_dir: str = "./lora_adapters",
        
        # Training parameters
        num_train_epochs: int = 2,  # Optimized for shorter runs
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        warmup_steps: int = 50,  # Reduced warmup
        max_seq_length: int = 1024,
        
        # Data parameters
        max_samples_per_author: int = 3000,  # Reduced for faster training
        validation_split: float = 0.1,
        min_text_length: int = 100,
        
        # Logging
        wandb_project: str = "StyleLORA",
        wandb_entity: Optional[str] = None,
        save_steps: int = 250,
        logging_steps: int = 25,
        
        # Hardware
        use_accelerate: bool = True,
        use_wandb: bool = True,
    ):
        self.authors = authors
        self.model_config = model_config
        self.output_base_dir = Path(output_base_dir)
        
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.max_seq_length = max_seq_length
        
        self.max_samples_per_author = max_samples_per_author
        self.validation_split = validation_split
        self.min_text_length = min_text_length
        
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        
        self.use_accelerate = use_accelerate
        self.use_wandb = use_wandb

#================================================================================#
# MULTI-AUTHOR TRAINER
#================================================================================#

class MultiAuthorTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.accelerator = Accelerator() if config.use_accelerate else None
        
        # Initialize W&B
        if config.use_wandb:
            wandb.login()
            
        # Create output directories
        for author in config.authors:
            author_dir = config.output_base_dir / author
            author_dir.mkdir(parents=True, exist_ok=True)
    
    def load_base_model_and_tokenizer(self):
        """Load the base model and tokenizer with quantization"""
        model_config = MODEL_CONFIGS[self.config.model_config]
        
        print(f"Loading base model: {model_config['model_name']}")
        
        # Quantization config for memory efficiency
        if model_config.get("quantization", True):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_config["model_name"],
            quantization_config=bnb_config,
            device_map="auto" if not self.config.use_accelerate else None,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_config["model_name"], 
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        print("Base model and tokenizer loaded successfully.")
        return model, tokenizer
    
    def train_author_adapter(self, author_key: str, model, tokenizer) -> str:
        """Train a LoRA adapter for a specific author"""
        author_config = AUTHOR_CONFIGS[author_key]
        model_config = MODEL_CONFIGS[self.config.model_config]
        
        print(f"\n{'='*60}")
        print(f"Training LoRA adapter for {author_config['name']}")
        print(f"{'='*60}")
        
        # Initialize W&B run for this author
        if self.config.use_wandb:
            run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=f"{author_key}_{self.config.model_config}",
                config={
                    "author": author_key,
                    "model": self.config.model_config,
                    **vars(self.config)
                },
                reinit=True,
            )
        
        # Collect and prepare dataset
        dataset_collector = DatasetCollector(self.config)
        dataset = dataset_collector.collect_author_data(author_key)
        
        print(f"Dataset collected. Total samples: {len(dataset)}")
        
        # Split dataset
        if len(dataset) > 100:  # Only split if we have enough data
            train_dataset, val_dataset = train_test_split(
                dataset, 
                test_size=self.config.validation_split,
                random_state=42
            )
        else:
            train_dataset = dataset
            val_dataset = None
        
        print(f"Training samples: {len(train_dataset)}")
        if val_dataset:
            print(f"Validation samples: {len(val_dataset)}")
        
        # Format dataset with author's style prompt
        def format_dataset(example):
            return {
                "text": author_config["style_prompt"].format(example["text"])
            }
        
        train_dataset = train_dataset.map(format_dataset, remove_columns=train_dataset.column_names)
        if val_dataset:
            val_dataset = val_dataset.map(format_dataset, remove_columns=val_dataset.column_names)
        
        # Show example
        print("\n--- Example of formatted training sample ---")
        print(train_dataset[0]['text'][:500] + "...")
        print("--------------------------------------------\n")
        
        # Create a fresh copy of the model for this author
        model_copy = copy.deepcopy(model)
        
        # Configure LoRA
        lora_params = author_config.get("lora_params", {"r": 16, "alpha": 32, "dropout": 0.05})
        peft_config = LoraConfig(
            r=lora_params["r"],
            lora_alpha=lora_params["alpha"],
            lora_dropout=lora_params["dropout"],
            target_modules=model_config["lora_target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Output directory for this author
        output_dir = self.config.output_base_dir / author_key
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            fp16=True,
            logging_steps=self.config.logging_steps,
            warmup_steps=self.config.warmup_steps,
            max_grad_norm=0.3,
            save_steps=self.config.save_steps,
            save_strategy="steps",
            evaluation_strategy="steps" if val_dataset else "no",
            eval_steps=self.config.save_steps if val_dataset else None,
            group_by_length=True,
            lr_scheduler_type="cosine",
            report_to="wandb" if self.config.use_wandb else None,
            run_name=f"{author_key}_{self.config.model_config}",
        )
        
        # Initialize trainer
        trainer = SFTTrainer(
            model=model_copy,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            tokenizer=tokenizer,
            args=training_args,
            packing=False,
        )
        
        # Train
        print("Starting LoRA training...")
        trainer.train()
        print("Training complete.")
        
        # Save the final adapter
        final_adapter_path = output_dir / "final_adapter"
        trainer.save_model(str(final_adapter_path))
        print(f"LoRA adapter saved to: {final_adapter_path}")
        
        # Save training config
        config_path = output_dir / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                "author": author_key,
                "author_name": author_config["name"],
                "model_config": self.config.model_config,
                "base_model": model_config["model_name"],
                "lora_params": lora_params,
                "training_params": {
                    "epochs": self.config.num_train_epochs,
                    "learning_rate": self.config.learning_rate,
                    "batch_size": self.config.per_device_train_batch_size,
                    "max_seq_length": self.config.max_seq_length,
                },
                "dataset_info": {
                    "train_samples": len(train_dataset),
                    "val_samples": len(val_dataset) if val_dataset else 0,
                }
            }, f, indent=2)
        
        # Save style prompt template
        prompt_path = output_dir / "style_prompt.txt"
        with open(prompt_path, 'w') as f:
            f.write(author_config["style_prompt"])
        
        if self.config.use_wandb:
            wandb.finish()
        
        return str(final_adapter_path)
    
    def train_all_authors(self):
        """Train LoRA adapters for all configured authors"""
        print(f"Training LoRA adapters for {len(self.config.authors)} authors")
        print(f"Authors: {', '.join(self.config.authors)}")
        print(f"Base model: {MODEL_CONFIGS[self.config.model_config]['model_name']}")
        
        # Load base model once
        base_model, tokenizer = self.load_base_model_and_tokenizer()
        
        # Train adapter for each author
        adapter_paths = {}
        for author_key in self.config.authors:
            if author_key not in AUTHOR_CONFIGS:
                print(f"Warning: Author '{author_key}' not found in configurations, skipping...")
                continue
                
            try:
                adapter_path = self.train_author_adapter(author_key, base_model, tokenizer)
                adapter_paths[author_key] = adapter_path
                print(f"✅ Successfully trained adapter for {author_key}")
            except Exception as e:
                print(f"❌ Failed to train adapter for {author_key}: {str(e)}")
                continue
        
        # Save summary
        summary_path = self.config.output_base_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                "base_model": MODEL_CONFIGS[self.config.model_config]["model_name"],
                "trained_authors": list(adapter_paths.keys()),
                "adapter_paths": adapter_paths,
                "training_config": vars(self.config),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=2)
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Successfully trained adapters for: {list(adapter_paths.keys())}")
        print(f"Summary saved to: {summary_path}")
        
        return adapter_paths

#================================================================================#
# MAIN FUNCTION
#================================================================================#

def main():
    parser = argparse.ArgumentParser(description="Multi-Author LoRA Training")
    parser.add_argument("--authors", nargs="+", default=["steinbeck", "vonnegut"],
                       help="Authors to train adapters for")
    parser.add_argument("--model", default="llama-3-8b", choices=list(MODEL_CONFIGS.keys()),
                       help="Base model to use")
    parser.add_argument("--output-dir", default="./lora_adapters",
                       help="Output directory for adapters")
    parser.add_argument("--epochs", type=int, default=2,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--max-samples", type=int, default=3000,
                       help="Maximum samples per author")
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable W&B logging")
    parser.add_argument("--no-accelerate", action="store_true",
                       help="Disable accelerate")
    
    args = parser.parse_args()
    
    # Create training configuration
    config = TrainingConfig(
        authors=args.authors,
        model_config=args.model,
        output_base_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        max_samples_per_author=args.max_samples,
        use_wandb=not args.no_wandb,
        use_accelerate=not args.no_accelerate,
    )
    
    # Initialize trainer and run
    trainer = MultiAuthorTrainer(config)
    adapter_paths = trainer.train_all_authors()
    
    print("\nTo use these adapters for inference, see the evaluation script.")

if __name__ == "__main__":
    main()
