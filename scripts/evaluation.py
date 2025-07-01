"""
Multi-Author LoRA Evaluation Script

This script evaluates trained LoRA adapters for different author styles.
It provides both quantitative and qualitative assessments.
"""

import os
import json
import torch
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)
from peft import PeftModel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from textstat import flesch_reading_ease, flesch_kincaid_grade
import re
import wandb
from collections import Counter
import pandas as pd

class StyleAnalyzer:
    """Analyze text style characteristics"""
    
    def __init__(self):
        pass
    
    def analyze_text_features(self, text: str) -> Dict:
        """Extract various text features for style analysis"""
        
        sentences = self._split_into_sentences(text)
        words = text.split()
        
        features = {
            # Basic statistics
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_sentence_length': np.mean([len(s.split()) for s in sentences]) if sentences else 0,
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            
            # Readability
            'flesch_reading_ease': flesch_reading_ease(text),
            'flesch_kincaid_grade': flesch_kincaid_grade(text),
            
            # Vocabulary diversity
            'vocabulary_size': len(set(words)),
            'type_token_ratio': len(set(words)) / len(words) if words else 0,
            
            # Punctuation usage
            'exclamation_ratio': text.count('!') / len(text) if text else 0,
            'question_ratio': text.count('?') / len(text) if text else 0,
            'comma_ratio': text.count(',') / len(text) if text else 0,
            'semicolon_ratio': text.count(';') / len(text) if text else 0,
            
            # Dialogue detection
            'dialogue_ratio': self._estimate_dialogue_ratio(text),
        }
        
        return features
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _estimate_dialogue_ratio(self, text: str) -> float:
        """Estimate the ratio of dialogue in text"""
        dialogue_indicators = ['"', "'", '—', '–']
        dialogue_chars = sum(text.count(indicator) for indicator in dialogue_indicators)
        return dialogue_chars / len(text) if text else 0

class AuthorStyleEvaluator:
    """Evaluate author-specific LoRA adapters"""
    
    def __init__(self, base_model_name: str, use_quantization: bool = True):
        self.base_model_name = base_model_name
        self.use_quantization = use_quantization
        self.style_analyzer = StyleAnalyzer()
        
        # Load base model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Load base model
        if use_quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        
        self.generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            max_new_tokens=500,
            pad_token_id=self.tokenizer.eos_token_id,
        )
    
    def load_adapter(self, adapter_path: str) -> PeftModel:
        """Load a LoRA adapter"""
        print(f"Loading adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(self.base_model, adapter_path)
        return model
    
    def generate_text(self, model: PeftModel, prompt: str, **generation_kwargs) -> str:
        """Generate text using the model"""
        
        # Update generation config with any provided kwargs
        gen_config = GenerationConfig(**{**self.generation_config.to_dict(), **generation_kwargs})
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=gen_config,
            )
        
        # Decode output
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return generated_text
    
    def evaluate_style_consistency(self, adapter_path: str, test_prompts: List[str], 
                                 author_name: str) -> Dict:
        """Evaluate style consistency of an adapter"""
        
        print(f"Evaluating style consistency for {author_name}")
        
        # Load adapter
        model = self.load_adapter(adapter_path)
        
        # Load style prompt template
        config_path = Path(adapter_path).parent / "training_config.json"
        prompt_path = Path(adapter_path).parent / "style_prompt.txt"
        
        style_prompt_template = ""
        if prompt_path.exists():
            with open(prompt_path, 'r') as f:
                style_prompt_template = f.read()
        
        # Generate texts for each prompt
        generated_texts = []
        style_features = []
        
        for prompt in test_prompts:
            # Format with style template
            if style_prompt_template:
                full_prompt = style_prompt_template.format(prompt)
            else:
                full_prompt = prompt
            
            # Generate multiple samples for consistency
            for i in range(3):  # Generate 3 samples per prompt
                try:
                    generated = self.generate_text(model, full_prompt, temperature=0.7 + i*0.1)
                    generated_texts.append(generated)
                    
                    # Analyze style features
                    features = self.style_analyzer.analyze_text_features(generated)
                    features['prompt'] = prompt
                    features['sample'] = i
                    style_features.append(features)
                    
                except Exception as e:
                    print(f"Generation failed: {e}")
                    continue
        
        # Calculate consistency metrics
        consistency_metrics = self._calculate_consistency_metrics(style_features)
        
        # Clean up model to free memory
        del model
        torch.cuda.empty_cache()
        
        return {
            'generated_texts': generated_texts,
            'style_features': style_features,
            'consistency_metrics': consistency_metrics,
            'author': author_name,
        }
    
    def _calculate_consistency_metrics(self, features_list: List[Dict]) -> Dict:
        """Calculate consistency metrics across generated samples"""
        
        if not features_list:
            return {}
        
        # Extract numeric features
        numeric_features = [
            'avg_sentence_length', 'avg_word_length', 'type_token_ratio',
            'flesch_reading_ease', 'flesch_kincaid_grade',
            'exclamation_ratio', 'question_ratio', 'comma_ratio'
        ]
        
        consistency_metrics = {}
        
        for feature in numeric_features:
            values = [f[feature] for f in features_list if feature in f and f[feature] is not None]
            if values:
                consistency_metrics[f'{feature}_mean'] = np.mean(values)
                consistency_metrics[f'{feature}_std'] = np.std(values)
                consistency_metrics[f'{feature}_cv'] = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
        
        return consistency_metrics
    
    def compare_authors(self, adapter_paths: Dict[str, str], test_prompts: List[str]) -> Dict:
        """Compare multiple author adapters"""
        
        print("Comparing author styles...")
        
        all_results = {}
        
        for author, adapter_path in adapter_paths.items():
            try:
                results = self.evaluate_style_consistency(adapter_path, test_prompts, author)
                all_results[author] = results
            except Exception as e:
                print(f"Failed to evaluate {author}: {e}")
                continue
        
        # Create comparison metrics
        comparison_metrics = self._create_comparison_metrics(all_results)
        
        return {
            'individual_results': all_results,
            'comparison_metrics': comparison_metrics
        }
    
    def _create_comparison_metrics(self, results: Dict) -> Dict:
        """Create metrics comparing different authors"""
        
        comparison = {}
        
        # Extract style features for each author
        author_features = {}
        for author, data in results.items():
            features = data['style_features']
            if features:
                author_features[author] = {
                    'avg_sentence_length': np.mean([f['avg_sentence_length'] for f in features]),
                    'avg_word_length': np.mean([f['avg_word_length'] for f in features]),
                    'type_token_ratio': np.mean([f['type_token_ratio'] for f in features]),
                    'flesch_reading_ease': np.mean([f['flesch_reading_ease'] for f in features]),
                    'dialogue_ratio': np.mean([f['dialogue_ratio'] for f in features]),
                }
        
        # Calculate pairwise differences
        authors = list(author_features.keys())
        if len(authors) >= 2:
            feature_names = list(author_features[authors[0]].keys())
            
            for i, author1 in enumerate(authors):
                for j, author2 in enumerate(authors[i+1:], i+1):
                    pair_key = f"{author1}_vs_{author2}"
                    comparison[pair_key] = {}
                    
                    for feature in feature_names:
                        val1 = author_features[author1][feature]
                        val2 = author_features[author2][feature]
                        comparison[pair_key][f'{feature}_diff'] = abs(val1 - val2)
        
        return comparison

class QualitativeEvaluator:
    """Perform qualitative evaluation of generated texts"""
    
    def __init__(self):
        self.evaluation_prompts = {
            "narrative": [
                "The old man sat on the porch, watching the sun set over the fields.",
                "She walked through the empty streets, her footsteps echoing in the silence.",
                "The factory whistle blew at dawn, calling the workers to another day of labor.",
            ],
            "dialogue": [
                "Write a conversation between two friends discussing their dreams.",
                "Create a dialogue between a parent and child about growing up.",
                "Write an argument between neighbors about a fence.",
            ],
            "descriptive": [
                "Describe a small town during the Great Depression.",
                "Paint a picture of a busy city street in words.",
                "Describe the feeling of coming home after a long journey.",
            ]
        }
    
    def evaluate_author_authenticity(self, generated_texts: List[str], author_name: str) -> Dict:
        """Evaluate how authentic the generated text sounds for the given author"""
        
        # This is a simplified version - in practice, you might use more sophisticated
        # methods like training a classifier or using human evaluators
        
        author_characteristics = {
            "steinbeck": {
                "keywords": ["dust", "valley", "worker", "poor", "land", "migrant", "california"],
                "themes": ["social justice", "working class", "nature", "human dignity"],
                "style_markers": ["naturalism", "symbolism", "realistic dialogue"]
            },
            "vonnegut": {
                "keywords": ["so it goes", "war", "absurd", "machine", "time", "planet"],
                "themes": ["war criticism", "absurdity", "dark humor", "science fiction"],
                "style_markers": ["fragmented narrative", "dark humor", "repetitive phrases"]
            },
            "hemingway": {
                "keywords": ["war", "death", "courage", "honor", "clean", "true"],
                "themes": ["war", "death", "human dignity", "lost generation"],
                "style_markers": ["iceberg theory", "understated prose", "dialogue"]
            }
        }
        
        if author_name.lower() not in author_characteristics:
            return {"error": f"No characteristics defined for {author_name}"}
        
        characteristics = author_characteristics[author_name.lower()]
        
        # Calculate keyword presence
        all_text = " ".join(generated_texts).lower()
        keyword_scores = []
        
        for keyword in characteristics["keywords"]:
            count = all_text.count(keyword.lower())
            keyword_scores.append(count)
        
        # Simple authenticity metrics
        authenticity_metrics = {
            "keyword_presence_score": np.mean(keyword_scores),
            "total_keyword_matches": sum(keyword_scores),
            "avg_text_length": np.mean([len(text) for text in generated_texts]),
            "vocabulary_richness": len(set(all_text.split())) / len(all_text.split()) if all_text else 0,
        }
        
        return authenticity_metrics

def create_visualizations(comparison_results: Dict, output_dir: Path):
    """Create visualizations for the evaluation results"""
    
    output_dir = Path(output_dir) / "evaluation_plots"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract data for plotting
    authors = list(comparison_results['individual_results'].keys())
    
    if not authors:
        print("No results to visualize")
        return
    
    # 1. Style Feature Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    features_to_plot = [
        'avg_sentence_length', 'avg_word_length', 
        'type_token_ratio', 'flesch_reading_ease'
    ]
    
    for idx, feature in enumerate(features_to_plot):
        ax = axes[idx // 2, idx % 2]
        
        author_values = []
        author_names = []
        
        for author in authors:
            features = comparison_results['individual_results'][author]['style_features']
            if features:
                values = [f[feature] for f in features if feature in f and f[feature] is not None]
                if values:
                    author_values.append(values)
                    author_names.append(author)
        
        if author_values:
            ax.boxplot(author_values, labels=author_names)
            ax.set_title(f'{feature.replace("_", " ").title()}')
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / "style_features_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Consistency Metrics Heatmap
    if len(authors) > 1:
        consistency_data = []
        
        for author in authors:
            metrics = comparison_results['individual_results'][author]['consistency_metrics']
            consistency_row = []
            
            for feature in features_to_plot:
                cv_key = f'{feature}_cv'
                if cv_key in metrics:
                    consistency_row.append(metrics[cv_key])
                else:
                    consistency_row.append(0)
            
            consistency_data.append(consistency_row)
        
        if consistency_data:
            plt.figure(figsize=(10, 6))
            sns.heatmap(
                consistency_data,
                xticklabels=[f.replace('_', ' ').title() for f in features_to_plot],
                yticklabels=authors,
                annot=True,
                fmt='.3f',
                cmap='YlOrRd'
            )
            plt.title('Style Consistency (Lower = More Consistent)')
            plt.tight_layout()
            plt.savefig(output_dir / "consistency_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"Visualizations saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Multi-Author LoRA Adapters")
    parser.add_argument("--adapters-dir", required=True, help="Directory containing trained adapters")
    parser.add_argument("--base-model", default="meta-llama/Meta-Llama-3-8B", help="Base model name")
    parser.add_argument("--output-dir", default="./evaluation_results", help="Output directory for results")
    parser.add_argument("--authors", nargs="+", help="Specific authors to evaluate (default: all found)")
    parser.add_argument("--use-wandb", action="store_true", help="Log results to W&B")
    
    args = parser.parse_args()
    
    adapters_dir = Path(args.adapters_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Find adapter paths
    adapter_paths = {}
    
    if args.authors:
        # Use specified authors
        for author in args.authors:
            author_dir = adapters_dir / author
            adapter_path = author_dir / "final_adapter"
            if adapter_path.exists():
                adapter_paths[author] = str(adapter_path)
            else:
                print(f"Warning: Adapter not found for {author} at {adapter_path}")
    else:
        # Find all available adapters
        for author_dir in adapters_dir.iterdir():
            if author_dir.is_dir():
                adapter_path = author_dir / "final_adapter"
                if adapter_path.exists():
                    adapter_paths[author_dir.name] = str(adapter_path)
    
    if not adapter_paths:
        print("No adapters found!")
        return
    
    print(f"Found adapters for: {list(adapter_paths.keys())}")
    
    # Initialize evaluator
    evaluator = AuthorStyleEvaluator(args.base_model)
    qualitative_evaluator = QualitativeEvaluator()
    
    # Test prompts
    test_prompts = [
        "The workers gathered at dawn, their faces weathered by years of hard labor.",
        "In the small town, nothing ever seemed to change, until that day.",
        "The war had ended, but its shadows still lingered in every conversation.",
        "She looked out the window and saw a world that seemed both familiar and strange.",
        "The machine hummed with the indifference of progress.",
    ]
    
    # Initialize W&B if requested
    if args.use_wandb:
        wandb.init(
            project="StyleLORA-Evaluation",
            config={
                "base_model": args.base_model,
                "authors": list(adapter_paths.keys()),
                "num_test_prompts": len(test_prompts),
            }
        )
    
    # Run evaluation
    print("Starting evaluation...")
    comparison_results = evaluator.compare_authors(adapter_paths, test_prompts)
    
    # Generate sample texts for qualitative evaluation
    print("Generating sample texts...")
    sample_results = {}
    
    for author, adapter_path in adapter_paths.items():
        print(f"Generating samples for {author}...")
        model = evaluator.load_adapter(adapter_path)
        
        # Load style prompt
        prompt_path = Path(adapter_path).parent / "style_prompt.txt"
        style_prompt_template = ""
        if prompt_path.exists():
            with open(prompt_path, 'r') as f:
                style_prompt_template = f.read()
        
        samples = []
        for i, prompt in enumerate(test_prompts[:3]):  # Use first 3 prompts
            if style_prompt_template:
                full_prompt = style_prompt_template.format(prompt)
            else:
                full_prompt = prompt
            
            generated = evaluator.generate_text(model, full_prompt, temperature=0.8)
            samples.append({
                'prompt': prompt,
                'generated': generated,
                'full_prompt': full_prompt
            })
        
        sample_results[author] = samples
        del model
        torch.cuda.empty_cache()
    
    # Save results
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'comparison_results': comparison_results,
            'sample_results': sample_results,
            'config': {
                'base_model': args.base_model,
                'adapters_evaluated': list(adapter_paths.keys()),
                'test_prompts': test_prompts,
            }
        }, f, indent=2, default=str)
    
    # Create visualizations
    create_visualizations(comparison_results, output_dir)
    
    # Generate human-readable report
    report_file = output_dir / "evaluation_report.txt"
    with open(report_file, 'w') as f:
        f.write("Multi-Author LoRA Evaluation Report\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Base Model: {args.base_model}\n")
        f.write(f"Authors Evaluated: {', '.join(adapter_paths.keys())}\n")
        f.write(f"Number of Test Prompts: {len(test_prompts)}\n\n")
        
        for author, results in comparison_results['individual_results'].items():
            f.write(f"\n{author.upper()} RESULTS:\n")
            f.write("-" * 30 + "\n")
            
            # Style consistency metrics
            consistency = results['consistency_metrics']
            f.write("Style Consistency Metrics:\n")
            for metric, value in consistency.items():
                if isinstance(value, float):
                    f.write(f"  {metric}: {value:.3f}\n")
            
            f.write("\nSample Generated Texts:\n")
            for i, text in enumerate(results['generated_texts'][:2]):  # Show first 2 samples
                f.write(f"  Sample {i+1}: {text[:200]}...\n")
    
    # Log to W&B
    if args.use_wandb:
        # Log metrics
        for author, results in comparison_results['individual_results'].items():
            for metric, value in results['consistency_metrics'].items():
                if isinstance(value, (int, float)):
                    wandb.log({f"{author}/{metric}": value})
        
        # Log sample texts as a table
        sample_data = []
        for author, samples in sample_results.items():
            for sample in samples:
                sample_data.append([
                    author,
                    sample['prompt'][:50] + "...",
                    sample['generated'][:100] + "..."
                ])
        
        wandb.log({
            "sample_generations": wandb.Table(
                columns=["Author", "Prompt", "Generated"],
                data=sample_data
            )
        })
        
        wandb.finish()
    
    print(f"\nEvaluation complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Main results file: {results_file}")
    print(f"Human-readable report: {report_file}")

if __name__ == "__main__":
    main()
