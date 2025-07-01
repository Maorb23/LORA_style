"""
Example script showing how to use trained LoRA adapters for text generation
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import argparse
from pathlib import Path

def load_adapter(base_model_name: str, adapter_path: str, use_quantization: bool = True):
    """Load a LoRA adapter for inference"""
    
    print(f"Loading base model: {base_model_name}")
    
    # Quantization config for memory efficiency
    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load LoRA adapter
    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    return model, tokenizer

def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 200, 
                  temperature: float = 0.8, top_p: float = 0.9):
    """Generate text using the model"""
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode output
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return generated_text

def main():
    parser = argparse.ArgumentParser(description="Generate text using trained LoRA adapters")
    parser.add_argument("--adapter", required=True, help="Path to the LoRA adapter directory")
    parser.add_argument("--base-model", default="meta-llama/Meta-Llama-3-8B", help="Base model name")
    parser.add_argument("--prompt", help="Text prompt to generate from")
    parser.add_argument("--max-tokens", type=int, default=200, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Generation temperature")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    # Verify adapter path exists
    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        print(f"Error: Adapter path {adapter_path} does not exist")
        return
    
    # Load style prompt if available
    style_prompt_path = adapter_path.parent / "style_prompt.txt"
    style_prompt_template = ""
    
    if style_prompt_path.exists():
        with open(style_prompt_path, 'r') as f:
            style_prompt_template = f.read()
        print(f"Loaded style prompt template from: {style_prompt_path}")
    
    # Load model and adapter
    try:
        model, tokenizer = load_adapter(args.base_model, str(adapter_path))
        print("✅ Model and adapter loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Interactive mode
    if args.interactive:
        print("\n" + "="*60)
        print("Interactive Text Generation")
        print("="*60)
        print("Enter your prompts (type 'quit' to exit)")
        print("The model will generate text in the trained author's style")
        print("-"*60)
        
        while True:
            user_prompt = input("\nEnter prompt: ").strip()
            
            if user_prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            # Format with style template if available
            if style_prompt_template:
                full_prompt = style_prompt_template.format(user_prompt)
            else:
                full_prompt = user_prompt
            
            print(f"\nGenerating text...")
            try:
                generated = generate_text(
                    model, tokenizer, full_prompt, 
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature
                )
                
                print(f"\n{'='*60}")
                print("GENERATED TEXT:")
                print(f"{'='*60}")
                print(generated)
                print(f"{'='*60}")
                
            except Exception as e:
                print(f"❌ Generation failed: {e}")
    
    # Single prompt mode
    else:
        if not args.prompt:
            print("Error: --prompt is required in non-interactive mode")
            return
        
        # Format with style template if available
        if style_prompt_template:
            full_prompt = style_prompt_template.format(args.prompt)
        else:
            full_prompt = args.prompt
        
        print(f"Generating text for prompt: {args.prompt}")
        
        try:
            generated = generate_text(
                model, tokenizer, full_prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature
            )
            
            print(f"\n{'='*60}")
            print("GENERATED TEXT:")
            print(f"{'='*60}")
            print(generated)
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")

if __name__ == "__main__":
    main()
