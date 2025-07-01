"""
Steinbeck-Style Training Data Creator

Since Steinbeck's major works are still under copyright, this script demonstrates
how to create training data in his style using:
1. User-provided text samples
2. Paraphrased/rewritten content in Steinbeck's style
3. Public domain works rewritten to match his style
"""

import json
from pathlib import Path
from typing import List, Dict


# Classic Steinbeck style characteristics for training prompts
STEINBECK_STYLE_PROMPT = """### Instruction:
Rewrite the following text in the distinctive style of John Steinbeck. Focus on:
- Simple, direct prose with profound emotional depth
- Detailed descriptions of landscapes and working-class life
- Social commentary woven naturally into the narrative
- Characters who struggle against economic and social forces
- Compassionate portrayal of human dignity in hardship
- Vivid, sensory descriptions of nature and environment

### Text to rewrite:
{}

### Rewritten in Steinbeck's style:"""


def create_steinbeck_style_samples() -> List[Dict]:
    """Create sample training data in Steinbeck's style"""
    
    # These are original compositions inspired by Steinbeck's themes
    # Not direct copies of his copyrighted works
    sample_texts = [
        "The old farmer looked across his dry fields. The drought had lasted three months. His family depended on this harvest.",
        
        "Maria worked in the factory from dawn to dusk. The machines never stopped, and neither could she. Her children waited at home.",
        
        "The workers gathered at sunrise, their tools worn smooth by years of labor. They knew the land better than their own hands.",
        
        "The valley stretched wide and green, dotted with oak trees. But the migrant camp at its edge told a different story.",
        
        "Tom's hands were cracked from the sun and work. He'd been picking fruit since he was twelve. The boss paid little, but it was enough to keep going.",
        
        "The dust covered everything—the furniture, the clothes, the dreams they'd carried from Oklahoma. But they kept moving west.",
        
        "The ranch stretched as far as the eye could see. But for the workers, it might as well have been a prison with invisible walls.",
        
        "She saved every penny from her job at the cannery. Someday, she thought, her children would have better lives than this.",
        
        "The highway was full of trucks carrying families west. Each one carried the same hope and the same fear.",
        
        "In the evening, the workers sat around small fires, sharing stories and food. In their faces was the strength that comes from enduring.",
    ]
    
    samples = []
    for i, text in enumerate(sample_texts):
        sample = {
            "text": STEINBECK_STYLE_PROMPT.format(text),
            "author": "steinbeck_style",
            "sample_id": i,
            "raw_text": text,
            "note": "Original composition inspired by Steinbeck's themes"
        }
        samples.append(sample)
    
    return samples


def create_rewritten_classics() -> List[Dict]:
    """Create samples by rewriting classic texts in Steinbeck's style"""
    
    # Take some public domain passages and show how they'd be rewritten
    classic_passages = [
        "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness.",
        
        "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, nor yet a dry, bare, sandy hole.",
        
        "Call me Ishmael. Some years ago—never mind how long precisely—having little or no money in my purse.",
        
        "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
        
        "Happy families are all alike; every unhappy family is unhappy in its own way.",
    ]
    
    samples = []
    for i, text in enumerate(classic_passages):
        sample = {
            "text": STEINBECK_STYLE_PROMPT.format(text),
            "author": "steinbeck_style", 
            "sample_id": i + 100,
            "raw_text": text,
            "note": "Classic literature rewritten in Steinbeck style"
        }
        samples.append(sample)
    
    return samples


def create_user_template() -> str:
    """Create a template for users to add their own content"""
    
    template = """
# Add Your Own Content for Steinbeck-Style Training

To create more training data:

1. Add your own passages to the list below
2. Run this script to generate training samples
3. Make sure any content you add is either:
   - Your original writing
   - Public domain text
   - Text you have permission to use

USER_TEXTS = [
    "Your first text passage here...",
    "Your second text passage here...", 
    "And so on..."
]

# The script will automatically format these with Steinbeck-style prompts
"""
    
    return template


def main():
    """Create Steinbeck-style training dataset"""
    
    print("📝 Creating Steinbeck-Style Training Data")
    print("=" * 50)
    print("⚖️  Note: Using original content inspired by Steinbeck's themes")
    print("📚 Major Steinbeck works are still under copyright")
    print()
    
    # Create output directory
    output_dir = Path("datasets")
    output_dir.mkdir(exist_ok=True)
    
    # Generate samples
    all_samples = []
    
    # Add thematic samples
    thematic_samples = create_steinbeck_style_samples()
    all_samples.extend(thematic_samples)
    print(f"✅ Created {len(thematic_samples)} thematic samples")
    
    # Add rewritten classics
    rewritten_samples = create_rewritten_classics()
    all_samples.extend(rewritten_samples)
    print(f"✅ Created {len(rewritten_samples)} rewritten classic samples")
    
    # Save the dataset
    output_file = output_dir / "steinbeck_style_dataset.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"📁 Saved {len(all_samples)} samples to {output_file}")
    
    # Create user template
    template_file = output_dir / "add_your_content_template.py"
    with open(template_file, 'w', encoding='utf-8') as f:
        f.write(create_user_template())
    
    print(f"📋 Created template file: {template_file}")
    print()
    print("🎯 Summary:")
    print(f"   Total samples: {len(all_samples)}")
    print(f"   Output file: {output_file}")
    print(f"   Template: {template_file}")
    print()
    print("💡 To add more data:")
    print("   1. Edit the template file with your own content")
    print("   2. Re-run this script")
    print("   3. Make sure you have rights to use any text you add")


if __name__ == "__main__":
    main()
