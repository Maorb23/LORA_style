"""
Create focused single-author datasets for LoRA training
"""

import json
from pathlib import Path


def create_focused_dataset(author: str, max_samples: int = 1000):
    """Create a focused dataset for a single author"""
    
    input_file = Path(f"datasets/{author}_dataset.jsonl")
    output_file = Path(f"datasets/{author}_focused.jsonl")
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        return
    
    print(f"📚 Creating focused {author} dataset...")
    
    samples_written = 0
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile):
            if samples_written >= max_samples:
                break
                
            try:
                sample = json.loads(line.strip())
                raw_text = sample.get('raw_text', '')
                
                # Skip samples that are mostly metadata
                if (line_num < 3 or  # Skip first few samples which are often headers
                    len(raw_text) < 200 or  # Skip very short samples
                    'Contents' in raw_text or 
                    'Chapter' in raw_text[:100] or
                    'Produced by' in raw_text):
                    continue
                
                # Create a simpler training format
                focused_sample = {
                    "text": sample["text"],
                    "author": author,
                    "sample_id": samples_written
                }
                
                outfile.write(json.dumps(focused_sample, ensure_ascii=False) + '\n')
                samples_written += 1
                
            except Exception as e:
                print(f"⚠️  Error processing line {line_num}: {e}")
                continue
    
    print(f"✅ Created focused dataset: {output_file}")
    print(f"📊 Samples: {samples_written}")
    
    return samples_written


def main():
    """Create focused datasets for training"""
    
    print("🎯 Creating Focused Author Datasets")
    print("=" * 40)
    
    # Create focused datasets
    authors = ["dickens", "twain", "vonnegut"]
    total_samples = 0
    
    for author in authors:
        samples = create_focused_dataset(author, max_samples=800)
        total_samples += samples
        print()
    
    print(f"✅ Total focused samples: {total_samples}")
    print("🚀 Ready for LoRA training!")


if __name__ == "__main__":
    main()
