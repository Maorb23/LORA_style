"""
Verify collected data quality and show examples.
"""

import json
import argparse
from pathlib import Path
from collections import Counter
import re

def analyze_dataset(file_path):
    """Analyze a single dataset file."""
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # Skip malformed lines
    
    if not samples:
        return None
    
    # Basic statistics
    text_lengths = [len(sample['text']) for sample in samples]
    raw_lengths = [len(sample.get('raw_text', '')) for sample in samples]
    
    # Word count statistics
    word_counts = [len(sample.get('raw_text', '').split()) for sample in samples]
    
    # Quality checks
    quality_stats = analyze_text_quality(samples)
    
    return {
        "total_samples": len(samples),
        "avg_text_length": sum(text_lengths) / len(text_lengths),
        "avg_raw_length": sum(raw_lengths) / len(raw_lengths) if raw_lengths else 0,
        "avg_word_count": sum(word_counts) / len(word_counts) if word_counts else 0,
        "min_words": min(word_counts) if word_counts else 0,
        "max_words": max(word_counts) if word_counts else 0,
        "quality_stats": quality_stats,
        "sample_texts": samples[:3]  # First 3 samples
    }

def analyze_text_quality(samples):
    """Analyze quality aspects of the text samples"""
    
    quality_stats = {
        "has_dialogue": 0,
        "avg_sentences": 0,
        "avg_paragraphs": 0,
        "punctuation_density": 0,
        "readability_score": 0
    }
    
    total_sentences = 0
    total_paragraphs = 0
    total_punctuation = 0
    total_chars = 0
    
    for sample in samples:
        raw_text = sample.get('raw_text', '')
        if not raw_text:
            continue
            
        # Check for dialogue
        if '"' in raw_text or "'" in raw_text:
            quality_stats["has_dialogue"] += 1
        
        # Count sentences (approximate)
        sentence_count = len(re.findall(r'[.!?]+', raw_text))
        total_sentences += sentence_count
        
        # Count paragraphs
        paragraph_count = len([p for p in raw_text.split('\n\n') if p.strip()])
        total_paragraphs += paragraph_count
        
        # Punctuation density
        punctuation_count = len(re.findall(r'[.!?,:;]', raw_text))
        total_punctuation += punctuation_count
        total_chars += len(raw_text)
    
    num_samples = len(samples)
    if num_samples > 0:
        quality_stats["has_dialogue"] = (quality_stats["has_dialogue"] / num_samples) * 100
        quality_stats["avg_sentences"] = total_sentences / num_samples
        quality_stats["avg_paragraphs"] = total_paragraphs / num_samples
        quality_stats["punctuation_density"] = (total_punctuation / total_chars) * 100 if total_chars > 0 else 0
    
    return quality_stats

def show_text_examples(sample, detailed=False):
    """Show example text with analysis"""
    
    raw_text = sample.get('raw_text', '')
    full_text = sample.get('text', '')
    
    print(f"    📝 Sample ID: {sample.get('sample_id', 'Unknown')}")
    print(f"    📏 Length: {len(raw_text)} chars, {len(raw_text.split())} words")
    
    if detailed:
        # Show text characteristics
        has_dialogue = '"' in raw_text or "'" in raw_text
        sentence_count = len(re.findall(r'[.!?]+', raw_text))
        paragraph_count = len([p for p in raw_text.split('\n\n') if p.strip()])
        
        print(f"    💬 Has dialogue: {'Yes' if has_dialogue else 'No'}")
        print(f"    📖 Sentences: ~{sentence_count}")
        print(f"    📄 Paragraphs: {paragraph_count}")
    
    # Show preview
    preview = raw_text[:300] + "..." if len(raw_text) > 300 else raw_text
    print(f"\n    📖 Preview:")
    print(f"    {preview}")
    
    if detailed and full_text != raw_text:
        print(f"\n    🎯 Formatted (first 200 chars):")
        formatted_preview = full_text[:200] + "..." if len(full_text) > 200 else full_text
        print(f"    {formatted_preview}")

def main():
    parser = argparse.ArgumentParser(description="Verify collected author data")
    parser.add_argument("--data-dir", default="./datasets",
                       help="Directory containing collected data")
    parser.add_argument("--show-examples", action="store_true",
                       help="Show example texts")
    parser.add_argument("--detailed", action="store_true",
                       help="Show detailed analysis")
    parser.add_argument("--author", type=str,
                       help="Analyze specific author only")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("   Run data collection first!")
        return
    
    # Find dataset files
    if args.author:
        dataset_files = list(data_dir.glob(f"*{args.author}*.jsonl"))
        if not dataset_files:
            print(f"❌ No dataset files found for author: {args.author}")
            return
    else:
        dataset_files = list(data_dir.glob("*dataset.jsonl")) + list(data_dir.glob("*minimal.jsonl"))
    
    if not dataset_files:
        print(f"❌ No dataset files found in {data_dir}")
        print("   Run collect_data_standalone.py or quick_collect_data.py first!")
        return
    
    print(f"📊 Analyzing {len(dataset_files)} dataset files...\n")
    
    total_samples = 0
    
    for file_path in dataset_files:
        author = file_path.stem.replace("_dataset", "").replace("_minimal", "")
        print(f"📚 {author.upper()} Dataset Analysis")
        print("=" * 50)
        
        stats = analyze_dataset(file_path)
        if not stats:
            print("❌ Empty or invalid dataset!")
            continue
        
        total_samples += stats['total_samples']
        
        print(f"📈 Total samples: {stats['total_samples']:,}")
        print(f"📝 Average word count: {stats['avg_word_count']:.1f}")
        print(f"📏 Word count range: {stats['min_words']} - {stats['max_words']}")
        print(f"💬 Average text length: {stats['avg_text_length']:.0f} chars")
        
        if stats['avg_raw_length'] > 0:
            print(f"📄 Average raw length: {stats['avg_raw_length']:.0f} chars")
        
        # Quality metrics
        quality = stats['quality_stats']
        print(f"\n📊 Quality Metrics:")
        print(f"  💬 Samples with dialogue: {quality['has_dialogue']:.1f}%")
        print(f"  📖 Average sentences per sample: {quality['avg_sentences']:.1f}")
        print(f"  📄 Average paragraphs per sample: {quality['avg_paragraphs']:.1f}")
        print(f"  ✏️  Punctuation density: {quality['punctuation_density']:.1f}%")
        
        if args.show_examples and stats['sample_texts']:
            print(f"\n📖 Example texts:")
            for i, sample in enumerate(stats['sample_texts'], 1):
                print(f"\n  Example {i}:")
                show_text_examples(sample, detailed=args.detailed)
        
        print("\n" + "="*50 + "\n")
    
    # Summary
    print(f"🎯 SUMMARY")
    print(f"📊 Total samples across all authors: {total_samples:,}")
    print(f"👥 Authors analyzed: {len(dataset_files)}")
    
    if total_samples < 1000:
        print(f"⚠️  Low sample count - consider collecting more data")
    elif total_samples > 10000:
        print(f"✅ Good sample count for training")
    else:
        print(f"✅ Adequate sample count for training")

if __name__ == "__main__":
    main()
