"""
Standalone data collection script for author corpora.
Run this first to collect and prepare data before training.
"""

import os
import json
import argparse
import time
from pathlib import Path
from data_collection import ProjectGutenbergScraper, HuggingFaceDatasetLoader, AUTHOR_CONFIGS

class AuthorDataCollector:
    """Simplified collector for standalone data collection"""
    
    def __init__(self):
        self.gutenberg_scraper = ProjectGutenbergScraper()
        self.hf_loader = HuggingFaceDatasetLoader()
    
    def collect_author_data(self, author_key: str, max_samples: int = 3000) -> list:
        """Collect data for a specific author"""
        
        if author_key not in AUTHOR_CONFIGS:
            raise ValueError(f"Author '{author_key}' not found in configurations")
        
        author_config = AUTHOR_CONFIGS[author_key]
        print(f"Collecting data for {author_config['name']}...")
        
        all_texts = []
        
        # 1. Try Hugging Face datasets first
        print("Step 1: Checking Hugging Face datasets...")
        hf_dataset = self.hf_loader.load_author_specific_dataset(author_key)
        
        if hf_dataset is None:
            # Try general Gutenberg dataset with author filter
            hf_dataset = self.hf_loader.load_gutenberg_dataset(author_config['name'])
        
        if hf_dataset and len(hf_dataset) > 0:
            # Extract text from HF dataset
            text_column = self._identify_text_column(hf_dataset)
            if text_column:
                hf_texts = [item[text_column] for item in hf_dataset]
                all_texts.extend(hf_texts)
                print(f"✅ Collected {len(hf_texts)} texts from Hugging Face")
        
        # 2. Scrape from Project Gutenberg if needed
        if len(all_texts) < max_samples // 2:
            print("Step 2: Scraping from Project Gutenberg...")
            
            # Use specific book IDs if available
            gutenberg_ids = author_config.get("gutenberg_ids", [])
            for book_id in gutenberg_ids:
                print(f"Downloading book ID {book_id}...")
                text = self.gutenberg_scraper.get_text_by_id(book_id)
                if text:
                    # Split long texts into chunks
                    chunks = self._split_text_into_chunks(text)
                    all_texts.extend(chunks)
                    print(f"  Added {len(chunks)} chunks")
                
                if len(all_texts) >= max_samples:
                    break
            
            # Search for more books if still need more data
            if len(all_texts) < max_samples:
                print("Searching for additional books...")
                search_results = self.gutenberg_scraper.search_by_author(
                    author_config['name'], limit=5
                )
                
                for book_info in search_results:
                    if book_info['id'] not in gutenberg_ids:  # Avoid duplicates
                        print(f"Downloading additional book: {book_info['title']} (ID: {book_info['id']})")
                        text = self.gutenberg_scraper.get_text_by_id(book_info['id'])
                        if text:
                            chunks = self._split_text_into_chunks(text)
                            all_texts.extend(chunks)
                            print(f"  Added {len(chunks)} chunks")
                    
                    if len(all_texts) >= max_samples:
                        break
        
        # 3. Filter and clean the collected texts
        cleaned_texts = self._clean_and_filter_texts(all_texts)
        
        # 4. Limit to max samples
        if len(cleaned_texts) > max_samples:
            cleaned_texts = cleaned_texts[:max_samples]
        
        print(f"Final dataset: {len(cleaned_texts)} text samples")
        return cleaned_texts
    
    def _identify_text_column(self, dataset) -> str:
        """Identify the main text column in a dataset"""
        possible_columns = ['text', 'content', 'body', 'paragraph', 'sentence']
        
        for col in possible_columns:
            if col in dataset.column_names:
                return col
        
        # If none found, return the first string column
        for col in dataset.column_names:
            if isinstance(dataset[0][col], str):
                return col
        
        return None
    
    def _split_text_into_chunks(self, text: str, chunk_size: int = 2000) -> list:
        """Split long text into manageable chunks"""
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size, save current chunk
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                if len(current_chunk.strip()) > 100:  # Only add if substantial
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        # Add the last chunk
        if current_chunk.strip() and len(current_chunk.strip()) > 100:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _clean_and_filter_texts(self, texts: list, min_length: int = 100) -> list:
        """Clean and filter text samples"""
        
        cleaned_texts = []
        
        for text in texts:
            # Skip if too short
            if len(text) < min_length:
                continue
            
            # Basic cleaning
            text = text.strip()
            
            # Remove texts that are mostly non-alphabetic
            alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / len(text)
            if alpha_ratio < 0.8:
                continue
            
            # Remove duplicates (approximate)
            is_duplicate = False
            for existing_text in cleaned_texts[-10:]:  # Check last 10 to avoid O(n²)
                if self._texts_similar(text, existing_text):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                cleaned_texts.append(text)
        
        return cleaned_texts
    
    def _texts_similar(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """Check if two texts are similar (simple overlap check)"""
        
        if abs(len(text1) - len(text2)) / max(len(text1), len(text2)) > 0.5:
            return False
        
        # Simple word overlap check
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return overlap / union > threshold
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        return time.strftime("%Y-%m-%d %H:%M:%S")

def main():
    parser = argparse.ArgumentParser(description="Collect author data for LoRA training")
    parser.add_argument("--authors", nargs="+", default=["steinbeck", "vonnegut"],
                       help="Authors to collect data for")
    parser.add_argument("--output-dir", default="./datasets", 
                       help="Directory to save collected data")
    parser.add_argument("--max-samples", type=int, default=3000,
                       help="Maximum samples per author")
    parser.add_argument("--force-download", action="store_true",
                       help="Force re-download even if data exists")
    parser.add_argument("--test-size", type=float, default=0.1,
                       help="Fraction of data to use for testing")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🚀 Starting data collection process...")
    print(f"📁 Output directory: {output_dir}")
    print(f"👥 Authors: {', '.join(args.authors)}")
    print(f"📊 Max samples per author: {args.max_samples}")
    
    collector = AuthorDataCollector()
    results = {}
    
    for author in args.authors:
        if author not in AUTHOR_CONFIGS:
            print(f"❌ Unknown author: {author}")
            continue
            
        print(f"\n📚 Collecting data for {AUTHOR_CONFIGS[author]['name']}...")
        
        # Check if data already exists
        author_file = output_dir / f"{author}_dataset.jsonl"
        if author_file.exists() and not args.force_download:
            print(f"✅ Data already exists for {author} at {author_file}")
            print("   Use --force-download to re-collect")
            
            # Count existing samples
            with open(author_file, 'r', encoding='utf-8') as f:
                sample_count = sum(1 for _ in f)
            results[author] = {
                "status": "exists",
                "samples": sample_count,
                "file": str(author_file)
            }
            continue
        
        try:
            # Collect data
            texts = collector.collect_author_data(
                author, 
                max_samples=args.max_samples
            )
            
            if not texts:
                print(f"❌ No data collected for {author}")
                results[author] = {"status": "failed", "samples": 0}
                continue
            
            # Save as JSONL
            samples_written = 0
            with open(author_file, 'w', encoding='utf-8') as f:
                for i, text in enumerate(texts):
                    # Create training sample with style prompt
                    style_prompt = AUTHOR_CONFIGS[author].get("style_prompt", "### Text:\n{}")
                    formatted_text = style_prompt.format(text.strip())
                    
                    sample = {
                        "text": formatted_text,
                        "author": author,
                        "sample_id": i,
                        "raw_text": text.strip()
                    }
                    
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                    samples_written += 1
            
            print(f"✅ Collected {samples_written} samples for {author}")
            print(f"💾 Saved to: {author_file}")
            
            results[author] = {
                "status": "success",
                "samples": samples_written,
                "file": str(author_file)
            }
            
        except Exception as e:
            print(f"❌ Error collecting data for {author}: {str(e)}")
            results[author] = {"status": "error", "error": str(e)}
    
    # Save collection summary
    summary_file = output_dir / "collection_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            "collection_date": collector._get_timestamp(),
            "total_authors": len(args.authors),
            "successful_authors": len([r for r in results.values() if r.get("status") == "success"]),
            "total_samples": sum(r.get("samples", 0) for r in results.values()),
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n📋 Collection Summary:")
    print(f"✅ Successful: {len([r for r in results.values() if r.get('status') in ['success', 'exists']])}/{len(args.authors)} authors")
    print(f"📊 Total samples: {sum(r.get('samples', 0) for r in results.values())}")
    print(f"💾 Summary saved to: {summary_file}")
    
    return results

if __name__ == "__main__":
    main()
