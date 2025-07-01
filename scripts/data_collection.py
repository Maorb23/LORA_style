"""
Data Collection Utilities for Multi-Author LoRA Training

This module provides utilities to collect text data from various sources:
1. Hugging Face datasets
2. Project Gutenberg scraping
3. Data preprocessing and formatting
"""

import os
import re
import time
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datasets import Dataset, load_dataset
import numpy as np
from bs4 import BeautifulSoup
import urllib.request
import urllib.error

class ProjectGutenbergScraper:
    """Scraper for Project Gutenberg texts"""
    
    BASE_URL = "https://www.gutenberg.org"
    MIRROR_URLS = [
        "https://www.gutenberg.org",
        "https://gutenberg.pglaf.org",
        "https://aleph.gutenberg.org",
    ]
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Educational Research Tool for Literary Analysis)'
        })
    
    def get_text_by_id(self, book_id: int) -> Optional[str]:
        """Download a book from Project Gutenberg by ID"""
        
        # Try different text formats and mirror URLs
        formats = ['txt', 'txt-utf8']
        
        for mirror_url in self.MIRROR_URLS:
            for fmt in formats:
                try:
                    # Construct URL
                    if fmt == 'txt-utf8':
                        url = f"{mirror_url}/files/{book_id}/{book_id}-0.txt"
                    else:
                        url = f"{mirror_url}/files/{book_id}/{book_id}.txt"
                    
                    print(f"Trying to download from: {url}")
                    
                    # Download with timeout
                    response = self.session.get(url, timeout=30)
                    
                    if response.status_code == 200:
                        text = response.text
                        
                        # Clean the text
                        cleaned_text = self._clean_gutenberg_text(text)
                        
                        if len(cleaned_text) > 1000:  # Ensure we got substantial content
                            print(f"✅ Successfully downloaded book {book_id}")
                            return cleaned_text
                    
                except Exception as e:
                    print(f"Failed to download from {url}: {str(e)}")
                    continue
                
                # Rate limiting
                time.sleep(1)
        
        print(f"❌ Failed to download book {book_id} from all sources")
        return None
    
    def _clean_gutenberg_text(self, text: str) -> str:
        """Clean Project Gutenberg text by removing headers/footers"""
        
        # Remove common Gutenberg headers and footers
        patterns_to_remove = [
            r'\*\*\*.*?START OF.*?PROJECT GUTENBERG.*?\*\*\*.*?\n',
            r'\*\*\*.*?END OF.*?PROJECT GUTENBERG.*?\*\*\*.*',
            r'Project Gutenberg.*?public domain.*?\n',
            r'Most people start at our Web sites.*?\n',
            r'http://www\.gutenberg\..*?\n',
            r'contact the Foundation.*?\n',
            r'comply with the terms.*?\n',
        ]
        
        cleaned_text = text
        for pattern in patterns_to_remove:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove excessive whitespace
        cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)
        cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)
        
        return cleaned_text.strip()
    
    def search_by_author(self, author_name: str, limit: int = 10) -> List[Dict]:
        """Search for books by author name"""
        
        try:
            search_url = f"{self.BASE_URL}/ebooks/search/"
            params = {
                'query': author_name,
                'submit_search': 'Go!',
                'sort_order': 'downloads'
            }
            
            response = self.session.get(search_url, params=params, timeout=30)
            
            if response.status_code != 200:
                print(f"Search failed with status code: {response.status_code}")
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            books = []
            book_links = soup.find_all('a', href=re.compile(r'/ebooks/\d+'))
            
            for link in book_links[:limit]:
                href = link.get('href')
                book_id_match = re.search(r'/ebooks/(\d+)', href)
                
                if book_id_match:
                    book_id = int(book_id_match.group(1))
                    title = link.get_text(strip=True)
                    
                    books.append({
                        'id': book_id,
                        'title': title,
                        'author': author_name
                    })
            
            print(f"Found {len(books)} books for {author_name}")
            return books
            
        except Exception as e:
            print(f"Search failed for {author_name}: {str(e)}")
            return []

class HuggingFaceDatasetLoader:
    """Loader for Hugging Face datasets"""
    
    def __init__(self):
        self.cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
    
    def load_gutenberg_dataset(self, author_filter: Optional[str] = None) -> Optional[Dataset]:
        """Load Project Gutenberg dataset from Hugging Face"""
        
        try:
            # Try different Gutenberg datasets available on HF
            dataset_names = [
                "sedthh/gutenberg_english",
                "storytracer/gutenberg",
                "EleutherAI/gutenberg",
            ]
            
            for dataset_name in dataset_names:
                try:
                    print(f"Trying to load dataset: {dataset_name}")
                    dataset = load_dataset(dataset_name, split='train', streaming=False)
                    
                    # Filter by author if specified
                    if author_filter and 'author' in dataset.column_names:
                        dataset = dataset.filter(
                            lambda x: author_filter.lower() in x['author'].lower()
                        )
                    
                    if len(dataset) > 0:
                        print(f"✅ Successfully loaded {len(dataset)} samples from {dataset_name}")
                        return dataset
                        
                except Exception as e:
                    print(f"Failed to load {dataset_name}: {str(e)}")
                    continue
            
            print("❌ No Gutenberg datasets found on Hugging Face")
            return None
            
        except Exception as e:
            print(f"Error loading Gutenberg dataset: {str(e)}")
            return None
    
    def load_author_specific_dataset(self, author_key: str) -> Optional[Dataset]:
        """Try to load author-specific datasets from Hugging Face"""
        
        # Author-specific dataset mappings
        author_datasets = {
            "steinbeck": [
                "steinbeck/complete_works",
                "literature/steinbeck",
            ],
            "vonnegut": [
                "vonnegut/novels",
                "literature/vonnegut",
            ],
            "hemingway": [
                "hemingway/complete_works",
                "literature/hemingway",
            ]
        }
        
        if author_key not in author_datasets:
            return None
        
        for dataset_name in author_datasets[author_key]:
            try:
                print(f"Trying to load author dataset: {dataset_name}")
                dataset = load_dataset(dataset_name, split='train')
                print(f"✅ Successfully loaded {len(dataset)} samples from {dataset_name}")
                return dataset
            except Exception as e:
                print(f"Failed to load {dataset_name}: {str(e)}")
                continue
        
        return None

class DatasetCollector:
    """Main class for collecting and preprocessing author datasets"""
    
    def __init__(self, config):
        self.config = config
        self.gutenberg_scraper = ProjectGutenbergScraper()
        self.hf_loader = HuggingFaceDatasetLoader()
    
    def collect_author_data(self, author_key: str) -> Dataset:
        """Collect data for a specific author using hybrid approach"""
        
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
        if len(all_texts) < self.config.max_samples_per_author // 2:
            print("Step 2: Scraping from Project Gutenberg...")
            
            # Use specific book IDs if available
            gutenberg_ids = author_config.get("gutenberg_ids", [])
            for book_id in gutenberg_ids:
                text = self.gutenberg_scraper.get_text_by_id(book_id)
                if text:
                    # Split long texts into chunks
                    chunks = self._split_text_into_chunks(text)
                    all_texts.extend(chunks)
                
                if len(all_texts) >= self.config.max_samples_per_author:
                    break
            
            # Search for more books if still need more data
            if len(all_texts) < self.config.max_samples_per_author:
                search_results = self.gutenberg_scraper.search_by_author(
                    author_config['name'], limit=5
                )
                
                for book_info in search_results:
                    if book_info['id'] not in gutenberg_ids:  # Avoid duplicates
                        text = self.gutenberg_scraper.get_text_by_id(book_info['id'])
                        if text:
                            chunks = self._split_text_into_chunks(text)
                            all_texts.extend(chunks)
                    
                    if len(all_texts) >= self.config.max_samples_per_author:
                        break
        
        # 3. Filter and clean the collected texts
        cleaned_texts = self._clean_and_filter_texts(all_texts)
        
        # 4. Limit to max samples
        if len(cleaned_texts) > self.config.max_samples_per_author:
            cleaned_texts = cleaned_texts[:self.config.max_samples_per_author]
        
        print(f"Final dataset: {len(cleaned_texts)} text samples")
        
        # Create Dataset object
        dataset = Dataset.from_dict({"text": cleaned_texts})
        
        return dataset
    
    def _identify_text_column(self, dataset: Dataset) -> Optional[str]:
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
    
    def _split_text_into_chunks(self, text: str, chunk_size: int = 2000) -> List[str]:
        """Split long text into manageable chunks"""
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size, save current chunk
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _clean_and_filter_texts(self, texts: List[str]) -> List[str]:
        """Clean and filter text samples"""
        
        cleaned_texts = []
        
        for text in texts:
            # Skip if too short
            if len(text) < self.config.min_text_length:
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

# Author configurations - duplicated here to avoid circular imports
AUTHOR_CONFIGS = {
    "steinbeck": {
        "name": "John Steinbeck",
        "style_prompt": """### Instruction:
Adopt the writing style of John Steinbeck, focusing on themes of social justice, working-class struggles, vivid descriptions of California landscapes, and realistic, empathetic dialogue. Use his characteristic blend of naturalism and symbolism.

### Text:
{}""",
        "gutenberg_ids": [132, 1023, 2168, 947, 4217],
        "gutenberg_search": ["John Steinbeck"],
    },
    "vonnegut": {
        "name": "Kurt Vonnegut",
        "style_prompt": """### Instruction:
Adopt the writing style of Kurt Vonnegut, incorporating dark humor, satirical commentary on war and society, fragmented narrative structure, and his characteristic phrase "So it goes." Use his blend of science fiction and social criticism.

### Text:
{}""",
        "gutenberg_ids": [1413, 2446, 9363],
        "gutenberg_search": ["Kurt Vonnegut"],
    },
    "hemingway": {
        "name": "Ernest Hemingway",
        "style_prompt": """### Instruction:
Adopt the writing style of Ernest Hemingway, using his iceberg theory with understated prose, sparse dialogue, and themes of war, death, and human dignity. Focus on his economical, direct style.

### Text:
{}""",
        "gutenberg_ids": [4300, 61],
        "gutenberg_search": ["Ernest Hemingway"],
    },
}
