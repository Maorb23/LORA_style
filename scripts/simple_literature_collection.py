"""
Simple Literature Collection - Just collect raw text from authors
No training prompts, just the actual literary works
"""

import os
import json
import requests
import re
from pathlib import Path
from typing import List, Dict
import time

# Direct Project Gutenberg book IDs
AUTHOR_BOOKS = {
    "dickens": {
        "name": "Charles Dickens",
        "books": [
            {"id": 1400, "title": "Great Expectations"},
            {"id": 766, "title": "David Copperfield"},
            {"id": 98, "title": "A Tale of Two Cities"},
            {"id": 730, "title": "Oliver Twist"},
        ]
    },
    "vonnegut": {
        "name": "Kurt Vonnegut (Early Stories)",
        "books": [
            {"id": 21279, "title": "2 B R 0 2 B"},
            {"id": 30240, "title": "The Big Trip Up Yonder"},
        ]
    },
    "twain": {
        "name": "Mark Twain",
        "books": [
            {"id": 74, "title": "The Adventures of Tom Sawyer"},
            {"id": 76, "title": "Adventures of Huckleberry Finn"},
            {"id": 102, "title": "A Connecticut Yankee in King Arthur's Court"},
            {"id": 119, "title": "The Prince and the Pauper"},
        ]
    }
}

def download_gutenberg_book(book_id: int) -> str:
    """Download a single book"""
    
    urls_to_try = [
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
        f"https://gutenberg.pglaf.org/files/{book_id}/{book_id}-0.txt",
    ]
    
    for url in urls_to_try:
        try:
            print(f"  📥 Downloading from {url}")
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                text = response.text
                if response.encoding:
                    text = response.content.decode(response.encoding, errors='ignore')
                return clean_gutenberg_text(text)
        except Exception as e:
            print(f"    ❌ Failed: {e}")
            continue
        
        time.sleep(0.5)
    
    return ""

def clean_gutenberg_text(text: str) -> str:
    """Clean Gutenberg text - remove headers/footers"""
    
    # Remove Gutenberg headers/footers
    start_markers = [
        r'\*\*\*.*?START OF.*?PROJECT GUTENBERG.*?\*\*\*',
        r'START OF THE PROJECT GUTENBERG',
        r'This etext was prepared'
    ]
    
    end_markers = [
        r'\*\*\*.*?END OF.*?PROJECT GUTENBERG.*?\*\*\*',
        r'END OF THE PROJECT GUTENBERG',
        r'End of Project Gutenberg'
    ]
    
    # Find actual content
    for marker in start_markers:
        match = re.search(marker, text, re.IGNORECASE | re.DOTALL)
        if match:
            text = text[match.end():]
            break
    
    for marker in end_markers:
        match = re.search(marker, text, re.IGNORECASE | re.DOTALL)
        if match:
            text = text[:match.start()]
            break
    
    # Basic cleanup
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    return text.strip()

def collect_author_literature(author: str) -> Dict:
    """Collect all literature for one author"""
    
    if author not in AUTHOR_BOOKS:
        print(f"❌ Unknown author: {author}")
        return {}
    
    author_info = AUTHOR_BOOKS[author]
    print(f"\n📚 Collecting {author_info['name']} literature...")
    
    # Create output directory
    output_dir = Path(f"literature/{author}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_books = {}
    
    # Download each book
    for book_info in author_info["books"]:
        print(f"  📖 Getting '{book_info['title']}'...")
        
        text = download_gutenberg_book(book_info["id"])
        if text:
            # Save individual book
            book_file = output_dir / f"{book_info['title'].replace(' ', '_')}.txt"
            with open(book_file, 'w', encoding='utf-8') as f:
                f.write(text)
            
            all_books[book_info['title']] = {
                'text': text,
                'word_count': len(text.split()),
                'char_count': len(text),
                'file_path': str(book_file)
            }
            print(f"    ✅ Saved {len(text.split()):,} words to {book_file}")
        else:
            print(f"    ❌ Failed to get book {book_info['id']}")
    
    # Save combined collection info
    collection_info = {
        'author': author_info['name'],
        'books': all_books,
        'total_books': len(all_books),
        'total_words': sum(book['word_count'] for book in all_books.values()),
        'collection_date': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    info_file = output_dir / "collection_info.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(collection_info, f, indent=2, ensure_ascii=False)
    
    print(f"  ✅ Total: {len(all_books)} books, {collection_info['total_words']:,} words")
    return collection_info

def main():
    """Collect raw literature from authors"""
    
    print("📚 Simple Literature Collection")
    print("==============================")
    print("🎯 Goal: Collect raw text of classic literature")
    print()
    
    all_collections = {}
    total_books = 0
    total_words = 0
    
    for author in ["dickens", "vonnegut", "twain"]:
        collection = collect_author_literature(author)
        if collection:
            all_collections[author] = collection
            total_books += collection['total_books']
            total_words += collection['total_words']
    
    print(f"\n✅ Collection Complete!")
    print(f"📖 Total books: {total_books}")
    print(f"📝 Total words: {total_words:,}")
    print(f"📁 Files saved in: ./literature/")
    
    # Save overall summary
    summary = {
        'collections': all_collections,
        'totals': {
            'books': total_books,
            'words': total_words,
            'authors': len(all_collections)
        },
        'collection_date': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open("literature/collection_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"📋 Summary saved to: literature/collection_summary.json")

if __name__ == "__main__":
    main()
