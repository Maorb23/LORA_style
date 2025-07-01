"""
Fast data collection - skips large dataset downloads, goes straight to Project Gutenberg
"""

import os
import json
import requests
import re
from pathlib import Path
from typing import List, Dict
import time

# Direct Project Gutenberg book IDs for the specified authors
AUTHOR_BOOKS = {
    "hemingway": {
        "name": "Ernest Hemingway",
        "books": [
            # Note: Most Hemingway works are still under copyright
            # These are his very early works that entered public domain
            {"id": 59603, "title": "Three Stories and Ten Poems"},
            {'id': 75201, "title": "A farewell to Arms"},
            {'id': 67138, "title": "The Sun Also Rises"},
            {"id": 61085, "title": "In Our Time"},
            {"id": 69683, "title": "Men Without Women"},
        ],
        "style_prompt": """### Instruction:
Rewrite the following text in the distinctive style of Ernest Hemingway. Focus on:
- Spare, economical prose with understated emotion
- "Iceberg Theory" - deep meaning beneath simple surface
- Dialogue that reveals character without explanation
- Focus on action and concrete details rather than abstract description
- Themes of war, love, death, and human endurance

### Text to rewrite:
{}

### Rewritten in Hemingway's style:"""
    },
    "dickens": {
        "name": "Charles Dickens",
        "books": [
            {"id": 1400, "title": "Great Expectations"},
            {"id": 766, "title": "David Copperfield"},
            {"id": 98, "title": "A Tale of Two Cities"},
            {"id": 730, "title": "Oliver Twist"},
            {"id": 580, "title": "A Christmas Carol"},
            {"id": 883, "title": "The Old Curiosity Shop"},
            {"id": 917, "title": "Little Dorrit"},
        ],
        "style_prompt": """### Instruction:
Rewrite the following text in the distinctive style of Charles Dickens. Focus on:
- Rich, detailed descriptions of characters and settings
- Social commentary on class inequality and poverty
- Vivid portrayal of both humor and pathos
- Complex sentence structures with dramatic flair
- Memorable, often exaggerated character names and traits
- London as a living, breathing character in the narrative

### Text to rewrite:
{}

### Rewritten in Dickens' style:"""
    },
    "wells": {
        "name": "H.G. Wells",
        "books": [
            {"id": 35, "title": "The Time Machine"},
            {"id": 36, "title": "The War of the Worlds"},
            {"id": 5230, "title": "The Invisible Man"},
            {"id": 159, "title": "The Island of Dr. Moreau"},
            {"id": 12750, "title": "The First Men in the Moon"},
            {"id": 30057, "title": "When the Sleeper Wakes"},
            {"id": 718, "title": "The Food of the Gods"},
        ],
        "style_prompt": """### Instruction:
Rewrite the following text in the distinctive style of H.G. Wells. Focus on:
- Scientific speculation grounded in logical extrapolation
- Clear, accessible prose that explains complex concepts
- Social commentary woven into science fiction scenarios
- Focus on how technology and science affect humanity
- First-person narrative perspective with scientific curiosity
- Blend of adventure with philosophical reflection

### Text to rewrite:
{}

### Rewritten in Wells' style:"""
    },
    "twain": {
        "name": "Mark Twain",
        "books": [
            {"id": 74, "title": "The Adventures of Tom Sawyer"},
            {"id": 76, "title": "Adventures of Huckleberry Finn"},
            {"id": 102, "title": "A Connecticut Yankee in King Arthur's Court"},
            {"id": 119, "title": "The Prince and the Pauper"},
            {"id": 3176, "title": "The Gilded Age"},
            {"id": 245, "title": "Life on the Mississippi"},
            {"id": 3177, "title": "The American Claimant"},
        ],
        "style_prompt": """### Instruction:
Rewrite the following text in the distinctive style of Mark Twain. Focus on:
- Colloquial, vernacular speech patterns and regional dialects
- Sharp wit and satirical humor targeting social hypocrisy
- Social commentary delivered through humor and irony
- Folksy, down-to-earth narrative voice with deep wisdom
- American frontier and river life settings
- Coming-of-age themes and moral awakening

### Text to rewrite:
{}

### Rewritten in Twain's style:"""
    }
}

def download_gutenberg_book(book_id: int) -> str:
    """Download a single book quickly"""
    
    urls_to_try = [
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://gutenberg.pglaf.org/files/{book_id}/{book_id}-0.txt",
    ]
    
    for url in urls_to_try:
        try:
            print(f"  📥 Downloading from {url}")
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                text = response.text
                # Basic encoding fix
                if response.encoding:
                    text = response.content.decode(response.encoding, errors='ignore')
                return clean_gutenberg_text(text)
        except Exception as e:
            print(f"    ❌ Failed: {e}")
            continue
        
        time.sleep(0.5)  # Be nice to servers
    
    return ""

def clean_gutenberg_text(text: str) -> str:
    """Quick cleaning of Gutenberg text"""
    
    # Remove Gutenberg headers/footers
    start_markers = [
        r'\*\*\*.*?START OF.*?PROJECT GUTENBERG.*?\*\*\*',
        r'START OF THE PROJECT GUTENBERG',
        r'This etext was prepared',
        r'Project Gutenberg.*?Etexts',
        r'This eBook is for the use of anyone anywhere'
    ]
    
    end_markers = [
        r'\*\*\*.*?END OF.*?PROJECT GUTENBERG.*?\*\*\*',
        r'END OF THE PROJECT GUTENBERG',
        r'End of Project Gutenberg',
        r'End of the Project Gutenberg EBook'
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
    
    # Remove chapter headers and other formatting
    text = re.sub(r'\n\s*CHAPTER [IVXLC\d]+.*?\n', '\n\n', text)
    text = re.sub(r'\n\s*Chapter \d+.*?\n', '\n\n', text)
    
    return text.strip()

def split_into_chunks(text: str, chunk_size: int = 1200) -> List[str]:
    """Split text into training chunks"""
    
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        # Skip very short paragraphs (likely formatting artifacts)
        if len(para.strip()) < 50:
            continue
            
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            if len(current_chunk.strip()) > 300:  # Min length for quality
                chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            current_chunk += "\n\n" + para if current_chunk else para
    
    if current_chunk.strip() and len(current_chunk.strip()) > 300:
        chunks.append(current_chunk.strip())
    
    return chunks

def quick_collect_author(author: str, max_samples: int = 2000) -> int:
    """Quickly collect data for one author"""
    
    if author not in AUTHOR_BOOKS:
        print(f"❌ Unknown author: {author}")
        return 0
    
    author_info = AUTHOR_BOOKS[author]
    print(f"\n📚 Collecting {author_info['name']} data...")
    
    output_file = Path(f"datasets/{author}_dataset.jsonl")
    output_file.parent.mkdir(exist_ok=True)
    
    all_chunks = []
    
    # Download each book
    for book_info in author_info["books"]:
        print(f"  📖 Getting '{book_info['title']}'...")
        
        text = download_gutenberg_book(book_info["id"])
        if text:
            chunks = split_into_chunks(text)
            all_chunks.extend(chunks)
            print(f"    ✅ Got {len(chunks)} chunks")
        else:
            print(f"    ❌ Failed to get book {book_info['id']}")
        
        # Small delay between downloads
        time.sleep(1)
    
    # Limit samples
    if len(all_chunks) > max_samples:
        all_chunks = all_chunks[:max_samples]
    
    # Save as JSONL
    samples_written = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(all_chunks):
            # Use a simple style prefix for direct style conditioning
            formatted_text = f"[{author_info['name']} Style]\n{chunk}"

            sample = {
                "text": formatted_text,
                "author": author,
                "sample_id": i,
                "word_count": len(chunk.split())
            }

            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            samples_written += 1
    
    print(f"  ✅ Saved {samples_written} samples to {output_file}")
    return samples_written

def main():
    """Quick collection main function"""
    
    print("🚀 Quick Data Collection - Classic Literary Authors")
    print("=" * 60)
    print("📚 Authors: Hemingway, Dickens, H.G. Wells, Mark Twain")
    print("⚠️  Note: Limited Hemingway works (most still under copyright)")
    print()
    
    authors = ["hemingway", "dickens", "wells", "twain"]
    total_samples = 0
    
    for author in authors:
        samples = quick_collect_author(author, max_samples=2000)
        total_samples += samples
        print()  # Add spacing between authors
    
    print("=" * 60)
    print(f"✅ Collection Complete!")
    print(f"📊 Total samples: {total_samples:,}")
    print(f"👥 Authors collected: {len([a for a in authors if Path(f'datasets/{a}_dataset.jsonl').exists()])}")
    print(f"⏱️  Efficient direct download approach!")
    
    # Create comprehensive summary
    summary = {
        "total_samples": total_samples,
        "authors": authors,
        "method": "direct_gutenberg_download",
        "focus": "Classic literary authors with distinctive styles",
        "notes": {
            "hemingway": "Limited early works (most under copyright until ~2050)",
            "dickens": "Full range of major novels available",
            "wells": "Complete science fiction classics",
            "twain": "Major American literature works"
        },
        "timestamp": str(time.time())
    }
    
    with open("datasets/collection_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📄 Summary saved to datasets/collection_summary.json")

if __name__ == "__main__":
    main()