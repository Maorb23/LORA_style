"""
Quick storage check before starting the project.
"""

import shutil
import json
import os
from pathlib import Path

def get_directory_size(path):
    """Get total size of directory in bytes"""
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total += os.path.getsize(filepath)
                except (OSError, FileNotFoundError):
                    pass
    except (OSError, FileNotFoundError):
        pass
    return total

def format_bytes(bytes_value):
    """Format bytes into human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} TB"

def quick_storage_check():
    """Quick check if we have enough storage"""
    
    print("💾 Quick Storage Check")
    print("=" * 40)
    
    # Check current directory free space
    try:
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)
        total_gb = total / (1024**3)
        used_gb = used / (1024**3)
        
        print(f"Disk Space:")
        print(f"  Total: {total_gb:.1f} GB")
        print(f"  Used:  {used_gb:.1f} GB")
        print(f"  Free:  {free_gb:.1f} GB")
    except Exception as e:
        print(f"❌ Could not check disk space: {e}")
        return False
    
    # Requirements
    requirements = {
        "Text Data Collection": 0.5,  # 500 MB
        "LoRA Training Artifacts": 2.0,         # 2 GB
        "Base Model Cache": 20.0,     # 20 GB (one-time)
        "Safety Buffer": 2.5,               # 2.5 GB safety margin
    }
    
    total_needed = sum(requirements.values())
    
    print(f"\nEstimated Storage Needs:")
    for item, size in requirements.items():
        print(f"  {item:<25}: {size:>6.1f} GB")
    print(f"  {'Total':<25}: {total_needed:>6.1f} GB")
    
    print(f"\nStorage Assessment:")
    if free_gb >= total_needed:
        print(f"✅ SUFFICIENT SPACE")
        print(f"   Available: {free_gb:.1f} GB")
        print(f"   Required:  {total_needed:.1f} GB")
        print(f"   Margin:    {free_gb - total_needed:.1f} GB")
    else:
        print(f"⚠️  SPACE MIGHT BE TIGHT")
        print(f"   Available: {free_gb:.1f} GB")
        print(f"   Required:  {total_needed:.1f} GB")
        print(f"   Shortage:  {total_needed - free_gb:.1f} GB")
        
        print(f"\n💡 Recommendations:")
        print(f"   • Free up {total_needed - free_gb:.1f} GB of space")
        print(f"   • Or start with smaller dataset (--max-samples 1000)")
        print(f"   • Model cache is reusable across projects")
    
    print(f"\n📊 Project Size Estimates:")
    print(f"   Text datasets (2 authors): ~100 MB")
    print(f"   LoRA adapters: ~200 MB")  
    print(f"   Training checkpoints: ~1 GB")
    print(f"   Base model (cached): ~20 GB")
    
    return free_gb >= total_needed

def check_existing_project():
    """Check current project storage usage"""
    
    print(f"\n📁 Current Project Usage")
    print("=" * 40)
    
    directories = {
        "datasets": "./datasets",
        "lora_adapters": "./lora_adapters",
        "evaluation_results": "./evaluation_results",
        "scripts": "./scripts"
    }
    
    total_project_size = 0
    
    for name, path in directories.items():
        path_obj = Path(path)
        if path_obj.exists():
            size = get_directory_size(path_obj)
            total_project_size += size
            print(f"  {name:<20}: {format_bytes(size)}")
        else:
            print(f"  {name:<20}: Not found")
    
    if total_project_size > 0:
        print(f"  {'Total Project':<20}: {format_bytes(total_project_size)}")
    
    # Check HuggingFace cache
    hf_cache = Path.home() / ".cache" / "huggingface"
    if hf_cache.exists():
        cache_size = get_directory_size(hf_cache)
        print(f"  {'HF Cache':<20}: {format_bytes(cache_size)}")
        print(f"  {'Grand Total':<20}: {format_bytes(total_project_size + cache_size)}")

def main():
    """Main function"""
    print("🔍 LoRA Training Storage Assessment")
    print("=" * 50)
    
    # Basic storage check
    has_space = quick_storage_check()
    
    # Check existing project
    check_existing_project()
    
    print(f"\n🎯 Quick Start Recommendations")
    print("=" * 40)
    
    if has_space:
        print("✅ You're ready to start!")
        print("\nNext steps:")
        print("1. python scripts/collect_data_standalone.py")
        print("2. python scripts/verify_data.py --show-examples")
        print("3. python scripts/multi_author_train.py")
    else:
        print("⚠️  Consider freeing up space first")
        print("\nAlternatives:")
        print("1. Start with smaller dataset:")
        print("   python scripts/collect_data_standalone.py --max-samples 1000")
        print("2. Free up disk space and try again")
        print("3. Use external storage for model cache")
    
    print(f"\n💡 Tips:")
    print("• Model cache (~20GB) is reused across all projects")
    print("• Text data is small (~100MB for 2 authors)")
    print("• Training artifacts can be deleted after training")
    print("• Only LoRA adapters (~200MB) need to be kept long-term")

if __name__ == "__main__":
    main()
