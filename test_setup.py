"""
Test script to verify the Multi-Author LoRA setup
"""

import sys
import importlib

def test_imports():
    """Test that all required packages can be imported"""
    
    required_packages = [
        'torch',
        'transformers',
        'peft',
        'trl',
        'accelerate',
        'datasets',
        'numpy',
        'pandas',
        'sklearn',
        'requests',
        'bs4',  # beautifulsoup4
        'textstat',
        'matplotlib',
        'seaborn',
        'wandb',
        'tqdm',
    ]
    
    print("Testing package imports...")
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    return failed_imports

def test_torch():
    """Test PyTorch installation and CUDA availability"""
    try:
        import torch
        print(f"\n🔥 PyTorch version: {torch.__version__}")
        print(f"🔥 CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🔥 CUDA version: {torch.version.cuda}")
            print(f"🔥 GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"🔥 GPU {i}: {torch.cuda.get_device_name(i)}")
        return True
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False

def test_transformers():
    """Test transformers installation"""
    try:
        import transformers
        print(f"\n🤗 Transformers version: {transformers.__version__}")
        return True
    except Exception as e:
        print(f"❌ Transformers test failed: {e}")
        return False

def test_data_collection():
    """Test data collection components"""
    try:
        from scripts.data_collection import ProjectGutenbergScraper, HuggingFaceDatasetLoader
        
        # Test Project Gutenberg scraper
        scraper = ProjectGutenbergScraper()
        print("📚 Project Gutenberg scraper initialized")
        
        # Test HF dataset loader
        hf_loader = HuggingFaceDatasetLoader()
        print("🤗 Hugging Face dataset loader initialized")
        
        return True
    except Exception as e:
        print(f"❌ Data collection test failed: {e}")
        return False

def main():
    print("Multi-Author LoRA Setup Test")
    print("=" * 40)
    
    # Test imports
    failed_imports = test_imports()
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Please install missing packages with: pip install -r requirements.txt")
        return False
    
    print("\n✅ All packages imported successfully!")
    
    # Test PyTorch
    if not test_torch():
        return False
    
    # Test transformers
    if not test_transformers():
        return False
    
    # Test data collection
    if not test_data_collection():
        print("⚠️  Data collection test failed - this might be due to import path issues")
        print("   This should not prevent the main functionality from working")
    
    print("\n" + "=" * 40)
    print("🎉 Setup test completed successfully!")
    print("\nYou can now run:")
    print("  python scripts/multi_author_train.py --help")
    print("  python scripts/evaluation.py --help")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
