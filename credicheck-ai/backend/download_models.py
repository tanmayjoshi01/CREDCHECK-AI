"""
Script to download all required models for CrediCheck AI backend.
Run this before starting the application.
"""

import os
import sys

print("=" * 60)
print("CrediCheck AI - Model Download Script")
print("=" * 60)

# 1. Download Transformers models
print("\n[1/4] Downloading Transformers models...")
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    
    print("  → Downloading facebook/bart-large-mnli (zero-shot classification)...")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    print("  ✓ facebook/bart-large-mnli downloaded")
    
    print("  → Downloading hamzab/roberta-fake-news-classification...")
    tokenizer = AutoTokenizer.from_pretrained("hamzab/roberta-fake-news-classification")
    model = AutoModelForSequenceClassification.from_pretrained("hamzab/roberta-fake-news-classification")
    print("  ✓ roberta-fake-news-classification downloaded")
    
except Exception as e:
    print(f"  ✗ Error downloading Transformers models: {e}")
    sys.exit(1)

# 2. Download spaCy model
print("\n[2/4] Downloading spaCy English model...")
try:
    import spacy
    print("  → Downloading en_core_web_sm...")
    os.system("python -m spacy download en_core_web_sm")
    # Verify download
    nlp = spacy.load("en_core_web_sm")
    print("  ✓ en_core_web_sm downloaded and verified")
except Exception as e:
    print(f"  ✗ Error downloading spaCy model: {e}")
    print("  Try running: python -m spacy download en_core_web_sm")

# 3. Check for optional XAI libraries
print("\n[3/4] Checking optional XAI libraries...")
try:
    import lime
    print("  ✓ LIME is installed")
except ImportError:
    print("  ⚠ LIME is not installed (optional)")
    print("    Install with: pip install lime")

try:
    import shap
    print("  ✓ SHAP is installed")
except ImportError:
    print("  ⚠ SHAP is not installed (optional)")
    print("    Install with: pip install shap")

# 4. Verify duckduckgo_search
print("\n[4/4] Checking duckduckgo_search...")
try:
    from duckduckgo_search import DDGS
    print("  ✓ duckduckgo_search is available")
except ImportError:
    print("  ⚠ duckduckgo_search is not installed")
    print("    Install with: pip install duckduckgo-search")

print("\n" + "=" * 60)
print("Model download complete!")
print("=" * 60)
print("\nNote: Models are cached in ~/.cache/huggingface/ (transformers)")
print("      and will be reused on subsequent runs.")
print("\nOptional XAI libraries (LIME/SHAP) can be installed with:")
print("  pip install lime shap")


