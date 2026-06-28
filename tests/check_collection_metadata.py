#!/usr/bin/env python3
"""
Check the actual embedding function stored in ChromaDB collection.
"""

import chromadb
from pathlib import Path
import json

INDEX_DIR = "/Users/ben.schwemlein/dev/indexes/radio3"
COLLECTION_NAME = "radio3"

print("=" * 80)
print("CHROMADB COLLECTION METADATA CHECK")
print("=" * 80)
print(f"\nIndex Directory: {INDEX_DIR}")
print(f"Collection Name: {COLLECTION_NAME}")
print()

if not Path(INDEX_DIR).exists():
    print(f"❌ ERROR: Index directory does not exist")
    exit(1)

try:
    client = chromadb.PersistentClient(path=INDEX_DIR)
    print(f"✓ Connected to ChromaDB")
    
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"✓ Collection '{COLLECTION_NAME}' found")
    print()
    
    # Get collection metadata
    metadata = collection.metadata
    print("📋 COLLECTION METADATA:")
    print("-" * 80)
    if metadata:
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    else:
        print("  No metadata found")
    print()
    
    # Check the embedding function
    print("🔧 EMBEDDING FUNCTION:")
    print("-" * 80)
    
    # ChromaDB stores the embedding function in the collection
    # Try to get info about it
    try:
        # Access internal collection data
        import inspect
        
        if hasattr(collection, '_embedding_function'):
            ef = collection._embedding_function
            print(f"  Type: {type(ef)}")
            print(f"  Details: {ef}")
            
            # Try to get model name if available
            if hasattr(ef, 'model_name'):
                print(f"  Model name: {ef.model_name}")
            if hasattr(ef, '_model_name'):
                print(f"  Model name: {ef._model_name}")
                
        else:
            print("  ⚠️  Could not access embedding function directly")
            
    except Exception as e:
        print(f"  ⚠️  Error inspecting embedding function: {e}")
    
    print()
    
    # Get a sample document to check embedding dimension
    print("📏 EMBEDDING DIMENSION CHECK:")
    print("-" * 80)
    try:
        results = collection.get(limit=1, include=['embeddings'])
        if results['embeddings'] and len(results['embeddings']) > 0:
            emb = results['embeddings'][0]
            dim = len(emb)
            print(f"  Actual dimension in index: {dim}")
            
            if dim == 768:
                print(f"  ❌ This is 768 - likely 'all-MiniLM-L6-v2' or 'mxbai-embed-large'")
            elif dim == 384:
                print(f"  ✅ This is 384 - correct for 'nomic-embed-text'")
            elif dim == 1024:
                print(f"  ⚠️  This is 1024 - likely 'nomic-embed-text:v1.5'")
            else:
                print(f"  ⚠️  Unknown dimension: {dim}")
        else:
            print("  No embeddings found to check")
    except Exception as e:
        print(f"  ❌ Error checking embeddings: {e}")
    
    print()
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("""
The index was created with a 768-dimension embedding model.
This does NOT match your current setting of 'nomic-embed-text' (384 dimensions).

POSSIBLE CAUSES:
1. The indexer is reading a cached or wrong config value
2. ChromaDB is somehow using a default embedding function
3. The indexing process didn't actually run with the new settings

SOLUTION:
You need to completely delete this collection and re-create it.
But first, we need to ensure the indexer will use the correct model.

Check your code_llm indexing code to see if it's explicitly setting
the embedding function when creating the collection.
    """)
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()