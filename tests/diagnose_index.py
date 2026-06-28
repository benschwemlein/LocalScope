#!/usr/bin/env python3
"""
Diagnostic script to analyze ChromaDB index and understand why documentation files weren't retrieved.
"""

import chromadb
from pathlib import Path
import json

# Configuration - adjust these to match your setup
INDEX_DIR = "/Users/ben.schwemlein/dev/indexes/radio3"
COLLECTION_NAME = "radio3"

def analyze_index():
    """Analyze the ChromaDB index to diagnose retrieval issues."""
    
    print("=" * 80)
    print("CHROMADB INDEX DIAGNOSTIC")
    print("=" * 80)
    print(f"\nIndex Directory: {INDEX_DIR}")
    print(f"Collection Name: {COLLECTION_NAME}")
    print()
    
    # Check if index directory exists
    if not Path(INDEX_DIR).exists():
        print(f"❌ ERROR: Index directory does not exist: {INDEX_DIR}")
        print(f"   This means the index hasn't been created yet.")
        print(f"   Go to the Index tab in code_llm and click 'Index' to create it.")
        return
    
    print(f"✓ Index directory exists")
    
    # Check if ChromaDB database file exists (don't create it!)
    db_file = Path(INDEX_DIR) / "chroma.sqlite3"
    if not db_file.exists():
        print(f"❌ ERROR: ChromaDB database file does not exist: {db_file}")
        print(f"   The directory exists but it's empty.")
        print(f"   Go to the Index tab in code_llm and click 'Index' to populate it.")
        return
    
    print(f"✓ ChromaDB database file exists ({db_file.stat().st_size:,} bytes)")
    
    # Connect to ChromaDB
    try:
        client = chromadb.PersistentClient(path=INDEX_DIR)
        print(f"✓ Connected to ChromaDB")
    except Exception as e:
        print(f"❌ ERROR: Could not connect to ChromaDB: {e}")
        return
    
    # Get collection
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        print(f"✓ Collection '{COLLECTION_NAME}' found")
    except Exception as e:
        print(f"❌ ERROR: Could not get collection: {e}")
        print("\nAvailable collections:")
        try:
            collections = client.list_collections()
            for c in collections:
                print(f"  - {c.name}")
        except:
            pass
        return
    
    # Get collection stats
    count = collection.count()
    print(f"\n📊 Total chunks in collection: {count}")
    
    if count == 0:
        print("❌ Collection is empty! No files were indexed.")
        return
    
    # Get all documents to analyze
    print("\n🔍 Analyzing indexed documents...")
    try:
        results = collection.get(
            include=['documents', 'metadatas']
        )
        
        documents = results['documents']
        metadatas = results['metadatas']
        
        # Analyze file types
        file_extensions = {}
        file_paths = set()
        
        for meta in metadatas:
            if meta and 'source' in meta:
                source = meta['source']
                file_paths.add(source)
                
                # Extract extension
                if '.' in source:
                    ext = '.' + source.rsplit('.', 1)[-1].lower()
                    file_extensions[ext] = file_extensions.get(ext, 0) + 1
        
        print(f"\n📁 Unique files indexed: {len(file_paths)}")
        print(f"📄 Total chunks: {len(documents)}")
        
        print("\n📋 File types distribution:")
        for ext, count in sorted(file_extensions.items(), key=lambda x: x[1], reverse=True):
            print(f"  {ext:12} : {count:4} chunks")
        
        # Check for documentation files
        print("\n📚 Documentation files check:")
        doc_files = [f for f in file_paths if f.endswith(('.md', '.txt', '.rst', '.adoc'))]
        
        if not doc_files:
            print("  ❌ NO documentation files (.md, .txt, .rst, .adoc) found in index!")
            print("  ⚠️  This is why your query didn't retrieve documentation.")
        else:
            print(f"  ✓ Found {len(doc_files)} documentation files:")
            for doc_file in sorted(doc_files)[:20]:  # Show first 20
                print(f"    - {doc_file}")
            if len(doc_files) > 20:
                print(f"    ... and {len(doc_files) - 20} more")
        
        # Specific files check
        print("\n🎯 Specific file checks:")
        target_files = [
            'setup.txt',
            'README.md',
            'scripts/motion_sensor/README.md'
        ]
        
        for target in target_files:
            found = any(target in f for f in file_paths)
            status = "✓ FOUND" if found else "❌ MISSING"
            print(f"  {status}: {target}")
        
        # Sample some chunks to see their content
        print("\n📝 Sample chunk analysis:")
        
        # Find a .py chunk
        py_chunks = [(i, m) for i, m in enumerate(metadatas) if m and m.get('source', '').endswith('.py')]
        if py_chunks:
            idx, meta = py_chunks[0]
            print(f"\nSample .py chunk from: {meta.get('source', 'unknown')}")
            print(f"Chunk length: {len(documents[idx])} chars")
            print(f"Preview: {documents[idx][:200]}...")
        
        # Find a doc chunk if any
        doc_chunks = [(i, m) for i, m in enumerate(metadatas) 
                      if m and any(m.get('source', '').endswith(ext) for ext in ['.md', '.txt'])]
        if doc_chunks:
            idx, meta = doc_chunks[0]
            print(f"\nSample doc chunk from: {meta.get('source', 'unknown')}")
            print(f"Chunk length: {len(documents[idx])} chars")
            print(f"Preview: {documents[idx][:200]}...")
        else:
            print("\n❌ No documentation chunks to sample (confirms they weren't indexed)")
        
        # Test query to see what would be retrieved
        print("\n🔎 Test query: 'what does this app do'")
        try:
            test_results = collection.query(
                query_texts=["what does this app do"],
                n_results=5
            )
            
            print("\nTop 5 results that WOULD be retrieved:")
            for i, (doc, meta) in enumerate(zip(test_results['documents'][0], 
                                                 test_results['metadatas'][0])):
                distance = test_results['distances'][0][i] if 'distances' in test_results else 'N/A'
                source = meta.get('source', 'unknown') if meta else 'unknown'
                print(f"\n  [{i+1}] Distance: {distance}")
                print(f"      Source: {source}")
                print(f"      Preview: {doc[:150]}...")
                
        except Exception as e:
            print(f"❌ Query test failed: {e}")
        
    except Exception as e:
        print(f"❌ ERROR during analysis: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    analyze_index()