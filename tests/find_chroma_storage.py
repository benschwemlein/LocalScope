#!/usr/bin/env python3
"""
Find where ChromaDB is actually storing the radio2 collection.
"""

import os
from pathlib import Path

print("=" * 80)
print("CHROMADB STORAGE LOCATION FINDER")
print("=" * 80)
print()

# Possible locations
possible_locations = [
    "/Users/ben.schwemlein/dev/indexes/radio2",
    "/Users/ben.schwemlein/dev/indexes",
    "/Users/ben.schwemlein/dev/repos/code_llm/chroma_repo",
    Path.home() / "dev" / "indexes" / "radio2",
    Path.home() / "chroma_repo",
    Path.cwd() / "chroma_repo",
]

print("🔍 Checking possible ChromaDB locations:")
print()

found_locations = []

for loc in possible_locations:
    loc_path = Path(loc)
    if loc_path.exists():
        print(f"✅ FOUND: {loc_path}")
        
        # Check for ChromaDB files
        chroma_files = list(loc_path.rglob("chroma.sqlite3"))
        if chroma_files:
            print(f"   Contains ChromaDB database: {len(chroma_files)} file(s)")
            for cf in chroma_files:
                size = cf.stat().st_size
                print(f"   - {cf} ({size:,} bytes)")
        
        # Check for collection directories
        if (loc_path / "chroma.sqlite3").exists():
            print(f"   This IS a ChromaDB database directory")
            found_locations.append(loc_path)
        
        # List subdirectories
        subdirs = [d for d in loc_path.iterdir() if d.is_dir()]
        if subdirs:
            print(f"   Subdirectories: {len(subdirs)}")
            for sd in subdirs[:5]:  # Show first 5
                print(f"   - {sd.name}")
        
        print()
    else:
        print(f"❌ NOT FOUND: {loc_path}")

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

if found_locations:
    print(f"\n✅ Found {len(found_locations)} ChromaDB location(s):")
    for loc in found_locations:
        print(f"   {loc}")
    print("\nTo delete the radio2 collection, you need to delete:")
    for loc in found_locations:
        print(f"   rm -rf {loc}")
else:
    print("\n❌ No ChromaDB storage locations found!")
    print("   This is very strange if diagnose_index.py found data.")
    print("\n   The collection might be stored in ChromaDB's default location.")
    print("   Try checking: ~/.chroma")

# Check ChromaDB default location
default_chroma = Path.home() / ".chroma"
if default_chroma.exists():
    print(f"\n⚠️  Found ChromaDB default directory: {default_chroma}")
    print("   This might be where your data is stored!")

print()