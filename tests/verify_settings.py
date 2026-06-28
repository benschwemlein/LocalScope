#!/usr/bin/env python3
"""
Verify what settings code_llm is actually using for indexing.
"""

import json
from pathlib import Path

# Path to config file (adjust if needed)
CONFIG_PATH = Path(__file__).parent.parent / "config.json"

print("=" * 80)
print("CODE_LLM SETTINGS VERIFICATION")
print("=" * 80)
print()

if not CONFIG_PATH.exists():
    print(f"❌ Config file not found at: {CONFIG_PATH}")
    print("   Please run this script from the code_llm/tests directory")
    exit(1)

print(f"📄 Config file: {CONFIG_PATH}")
print()

# Load config
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

print("🔧 SETTINGS TAB (used for embeddings and queries):")
print("-" * 80)
settings = config.get('settings_tab', {})
print(f"  Ollama URL:      {settings.get('ollama_url', 'NOT SET')}")
print(f"  Embedding model: {settings.get('embed_model', 'NOT SET')}")
print(f"  Chat model:      {settings.get('chat_model', 'NOT SET')}")
print()

print("📂 INDEX TAB (used for indexing):")
print("-" * 80)
index_settings = config.get('index_tab', {})
print(f"  Repo root:       {index_settings.get('repo_root', 'NOT SET')}")
print(f"  Index dir base:  {index_settings.get('index_dir_base', 'NOT SET')}")
print(f"  Collection name: {index_settings.get('collection_name', 'NOT SET')}")
print(f"  Chars per chunk: {index_settings.get('chars_per_chunk', 'NOT SET')}")
print(f"  Chunk overlap:   {index_settings.get('chunk_overlap', 'NOT SET')}")
print(f"  Max file bytes:  {index_settings.get('max_file_bytes', 'NOT SET')}")
print()

print("📝 FILE TYPES (what gets indexed):")
print("-" * 80)
filetypes = index_settings.get('filetypes', {})
important_types = ['.md', '.txt', '.rst', '.adoc', '.py', '.js', '.java']
for ext in important_types:
    enabled = filetypes.get(ext, False)
    status = "✅ ENABLED" if enabled else "❌ DISABLED"
    print(f"  {ext:8} {status}")
print()

# Count total enabled types
enabled_count = sum(1 for v in filetypes.values() if v)
print(f"  Total file types enabled: {enabled_count}")
print()

print("🔍 QUERY TAB (used for queries):")
print("-" * 80)
query_settings = config.get('query_tab', {})
print(f"  Index dir:       {query_settings.get('index_dir', 'NOT SET')}")
print(f"  Repo root:       {query_settings.get('repo_root', 'NOT SET')}")
print(f"  Top K:           {query_settings.get('top_k', 'NOT SET')}")
print(f"  Max chars:       {query_settings.get('max_chars', 'NOT SET')}")
print()

print("=" * 80)
print("IMPORTANT NOTES:")
print("=" * 80)
print("""
1. The embedding model in SETTINGS TAB must match what was used to create the index
2. If you changed the embedding model, you MUST re-index
3. These settings are read when:
   - The app starts
   - You click "Start Indexing" or "Re-index"
   - You run a query

To fix the dimension mismatch error:
1. Delete the old index: rm -rf /Users/ben.schwemlein/dev/indexes/radio2
2. Go to Index tab in code_llm
3. Click "Start Indexing" to rebuild with current settings (nomic-embed-text)
""")
print()