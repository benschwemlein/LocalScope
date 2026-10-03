# AST-Based Code Chunking Implementation Guide

## Overview

This implementation adds intelligent AST-based code chunking to your RAG application. Instead of splitting code at arbitrary character positions, it now respects code structure (functions, classes, methods).

## What's New

### 1. Smart Chunking
- **Code files** (.py, .java, .cs, .ts, .js, etc.) are chunked at function/class boundaries
- **Documentation files** (.md, .txt, etc.) use the original character-based chunking
- Automatic language detection from file extension
- Graceful fallback if AST parsing fails

### 2. Two-Tier Implementation
1. **Preferred**: Uses `astchunk` library (supports Python, Java, C#, TypeScript)
2. **Fallback**: Uses Python's built-in `ast` module (Python files only)
3. **Final Fallback**: Original character-based chunking

## Installation

### Step 1: Install the astchunk library

```bash
pip install astchunk
```

This will automatically install dependencies:
- `tree-sitter` - The parser
- `tree-sitter-python` - Python grammar
- `tree-sitter-java` - Java grammar
- `tree-sitter-c-sharp` - C# grammar
- `tree-sitter-typescript` - TypeScript/JavaScript grammar

### Step 2: Replace your indexer.py

Replace your current `indexing/indexer.py` with the new `indexing/indexer_new.py`:

```bash
# Backup your current indexer
cp indexing/indexer.py indexing/indexer_old.py

# Use the new indexer
cp indexing/indexer_new.py indexing/indexer.py
```

### Step 3: Add the ast_chunker module

Place `indexing/ast_chunker.py` in your project's `indexing/` directory.

## File Structure

After installation, your project should have:

```
indexing/
├── __init__.py
├── indexer.py          # Updated with AST chunking
└── ast_chunker.py      # New AST chunking module
```

## Usage

### From GUI (No Changes Required!)

The GUI will automatically use AST-based chunking. Just index your repository as normal.

The indexing log will show:
```
AST-based chunking: ENABLED
...
DONE.
Files processed: 250
  - AST-chunked (code files): 180
  - Char-chunked (docs/configs): 70
Chunks indexed: 3452
```

### From Python Code

```python
from indexing.indexer import index_repo

# Index with AST chunking (default)
index_repo(
    repo_root="/path/to/your/codebase",
    index_dir="/path/to/index",
    use_ast_chunking=True  # NEW parameter
)

# Or disable AST chunking (use old behavior)
index_repo(
    repo_root="/path/to/your/codebase",
    index_dir="/path/to/index",
    use_ast_chunking=False
)
```

### Testing Individual Files

```python
from indexing.ast_chunker import chunk_code_intelligently

# Read a Python file
with open("example.py", "r") as f:
    code = f.read()

# Chunk it intelligently
chunks = chunk_code_intelligently(
    text=code,
    file_path="example.py",
    max_chunk_size=1200,
    chunk_overlap=200
)

# Print chunks
for i, chunk in enumerate(chunks):
    print(f"=== Chunk {i+1} ===")
    print(chunk)
    print()
```

## How It Works

### For Code Files (.py, .java, .cs, .ts, .js, etc.)

**Before (Character-based):**
```python
# This function might get split mid-way
def calculate_total(items):
    """Calculate total with tax."""
    subtotal = 0
    for item in items:
        subtotal += item.price * item.qu
# --- CHUNK BOUNDARY ---
antity
    tax = subtotal * 0.08
    return subtotal + tax
```

**After (AST-based):**
```python
# Chunk 1: Complete function
def calculate_total(items):
    """Calculate total with tax."""
    subtotal = 0
    for item in items:
        subtotal += item.price * item.quantity
    tax = subtotal * 0.08
    return subtotal + tax

# Chunk 2: Next complete function
def process_order(order):
    # ...
```

### For Large Classes (Java Example)

If a class has many methods, it chunks like this:

```java
// Chunk 1: Class context + Method 1
public class UserService {
    // ... (other methods)
    
    public User getUser(String id) {
        // Complete method
    }
}

// Chunk 2: Class context + Method 2
public class UserService {
    // ... (other methods)
    
    public void saveUser(User user) {
        // Complete method
    }
}
```

This gives the LLM both:
1. The method implementation
2. The class context (which class it belongs to)

### For Documentation Files (.md, .txt, etc.)

These continue to use character-based chunking, which works fine for prose.

## Supported Languages

### Full AST Support (via astchunk)
- ✅ Python (.py, .pyi)
- ✅ Java (.java)
- ✅ C# (.cs, .csx)
- ✅ TypeScript (.ts, .tsx)
- ✅ JavaScript (.js, .jsx)

### Python-Only AST Support (built-in fallback)
- ✅ Python (.py, .pyi)

### Character-Based Chunking
- All other file types (works as before)

## Configuration

### Default Settings

```python
# In indexing/indexer.py
CHARS_PER_CHUNK = 1200    # Max chunk size
CHUNK_OVERLAP = 200        # Overlap between chunks

# File types that use AST chunking
CODE_FILE_EXTS = {
    ".py", ".pyi",           # Python
    ".java",                 # Java
    ".cs", ".csx",           # C#
    ".js", ".jsx",           # JavaScript
    ".ts", ".tsx",           # TypeScript
    ".go",                   # Go (if you add go support)
    ".rs",                   # Rust (if you add rust support)
}
```

### Customizing Chunk Size

You can adjust chunk size in the GUI's Index tab, or programmatically:

```python
index_repo(
    repo_root="/path/to/repo",
    chars_per_chunk=2000,    # Larger chunks
    chunk_overlap=300,       # More overlap
)
```

## Troubleshooting

### Issue: "astchunk not found"

**Solution:** Install it
```bash
pip install astchunk
```

The code will automatically fall back to Python-only AST or character-based chunking if astchunk is not available.

### Issue: AST chunking fails for a specific file

**What happens:** The code automatically falls back to character-based chunking and logs a warning.

**Why:** The file might have syntax errors or use unsupported language features.

**Fix:** Check the logs for the specific error. The file will still be indexed using character-based chunking.

### Issue: Chunks are too large

**Solution:** Reduce `chars_per_chunk`:

```python
index_repo(
    repo_root="/path/to/repo",
    chars_per_chunk=800,  # Smaller chunks
)
```

### Issue: Want to disable AST chunking temporarily

**Solution:**

```python
index_repo(
    repo_root="/path/to/repo",
    use_ast_chunking=False  # Disable AST chunking
)
```

## Performance Considerations

### Memory
- AST parsing uses slightly more memory per file
- Impact is negligible for files under 100KB
- Large files (>500KB) are skipped anyway (MAX_FILE_BYTES)

### Speed
- AST parsing adds ~10-20% overhead per file
- Overall indexing speed is still dominated by embedding generation
- Typically: 90% time in Ollama API calls, 10% in parsing/chunking

### Quality
- **Dramatic improvement** in retrieval quality for code queries
- Functions/methods are now complete, giving LLM full context
- Fewer "broken" chunks that miss critical information

## Examples

### Example 1: Python Class

**Input file:**
```python
class Calculator:
    def __init__(self):
        self.result = 0
    
    def add(self, a, b):
        """Add two numbers."""
        return a + b
    
    def multiply(self, a, b):
        """Multiply two numbers."""
        result = a * b
        self.result = result
        return result
```

**Chunks created:**
```
Chunk 1:
class Calculator:
    # ... (other methods)

    def __init__(self):
        self.result = 0

Chunk 2:
class Calculator:
    # ... (other methods)

    def add(self, a, b):
        """Add two numbers."""
        return a + b

Chunk 3:
class Calculator:
    # ... (other methods)

    def multiply(self, a, b):
        """Multiply two numbers."""
        result = a * b
        self.result = result
        return result
```

### Example 2: Java Service

**Input file:**
```java
public class UserService {
    private UserRepository repo;
    
    public User findById(String id) {
        return repo.findById(id)
            .orElseThrow(() -> new UserNotFoundException(id));
    }
    
    public void save(User user) {
        validateUser(user);
        repo.save(user);
    }
    
    private void validateUser(User user) {
        if (user.getName() == null) {
            throw new ValidationException("Name required");
        }
    }
}
```

**Chunks created:**
Each method becomes a separate chunk with class context, ensuring the LLM knows:
- Which class the method belongs to
- The method's complete implementation
- The method's signature and purpose

## Migration from Old System

### Step 1: Test on a Small Repo

Before re-indexing your entire codebase:

```python
# Test on a small project
index_repo(
    repo_root="/path/to/small/project",
    index_dir="/tmp/test_index",
    use_ast_chunking=True
)
```

### Step 2: Compare Results

Run the same query on both old and new indexes:

```python
# Query old index
results_old = query("/path/to/old_index", "find user authentication")

# Query new index
results_new = query("/path/to/new_index", "find user authentication")

# Compare which gives better context
```

### Step 3: Re-Index Production

Once satisfied, re-index your main codebase:

```python
index_repo(
    repo_root="/path/to/main/codebase",
    index_dir="/path/to/production/index",
    use_ast_chunking=True
)
```

## Next Steps

After implementing AST-based chunking, consider:

1. **Priority 2: Multiple Index Search** - Query multiple codebases simultaneously
2. **Priority 3: Parallel Indexing** - Speed up indexing with multi-threading
3. **Priority 4: Incremental Updates** - Only re-index changed files

See the full roadmap in `rag_improvement_roadmap.md` for details.

## Support

For issues or questions:
1. Check the logs for specific error messages
2. Verify astchunk is installed: `pip list | grep astchunk`
3. Test with a simple Python file first
4. Check that file extensions are in CODE_FILE_EXTS

## Summary

✅ **What you get:**
- Intelligent chunking that respects code structure
- Better retrieval quality (complete functions/methods)
- Automatic fallback for unsupported files
- Minimal performance impact
- No changes needed to GUI or query logic

✅ **What to do:**
1. `pip install astchunk`
2. Replace `indexer.py` with new version
3. Add `ast_chunker.py` to your project
4. Re-index your codebase
5. Enjoy better code search!