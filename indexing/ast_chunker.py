"""
AST-based code chunking for intelligent code segmentation.

This module provides smart chunking that respects code structure (functions, classes, methods)
rather than splitting at arbitrary character positions.

Supports two modes:
1. Full AST chunking using astchunk library (preferred - supports multiple languages)
2. Fallback Python-only AST chunking using Python's built-in ast module
"""

import os
import ast
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CodeChunk:
    """Represents a chunk of code with metadata."""
    content: str
    start_line: int
    end_line: int
    chunk_type: str  # 'function', 'class', 'method', 'module', 'fragment'
    name: Optional[str] = None
    parent_context: Optional[str] = None


class ASTChunker:
    """
    Intelligent code chunker that uses AST to split code at logical boundaries.
    """
    
    def __init__(self, max_chunk_size: int = 1200, chunk_overlap: int = 200):
        """
        Initialize the AST chunker.
        
        Args:
            max_chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks (for context)
        """
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self._astchunk_available = self._check_astchunk()
    
    def _check_astchunk(self) -> bool:
        """Check if astchunk library is available."""
        try:
            import astchunk
            return True
        except ImportError:
            return False
    
    def chunk_code(self, code: str, file_path: str, language: Optional[str] = None) -> List[str]:
        """
        Chunk code intelligently based on its structure.
        
        Args:
            code: Source code to chunk
            file_path: Path to the file (used for language detection)
            language: Explicit language override
        
        Returns:
            List of code chunks as strings
        """
        if not code or not code.strip():
            return []
        
        # Detect language from file extension if not provided
        if language is None:
            language = self._detect_language(file_path)
        
        # Try astchunk first if available
        if self._astchunk_available and language in ['python', 'java', 'csharp', 'typescript', 'javascript']:
            try:
                return self._chunk_with_astchunk(code, file_path, language)
            except Exception as e:
                print(f"astchunk failed for {file_path}: {e}, falling back...")
        
        # Fallback to Python AST for Python files
        if language == 'python':
            try:
                return self._chunk_python_ast(code)
            except Exception as e:
                print(f"Python AST chunking failed for {file_path}: {e}, using character-based...")
        
        # Final fallback: character-based chunking
        return self._chunk_by_characters(code)
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        ext = os.path.splitext(file_path)[1].lower()
        
        ext_to_lang = {
            '.py': 'python',
            '.pyi': 'python',
            '.java': 'java',
            '.cs': 'csharp',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.go': 'go',
            '.rs': 'rust',
            '.cpp': 'cpp',
            '.cc': 'cpp',
            '.cxx': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
        }
        
        return ext_to_lang.get(ext, 'unknown')
    
    def _chunk_with_astchunk(self, code: str, file_path: str, language: str) -> List[str]:
        """Chunk using the astchunk library."""
        from astchunk import ASTChunkBuilder
        
        # Map our language names to astchunk's expected names
        lang_map = {
            'javascript': 'typescript',  # astchunk uses typescript for JS too
        }
        language = lang_map.get(language, language)
        
        chunk_builder = ASTChunkBuilder(
            max_chunk_size=self.max_chunk_size,
            language=language,
            metadata_template="default"
        )
        
        chunks = chunk_builder.chunkify(
            code,
            repo_level_metadata={"filepath": file_path},
            chunk_expansion=True,  # Adds context headers
            chunk_overlap=1 if self.chunk_overlap > 0 else 0
        )
        
        # Extract content and ENFORCE SIZE LIMIT
        result_chunks = []
        for chunk in chunks:
            content = chunk['content']
            # CRITICAL: astchunk may create oversized chunks with context headers
            # Split any chunk that exceeds our limit
            if len(content) > self.max_chunk_size:
                # Split this oversized chunk
                result_chunks.extend(self._chunk_by_characters(content))
            else:
                result_chunks.append(content)
        
        return result_chunks
    
    def _chunk_python_ast(self, code: str) -> List[str]:
        """
        Chunk Python code using Python's built-in AST module.
        Extracts top-level functions and classes as chunks.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # If parsing fails, fall back to character chunking
            return self._chunk_by_characters(code)
        
        code_lines = code.splitlines(keepends=True)
        chunks = []
        
        # Extract top-level nodes (functions, classes)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                chunk_content = self._extract_node_content(node, code_lines)
                
                # STRICT SIZE ENFORCEMENT: If chunk is too large, split it
                if len(chunk_content) > self.max_chunk_size:
                    # For classes, try to split by methods
                    if isinstance(node, ast.ClassDef):
                        sub_chunks = self._chunk_class(node, code_lines)
                        # Check if sub-chunks are still too large
                        for sub_chunk in sub_chunks:
                            if len(sub_chunk) > self.max_chunk_size:
                                chunks.extend(self._chunk_by_characters(sub_chunk))
                            else:
                                chunks.append(sub_chunk)
                    else:
                        # For large functions, use character chunking
                        chunks.extend(self._chunk_by_characters(chunk_content))
                else:
                    chunks.append(chunk_content)
            elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                # Keep imports with the next meaningful chunk
                continue
            else:
                # Other top-level code (assignments, etc.)
                chunk_content = self._extract_node_content(node, code_lines)
                if chunk_content.strip():
                    # Enforce size limit on all chunks
                    if len(chunk_content) > self.max_chunk_size:
                        chunks.extend(self._chunk_by_characters(chunk_content))
                    else:
                        chunks.append(chunk_content)
        
        # If no chunks were created (e.g., file is just imports), fall back
        if not chunks:
            return self._chunk_by_characters(code)
        
        # FINAL SAFETY CHECK: Ensure no chunk exceeds max size
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.max_chunk_size:
                final_chunks.extend(self._chunk_by_characters(chunk))
            else:
                final_chunks.append(chunk)
        
        return final_chunks
    
    def _chunk_class(self, class_node: ast.ClassDef, code_lines: List[str]) -> List[str]:
        """
        Chunk a class by extracting each method separately.
        Enforces strict size limits.
        """
        chunks = []
        
        # Get class definition header (class name, inheritance, etc.)
        class_header_end = class_node.body[0].lineno - 1 if class_node.body else class_node.lineno
        class_header = ''.join(code_lines[class_node.lineno - 1:class_header_end])
        
        # Extract each method
        for node in class_node.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_content = self._extract_node_content(node, code_lines)
                
                # Try to add class context to method
                full_chunk = f"{class_header}\n    # ... (other methods)\n\n{method_content}"
                
                if len(full_chunk) > self.max_chunk_size:
                    # If method + context is too large, try just the method
                    if len(method_content) > self.max_chunk_size:
                        # If method alone is too large, split it with character chunking
                        chunks.extend(self._chunk_by_characters(method_content))
                    else:
                        chunks.append(method_content)
                else:
                    chunks.append(full_chunk)
            else:
                # Class variables, etc.
                content = self._extract_node_content(node, code_lines)
                if content.strip():
                    chunk = f"{class_header}\n{content}"
                    if len(chunk) > self.max_chunk_size:
                        chunks.extend(self._chunk_by_characters(content))
                    else:
                        chunks.append(chunk)
        
        return chunks
    
    def _extract_node_content(self, node: ast.AST, code_lines: List[str]) -> str:
        """Extract the source code for an AST node."""
        if not hasattr(node, 'lineno') or not hasattr(node, 'end_lineno'):
            return ""
        
        start_line = node.lineno - 1  # Convert to 0-indexed
        end_line = node.end_lineno  # Already inclusive in AST
        
        if start_line < 0 or end_line > len(code_lines):
            return ""
        
        return ''.join(code_lines[start_line:end_line])
    
    def _chunk_by_characters(self, text: str) -> List[str]:
        """
        Fallback: Simple character-based chunking with overlap.
        This is the original chunking strategy.
        """
        chunks = []
        start = 0
        n = len(text)
        
        while start < n:
            end = min(n, start + self.max_chunk_size)
            chunks.append(text[start:end])
            
            if end == n:
                break
            
            start = end - self.chunk_overlap
        
        return chunks


def chunk_code_intelligently(
    text: str,
    file_path: str,
    max_chunk_size: int = 1200,
    chunk_overlap: int = 200
) -> List[str]:
    """
    Convenience function for chunking code.
    
    This is a drop-in replacement for the original chunk_text() function
    but with AST awareness.
    
    Args:
        text: Source code to chunk
        file_path: Path to the file (for language detection)
        max_chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of code chunks
    """
    chunker = ASTChunker(max_chunk_size=max_chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk_code(text, file_path)


# For backward compatibility
def chunk_text(text: str, max_len: int, overlap: int) -> List[str]:
    """
    Original character-based chunking (kept for non-code files).
    """
    chunks = []
    start = 0
    n = len(text)
    
    while start < n:
        end = min(n, start + max_len)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
    
    return chunks