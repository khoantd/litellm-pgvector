from typing import List, Literal
import re
import logging

logger = logging.getLogger(__name__)


def chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    splitter: Literal["tokens", "chars", "lines", "paragraphs"] = "chars",
    normalize_whitespace: bool = True,
    lowercase: bool = False,
    max_chunks: int = None
) -> List[str]:
    """
    Chunk text into smaller pieces based on the specified splitter strategy.
    
    Args:
        text: The text to chunk
        chunk_size: Target size for each chunk (in chars/tokens/lines/paragraphs depending on splitter)
        chunk_overlap: Number of overlapping units between chunks
        splitter: Strategy to use for chunking ("tokens", "chars", "lines", "paragraphs")
        normalize_whitespace: Whether to normalize whitespace before chunking
        lowercase: Whether to lowercase the text (applied after chunking)
        max_chunks: Maximum number of chunks to return (None for no limit)
        
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []
    
    # Normalize whitespace if requested
    if normalize_whitespace:
        text = re.sub(r'\s+', ' ', text.strip())
    
    # Apply splitter strategy
    if splitter == "chars":
        chunks = chunk_by_chars(text, chunk_size, chunk_overlap)
    elif splitter == "tokens":
        chunks = chunk_by_tokens(text, chunk_size, chunk_overlap)
    elif splitter == "lines":
        chunks = chunk_by_lines(text, chunk_size, chunk_overlap)
    elif splitter == "paragraphs":
        chunks = chunk_by_paragraphs(text, chunk_size, chunk_overlap)
    else:
        raise ValueError(f"Unknown splitter: {splitter}. Must be one of: tokens, chars, lines, paragraphs")
    
    # Apply lowercase if requested
    if lowercase:
        chunks = [chunk.lower() for chunk in chunks]
    
    # Apply max_chunks limit
    if max_chunks is not None and max_chunks > 0:
        chunks = chunks[:max_chunks]
    
    # Filter out empty chunks
    chunks = [chunk for chunk in chunks if chunk.strip()]
    
    return chunks


def chunk_by_chars(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Chunk text by character count"""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    
    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size - 1
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        if not chunk.strip():
            break
        
        chunks.append(chunk)
        
        # Move start position with overlap
        start += chunk_size - chunk_overlap
        
        # Break if we've covered all text
        if end >= len(text):
            break
    
    return chunks


def chunk_by_tokens(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Chunk text by approximate token count.
    Uses a simple heuristic: ~4 characters per token.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    
    # Approximate token size (4 chars per token is a reasonable heuristic)
    char_size = chunk_size * 4
    char_overlap = chunk_overlap * 4
    
    return chunk_by_chars(text, char_size, char_overlap)


def chunk_by_lines(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Chunk text by line count"""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    
    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size - 1
    
    lines = text.split('\n')
    chunks = []
    start = 0
    
    while start < len(lines):
        end = start + chunk_size
        chunk_lines = lines[start:end]
        chunk = '\n'.join(chunk_lines)
        
        if not chunk.strip():
            break
        
        chunks.append(chunk)
        
        # Move start position with overlap
        start += chunk_size - chunk_overlap
        
        # Break if we've covered all lines
        if end >= len(lines):
            break
    
    return chunks


def chunk_by_paragraphs(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Chunk text by paragraph count"""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    
    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size - 1
    
    # Split by double newlines (paragraph separator)
    paragraphs = re.split(r'\n\s*\n', text)
    # Filter out empty paragraphs
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    chunks = []
    start = 0
    
    while start < len(paragraphs):
        end = start + chunk_size
        chunk_paragraphs = paragraphs[start:end]
        chunk = '\n\n'.join(chunk_paragraphs)
        
        if not chunk.strip():
            break
        
        chunks.append(chunk)
        
        # Move start position with overlap
        start += chunk_size - chunk_overlap
        
        # Break if we've covered all paragraphs
        if end >= len(paragraphs):
            break
    
    return chunks

