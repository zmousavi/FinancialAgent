"""
chunk_utils.py
Text chunking utilities for SEC filings to prepare documents for RAG.

This module provides functions to:
1. Split large documents into manageable chunks
2. Preserve semantic boundaries (sections, paragraphs)
3. Add metadata for better retrieval
4. Handle overlaps between chunks for context preservation
"""

import os
import re
import yaml
from typing import List, Dict, Tuple, Optional


def load_cfg(path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def read_text_file(file_path: str) -> str:
    """Read text file with UTF-8 encoding."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def detect_sections(text: str) -> List[Dict[str, any]]:
    """
    Detect major sections in SEC filing text based on common patterns.
    
    Returns list of dictionaries with section info:
    - title: section title
    - start_pos: character position where section starts
    - content: section content
    """
    sections = []
    
    # Common SEC filing section patterns
    section_patterns = [
        r"^PART\s+I+\b",
        r"^PART\s+\d+\b",
        r"^Item\s+\d+[A-Z]?\.",
        r"^ITEM\s+\d+[A-Z]?\.",
        r"^\d+\.\s+[A-Z][A-Za-z\s]+$",
        r"^[A-Z\s]{10,}$",  # All caps headers
        r"^BUSINESS$",
        r"^RISK\s+FACTORS$",
        r"^FINANCIAL\s+STATEMENTS",
        r"^MANAGEMENT'S\s+DISCUSSION",
    ]
    
    lines = text.split('\n')
    current_pos = 0
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        # Skip empty lines
        if not line_stripped:
            current_pos += len(line) + 1
            continue
            
        # Check if line matches section pattern
        is_section = False
        for pattern in section_patterns:
            if re.match(pattern, line_stripped, re.IGNORECASE):
                is_section = True
                break
        
        if is_section:
            # Find content until next section or end
            content_lines = []
            j = i + 1
            next_section_pos = len(text)
            
            while j < len(lines):
                next_line = lines[j].strip()
                if next_line:
                    # Check if this is start of next section
                    is_next_section = False
                    for pattern in section_patterns:
                        if re.match(pattern, next_line, re.IGNORECASE):
                            is_next_section = True
                            break
                    
                    if is_next_section:
                        next_section_pos = sum(len(lines[k]) + 1 for k in range(j))
                        break
                
                content_lines.append(lines[j])
                j += 1
            
            section_content = '\n'.join(content_lines)
            
            sections.append({
                'title': line_stripped,
                'start_pos': current_pos,
                'content': section_content,
                'line_number': i + 1
            })
        
        current_pos += len(line) + 1
    
    return sections


def chunk_by_sentences(text: str, max_chunk_size: int = 1000, overlap_size: int = 100) -> List[Dict[str, any]]:
    """
    Split text into chunks based on sentence boundaries.
    
    Args:
        text: Input text to chunk
        max_chunk_size: Maximum characters per chunk
        overlap_size: Characters to overlap between chunks
        
    Returns:
        List of chunk dictionaries with text, start_pos, end_pos, chunk_id
    """
    chunks = []
    
    # Simple sentence splitting on periods, exclamation marks, question marks
    sentences = re.split(r'[.!?]+\s+', text)
    
    current_chunk = ""
    chunk_start_pos = 0
    chunk_id = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Check if adding this sentence would exceed max size
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            # Save current chunk
            chunks.append({
                'text': current_chunk.strip(),
                'start_pos': chunk_start_pos,
                'end_pos': chunk_start_pos + len(current_chunk),
                'chunk_id': chunk_id,
                'word_count': len(current_chunk.split())
            })
            
            # Start new chunk with overlap
            if overlap_size > 0 and len(current_chunk) > overlap_size:
                overlap_text = current_chunk[-overlap_size:]
                current_chunk = overlap_text + " " + sentence
                chunk_start_pos = chunk_start_pos + len(current_chunk) - overlap_size
            else:
                current_chunk = sentence
                chunk_start_pos = chunk_start_pos + len(current_chunk)
            
            chunk_id += 1
        else:
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    # Add final chunk if there's remaining content
    if current_chunk.strip():
        chunks.append({
            'text': current_chunk.strip(),
            'start_pos': chunk_start_pos,
            'end_pos': chunk_start_pos + len(current_chunk),
            'chunk_id': chunk_id,
            'word_count': len(current_chunk.split())
        })
    
    return chunks


def chunk_by_paragraphs(text: str, max_chunk_size: int = 1500, overlap_size: int = 150) -> List[Dict[str, any]]:
    """
    Split text into chunks based on paragraph boundaries.
    
    Args:
        text: Input text to chunk
        max_chunk_size: Maximum characters per chunk
        overlap_size: Characters to overlap between chunks
        
    Returns:
        List of chunk dictionaries with text, start_pos, end_pos, chunk_id
    """
    chunks = []
    
    # Split by double newlines (paragraph breaks)
    paragraphs = re.split(r'\n\s*\n', text)
    
    current_chunk = ""
    chunk_start_pos = 0
    chunk_id = 0
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # Check if adding this paragraph would exceed max size
        if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
            # Save current chunk
            chunks.append({
                'text': current_chunk.strip(),
                'start_pos': chunk_start_pos,
                'end_pos': chunk_start_pos + len(current_chunk),
                'chunk_id': chunk_id,
                'word_count': len(current_chunk.split()),
                'paragraph_count': len(re.split(r'\n\s*\n', current_chunk))
            })
            
            # Start new chunk with overlap
            if overlap_size > 0 and len(current_chunk) > overlap_size:
                overlap_text = current_chunk[-overlap_size:]
                current_chunk = overlap_text + "\n\n" + paragraph
                chunk_start_pos = chunk_start_pos + len(current_chunk) - overlap_size
            else:
                current_chunk = paragraph
                chunk_start_pos = chunk_start_pos + len(current_chunk)
            
            chunk_id += 1
        else:
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
    
    # Add final chunk if there's remaining content
    if current_chunk.strip():
        chunks.append({
            'text': current_chunk.strip(),
            'start_pos': chunk_start_pos,
            'end_pos': chunk_start_pos + len(current_chunk),
            'chunk_id': chunk_id,
            'word_count': len(current_chunk.split()),
            'paragraph_count': len(re.split(r'\n\s*\n', current_chunk))
        })
    
    return chunks


def chunk_by_fixed_size(text: str, chunk_size: int = 1000, overlap_size: int = 100) -> List[Dict[str, any]]:
    """
    Split text into fixed-size chunks with overlap.
    
    Args:
        text: Input text to chunk
        chunk_size: Characters per chunk
        overlap_size: Characters to overlap between chunks
        
    Returns:
        List of chunk dictionaries with text, start_pos, end_pos, chunk_id
    """
    chunks = []
    chunk_id = 0
    start_pos = 0
    
    while start_pos < len(text):
        # Calculate end position
        end_pos = min(start_pos + chunk_size, len(text))
        
        # Extract chunk text
        chunk_text = text[start_pos:end_pos]
        
        # Try to break at word boundary if not at end of text
        if end_pos < len(text):
            # Look for last space within the chunk
            last_space = chunk_text.rfind(' ')
            if last_space > chunk_size * 0.7:  # Only if space is reasonably close to end
                end_pos = start_pos + last_space
                chunk_text = text[start_pos:end_pos]
        
        chunks.append({
            'text': chunk_text.strip(),
            'start_pos': start_pos,
            'end_pos': end_pos,
            'chunk_id': chunk_id,
            'word_count': len(chunk_text.split())
        })
        
        # Move start position for next chunk with overlap
        start_pos = max(start_pos + chunk_size - overlap_size, end_pos)
        chunk_id += 1
    
    return chunks


def add_document_metadata(chunks: List[Dict[str, any]], 
                         document_name: str, 
                         company_ticker: str = None,
                         filing_type: str = "10-K",
                         filing_year: str = None) -> List[Dict[str, any]]:
    """
    Add metadata to chunks for better retrieval and context.
    
    Args:
        chunks: List of chunk dictionaries
        document_name: Name of the source document
        company_ticker: Stock ticker symbol
        filing_type: Type of SEC filing
        filing_year: Year of the filing
        
    Returns:
        Enhanced chunks with metadata
    """
    enhanced_chunks = []
    
    for chunk in chunks:
        enhanced_chunk = chunk.copy()
        enhanced_chunk.update({
            'document_name': document_name,
            'company_ticker': company_ticker,
            'filing_type': filing_type,
            'filing_year': filing_year,
            'chunk_length': len(chunk['text']),
        })
        enhanced_chunks.append(enhanced_chunk)
    
    return enhanced_chunks


def process_filing_to_chunks(file_path: str, 
                           chunking_method: str = "sections",
                           max_chunk_size: int = 2500,
                           overlap_size: int = 250) -> List[Dict[str, any]]:
    """
    Main function to process a filing text file into chunks.
    
    Args:
        file_path: Path to the clean text file
        chunking_method: "sections", "paragraphs", or "fixed" 
                        (sections recommended based on analysis)
        max_chunk_size: Maximum characters per chunk
        overlap_size: Characters to overlap between chunks
        
    Returns:
        List of processed chunks with metadata
    """
    # Read the text file
    text = read_text_file(file_path)
    
    # Extract metadata from filename
    filename = file_path.split('/')[-1]
    document_name = filename.replace('.clean.txt', '')
    
    # Try to extract ticker from filename (e.g., "AAPL_10-K.clean.txt")
    ticker_match = re.match(r'^([A-Z]+)_', filename)
    company_ticker = ticker_match.group(1) if ticker_match else None
    
    # Choose chunking method based on analysis recommendations
    if chunking_method == "sections":
        # Primary method: Section-based chunking with paragraph fallback
        sections = detect_sections(text)
        chunks = []
        for section in sections:
            # If section is small enough, keep as single chunk
            if len(section['content']) <= max_chunk_size:
                chunks.append({
                    'text': section['content'].strip(),
                    'start_pos': section['start_pos'],
                    'end_pos': section['start_pos'] + len(section['content']),
                    'chunk_id': len(chunks),
                    'word_count': len(section['content'].split()),
                    'section_title': section['title'],
                    'section_line': section['line_number'],
                    'is_complete_section': True
                })
            else:
                # Large section: split using paragraph chunking
                section_chunks = chunk_by_paragraphs(section['content'], max_chunk_size, overlap_size)
                
                # Check if any chunks are still too large (paragraph chunking failed)
                oversized_chunks = [c for c in section_chunks if len(c['text']) > max_chunk_size]
                
                if oversized_chunks:
                    # Paragraph chunking failed, use fixed-size chunking as final fallback
                    print(f"Warning: Section '{section['title']}' has {len(oversized_chunks)} oversized chunks after paragraph split, using fixed-size fallback")
                    section_chunks = chunk_by_fixed_size(section['content'], max_chunk_size, overlap_size)
                    for chunk in section_chunks:
                        chunk['section_title'] = section['title']
                        chunk['section_line'] = section['line_number']
                        chunk['is_complete_section'] = False
                        chunk['chunking_method'] = 'fixed_fallback'
                        chunk['chunk_id'] = len(chunks) + chunk['chunk_id']
                else:
                    # Paragraph chunking worked fine
                    for chunk in section_chunks:
                        chunk['section_title'] = section['title']
                        chunk['section_line'] = section['line_number']
                        chunk['is_complete_section'] = False
                        chunk['chunking_method'] = 'paragraph'
                        chunk['chunk_id'] = len(chunks) + chunk['chunk_id']
                
                chunks.extend(section_chunks)
    elif chunking_method == "paragraphs":
        # Secondary method: Paragraph-based chunking
        chunks = chunk_by_paragraphs(text, max_chunk_size, overlap_size)
    elif chunking_method == "fixed":
        # Tertiary method: Fixed-size chunking (for table-heavy content)
        chunks = chunk_by_fixed_size(text, max_chunk_size, overlap_size)
    else:
        raise ValueError(f"Unknown chunking method: {chunking_method}. Use 'sections', 'paragraphs', or 'fixed'")
    
    # Add document metadata
    enhanced_chunks = add_document_metadata(
        chunks, 
        document_name, 
        company_ticker,
        filing_type="10-K"
    )
    
    return enhanced_chunks


def save_chunks_to_file(chunks: List[Dict[str, any]], output_path: str):
    """
    Save chunks to a text file for inspection.
    
    Args:
        chunks: List of chunk dictionaries
        output_path: Path to save the chunks
    """
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Total chunks: {len(chunks)}\n")
        f.write("=" * 50 + "\n\n")
        
        for i, chunk in enumerate(chunks):
            f.write(f"CHUNK {i + 1} (ID: {chunk['chunk_id']})\n")
            f.write(f"Document: {chunk.get('document_name', 'Unknown')}\n")
            f.write(f"Company: {chunk.get('company_ticker', 'Unknown')}\n")
            f.write(f"Word count: {chunk.get('word_count', 0)}\n")
            f.write(f"Character count: {chunk.get('chunk_length', 0)}\n")
            if 'section_title' in chunk:
                f.write(f"Section: {chunk['section_title']}\n")
                if chunk.get('is_complete_section', False):
                    f.write(f"Status: Complete section\n")
                else:
                    f.write(f"Status: Partial section (large section split)\n")
            f.write("-" * 30 + "\n")
            f.write(chunk['text'])
            f.write("\n\n" + "=" * 50 + "\n\n")


if __name__ == "__main__":
    """
    Example usage of the chunking utilities.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Chunk SEC filing text files")
    parser.add_argument("--input", required=True, help="Input clean text file")
    parser.add_argument("--output", help="Output chunks file (auto-generated if not provided)")
    parser.add_argument("--method", default="sections", 
                       choices=["sections", "paragraphs", "fixed"],
                       help="Chunking method (sections recommended based on analysis)")
    parser.add_argument("--chunk-size", type=int, default=2500, 
                       help="Maximum chunk size in characters")
    parser.add_argument("--overlap", type=int, default=250,
                       help="Overlap size in characters")
    
    args = parser.parse_args()
    
    # Create chunks directory if it doesn't exist
    chunks_dir = "chunks"
    os.makedirs(chunks_dir, exist_ok=True)
    
    # Generate output filename if not provided
    if args.output:
        output_path = args.output
    else:
        # Extract base filename from input path
        input_filename = os.path.basename(args.input)
        base_name = input_filename.replace('.clean.txt', '').replace('.txt', '')
        output_filename = f"{base_name}_chunks_{args.method}.txt"
        output_path = os.path.join(chunks_dir, output_filename)
    
    # Process the file
    chunks = process_filing_to_chunks(
        args.input, 
        args.method, 
        args.chunk_size, 
        args.overlap
    )
    
    # Save results
    save_chunks_to_file(chunks, output_path)
    
    print(f"Processed {len(chunks)} chunks from {args.input}")
    print(f"Results saved to {output_path}")