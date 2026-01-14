#!/usr/bin/env python3
"""
04_create_chunks.py
Create semantic chunks from cleaned SEC filing text for RAG pipeline.

This script processes cleaned text files and creates chunks optimized for 
semantic search and retrieval. It's designed to run after text cleaning 
and before embedding creation.

Pipeline position: 01_download -> 02_clean -> 04_create_chunks -> 05_embeddings -> ...
"""

import os
import json
import yaml
import argparse
from typing import List, Dict, Any
from pathlib import Path

# Import chunking functions from our utility module
from chunk_utils import (
    process_filing_to_chunks,
    save_chunks_to_file,
    load_cfg,
    read_text_file
)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    return load_cfg(config_path)


def save_chunks_as_json(chunks: List[Dict[str, Any]], output_path: str):
    """
    Save chunks as JSON for embedding pipeline.
    
    Args:
        chunks: List of chunk dictionaries
        output_path: Path to save JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)


def process_single_file(input_path: str, 
                       output_dir: str,
                       chunking_method: str = "sections",
                       max_chunk_size: int = 2500,
                       overlap_size: int = 250,
                       save_text: bool = True) -> int:
    """
    Process a single clean text file into chunks.
    
    Args:
        input_path: Path to clean text file
        output_dir: Directory to save chunks
        chunking_method: Method to use for chunking
        max_chunk_size: Maximum characters per chunk
        overlap_size: Characters to overlap between chunks
        save_text: Whether to save human-readable text version
        
    Returns:
        Number of chunks created
    """
    # Extract filename without extension
    filename = Path(input_path).stem
    base_name = filename.replace('.clean', '')
    
    # Create output paths
    json_output_path = os.path.join(output_dir, f"{base_name}_chunks.json")
    if save_text:
        text_output_path = os.path.join(output_dir, f"{base_name}_chunks.txt")
    
    print(f"Processing {filename}...")
    
    try:
        # Process file to chunks
        chunks = process_filing_to_chunks(
            input_path,
            chunking_method=chunking_method,
            max_chunk_size=max_chunk_size,
            overlap_size=overlap_size
        )
        
        # Save as JSON (for embedding pipeline)
        save_chunks_as_json(chunks, json_output_path)
        
        # Save as text (for human inspection)
        if save_text:
            save_chunks_to_file(chunks, text_output_path)
        
        print(f"  ✓ Created {len(chunks)} chunks")
        print(f"  ✓ Saved to {json_output_path}")
        if save_text:
            print(f"  ✓ Text version: {text_output_path}")
        
        return len(chunks)
        
    except Exception as e:
        print(f"  ✗ Error processing {filename}: {str(e)}")
        return 0


def process_batch(config: dict, 
                 chunking_method: str = "sections",
                 max_chunk_size: int = 2500,
                 overlap_size: int = 250,
                 save_text: bool = True,
                 skip_existing: bool = True) -> Dict[str, int]:
    """
    Process all clean text files in batch mode.
    Auto-detects files in the clean directory instead of relying on config tickers.
    
    Args:
        config: Configuration dictionary
        chunking_method: Method to use for chunking
        max_chunk_size: Maximum characters per chunk
        overlap_size: Characters to overlap between chunks
        save_text: Whether to save human-readable text versions
        skip_existing: Whether to skip files that already have chunks
        
    Returns:
        Dictionary mapping ticker to number of chunks created
    """
    # Get directories from config
    clean_annual_dir = config.get("output_dirs", {}).get("sec_txt_clean_annual", "sec_txt_clean/annual")
    clean_quarterly_dir = config.get("output_dirs", {}).get("sec_txt_clean_quarterly", "sec_txt_clean/quarterly")
    chunks_dir = config.get("output_dirs", {}).get("chunks", "chunks")
    
    # Create chunks directory
    os.makedirs(chunks_dir, exist_ok=True)
    
    # Find all .clean.txt files from both annual and quarterly directories
    clean_files = []
    
    # Process annual files
    if os.path.exists(clean_annual_dir):
        print(f"📁 Scanning annual files in: {clean_annual_dir}")
        for filename in os.listdir(clean_annual_dir):
            if filename.endswith('.clean.txt'):
                # Extract ticker from filename (e.g., "AAPL_10-K.clean.txt" -> "AAPL")
                ticker = filename.split('_')[0]
                clean_files.append((ticker, filename, os.path.join(clean_annual_dir, filename)))
    else:
        print(f"⚠️  Annual clean directory not found: {clean_annual_dir}")
    
    # Process quarterly files  
    if os.path.exists(clean_quarterly_dir):
        print(f"📁 Scanning quarterly files in: {clean_quarterly_dir}")
        for filename in os.listdir(clean_quarterly_dir):
            if filename.endswith('.clean.txt'):
                # Extract ticker from filename (e.g., "AAPL_2025_Q1_10-Q.clean.txt" -> "AAPL")
                ticker = filename.split('_')[0]
                clean_files.append((ticker, filename, os.path.join(clean_quarterly_dir, filename)))
    else:
        print(f"⚠️  Quarterly clean directory not found: {clean_quarterly_dir}")
    
    if not clean_files:
        print(f"❌ No .clean.txt files found in annual or quarterly directories")
        return {}
    
    # Sort by ticker for consistent processing order
    clean_files.sort(key=lambda x: x[0])
    
    results = {}
    total_processed = 0
    total_chunks = 0
    
    print(f"Starting batch chunk creation for {len(clean_files)} companies...")
    print(f"Auto-detected files: {[ticker for ticker, _, _ in clean_files]}")
    print(f"Method: {chunking_method}")
    print(f"Max chunk size: {max_chunk_size} characters")
    print(f"Overlap: {overlap_size} characters")
    print("-" * 60)
    
    for ticker, filename, input_file in clean_files:
        # Determine if this is annual or quarterly based on filename and path
        if "quarterly" in input_file or "_Q" in filename:
            # Quarterly file: AAPL_2025_Q1_10-Q.clean.txt -> AAPL_2025_Q1_10-Q_chunks.json
            output_dir = config.get("output_dirs", {}).get("chunks_quarterly", "chunks/quarterly")
            chunk_filename = filename.replace(".clean.txt", "_chunks.json")
        else:
            # Annual file: AAPL_10-K.clean.txt -> AAPL_10-K_chunks.json  
            output_dir = config.get("output_dirs", {}).get("chunks_annual", "chunks/annual")
            chunk_filename = filename.replace(".clean.txt", "_chunks.json")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        json_output_file = os.path.join(output_dir, chunk_filename)
        
        # File existence already verified in auto-detection
        
        # Skip if chunks already exist (optional)
        if skip_existing and os.path.exists(json_output_file):
            print(f"⏭️  Chunks already exist for {ticker}, skipping")
            # Count existing chunks for reporting
            try:
                with open(json_output_file, 'r') as f:
                    existing_chunks = json.load(f)
                    results[ticker] = len(existing_chunks)
                    total_chunks += len(existing_chunks)
            except:
                results[ticker] = 0
            continue
        
        # Process the file
        chunk_count = process_single_file(
            input_file,
            output_dir,
            chunking_method=chunking_method,
            max_chunk_size=max_chunk_size,
            overlap_size=overlap_size,
            save_text=save_text
        )
        
        results[ticker] = chunk_count
        if chunk_count > 0:
            total_processed += 1
            total_chunks += chunk_count
    
    print("-" * 60)
    print(f"Batch processing complete!")
    print(f"Files processed: {total_processed}/{len(clean_files)}")
    print(f"Total chunks created: {total_chunks}")
    print("\nResults by company:")
    for ticker, count in results.items():
        status = "✓" if count > 0 else "✗"
        print(f"  {status} {ticker}: {count} chunks")
    
    return results


def analyze_chunk_distribution(chunks_dir: str, tickers: List[str] = None) -> Dict[str, Any]:
    """
    Analyze the distribution of chunks across companies.
    Auto-detects chunk files if tickers not provided.
    
    Args:
        chunks_dir: Directory containing chunk files
        tickers: List of company tickers (auto-detected if None)
        
    Returns:
        Analysis results
    """
    analysis = {
        'total_files': 0,
        'total_chunks': 0,
        'by_company': {},
        'chunk_size_stats': {
            'min_size': float('inf'),
            'max_size': 0,
            'avg_size': 0,
            'sizes': []
        }
    }
    
    # Auto-detect chunk files if tickers not provided
    if tickers is None:
        if not os.path.exists(chunks_dir):
            return analysis
        
        tickers = []
        for filename in os.listdir(chunks_dir):
            if filename.endswith('_chunks.json'):
                # Extract ticker from filename (e.g., "AAPL_10-K_chunks.json" -> "AAPL")
                ticker = filename.split('_')[0]
                if ticker not in tickers:
                    tickers.append(ticker)
        tickers.sort()
    
    for ticker in tickers:
        json_file = os.path.join(chunks_dir, f"{ticker}_10-K_chunks.json")
        
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r') as f:
                    chunks = json.load(f)
                    
                count = len(chunks)
                analysis['total_files'] += 1
                analysis['total_chunks'] += count
                analysis['by_company'][ticker] = {
                    'chunk_count': count,
                    'avg_chunk_size': 0,
                    'total_characters': 0
                }
                
                # Analyze chunk sizes
                company_sizes = []
                company_chars = 0
                
                for chunk in chunks:
                    chunk_size = len(chunk.get('text', ''))
                    company_sizes.append(chunk_size)
                    company_chars += chunk_size
                    analysis['chunk_size_stats']['sizes'].append(chunk_size)
                    
                    analysis['chunk_size_stats']['min_size'] = min(
                        analysis['chunk_size_stats']['min_size'], chunk_size
                    )
                    analysis['chunk_size_stats']['max_size'] = max(
                        analysis['chunk_size_stats']['max_size'], chunk_size
                    )
                
                if company_sizes:
                    analysis['by_company'][ticker]['avg_chunk_size'] = int(
                        sum(company_sizes) / len(company_sizes)
                    )
                    analysis['by_company'][ticker]['total_characters'] = company_chars
                
            except Exception as e:
                print(f"Error analyzing {ticker}: {e}")
                analysis['by_company'][ticker] = {'chunk_count': 0, 'error': str(e)}
    
    # Calculate overall averages
    if analysis['chunk_size_stats']['sizes']:
        analysis['chunk_size_stats']['avg_size'] = int(
            sum(analysis['chunk_size_stats']['sizes']) / len(analysis['chunk_size_stats']['sizes'])
        )
    
    if analysis['chunk_size_stats']['min_size'] == float('inf'):
        analysis['chunk_size_stats']['min_size'] = 0
    
    return analysis


def print_analysis(analysis: Dict[str, Any]):
    """Print chunk analysis in a readable format."""
    print("\n" + "=" * 60)
    print("CHUNK ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"Total files processed: {analysis['total_files']}")
    print(f"Total chunks created: {analysis['total_chunks']}")
    
    if analysis['total_chunks'] > 0:
        print(f"\nChunk size statistics:")
        print(f"  Minimum size: {analysis['chunk_size_stats']['min_size']} characters")
        print(f"  Maximum size: {analysis['chunk_size_stats']['max_size']} characters")
        print(f"  Average size: {analysis['chunk_size_stats']['avg_size']} characters")
    
    print(f"\nBy company:")
    for ticker, stats in analysis['by_company'].items():
        if 'error' in stats:
            print(f"  ✗ {ticker}: Error - {stats['error']}")
        else:
            chunks = stats['chunk_count']
            avg_size = stats['avg_chunk_size']
            total_chars = stats.get('total_characters', 0)
            print(f"  ✓ {ticker}: {chunks} chunks, avg {avg_size} chars, total {total_chars:,} chars")


def main():
    """Main function to handle command line arguments and execute chunking."""
    parser = argparse.ArgumentParser(
        description="Create semantic chunks from cleaned SEC filing text"
    )
    
    parser.add_argument(
        "--file", 
        help="Process single file (provide path to clean text file)"
    )
    
    parser.add_argument(
        "--batch", 
        action="store_true",
        help="Process all files from config in batch mode"
    )
    
    parser.add_argument(
        "--method",
        choices=["sections", "paragraphs", "fixed"],
        default="sections",
        help="Chunking method (default: sections, recommended based on analysis)"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2500,
        help="Maximum chunk size in characters (default: 2500)"
    )
    
    parser.add_argument(
        "--overlap",
        type=int,
        default=250,
        help="Overlap size in characters (default: 250)"
    )
    
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Don't save human-readable text versions of chunks"
    )
    
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Don't skip files that already have chunks (reprocess all)"
    )
    
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze existing chunks without processing new ones"
    )
    
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        return 1
    
    # Handle analysis mode
    if args.analyze:
        chunks_dir = config.get("output_dirs", {}).get("chunks", "chunks")
        analysis = analyze_chunk_distribution(chunks_dir)  # Auto-detect tickers
        print_analysis(analysis)
        return 0
    
    # Handle single file mode
    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
            return 1
        
        chunks_dir = config.get("output_dirs", {}).get("chunks", "chunks")
        
        chunk_count = process_single_file(
            args.file,
            chunks_dir,
            chunking_method=args.method,
            max_chunk_size=args.chunk_size,
            overlap_size=args.overlap,
            save_text=not args.no_text
        )
        
        if chunk_count > 0:
            print(f"\n✓ Successfully created {chunk_count} chunks")
            return 0
        else:
            print(f"\n✗ Failed to create chunks")
            return 1
    
    # Handle batch mode or default batch processing
    if args.batch or (not args.file and not args.analyze):
        results = process_batch(
            config,
            chunking_method=args.method,
            max_chunk_size=args.chunk_size,
            overlap_size=args.overlap,
            save_text=not args.no_text,
            skip_existing=not args.no_skip
        )
        
        # Run analysis after batch processing
        chunks_dir = config.get("output_dirs", {}).get("chunks", "chunks")
        analysis = analyze_chunk_distribution(chunks_dir)  # Auto-detect tickers
        print_analysis(analysis)
        
        # Return success if any files were processed
        successful = sum(1 for count in results.values() if count > 0)
        return 0 if successful > 0 else 1
    
    # If no mode specified, show help
    parser.print_help()
    return 1


if __name__ == "__main__":
    exit(main())