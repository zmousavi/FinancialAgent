"""
03_analyze_documents.py
Analyze cleaned SEC filing documents to determine optimal chunking strategy.

This script examines document structure, sections, paragraphs, and content patterns
to help choose the best chunking method for RAG implementation.

Usage:
  python 03_analyze_documents.py --file docs_txt_clean/AAPL_10-K.clean.txt
  python 03_analyze_documents.py --batch  # analyze all files in docs_txt_clean/
"""

import os
import re
import argparse
import yaml
from typing import Dict, List, Tuple


def load_cfg(path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def read_text_file(file_path: str) -> str:
    """Read text file with UTF-8 encoding."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def analyze_basic_stats(text: str) -> Dict[str, int]:
    """Get basic document statistics."""
    lines = text.split('\n')
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    
    return {
        'total_characters': len(text),
        'total_lines': len(lines),
        'blank_lines': sum(1 for line in lines if not line.strip()),
        'content_lines': sum(1 for line in lines if line.strip()),
        'total_paragraphs': len(paragraphs),
        'total_sentences': len(sentences),
        'total_words': len(text.split()),
        'avg_paragraph_length': sum(len(p) for p in paragraphs) // len(paragraphs) if paragraphs else 0,
        'avg_sentence_length': sum(len(s) for s in sentences) // len(sentences) if sentences else 0
    }


def find_sec_sections(text: str) -> List[Dict[str, any]]:
    """Find SEC filing sections and analyze their structure."""
    sections = []
    lines = text.split('\n')
    
    # SEC section patterns
    section_patterns = [
        (r'^PART\s+(I+|IV?)\s*$', 'PART'),
        (r'^Item\s+\d+[A-Z]?\.\s*(.+)', 'ITEM'),
        (r'^ITEM\s+\d+[A-Z]?\.\s*(.+)', 'ITEM'),
        (r'^(BUSINESS|RISK FACTORS|FINANCIAL STATEMENTS|PROPERTIES)$', 'MAJOR_SECTION'),
        (r'^[A-Z\s]{15,}$', 'HEADER'),  # Long all-caps headers
    ]
    
    current_pos = 0
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped:
            current_pos += len(line) + 1
            continue
        
        for pattern, section_type in section_patterns:
            match = re.match(pattern, line_stripped, re.IGNORECASE)
            if match:
                # Calculate content until next section
                content_length = 0
                content_lines = 0
                j = i + 1
                
                while j < len(lines):
                    next_line = lines[j].strip()
                    if next_line:
                        # Check if this starts a new section
                        is_new_section = any(re.match(p, next_line, re.IGNORECASE) 
                                           for p, _ in section_patterns)
                        if is_new_section:
                            break
                    
                    content_length += len(lines[j]) + 1
                    if lines[j].strip():
                        content_lines += 1
                    j += 1
                
                sections.append({
                    'line_number': i + 1,
                    'title': line_stripped,
                    'type': section_type,
                    'content_length': content_length,
                    'content_lines': content_lines,
                    'position': current_pos
                })
                break
        
        current_pos += len(line) + 1
    
    return sections


def analyze_paragraph_distribution(text: str) -> Dict[str, any]:
    """Analyze paragraph length distribution."""
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    
    if not paragraphs:
        return {}
    
    lengths = [len(p) for p in paragraphs]
    lengths.sort()
    
    return {
        'paragraph_count': len(paragraphs),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'median_length': lengths[len(lengths) // 2],
        'avg_length': sum(lengths) // len(lengths),
        'short_paragraphs': sum(1 for l in lengths if l < 200),  # < 200 chars
        'medium_paragraphs': sum(1 for l in lengths if 200 <= l <= 1000),  # 200-1000 chars
        'long_paragraphs': sum(1 for l in lengths if l > 1000),  # > 1000 chars
        'very_long_paragraphs': sum(1 for l in lengths if l > 2000),  # > 2000 chars
    }


def analyze_table_content(text: str) -> Dict[str, int]:
    """Detect and analyze table-like content."""
    lines = text.split('\n')
    
    # Patterns that suggest tabular data
    table_indicators = {
        'pipe_tables': 0,  # Lines with multiple | characters
        'aligned_numbers': 0,  # Lines with aligned numeric data
        'dollar_amounts': 0,  # Lines with dollar amounts
        'percentage_lines': 0,  # Lines with percentages
        'date_lines': 0,  # Lines with dates
    }
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Count pipe characters (table formatting)
        if line.count('|') >= 3:
            table_indicators['pipe_tables'] += 1
        
        # Look for dollar amounts
        if re.search(r'\$\s*[\d,]+', line):
            table_indicators['dollar_amounts'] += 1
        
        # Look for percentages
        if re.search(r'\d+\.?\d*\s*%', line):
            table_indicators['percentage_lines'] += 1
        
        # Look for dates
        if re.search(r'\b(January|February|March|April|May|June|July|August|September|October|November|December|\d{1,2}/\d{1,2}/\d{4}|\d{4})', line):
            table_indicators['date_lines'] += 1
        
        # Look for aligned numbers (multiple numbers with spacing)
        if re.search(r'\d+\s+\d+\s+\d+', line):
            table_indicators['aligned_numbers'] += 1
    
    return table_indicators


def suggest_chunking_strategy(analysis: Dict[str, any]) -> Dict[str, any]:
    """Analyze document structure and suggest optimal chunking strategy."""
    basic_stats = analysis['basic_stats']
    sections = analysis['sections']
    paragraphs = analysis['paragraphs']
    tables = analysis['tables']
    
    recommendations = {
        'primary_method': '',
        'chunk_size': 0,
        'overlap_size': 0,
        'reasoning': [],
        'secondary_method': '',
        'section_based_feasible': False
    }
    
    # Check if section-based chunking is feasible
    if len(sections) > 5:  # Has meaningful section structure
        avg_section_size = sum(s['content_length'] for s in sections) // len(sections)
        
        if avg_section_size < 3000:  # Sections are reasonably sized
            recommendations['primary_method'] = 'sections'
            recommendations['chunk_size'] = 2500
            recommendations['overlap_size'] = 250
            recommendations['section_based_feasible'] = True
            recommendations['reasoning'].append(f"Document has {len(sections)} well-defined sections")
            recommendations['reasoning'].append(f"Average section size: {avg_section_size} characters")
        else:
            recommendations['reasoning'].append(f"Sections too large (avg: {avg_section_size} chars)")
    
    # Check paragraph distribution
    if paragraphs['medium_paragraphs'] > paragraphs['paragraph_count'] * 0.6:
        if not recommendations['primary_method']:
            recommendations['primary_method'] = 'paragraphs'
            recommendations['chunk_size'] = 1500
            recommendations['overlap_size'] = 150
        else:
            recommendations['secondary_method'] = 'paragraphs'
        recommendations['reasoning'].append(f"Good paragraph distribution: {paragraphs['medium_paragraphs']} medium-sized paragraphs")
    
    # Check for heavy table content
    table_density = (tables['pipe_tables'] + tables['dollar_amounts']) / basic_stats['content_lines']
    if table_density > 0.1:  # More than 10% table content
        recommendations['reasoning'].append(f"High table content density: {table_density:.1%}")
        if not recommendations['primary_method']:
            recommendations['primary_method'] = 'fixed'
            recommendations['chunk_size'] = 1000
            recommendations['overlap_size'] = 100
        recommendations['reasoning'].append("Fixed-size chunking recommended for table-heavy content")
    
    # Default fallback
    if not recommendations['primary_method']:
        recommendations['primary_method'] = 'paragraphs'
        recommendations['chunk_size'] = 1200
        recommendations['overlap_size'] = 120
        recommendations['reasoning'].append("Default paragraph-based chunking")
    
    return recommendations


def analyze_document(file_path: str) -> Dict[str, any]:
    """Perform comprehensive analysis of a single document."""
    print(f"\nAnalyzing: {file_path}")
    print("=" * 60)
    
    text = read_text_file(file_path)
    
    # Perform all analyses
    basic_stats = analyze_basic_stats(text)
    sections = find_sec_sections(text)
    paragraphs = analyze_paragraph_distribution(text)
    tables = analyze_table_content(text)
    
    analysis = {
        'file_path': file_path,
        'basic_stats': basic_stats,
        'sections': sections,
        'paragraphs': paragraphs,
        'tables': tables
    }
    
    recommendations = suggest_chunking_strategy(analysis)
    analysis['recommendations'] = recommendations
    
    return analysis


def print_analysis_report(analysis: Dict[str, any]):
    """Print a formatted analysis report."""
    file_name = os.path.basename(analysis['file_path'])
    stats = analysis['basic_stats']
    sections = analysis['sections']
    paragraphs = analysis['paragraphs']
    tables = analysis['tables']
    rec = analysis['recommendations']
    
    print(f"\nDOCUMENT: {file_name}")
    print("-" * 40)
    
    print(f"\nBASIC STATISTICS:")
    print(f"  Total characters: {stats['total_characters']:,}")
    print(f"  Total lines: {stats['total_lines']:,}")
    print(f"  Content lines: {stats['content_lines']:,}")
    print(f"  Paragraphs: {stats['total_paragraphs']:,}")
    print(f"  Words: {stats['total_words']:,}")
    print(f"  Avg paragraph length: {stats['avg_paragraph_length']} chars")
    
    print(f"\nSECTION STRUCTURE:")
    print(f"  Total sections found: {len(sections)}")
    if sections:
        section_types = {}
        for section in sections:
            section_types[section['type']] = section_types.get(section['type'], 0) + 1
        
        for section_type, count in section_types.items():
            print(f"    {section_type}: {count}")
        
        avg_section_size = sum(s['content_length'] for s in sections) // len(sections)
        print(f"  Average section size: {avg_section_size} chars")
    
    print(f"\nPARAGRAPH DISTRIBUTION:")
    print(f"  Short paragraphs (<200 chars): {paragraphs.get('short_paragraphs', 0)}")
    print(f"  Medium paragraphs (200-1000 chars): {paragraphs.get('medium_paragraphs', 0)}")
    print(f"  Long paragraphs (1000-2000 chars): {paragraphs.get('long_paragraphs', 0)}")
    print(f"  Very long paragraphs (>2000 chars): {paragraphs.get('very_long_paragraphs', 0)}")
    print(f"  Median paragraph length: {paragraphs.get('median_length', 0)} chars")
    
    print(f"\nTABLE CONTENT INDICATORS:")
    print(f"  Pipe table lines: {tables['pipe_tables']}")
    print(f"  Dollar amount lines: {tables['dollar_amounts']}")
    print(f"  Percentage lines: {tables['percentage_lines']}")
    print(f"  Date lines: {tables['date_lines']}")
    
    print(f"\nRECOMMENDED CHUNKING STRATEGY:")
    print(f"  Primary method: {rec['primary_method']}")
    print(f"  Chunk size: {rec['chunk_size']} characters")
    print(f"  Overlap size: {rec['overlap_size']} characters")
    if rec['secondary_method']:
        print(f"  Secondary method: {rec['secondary_method']}")
    
    print(f"\nREASONING:")
    for reason in rec['reasoning']:
        print(f"  - {reason}")


def main():
    """Main function to run document analysis."""
    parser = argparse.ArgumentParser(description="Analyze SEC filing documents for chunking strategy")
    parser.add_argument("--file", help="Analyze a specific file")
    parser.add_argument("--batch", action="store_true", help="Analyze all files in docs_txt_clean/")
    
    args = parser.parse_args()
    
    if args.file:
        # Analyze single file
        if not os.path.exists(args.file):
            print(f"Error: File {args.file} not found")
            return
        
        analysis = analyze_document(args.file)
        print_analysis_report(analysis)
        
    elif args.batch:
        # Analyze all files in both annual and quarterly clean directories
        cfg = load_cfg()
        clean_annual_dir = cfg['output_dirs']['clean_txt_annual']
        clean_quarterly_dir = cfg['output_dirs']['clean_txt_quarterly']
        
        all_analyses = []
        txt_files = []
        
        # Collect files from annual directory
        if os.path.exists(clean_annual_dir):
            annual_files = [f for f in os.listdir(clean_annual_dir) if f.endswith('.txt')]
            print(f"Found {len(annual_files)} annual files in {clean_annual_dir}")
            for txt_file in annual_files:
                txt_files.append(('annual', os.path.join(clean_annual_dir, txt_file), txt_file))
        else:
            print(f"Annual directory {clean_annual_dir} not found")
        
        # Collect files from quarterly directory  
        if os.path.exists(clean_quarterly_dir):
            quarterly_files = [f for f in os.listdir(clean_quarterly_dir) if f.endswith('.txt')]
            print(f"Found {len(quarterly_files)} quarterly files in {clean_quarterly_dir}")
            for txt_file in quarterly_files:
                txt_files.append(('quarterly', os.path.join(clean_quarterly_dir, txt_file), txt_file))
        else:
            print(f"Quarterly directory {clean_quarterly_dir} not found")
        
        if not txt_files:
            print("No .txt files found in annual or quarterly directories")
            return
        
        print(f"\nAnalyzing {len(txt_files)} total files...\n")
        
        # Analyze each file
        for file_type, file_path, filename in txt_files:
            print(f"\n{'='*60}")
            print(f"Analyzing {file_type.upper()}: {filename}")
            print('='*60)
            analysis = analyze_document(file_path)
            analysis['file_type'] = file_type  # Add file type to analysis
            print_analysis_report(analysis)
            all_analyses.append(analysis)
        
        # Summary across all documents
        print("\n" + "=" * 80)
        print("SUMMARY ACROSS ALL DOCUMENTS")
        print("=" * 80)
        
        # Separate annual and quarterly analyses
        annual_analyses = [a for a in all_analyses if a.get('file_type') == 'annual']
        quarterly_analyses = [a for a in all_analyses if a.get('file_type') == 'quarterly']
        
        print(f"\nDocument breakdown:")
        print(f"  Annual documents: {len(annual_analyses)}")
        print(f"  Quarterly documents: {len(quarterly_analyses)}")
        print(f"  Total documents: {len(all_analyses)}")
        
        # Recommended methods
        primary_methods = [a['recommendations']['primary_method'] for a in all_analyses]
        method_counts = {}
        for method in primary_methods:
            method_counts[method] = method_counts.get(method, 0) + 1
        
        print(f"\nRecommended primary methods:")
        for method, count in method_counts.items():
            print(f"  {method}: {count} documents")
        
        # Calculate statistics by type
        def calc_stats(analyses, doc_type):
            if not analyses:
                return
            total_chars = sum(a['basic_stats']['total_characters'] for a in analyses)
            total_sections = sum(len(a['sections']) for a in analyses)
            print(f"\n{doc_type} statistics:")
            print(f"  Documents: {len(analyses)}")
            print(f"  Total characters: {total_chars:,}")
            print(f"  Total sections: {total_sections}")
            print(f"  Average document size: {total_chars // len(analyses):,} characters")
        
        calc_stats(annual_analyses, "Annual")
        calc_stats(quarterly_analyses, "Quarterly")
        
        # Overall statistics
        total_chars = sum(a['basic_stats']['total_characters'] for a in all_analyses)
        total_sections = sum(len(a['sections']) for a in all_analyses)
        
        print(f"\nOverall statistics:")
        print(f"  Total documents: {len(all_analyses)}")
        print(f"  Total characters: {total_chars:,}")
        print(f"  Total sections: {total_sections}")
        print(f"  Average document size: {total_chars // len(all_analyses):,} characters")
        
    else:
        print("Error: Please specify --file <path> or --batch")
        parser.print_help()


if __name__ == "__main__":
    main()