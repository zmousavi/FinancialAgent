#!/usr/bin/env python3
"""
06_setup_vector_db.py

Sets up a FAISS vector database with incremental update support.
Loads embeddings and creates searchable index with metadata filtering.
"""

import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple, Optional
import yaml
from pathlib import Path

def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_all_embeddings(embeddings_dir: str) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Load all embeddings from the combined file created by 05_create_embeddings.py
    
    Returns:
        chunks: List of chunk metadata (company, section, chunk_id, text, etc.)
        embeddings_matrix: NumPy array of vector embeddings
    """
    combined_file = os.path.join(embeddings_dir, "all_embeddings.json")
    
    if not os.path.exists(combined_file):
        print(f"No embeddings found at {combined_file}")
        print("Please run 05_create_embeddings.py first to generate embeddings.")
        return [], np.array([])
    
    print(f"Loading embeddings from {combined_file}")
    
    with open(combined_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract chunks and their embeddings
    chunks = data['chunks']
    embeddings_matrix = np.array([chunk['embedding'] for chunk in chunks], dtype=np.float32)
    
    print(f"Loaded {len(chunks)} chunks with {embeddings_matrix.shape[1]}-dimensional embeddings")
    return chunks, embeddings_matrix

def create_faiss_index(embeddings: np.ndarray, index_type: str = "IndexFlatIP") -> faiss.Index:
    """
    Create a FAISS index for similarity search
    
    Args:
        embeddings: Matrix of all vector embeddings [num_chunks, embedding_dimension]
        index_type: Type of FAISS index ("IndexFlatIP" for cosine similarity)
    
    Returns:
        FAISS index ready for similarity search
    """
    if embeddings.size == 0:
        raise ValueError("No embeddings provided")
    
    dimension = embeddings.shape[1]  # Usually 1536 for OpenAI embeddings
    print(f"Creating FAISS index ({index_type}) for {embeddings.shape[0]} vectors of dimension {dimension}")
    
    # Normalize embeddings for cosine similarity (required for IndexFlatIP)
    # This converts dot product to cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Create index based on type
    if index_type == "IndexFlatIP":
        # Inner Product index (cosine similarity after normalization)
        index = faiss.IndexFlatIP(dimension)
    elif index_type == "IndexFlatL2":
        # L2 (Euclidean) distance
        index = faiss.IndexFlatL2(dimension)
    else:
        raise ValueError(f"Unsupported index type: {index_type}")
    
    # Add all embeddings to index at once
    # Each vector gets assigned position: 0, 1, 2, 3, ... (vector_id)
    index.add(embeddings)
    
    print(f"FAISS index created with {index.ntotal} vectors")
    return index

def save_vector_db(index: faiss.Index, chunks: List[Dict[str, Any]], vector_db_dir: str):
    """
    Save FAISS index and chunks metadata to disk
    
    This creates two files:
    1. faiss_index.index - The actual FAISS search index
    2. chunks_metadata.json - Metadata for each chunk (company, section, text, etc.)
    
    The metadata includes vector_id which maps to FAISS index positions
    """
    os.makedirs(vector_db_dir, exist_ok=True)
    
    # Save FAISS index (the actual vector search structure)
    index_path = os.path.join(vector_db_dir, "faiss_index.index")
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to {index_path}")
    
    # Save chunks metadata (without embeddings to save space)
   
    chunks_metadata = []
    for i, chunk in enumerate(chunks):
        # Copy all metadata EXCEPT the embedding vector (too big to store twice)
        metadata = {k: v for k, v in chunk.items() if k != 'embedding'}
        
        # CRITICAL: Add vector_id that maps to FAISS index position
        # vector_id=0 corresponds to FAISS index position 0, etc.
        metadata['vector_id'] = i  # This links metadata to FAISS results
        chunks_metadata.append(metadata)
    
    # Save metadata file with all the filtering information
    metadata_path = os.path.join(vector_db_dir, "chunks_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump({
            'chunks': chunks_metadata,  # Each chunk has: company, section, chunk_id, text, vector_id
            'metadata': {
                'total_chunks': len(chunks_metadata),
                'index_type': 'IndexFlatIP',
                'similarity_metric': 'cosine',
                'embedding_dimension': index.d,
                'created_at': json.dumps(None)  # Will be filled by json.dumps
            }
        }, f, indent=2)
    
    print(f"Chunks metadata saved to {metadata_path}")

class FinancialVectorDB:
    """
    Vector database for financial document retrieval with filtering
    
    This class loads the FAISS index and metadata, then provides search
    functionality with company and section filtering capabilities.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.index = None
        self.chunks = None
        self._load_db()
    
    def _load_db(self):
        """Load FAISS index and chunks metadata"""
        index_path = os.path.join(self.db_path, "faiss_index.index")
        metadata_path = os.path.join(self.db_path, "chunks_metadata.json")
        
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Vector database not found at {self.db_path}")
        
        # Load FAISS index (the vector search engine)
        self.index = faiss.read_index(index_path)
        print(f"Loaded FAISS index with {self.index.ntotal} vectors")
        
        # Load chunks metadata 
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.chunks = data['chunks']  # Each chunk has: company, section, chunk_id, text, vector_id
        
        print(f"Loaded metadata for {len(self.chunks)} chunks")
    
    def search(self, query_embedding: np.ndarray, k: int = 5,
               company_filter: Optional[str] = None,
               section_filter: Optional[str] = None,
               document_type_filter: Optional[str] = None,
               query_text: Optional[str] = None,
               keyword_boost: float = 0.1) -> List[Dict[str, Any]]:
        """
        Hybrid search: combines semantic similarity with keyword matching.

        HOW IT WORKS:
        1. FAISS finds similar vectors by embedding similarity
        2. If query_text provided, boost scores for chunks containing query keywords
        3. Apply metadata filters (company, section, document_type)
        4. Re-rank by boosted score and return top k

        Args:
            query_embedding: Query vector (will be normalized)
            k: Number of TOP results to return
            company_filter: Filter by company (e.g., 'AAPL', 'MSFT')
            section_filter: Filter by section (e.g., 'Business', 'Risk Factors')
            document_type_filter: Filter by document type (e.g., '10-K')
            query_text: Original query text for keyword boosting (optional)
            keyword_boost: How much to boost score per keyword match (default 0.1)

        Returns:
            List of chunk dictionaries with similarity scores, filtered and re-ranked
        """
        # Normalize query embedding for cosine similarity
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_embedding)

        # Get more results for hybrid re-ranking (need candidates to re-rank)
        # Fetch more if we're doing keyword boosting or filtering
        has_filters = company_filter or section_filter or document_type_filter
        search_k = min(k * 10, self.index.ntotal) if (has_filters or query_text) else k * 2

        # STEP 1: FAISS similarity search - returns vector positions and scores
        similarities, indices = self.index.search(query_embedding, search_k)

        # Extract query keywords for boosting (if query_text provided)
        query_keywords = []
        if query_text:
            import re
            # Stop words to ignore (common words that don't help with relevance)
            stop_words = {'what', 'are', 'the', 'is', 'a', 'an', 'of', 'in', 'for', 'to', 'and',
                         'or', 'how', 'does', 'do', 'their', 'its', 'this', 'that', 'with',
                         'from', 'about', 'say', 'main', 'key', 'company', 'companies'}
            # Ignore company names since we already filter by company
            company_names = {'apple', 'microsoft', 'google', 'amazon', 'meta', 'tesla',
                           'nvidia', 'netflix', 'walmart', 'facebook', 'alphabet'}
            # Phrase patterns to look for (more precise than single-word synonyms)
            # These are multi-word patterns that indicate relevance
            phrase_patterns = {
                'revenue': ['net sales', 'total revenue', 'revenue by', 'sales by category',
                           'products and services', 'segment revenue', 'revenue source'],
                'risk': ['risk factor', 'business risk', 'operational risk', 'market risk'],
                'competition': ['competitive', 'competitors', 'market position'],
                'business': ['business model', 'business segment', 'operations'],
            }
            # Tokenize: extract only alphabetic words
            words = re.findall(r'[a-zA-Z]+', query_text.lower())
            base_keywords = [w for w in words
                           if len(w) > 2 and w not in stop_words and w not in company_names]
            # Build keyword list with phrase patterns
            query_keywords = list(base_keywords)  # Start with base keywords
            for kw in base_keywords:
                if kw in phrase_patterns:
                    query_keywords.extend(phrase_patterns[kw])

        candidates = []
        # STEP 2: Process each FAISS result
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue

            chunk = self.chunks[idx].copy()
            base_score = float(similarity)

            # STEP 3: Apply metadata filters
            if company_filter and chunk.get('company_ticker') != company_filter:
                continue
            if section_filter and section_filter.lower() not in chunk.get('section_title', '').lower():
                continue
            if document_type_filter and chunk.get('filing_type') != document_type_filter:
                continue

            # STEP 4: Keyword boosting - check if query words appear in chunk
            boost = 0.0
            if query_keywords:
                chunk_text = (chunk.get('text', '') + ' ' + chunk.get('section_title', '')).lower()
                matches = sum(1 for kw in query_keywords if kw in chunk_text)
                boost = matches * keyword_boost

            chunk['similarity_score'] = base_score
            chunk['keyword_boost'] = boost
            chunk['final_score'] = base_score + boost
            candidates.append(chunk)

        # STEP 5: Re-rank by final_score (semantic + keyword boost)
        candidates.sort(key=lambda x: x['final_score'], reverse=True)

        # Return top k with rank assigned
        results = candidates[:k]
        for i, chunk in enumerate(results):
            chunk['rank'] = i + 1

        return results
    
    def get_companies(self) -> List[str]:
        """
        Get list of available companies in the database

        Scans all chunk metadata to find unique company values
        Used for: Validation, UI dropdowns, user info
        """
        companies = set()
        for chunk in self.chunks:
            if chunk.get('company_ticker'):
                companies.add(chunk['company_ticker'])
        return sorted(list(companies))

    def get_document_types(self) -> List[str]:
        """
        Get list of available document types in the database

        Returns:
            List of document types (e.g., ['10-K', '8-K', '10-Q'])
        """
        doc_types = set()
        for chunk in self.chunks:
            if chunk.get('filing_type'):
                doc_types.add(chunk['filing_type'])
        return sorted(list(doc_types))

    def get_sections(self, company: Optional[str] = None) -> List[str]:
        """
        Get list of available sections, optionally filtered by company

        Args:
            company: If provided, only return sections for this company

        Returns:
            List of section names (e.g., ['Business', 'Risk Factors', 'Financial Data'])
        """
        sections = set()
        for chunk in self.chunks:
            # Apply company filter if provided
            if company and chunk.get('company_ticker') != company:
                continue
            if chunk.get('section_title'):
                sections.add(chunk['section_title'])
        return sorted(list(sections))
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics for debugging and user info
        
        Returns:
            Dictionary with counts by company, section, totals, etc.
        """
        stats = {
            'total_chunks': len(self.chunks),
            'total_vectors': self.index.ntotal,
            'embedding_dimension': self.index.d,
            'companies': {},      # Will contain: {'AAPL': 154, 'MSFT': 361}
            'sections': {},       # Will contain: {'Business': 45, 'Risk Factors': 38, ...}
            'document_types': {}  # Will contain: {'10-K': 515, '8-K': 23, ...}
        }
        
        # Count chunks by company (for filtering validation)
        for chunk in self.chunks:
            company = chunk.get('company_ticker', 'Unknown')
            stats['companies'][company] = stats['companies'].get(company, 0) + 1

        # Count chunks by section (for filtering validation)
        for chunk in self.chunks:
            section = chunk.get('section_title', 'Unknown')
            stats['sections'][section] = stats['sections'].get(section, 0) + 1

        # Count chunks by document type (for filtering validation)
        for chunk in self.chunks:
            doc_type = chunk.get('filing_type', 'Unknown')
            stats['document_types'][doc_type] = stats['document_types'].get(doc_type, 0) + 1
        
        return stats

def main():
    """
    Main function to set up the vector database
    
    WORKFLOW:
    1. Load embeddings created by 04_create_embeddings.py
    2. Create FAISS search index from embeddings
    3. Save index + metadata to disk
    4. Test the database with a sample search
    5. Show statistics and confirm everything works
    """
    
    config = load_config()
    embeddings_dir = "embeddings"      # Where 04_create_embeddings.py saved embeddings
    vector_db_dir = "vector_db"        # Where we'll save the searchable database
    
    # Check if database already exists
    index_path = os.path.join(vector_db_dir, "faiss_index.index")
    metadata_path = os.path.join(vector_db_dir, "chunks_metadata.json")
    
    if os.path.exists(index_path) and os.path.exists(metadata_path):
        print(f"✅ Vector database already exists at {vector_db_dir}/")
        print(f"   To rebuild, delete the directory first: rm -rf {vector_db_dir}")
        
        # Show current database stats
        try:
            db = FinancialVectorDB(vector_db_dir)
            stats = db.get_stats()
            print(f"   Current database has {stats['total_chunks']} chunks from {len(stats['companies'])} companies")
            print(f"   Companies: {list(stats['companies'].keys())}")
            return
        except Exception as e:
            print(f"   Database exists but corrupted: {e}")
            print(f"   Rebuilding...")
    
    # STEP 1: Load embeddings from 05_create_embeddings.py output
    chunks, embeddings_matrix = load_all_embeddings(embeddings_dir)
    
    if len(chunks) == 0:
        return  # No embeddings found, exit
    
    # STEP 2: Create FAISS index for fast similarity search
    index = create_faiss_index(embeddings_matrix)
    
    # STEP 3: Save vector database (index + metadata for filtering)
    save_vector_db(index, chunks, vector_db_dir)
    
    print(f"\n Vector database setup complete!")
    print(f"   - Database saved to: {vector_db_dir}/")
    print(f"   - Total vectors indexed: {index.ntotal}")
    print(f"   - Vector dimension: {embeddings_matrix.shape[1]}")
    
    # STEP 4: Test the database to make sure everything works
    print(f"\n Testing database...")
    try:
        # Create a new instance to test loading from disk
        db = FinancialVectorDB(vector_db_dir)
        
        # Show available companies and sections for filtering
        stats = db.get_stats()
        print(f"   - Companies: {list(stats['companies'].keys())}")
        print(f"   - Sections available: {len(stats['sections'])}")
        
        # Test search with first chunk's embedding (should return itself as top result)
        test_query = embeddings_matrix[0:1]  # Use first chunk as test query
        results = db.search(test_query, k=3)  # k=3 means return TOP 3 most similar chunks
        
        print(f"   - Test search returned {len(results)} results")
        if results:
            top_result = results[0]
            print(f"   - Top result: {top_result['chunk_id']} (score: {top_result['similarity_score']:.4f})")
            print(f"   - Section: {top_result.get('section', 'Unknown')}")
        
        print(f"\n Database ready for queries!")
        print(f"   Next steps:")
        print(f"   1. Use the database in your RAG pipeline")
        print(f"   2. Create query interface (Phase 2)")
        
    except Exception as e:
        print(f"    Test failed: {e}")
        return
    
    # STEP 5: Show final statistics
    print(f"\n Final Statistics:")
    print(f"   - Total chunks: {stats['total_chunks']}")
    for company, count in stats['companies'].items():
        print(f"   - {company}: {count} chunks")

if __name__ == "__main__":
    main()