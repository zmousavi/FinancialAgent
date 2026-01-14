#!/usr/bin/env python3
"""
07_generation_system.py

Phase 3: Generation System - RAG Pipeline with LLM Integration
Combines vector search with LLM generation for financial Q&A.

This script creates a complete RAG (Retrieval-Augmented Generation) pipeline:
1. Takes user queries about financial data
2. Converts queries to embeddings and searches vector database
3. Retrieves relevant chunks with filtering
4. Sends context + query to LLM for generation
5. Returns formatted response with citations
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
# CHANGED: Replaced OpenAI with Google's genai SDK for Gemini
from google import genai
from dotenv import load_dotenv
import yaml
from datetime import datetime
import sys


# Import the vector DB class by executing the file
exec(open('06_setup_vector_db.py').read())
# FinancialVectorDB class is now available in the global namespace

# Load environment variables
load_dotenv()

def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

class FinancialRAGSystem:
    """
    Complete RAG (Retrieval-Augmented Generation) system for financial documents.

    Combines:
    - Vector database search (from 06_setup_vector_db.py)
    - Google Gemini LLM generation (via Vertex AI)
    - Smart prompting for financial context
    - Citation generation

    Usage:
        rag = FinancialRAGSystem('vector_db/')
        response = rag.query("What are Apple's main revenue sources?")
        print(response['answer'])
        print(response['citations'])
    """

    def __init__(self, vector_db_path: str, model: str = "gemini-2.0-flash-exp"):
        """
        Initialize the RAG system

        Args:
            vector_db_path: Path to vector database directory
            model: Gemini model to use for generation
                   Options: 'gemini-2.0-flash-exp', 'gemini-1.5-flash', 'gemini-1.5-pro'
        """
        self.vector_db_path = vector_db_path
        self.model = model
        self.config = load_config()

        # CHANGED: Initialize Vertex AI (Google Cloud) instead of OpenAI
        self._initialize_vertex_ai()

        # Load vector database
        self._load_vector_db()

        # System prompts for financial context
        self.system_prompt = self._create_system_prompt()

    def _initialize_vertex_ai(self):
        """
        Initialize Vertex AI client for Gemini and embeddings

        WHY THIS IS NEEDED:
        - Connects to GCP project with proper authentication
        - Creates client for both embeddings (queries) and generation (Gemini)
        - Uses google-genai SDK (the one that works!)
        """
        # Get GCP project details from environment
        project_id = os.getenv(self.config['vertex_ai']['project_id_env_var'])
        location = self.config['vertex_ai']['location']

        if not project_id:
            raise ValueError(
                f"Environment variable {self.config['vertex_ai']['project_id_env_var']} "
                f"not set in .env file"
            )

        # Create Vertex AI client (same client for both embeddings and generation)
        self.genai_client = genai.Client(
            vertexai=True,  # Use Vertex AI (enterprise), not public API
            project=project_id,
            location=location
        )

        print(f"✓ Initialized Vertex AI")
        print(f"  Project: {project_id}")
        print(f"  Location: {location}")
        print(f"  Model: {self.model}")    
    def _load_vector_db(self):
        """Load the vector database using the imported FinancialVectorDB class"""
        # Create database instance directly
        self.vector_db = FinancialVectorDB(self.vector_db_path)
        print(f"Loaded vector database with {self.vector_db.index.ntotal} vectors")
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for financial Q&A"""
        return """You are a financial analyst AI assistant specialized in analyzing SEC filings and corporate financial documents.

Your role:
- Analyze financial documents (10-K filings, annual reports)
- Provide accurate, well-cited responses about company finances
- Focus on factual information from the provided context
- Integrate source information naturally and provide numbered references

IMPORTANT DATA CURRENCY NOTICE:
- Your responses are based on historical 10-K filings (typically 2023-2024 data)
- For the most current financial data, users should consult recent quarterly reports
- Always specify the fiscal year/period when citing financial metrics
- If asked about "current" or "latest" data, clarify the data vintage in your response

Guidelines:
1. ACCURACY: Only use information from the provided context
2. NATURAL CITATIONS: Embed source context naturally (e.g., "According to Apple's 2024 annual report...")
3. NUMBERED REFERENCES: Use [1], [2], etc. for specific factual claims
4. DATA VINTAGE: Always specify the fiscal year when citing financial metrics
5. FINANCIAL FOCUS: Emphasize financial metrics, business models, risks
6. CLARITY: Explain complex financial concepts clearly
7. COMPLETENESS: If context is insufficient, say so explicitly

Response format:
- Direct answer with natural source integration and data vintage
- Use numbered references [1], [2] for specific claims
- Include "Sources:" section at the end with full source details

Example:
According to Apple's 2024 annual report, the company's net profit margin for fiscal year 2024 was 23.96% [1]. Microsoft's business model focuses on cloud services [2].

Sources:
[1] Apple Inc. - 2024 Form 10-K Annual Report, Financial Performance section
[2] Microsoft Corp. - 2024 Form 10-K Annual Report, Business section"""

    def _create_embedding(self, text: str) -> np.ndarray:
        """
        Create embedding for text query using Vertex AI

        IMPORTANT DIFFERENCE FROM DOCUMENT EMBEDDINGS:
        - Uses task_type='RETRIEVAL_QUERY' (for search queries)
        - Documents use 'RETRIEVAL_DOCUMENT' (for content being searched)
        - This improves search accuracy

        Args:
            text: Query text to embed

        Returns:
            768-dimensional embedding vector (Vertex AI text-embedding-004)
        """
        # CHANGED: Use Vertex AI embeddings instead of OpenAI
        response = self.genai_client.models.embed_content(
            model='text-embedding-004',  # Google's latest embedding model
            contents=[text],
            config={
                'task_type': 'RETRIEVAL_QUERY',  # For queries (not documents!)
                'output_dimensionality': 768
            }
        )

        # Extract vector from response
        vector = response.embeddings[0].values
        return np.array(vector, dtype=np.float32)
    
    def _extract_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Extract intent and filters from user query
        
        Future enhancement: Use LLM to parse query for:
        - Company mentions (Apple, AAPL, Microsoft, MSFT)
        - Section interests (revenue, risk, business model)
        - Document type preferences
        
        For now, returns basic structure
        """
        intent = {
            'original_query': query,
            'company_filter': None,
            'section_filter': None,
            'document_type_filter': None,
            'search_k': 5  # Number of chunks to retrieve
        }
        
        # Simple keyword-based extraction (can be enhanced with LLM later)
        query_lower = query.lower()

        # Get company name mappings from config (ticker -> list of names)
        company_names = self.config.get('company_names', {})

        # Find all companies mentioned in the query
        mentioned_companies = []
        for ticker, names in company_names.items():
            # Check ticker (e.g., "aapl") or any company name (e.g., "apple")
            if ticker.lower() in query_lower or any(name in query_lower for name in names):
                mentioned_companies.append(ticker)

        # Store mentioned companies for multi-company retrieval
        if len(mentioned_companies) == 1:
            intent['company_filter'] = mentioned_companies[0]
        elif len(mentioned_companies) >= 2:
            # Multi-company query: store list for balanced retrieval
            intent['multi_company'] = mentioned_companies

        # Section filter detection (currently disabled in search)
        if any(word in query_lower for word in ['risk', 'risks', 'uncertainty']):
            intent['section_filter'] = 'Risk Factors'
        elif any(word in query_lower for word in ['business', 'operations', 'model']):
            intent['section_filter'] = 'Business'

        return intent
    
    def _retrieve_context(self, query: str, intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks from vector database

        Args:
            query: Original user query
            intent: Extracted intent with filters

        Returns:
            List of relevant chunks with metadata
        """
        # Create query embedding
        query_embedding = self._create_embedding(query)

        # Multi-company query: retrieve chunks from each company separately for balance
        if intent.get('multi_company'):
            companies = intent['multi_company']
            chunks_per_company = intent['search_k']  # Get full k per company

            all_results = []
            for company in companies:
                company_results = self.vector_db.search(
                    query_embedding,
                    k=chunks_per_company,
                    company_filter=company,
                    document_type_filter=intent.get('document_type_filter'),
                    query_text=query
                )
                all_results.extend(company_results)

            # Return all chunks (k * num_companies total)
            all_results.sort(key=lambda x: x.get('final_score', x.get('similarity_score', 0)), reverse=True)
            return all_results

        # Single company or no company filter: standard search
        results = self.vector_db.search(
            query_embedding,
            k=intent['search_k'],
            company_filter=intent.get('company_filter'),
            document_type_filter=intent.get('document_type_filter'),
            query_text=query
        )

        return results
    
    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into context for LLM
        
        Args:
            chunks: Retrieved chunks from vector database
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            # Use ticker symbol directly - scalable for any company
            # FIXED: Use correct field names from metadata
            source_description = f"{chunk['company_ticker']} - 2023 Form 10-K Annual Report, {chunk['section_title']} section"
            
            context_parts.append(f"""
sDocument [{i}] - {source_description}:
{chunk['text']}
""")
        
        # Add instruction for numbered citations
        context_parts.append("""

CITATION INSTRUCTIONS:
- Use natural language to mention sources (e.g., "According to Apple's annual report...")
- Add numbered references [1], [2], etc. for specific factual claims
- The numbers above correspond to the document sources
- Include a "Sources:" section at the end with full source details
""")
        
        return "\n".join(context_parts)
    
    def _generate_response(self, query: str, context: str) -> Dict[str, Any]:
        """
        Generate response using Google Gemini with context

        KEY DIFFERENCES FROM OPENAI:
        - Uses generate_content() instead of chat.completions.create()
        - System prompt goes in 'system_instruction' config parameter
        - Response is in .text property (not .choices[0].message.content)
        - Token count not returned (we estimate it)

        Args:
            query: User's original question
            context: Formatted context from retrieved chunks

        Returns:
            Response dictionary with answer and metadata
        """
        # Create user prompt (same as before)
        user_prompt = f"""
Based on the following financial documents, please answer this question:

QUESTION: {query}

CONTEXT:
{context}

Please provide a comprehensive answer with proper citations for all factual claims.
"""

        # CHANGED: Call Gemini API instead of OpenAI
        response = self.genai_client.models.generate_content(
            model=self.model,  # e.g., 'gemini-2.0-flash-exp'
            contents=[user_prompt],
            config={
                # KEY DIFFERENCE: system_instruction is a config parameter, not a message
                'system_instruction': self.system_prompt,
                'temperature': 0.1,  # Low temperature for factual accuracy
                'max_output_tokens': 2048,  # Gemini supports more tokens than OpenAI
                'top_p': 0.95
            }
        )

        # CHANGED: Extract answer from Gemini response format
        answer = response.text

        # CHANGED: Gemini doesn't return exact token count, so we estimate
        # Rough estimate: word count * 1.3 (accounting for tokens vs words)
        estimated_tokens = int(len(answer.split()) * 1.3)

        return {
            'answer': answer,
            'model': self.model,
            'tokens_used': estimated_tokens,  # Estimated (not exact like OpenAI)
            'timestamp': datetime.now().isoformat()
        }
    
    def _extract_citations(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Extract citation information from retrieved chunks
        
        Args:
            chunks: Retrieved chunks from vector database
            
        Returns:
            List of citation dictionaries
        """
        citations = []
        
        for i, chunk in enumerate(chunks, 1):
            # Use ticker symbol directly - scalable for any company
            # FIXED: Use correct field names from metadata
            user_friendly_citation = f"{chunk['company_ticker']} - 2023 Form 10-K Annual Report, {chunk['section_title']} section"

            citations.append({
                'reference_number': i,
                'company': chunk.get('company_ticker', 'Unknown'),
                'section': chunk.get('section_title', 'Unknown'),
                'document_type': chunk.get('filing_type', 'Unknown'),
                'source_file': chunk.get('document_name', 'Unknown'),
                'user_friendly_format': user_friendly_citation,
                'similarity_score': chunk.get('similarity_score', 0.0)
            })
        
        return citations
    
    def query(self, user_query: str, **kwargs) -> Dict[str, Any]:
        """
        Main query method - complete RAG pipeline
        
        Args:
            user_query: User's question about financial documents
            **kwargs: Optional overrides for filters
            
        Returns:
            Complete response with answer, citations, and metadata
        """
        print(f"Processing query: {user_query}")
        
        # Step 1: Extract intent and filters from query
        intent = self._extract_query_intent(user_query)
        
        # Allow manual overrides
        intent.update(kwargs)
        
        print(f"Query intent: {intent}")
        
        # Step 2: Retrieve relevant context from vector database
        retrieved_chunks = self._retrieve_context(user_query, intent)
        
        print(f"Retrieved {len(retrieved_chunks)} relevant chunks")
        
        # Step 3: Format context for LLM
        formatted_context = self._format_context(retrieved_chunks)
        
        # Step 4: Generate response with LLM
        response = self._generate_response(user_query, formatted_context)
        
        # Step 5: Extract citations
        citations = self._extract_citations(retrieved_chunks)
        
        # Combine everything into final response
        return {
            'query': user_query,
            'answer': response['answer'],
            'citations': citations,
            'retrieved_chunks': len(retrieved_chunks),
            'model_used': response['model'],
            'tokens_used': response['tokens_used'],
            'timestamp': response['timestamp'],
            'intent': intent
        }

def test_rag_system():
    """Test the RAG system with sample queries using Gemini"""
    print("Testing Financial RAG System with Gemini...")
    print("=" * 50)

    # Initialize RAG system (uses gemini-2.0-flash-exp by default)
    rag = FinancialRAGSystem('vector_db/')
    
    # Test queries
    test_queries = [
        "What are Apple's main revenue sources?",
        "What are the key risk factors for Microsoft?",
        "Compare Apple and Microsoft's business models",
        "What does Apple say about competition in their filing?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 30)
        
        try:
            result = rag.query(query)
            
            print(f"Answer: {result['answer']}")
            print(f"\nSources ({len(result['citations'])} total):")
            for i, citation in enumerate(result['citations'], 1):
                print(f"[{i}] {citation['user_friendly_format']} (similarity: {citation['similarity_score']:.3f})")
            print(f"\nTokens used: {result['tokens_used']}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "=" * 50)
    print("RAG system test complete!")

if __name__ == "__main__":
    test_rag_system()