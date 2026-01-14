"""
agent.py - Main FinancialAgent class for RAG-powered financial Q&A.

This module provides the FinancialAgent class which:
1. Takes user queries about financial documents
2. Converts queries to embeddings and searches vector database
3. Retrieves relevant chunks with filtering
4. Sends context + query to Gemini LLM for generation
5. Returns formatted response with citations
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime

from google import genai
from dotenv import load_dotenv

from .vector_db import FinancialVectorDB

# Load environment variables
load_dotenv()


class FinancialAgent:
    """
    RAG-powered agent for financial document Q&A.

    Combines:
    - Vector database search for relevant document chunks
    - Google Gemini LLM for generating answers
    - Smart prompting for financial context
    - Citation generation

    Usage:
        agent = FinancialAgent("path/to/vector_db")
        result = agent.query("What are Apple's main revenue sources?")
        print(result["answer"])
        print(result["citations"])
    """

    def __init__(self, vector_db_path: str, model: str = "gemini-2.0-flash-exp"):
        """
        Initialize the Financial Agent.

        Args:
            vector_db_path: Path to vector database directory
            model: Gemini model to use for generation
                   Options: 'gemini-2.0-flash-exp', 'gemini-1.5-flash', 'gemini-1.5-pro'
        """
        self.vector_db_path = vector_db_path
        self.model = model

        # Company name mappings for query parsing
        self.company_names = {
            'AAPL': ['apple'],
            'MSFT': ['microsoft'],
            'GOOGL': ['google', 'alphabet'],
            'AMZN': ['amazon'],
            'TSLA': ['tesla'],
            'META': ['meta', 'facebook'],
            'NFLX': ['netflix'],
            'NVDA': ['nvidia'],
            'WMT': ['walmart'],
        }

        # Initialize Gemini client
        self._initialize_gemini()

        # Load vector database
        self._load_vector_db()

        # System prompt for financial context
        self.system_prompt = self._create_system_prompt()

    def _initialize_gemini(self):
        """Initialize Google Gemini client."""
        api_key = os.getenv('GOOGLE_API_KEY')

        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable not set. "
                "Get a free API key at: https://aistudio.google.com/app/apikey"
            )

        self.genai_client = genai.Client(api_key=api_key)

        print(f"Initialized Gemini")
        print(f"  Model: {self.model}")

    def _load_vector_db(self):
        """Load the vector database."""
        self.vector_db = FinancialVectorDB(self.vector_db_path)
        print(f"Loaded vector database with {self.vector_db.index.ntotal} vectors")

    def _create_system_prompt(self) -> str:
        """Create system prompt for financial Q&A."""
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
- Include "Sources:" section at the end with full source details"""

    def _create_embedding(self, text: str) -> np.ndarray:
        """
        Create embedding for text query using Gemini.

        Args:
            text: Query text to embed

        Returns:
            768-dimensional embedding vector
        """
        response = self.genai_client.models.embed_content(
            model='text-embedding-004',
            contents=[text],
            config={
                'task_type': 'RETRIEVAL_QUERY',
                'output_dimensionality': 768
            }
        )

        vector = response.embeddings[0].values
        return np.array(vector, dtype=np.float32)

    def _extract_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Extract intent and filters from user query.

        Args:
            query: User's question

        Returns:
            Dictionary with query intent and filters
        """
        intent = {
            'original_query': query,
            'company_filter': None,
            'section_filter': None,
            'document_type_filter': None,
            'search_k': 5
        }

        query_lower = query.lower()

        # Find mentioned companies
        mentioned_companies = []
        for ticker, names in self.company_names.items():
            if ticker.lower() in query_lower or any(name in query_lower for name in names):
                mentioned_companies.append(ticker)

        if len(mentioned_companies) == 1:
            intent['company_filter'] = mentioned_companies[0]
        elif len(mentioned_companies) >= 2:
            intent['multi_company'] = mentioned_companies

        # Section filter detection
        if any(word in query_lower for word in ['risk', 'risks', 'uncertainty']):
            intent['section_filter'] = 'Risk Factors'
        elif any(word in query_lower for word in ['business', 'operations', 'model']):
            intent['section_filter'] = 'Business'

        return intent

    def _retrieve_context(self, query: str, intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks from vector database.

        Args:
            query: Original user query
            intent: Extracted intent with filters

        Returns:
            List of relevant chunks with metadata
        """
        query_embedding = self._create_embedding(query)

        # Multi-company query: retrieve from each company
        if intent.get('multi_company'):
            companies = intent['multi_company']
            chunks_per_company = intent['search_k']

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

            all_results.sort(key=lambda x: x.get('final_score', x.get('similarity_score', 0)), reverse=True)
            return all_results

        # Single company or no filter
        return self.vector_db.search(
            query_embedding,
            k=intent['search_k'],
            company_filter=intent.get('company_filter'),
            document_type_filter=intent.get('document_type_filter'),
            query_text=query
        )

    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into context for LLM."""
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            source_description = f"{chunk['company_ticker']} - 2023 Form 10-K Annual Report, {chunk['section_title']} section"

            context_parts.append(f"""
Document [{i}] - {source_description}:
{chunk['text']}
""")

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
        Generate response using Gemini with context.

        Args:
            query: User's original question
            context: Formatted context from retrieved chunks

        Returns:
            Response dictionary with answer and metadata
        """
        user_prompt = f"""
Based on the following financial documents, please answer this question:

QUESTION: {query}

CONTEXT:
{context}

Please provide a comprehensive answer with proper citations for all factual claims.
"""

        response = self.genai_client.models.generate_content(
            model=self.model,
            contents=[user_prompt],
            config={
                'system_instruction': self.system_prompt,
                'temperature': 0.1,
                'max_output_tokens': 2048,
                'top_p': 0.95
            }
        )

        answer = response.text
        estimated_tokens = int(len(answer.split()) * 1.3)

        return {
            'answer': answer,
            'model': self.model,
            'tokens_used': estimated_tokens,
            'timestamp': datetime.now().isoformat()
        }

    def _extract_citations(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract citation information from retrieved chunks."""
        citations = []

        for i, chunk in enumerate(chunks, 1):
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
        Main query method - complete RAG pipeline.

        Args:
            user_query: User's question about financial documents
            **kwargs: Optional overrides for filters

        Returns:
            Complete response with answer, citations, and metadata
        """
        print(f"Processing query: {user_query}")

        # Step 1: Extract intent
        intent = self._extract_query_intent(user_query)
        intent.update(kwargs)
        print(f"Query intent: {intent}")

        # Step 2: Retrieve context
        retrieved_chunks = self._retrieve_context(user_query, intent)
        print(f"Retrieved {len(retrieved_chunks)} relevant chunks")

        # Step 3: Format context
        formatted_context = self._format_context(retrieved_chunks)

        # Step 4: Generate response
        response = self._generate_response(user_query, formatted_context)

        # Step 5: Extract citations
        citations = self._extract_citations(retrieved_chunks)

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
