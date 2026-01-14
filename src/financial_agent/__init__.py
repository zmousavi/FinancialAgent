"""
Financial Agent - RAG-powered agent for SEC financial filings.

Usage:
    from financial_agent import FinancialAgent

    agent = FinancialAgent("path/to/vector_db")
    result = agent.query("What are Apple's revenues?")
    print(result["answer"])
"""

from .agent import FinancialAgent
from .vector_db import FinancialVectorDB

__all__ = ["FinancialAgent", "FinancialVectorDB"]
__version__ = "0.1.0"
