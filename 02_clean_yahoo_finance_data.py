#!/usr/bin/env python3
"""
02_clean_market_data.py

Clean and extract essential information from Yahoo Finance raw data.
Reduces token usage by extracting only key metrics and organizing them efficiently.

Processes:
- Stock prices: Key metrics, trends, and technical indicators
- Financial statements: Core financial metrics only
- Ratios: Essential valuation and performance ratios
- Company info: Key business information
- Removes redundant data and focuses on chatbot-relevant information
"""

import os
import json
import yaml
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def create_output_directories() -> Dict[str, Dict[str, str]]:
    """Create output directories matching Yahoo Finance structure with time periods"""
    current_date = datetime.now()
    year = current_date.year
    quarter = f"Q{((current_date.month - 1) // 3) + 1}"
    
    # Annual data (financials, prices)
    annual_types = ['financials', 'prices']
    # Current/quarterly data (company_info, ratios, dividends, recommendations)
    quarterly_types = ['company_info', 'ratios', 'dividends', 'recommendations']
    
    # Step 1: Convert to text (like yahoo_finance_data -> yahoo_finance_txt)
    txt_dirs = {}
    for data_type in annual_types:
        txt_dirs[data_type] = os.path.join("yahoo_finance_txt", data_type, str(year))
        os.makedirs(txt_dirs[data_type], exist_ok=True)
    
    for data_type in quarterly_types:
        txt_dirs[data_type] = os.path.join("yahoo_finance_txt", data_type, f"{year}_{quarter}")
        os.makedirs(txt_dirs[data_type], exist_ok=True)
    
    # Step 2: Clean text (like yahoo_finance_txt -> yahoo_finance_txt_clean) 
    clean_dirs = {}
    for data_type in annual_types:
        clean_dirs[data_type] = os.path.join("yahoo_finance_txt_clean", data_type, str(year))
        os.makedirs(clean_dirs[data_type], exist_ok=True)
    
    for data_type in quarterly_types:
        clean_dirs[data_type] = os.path.join("yahoo_finance_txt_clean", data_type, f"{year}_{quarter}")
        os.makedirs(clean_dirs[data_type], exist_ok=True)
    
    return {'txt': txt_dirs, 'clean': clean_dirs}

def clean_stock_prices(ticker: str, csv_file: str) -> Dict[str, Any]:
    """
    Extract key price metrics and trends from CSV price data
    """
    try:
        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        
        if df.empty:
            return {}
        
        # Current metrics (most recent data)
        current = df.iloc[-1]
        
        # Performance metrics
        start_price = df['Close'].iloc[0]
        current_price = current['Close']
        ytd_return = ((current_price - start_price) / start_price) * 100
        
        # Volatility and ranges
        high_52w = df['Close'].max()
        low_52w = df['Close'].min()
        avg_volume = df['Volume'].mean()
        
        # Recent performance (30 days)
        recent_30d = df.tail(30)
        recent_return = ((recent_30d['Close'].iloc[-1] - recent_30d['Close'].iloc[0]) / recent_30d['Close'].iloc[0]) * 100
        
        # Technical indicators (if available)
        current_rsi = current.get('RSI', None)
        ma_20 = current.get('MA_20', None)
        ma_50 = current.get('MA_50', None)
        
        price_summary = {
            'ticker': ticker,
            'current_price': round(current_price, 2),
            'ytd_return_percent': round(ytd_return, 2),
            'recent_30d_return_percent': round(recent_return, 2),
            '52_week_high': round(high_52w, 2),
            '52_week_low': round(low_52w, 2),
            'avg_daily_volume': int(avg_volume),
            'current_rsi': round(current_rsi, 2) if current_rsi else None,
            'price_vs_ma20': round(((current_price - ma_20) / ma_20) * 100, 2) if ma_20 else None,
            'price_vs_ma50': round(((current_price - ma_50) / ma_50) * 100, 2) if ma_50 else None,
            'last_updated': df.index[-1].isoformat()[:10]  # Just the date
        }
        
        print(f"✓ Cleaned price data for {ticker}")
        return price_summary
        
    except Exception as e:
        print(f"❌ Error cleaning prices for {ticker}: {e}")
        return {}

def clean_financial_statements(ticker: str, financials_file: str) -> Dict[str, Any]:
    """
    Extract key financial metrics from financial statements
    """
    try:
        with open(financials_file, 'r') as f:
            data = json.load(f)
        
        financial_summary = {'ticker': ticker}
        
        # Extract key metrics from income statement
        if 'income_statement' in data and data['income_statement']:
            income = data['income_statement']
            
            # Get the most recent year (first key)
            recent_year = list(income.keys())[0] if income else None
            
            if recent_year:
                # Get the year's data
                year_data = income.get(recent_year, {})
                
                # Core income statement metrics
                metrics = {
                    'revenue_ttm': year_data.get('Total Revenue', year_data.get('Operating Revenue', 0)),
                    'gross_profit_ttm': year_data.get('Gross Profit', 0),
                    'operating_income_ttm': year_data.get('Operating Income', 0),
                    'net_income_ttm': year_data.get('Net Income', 0),
                    'ebitda_ttm': year_data.get('EBITDA', 0),
                }
                
                # Calculate margins if revenue exists
                revenue = metrics['revenue_ttm']
                if revenue and revenue > 0:
                    metrics['gross_margin_percent'] = round((metrics['gross_profit_ttm'] / revenue) * 100, 2)
                    metrics['operating_margin_percent'] = round((metrics['operating_income_ttm'] / revenue) * 100, 2)
                    metrics['net_margin_percent'] = round((metrics['net_income_ttm'] / revenue) * 100, 2)
                
                financial_summary.update(metrics)
        
        # Extract key metrics from balance sheet
        if 'balance_sheet' in data and data['balance_sheet']:
            balance = data['balance_sheet']
            recent_year = list(balance.keys())[0] if balance else None
            
            if recent_year:
                year_balance_data = balance.get(recent_year, {})
                
                balance_metrics = {
                    'total_assets': year_balance_data.get('Total Assets', 0),
                    'total_debt': year_balance_data.get('Total Debt', 0),
                    'total_equity': year_balance_data.get('Total Stockholder Equity', year_balance_data.get('Stockholders Equity', 0)),
                    'cash_and_equivalents': year_balance_data.get('Cash And Cash Equivalents', year_balance_data.get('Cash Cash Equivalents And Short Term Investments', 0)),
                }
                financial_summary.update(balance_metrics)
        
        # Extract key metrics from cash flow
        if 'cash_flow' in data and data['cash_flow']:
            cashflow = data['cash_flow']
            recent_year = list(cashflow.keys())[0] if cashflow else None
            
            if recent_year:
                year_cashflow_data = cashflow.get(recent_year, {})
                
                cashflow_metrics = {
                    'operating_cashflow_ttm': year_cashflow_data.get('Operating Cash Flow', 0),
                    'free_cashflow_ttm': year_cashflow_data.get('Free Cash Flow', 0),
                    'capex_ttm': year_cashflow_data.get('Capital Expenditure', year_cashflow_data.get('Capital Expenditures', 0)),
                }
                financial_summary.update(cashflow_metrics)
        
        # Convert large numbers to millions/billions for readability
        for key, value in financial_summary.items():
            if isinstance(value, (int, float)) and abs(value) > 1000000:
                if abs(value) > 1000000000:
                    financial_summary[key] = f"{value/1000000000:.2f}B"
                else:
                    financial_summary[key] = f"{value/1000000:.2f}M"
        
        print(f"✓ Cleaned financial statements for {ticker}")
        return financial_summary
        
    except Exception as e:
        print(f"❌ Error cleaning financials for {ticker}: {e}")
        return {}

def clean_ratios_and_valuation(ticker: str, ratios_file: str) -> Dict[str, Any]:
    """
    Extract essential valuation and performance ratios
    """
    try:
        with open(ratios_file, 'r') as f:
            data = json.load(f)
        
        # Key ratios for valuation and performance analysis
        key_ratios = {
            'ticker': ticker,
            'pe_ratio': data.get('trailingPE'),
            'forward_pe': data.get('forwardPE'),
            'price_to_book': data.get('priceToBook'), 
            'price_to_sales': data.get('priceToSalesTrailing12Months'),
            'enterprise_to_revenue': data.get('enterpriseToRevenue'),
            'enterprise_to_ebitda': data.get('enterpriseToEbitda'),
            'debt_to_equity': data.get('debtToEquity'),
            'return_on_assets_percent': round(data.get('returnOnAssets', 0) * 100, 2) if data.get('returnOnAssets') else None,
            'return_on_equity_percent': round(data.get('returnOnEquity', 0) * 100, 2) if data.get('returnOnEquity') else None,
            'gross_margins_percent': round(data.get('grossMargins', 0) * 100, 2) if data.get('grossMargins') else None,
            'operating_margins_percent': round(data.get('operatingMargins', 0) * 100, 2) if data.get('operatingMargins') else None,
            'profit_margins_percent': round(data.get('profitMargins', 0) * 100, 2) if data.get('profitMargins') else None,
            'dividend_yield_percent': round(data.get('dividendYield', 0) * 100, 2) if data.get('dividendYield') else None,
        }
        
        # Remove None values and round floats
        clean_ratios = {}
        for key, value in key_ratios.items():
            if value is not None:
                if isinstance(value, float):
                    clean_ratios[key] = round(value, 2)
                else:
                    clean_ratios[key] = value
        
        print(f"✓ Cleaned ratios for {ticker}")
        return clean_ratios
        
    except Exception as e:
        print(f"❌ Error cleaning ratios for {ticker}: {e}")
        return {}

def clean_company_info(ticker: str, info_file: str) -> Dict[str, Any]:
    """
    Extract essential company information
    """
    try:
        with open(info_file, 'r') as f:
            data = json.load(f)
        
        # Key company information for financial analysis
        company_summary = {
            'ticker': ticker,
            'company_name': data.get('longName', ''),
            'sector': data.get('sector', ''),
            'industry': data.get('industry', ''),
            'country': data.get('country', ''),
            'employees': data.get('employees', 0),
            'market_cap': format_large_number(data.get('market_cap', 0)),
            'enterprise_value': format_large_number(data.get('enterprise_value', 0)),
            'beta': data.get('beta'),
            'business_summary': data.get('business_summary', '')[:500] + "..." if len(data.get('business_summary', '')) > 500 else data.get('business_summary', ''),  # Truncate long descriptions
        }
        
        print(f"✓ Cleaned company info for {ticker}")
        return company_summary
        
    except Exception as e:
        print(f"❌ Error cleaning company info for {ticker}: {e}")
        return {}

def format_large_number(number: float) -> str:
    """Format large numbers into readable format (M/B/T)"""
    if not number or number == 0:
        return "0"
    
    if abs(number) >= 1e12:
        return f"{number/1e12:.2f}T"
    elif abs(number) >= 1e9:
        return f"{number/1e9:.2f}B"
    elif abs(number) >= 1e6:
        return f"{number/1e6:.2f}M"
    else:
        return f"{number:,.0f}"

def create_financial_text(ticker: str, financial_data: Dict, year: int) -> str:
    """Create raw financial text content"""
    content = []
    content.append(f"FINANCIAL STATEMENTS - {ticker} ({year})")
    content.append("=" * 50)
    content.append("")
    
    if financial_data:
        content.append("INCOME STATEMENT METRICS (TTM)")
        content.append("-" * 30)
        content.append(f"Total Revenue: {financial_data.get('revenue_ttm', 'N/A')}")
        content.append(f"Gross Profit: {financial_data.get('gross_profit_ttm', 'N/A')}")
        content.append(f"Operating Income: {financial_data.get('operating_income_ttm', 'N/A')}")
        content.append(f"Net Income: {financial_data.get('net_income_ttm', 'N/A')}")
        content.append(f"EBITDA: {financial_data.get('ebitda_ttm', 'N/A')}")
        content.append("")
        
        content.append("PROFITABILITY MARGINS")
        content.append("-" * 20)
        content.append(f"Gross Margin: {financial_data.get('gross_margin_percent', 'N/A')}%")
        content.append(f"Operating Margin: {financial_data.get('operating_margin_percent', 'N/A')}%")
        content.append(f"Net Margin: {financial_data.get('net_margin_percent', 'N/A')}%")
        content.append("")
        
        content.append("BALANCE SHEET METRICS")
        content.append("-" * 22)
        content.append(f"Total Assets: {financial_data.get('total_assets', 'N/A')}")
        content.append(f"Total Debt: {financial_data.get('total_debt', 'N/A')}")
        content.append(f"Total Equity: {financial_data.get('total_equity', 'N/A')}")
        content.append(f"Cash and Equivalents: {financial_data.get('cash_and_equivalents', 'N/A')}")
        content.append("")
        
        content.append("CASH FLOW METRICS")
        content.append("-" * 18)
        content.append(f"Operating Cash Flow: {financial_data.get('operating_cashflow_ttm', 'N/A')}")
        content.append(f"Free Cash Flow: {financial_data.get('free_cashflow_ttm', 'N/A')}")
        content.append(f"Capital Expenditure: {financial_data.get('capex_ttm', 'N/A')}")
    
    return "\n".join(content)

def create_price_text(ticker: str, price_data: Dict, year: int) -> str:
    """Create raw price text content"""
    content = []
    content.append(f"MARKET PERFORMANCE - {ticker} ({year})")
    content.append("=" * 50)
    content.append("")
    
    if price_data:
        content.append("CURRENT PRICE METRICS")
        content.append("-" * 22)
        content.append(f"Current Price: ${price_data.get('current_price', 'N/A')}")
        content.append(f"52-Week High: ${price_data.get('52_week_high', 'N/A')}")
        content.append(f"52-Week Low: ${price_data.get('52_week_low', 'N/A')}")
        content.append("")
        
        content.append("PERFORMANCE RETURNS")
        content.append("-" * 18)
        content.append(f"Year-to-Date Return: {price_data.get('ytd_return_percent', 'N/A')}%")
        content.append(f"30-Day Return: {price_data.get('recent_30d_return_percent', 'N/A')}%")
        content.append("")
        
        content.append("TECHNICAL INDICATORS")
        content.append("-" * 19)
        content.append(f"RSI: {price_data.get('current_rsi', 'N/A')}")
        content.append(f"Price vs 20-day MA: {price_data.get('price_vs_ma20', 'N/A')}%")
        content.append(f"Price vs 50-day MA: {price_data.get('price_vs_ma50', 'N/A')}%")
        content.append("")
        
        content.append("TRADING VOLUME")
        content.append("-" * 14)
        content.append(f"Average Daily Volume: {price_data.get('avg_daily_volume', 'N/A'):,}")
    
    return "\n".join(content)

def create_ratios_text(ticker: str, ratios_data: Dict, period: str) -> str:
    """Create raw ratios text content"""
    content = []
    content.append(f"VALUATION RATIOS - {ticker} ({period})")
    content.append("=" * 50)
    content.append("")
    
    if ratios_data:
        content.append("VALUATION MULTIPLES")
        content.append("-" * 19)
        content.append(f"Price-to-Earnings (P/E): {ratios_data.get('pe_ratio', 'N/A')}")
        content.append(f"Forward P/E: {ratios_data.get('forward_pe', 'N/A')}")
        content.append(f"Price-to-Book (P/B): {ratios_data.get('price_to_book', 'N/A')}")
        content.append(f"Price-to-Sales (P/S): {ratios_data.get('price_to_sales', 'N/A')}")
        content.append(f"Enterprise Value/Revenue: {ratios_data.get('enterprise_to_revenue', 'N/A')}")
        content.append(f"Enterprise Value/EBITDA: {ratios_data.get('enterprise_to_ebitda', 'N/A')}")
        content.append("")
        
        content.append("PROFITABILITY RATIOS")
        content.append("-" * 19)
        content.append(f"Return on Assets (ROA): {ratios_data.get('return_on_assets_percent', 'N/A')}%")
        content.append(f"Return on Equity (ROE): {ratios_data.get('return_on_equity_percent', 'N/A')}%")
        content.append(f"Gross Margins: {ratios_data.get('gross_margins_percent', 'N/A')}%")
        content.append(f"Operating Margins: {ratios_data.get('operating_margins_percent', 'N/A')}%")
        content.append(f"Profit Margins: {ratios_data.get('profit_margins_percent', 'N/A')}%")
        content.append("")
        
        content.append("FINANCIAL HEALTH")
        content.append("-" * 16)
        content.append(f"Debt-to-Equity: {ratios_data.get('debt_to_equity', 'N/A')}")
        content.append(f"Dividend Yield: {ratios_data.get('dividend_yield_percent', 'N/A')}%")
    
    return "\n".join(content)

def create_company_text(ticker: str, company_data: Dict, period: str) -> str:
    """Create raw company info text content"""
    content = []
    content.append(f"COMPANY INFORMATION - {ticker} ({period})")
    content.append("=" * 50)
    content.append("")
    
    if company_data:
        content.append("BASIC INFORMATION")
        content.append("-" * 17)
        content.append(f"Company Name: {company_data.get('company_name', 'N/A')}")
        content.append(f"Sector: {company_data.get('sector', 'N/A')}")
        content.append(f"Industry: {company_data.get('industry', 'N/A')}")
        content.append(f"Country: {company_data.get('country', 'N/A')}")
        content.append(f"Employees: {company_data.get('employees', 'N/A'):,}")
        content.append("")
        
        content.append("MARKET METRICS")
        content.append("-" * 14)
        content.append(f"Market Capitalization: {company_data.get('market_cap', 'N/A')}")
        content.append(f"Enterprise Value: {company_data.get('enterprise_value', 'N/A')}")
        content.append(f"Beta: {company_data.get('beta', 'N/A')}")
        content.append("")
        
        content.append("BUSINESS DESCRIPTION")
        content.append("-" * 19)
        content.append(company_data.get('business_summary', 'No business summary available.'))
    
    return "\n".join(content)

def clean_financial_text(raw_text: str) -> str:
    """Clean financial text for chatbot consumption"""
    # Remove excessive formatting, normalize spacing
    lines = raw_text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if line and not line.startswith('=') and not line.startswith('-'):
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def clean_price_text(raw_text: str) -> str:
    """Clean price text for chatbot consumption"""
    lines = raw_text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if line and not line.startswith('=') and not line.startswith('-'):
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def clean_ratios_text(raw_text: str) -> str:
    """Clean ratios text for chatbot consumption"""
    lines = raw_text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if line and not line.startswith('=') and not line.startswith('-'):
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def clean_company_text(raw_text: str) -> str:
    """Clean company text for chatbot consumption"""
    lines = raw_text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if line and not line.startswith('=') and not line.startswith('-'):
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def create_comprehensive_summary(ticker: str, price_data: Dict, financial_data: Dict, ratios_data: Dict, company_data: Dict) -> str:
    """
    Create a comprehensive, human-readable summary for the chatbot
    """
    summary_parts = []
    
    # Company header
    company_name = company_data.get('company_name', ticker)
    sector = company_data.get('sector', 'Unknown')
    summary_parts.append(f"Company: {company_name} ({ticker})")
    summary_parts.append(f"Sector: {sector} | Industry: {company_data.get('industry', 'Unknown')}")
    summary_parts.append(f"Employees: {company_data.get('employees', 0):,}")
    
    # Market performance
    if price_data:
        summary_parts.append(f"\nMarket Performance:")
        summary_parts.append(f"Current Price: ${price_data.get('current_price', 0)}")
        summary_parts.append(f"YTD Return: {price_data.get('ytd_return_percent', 0)}%")
        summary_parts.append(f"52-Week Range: ${price_data.get('52_week_low', 0)} - ${price_data.get('52_week_high', 0)}")
    
    # Financial metrics
    if financial_data:
        summary_parts.append(f"\nFinancial Metrics (TTM):")
        summary_parts.append(f"Revenue: {financial_data.get('revenue_ttm', 'N/A')}")
        summary_parts.append(f"Net Income: {financial_data.get('net_income_ttm', 'N/A')}")
        summary_parts.append(f"Operating Margin: {financial_data.get('operating_margin_percent', 'N/A')}%")
        summary_parts.append(f"Net Margin: {financial_data.get('net_margin_percent', 'N/A')}%")
    
    # Valuation
    if ratios_data:
        summary_parts.append(f"\nValuation Metrics:")
        summary_parts.append(f"P/E Ratio: {ratios_data.get('pe_ratio', 'N/A')}")
        summary_parts.append(f"P/B Ratio: {ratios_data.get('price_to_book', 'N/A')}")
        summary_parts.append(f"ROE: {ratios_data.get('return_on_equity_percent', 'N/A')}%")
        summary_parts.append(f"Debt/Equity: {ratios_data.get('debt_to_equity', 'N/A')}")
    
    # Business summary
    if company_data.get('business_summary'):
        summary_parts.append(f"\nBusiness Overview:")
        summary_parts.append(company_data['business_summary'])
    
    return "\n".join(summary_parts)

def main():
    """Main function to clean all Yahoo Finance data"""
    print("Starting Yahoo Finance data cleaning...")
    
    config = load_config()
    tickers = config.get('tickers', [])
    
    if not tickers:
        print("❌ No tickers found in config.yaml")
        return
    
    # Create output directories following SEC pattern
    output_dirs = create_output_directories()
    
    # Input directories (organized by time period)
    current_date = datetime.now()
    year = current_date.year
    quarter = f"Q{((current_date.month - 1) // 3) + 1}"
    
    raw_data_base = "yahoo_finance_data"
    
    if not os.path.exists(raw_data_base):
        print(f"❌ Raw Yahoo Finance data not found: {raw_data_base}")
        print("Run 01c_download_yahoo_finance.py first")
        return
    
    print(f"Cleaning data for {len(tickers)} companies...")
    print("-" * 60)
    
    summary_report = {
        'cleaned_at': datetime.now().isoformat(),
        'companies_processed': [],
        'total_size_reduction': 0
    }
    
    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        
        try:
            # File paths in organized structure
            price_file = os.path.join(raw_data_base, 'prices', str(year), f"{ticker}_prices_2y.csv")
            financials_file = os.path.join(raw_data_base, 'financials', str(year), f"{ticker}_financials.json")
            ratios_file = os.path.join(raw_data_base, 'ratios', f"{year}_{quarter}", f"{ticker}_ratios.json")
            info_file = os.path.join(raw_data_base, 'company_info', f"{year}_{quarter}", f"{ticker}_info.json")
            
            # Clean each data type
            price_data = clean_stock_prices(ticker, price_file) if os.path.exists(price_file) else {}
            financial_data = clean_financial_statements(ticker, financials_file) if os.path.exists(financials_file) else {}
            ratios_data = clean_ratios_and_valuation(ticker, ratios_file) if os.path.exists(ratios_file) else {}
            company_data = clean_company_info(ticker, info_file) if os.path.exists(info_file) else {}
            
            # Step 1: Create raw text files by data type
            # Financial data
            financial_txt_content = create_financial_text(ticker, financial_data, year)
            financial_txt_file = os.path.join(output_dirs['txt']['financials'], f"{ticker}_financials.txt")
            with open(financial_txt_file, 'w') as f:
                f.write(financial_txt_content)
            
            # Price data
            price_txt_content = create_price_text(ticker, price_data, year)
            price_txt_file = os.path.join(output_dirs['txt']['prices'], f"{ticker}_prices.txt")
            with open(price_txt_file, 'w') as f:
                f.write(price_txt_content)
            
            # Ratios
            ratios_txt_content = create_ratios_text(ticker, ratios_data, f"{year}_{quarter}")
            ratios_txt_file = os.path.join(output_dirs['txt']['ratios'], f"{ticker}_ratios.txt")
            with open(ratios_txt_file, 'w') as f:
                f.write(ratios_txt_content)
            
            # Company info
            company_txt_content = create_company_text(ticker, company_data, f"{year}_{quarter}")
            company_txt_file = os.path.join(output_dirs['txt']['company_info'], f"{ticker}_info.txt")
            with open(company_txt_file, 'w') as f:
                f.write(company_txt_content)
            
            # Step 2: Create clean text files
            # Financial clean
            financial_clean_content = clean_financial_text(financial_txt_content)
            financial_clean_file = os.path.join(output_dirs['clean']['financials'], f"{ticker}_financials.clean.txt")
            with open(financial_clean_file, 'w') as f:
                f.write(financial_clean_content)
                
            # Price clean
            price_clean_content = clean_price_text(price_txt_content)  
            price_clean_file = os.path.join(output_dirs['clean']['prices'], f"{ticker}_prices.clean.txt")
            with open(price_clean_file, 'w') as f:
                f.write(price_clean_content)
            
            # Ratios clean
            ratios_clean_content = clean_ratios_text(ratios_txt_content)
            ratios_clean_file = os.path.join(output_dirs['clean']['ratios'], f"{ticker}_ratios.clean.txt")  
            with open(ratios_clean_file, 'w') as f:
                f.write(ratios_clean_content)
                
            # Company info clean
            company_clean_content = clean_company_text(company_txt_content)
            company_clean_file = os.path.join(output_dirs['clean']['company_info'], f"{ticker}_info.clean.txt")
            with open(company_clean_file, 'w') as f:
                f.write(company_clean_content)
            
            summary_report['companies_processed'].append(ticker)
            print(f"✅ Successfully cleaned data for {ticker}")
            
        except Exception as e:
            print(f"❌ Failed to clean data for {ticker}: {e}")
    
    # Save cleaning summary to main yahoo finance directory
    os.makedirs("yahoo_finance_txt_clean", exist_ok=True)
    summary_file = os.path.join("yahoo_finance_txt_clean", "cleaning_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary_report, f, indent=2)
    
    current_date = datetime.now()
    year = current_date.year
    quarter = f"Q{((current_date.month - 1) // 3) + 1}"
    
    print(f"\n{'='*60}")
    print("Yahoo Finance data cleaning complete!")
    print(f"✅ Processed: {len(summary_report['companies_processed'])} companies")
    print(f"📁 Raw text: yahoo_finance_txt/")
    print(f"  - financials/{year}/ prices/{year}/ (annual)")
    print(f"  - company_info/{year}_{quarter}/ ratios/{year}_{quarter}/ (current)")
    print(f"📁 Clean text: yahoo_finance_txt_clean/") 
    print(f"  - financials/{year}/ prices/{year}/ (annual)")
    print(f"  - company_info/{year}_{quarter}/ ratios/{year}_{quarter}/ (current)")
    print(f"📊 Following SEC filing pattern: raw -> .txt -> .clean.txt")
    print("\nReady for chunking pipeline! 🚀")

if __name__ == "__main__":
    main()