#!/usr/bin/env python3
"""
agent_fundamentals.py - Fundamentals Domain RAG Agent

Provides retrieval and query functions for the Fundamentals domain.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

import lancedb
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FundamentalsAgent:
    """Fundamentals Domain RAG Agent for intelligent financial data retrieval"""
    
    def __init__(
        self,
        lancedb_dir: str = "../lancedb_store",
        table_name: str = "fundamentals_embeddings",
        embedding_model: str = "all-MiniLM-L6-v2",
        default_limit: int = 10
    ):
        self.lancedb_dir = Path(lancedb_dir)
        self.table_name = table_name
        self.default_limit = default_limit
        
        # Connect to LanceDB
        self.db = lancedb.connect(str(self.lancedb_dir))
        
        try:
            self.table = self.db.open_table(table_name)
            logger.info(f"Connected to table '{table_name}' with {len(self.table)} records")
        except Exception as e:
            logger.error(f"Failed to open table '{table_name}': {e}")
            raise
        
        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
    def retrieve_by_setup_id(
        self, 
        setup_id: str, 
        limit: int = None,
        include_labels: bool = True
    ) -> List[Dict[str, Any]]:
        """Retrieve fundamentals records by setup_id"""
        limit = limit or self.default_limit
        
        try:
            results_df = self.table.search().where(f"setup_id = '{setup_id}'").limit(limit).to_pandas()
            
            if results_df.empty:
                logger.warning(f"No fundamentals records found for setup_id: {setup_id}")
                return []
            
            records = self._format_results(results_df, include_labels)
            logger.info(f"Retrieved {len(records)} fundamentals records for setup_id: {setup_id}")
            return records
            
        except Exception as e:
            logger.error(f"Error retrieving by setup_id '{setup_id}': {e}")
            return []
    
    def retrieve_by_ticker(
        self,
        ticker: str,
        limit: int = None,
        report_type: Optional[str] = None,
        period_start: Optional[str] = None,
        period_end: Optional[str] = None,
        include_labels: bool = True
    ) -> List[Dict[str, Any]]:
        """Retrieve fundamentals records by ticker symbol"""
        limit = limit or self.default_limit
        
        try:
            conditions = [f"ticker = '{ticker}'"]
            
            if report_type:
                conditions.append(f"report_type = '{report_type}'")
            if period_start:
                conditions.append(f"period_end >= '{period_start}'")
            if period_end:
                conditions.append(f"period_end <= '{period_end}'")
            
            where_clause = " AND ".join(conditions)
            results_df = self.table.search().where(where_clause).limit(limit).to_pandas()
            
            if results_df.empty:
                logger.warning(f"No fundamentals records found for ticker: {ticker}")
                return []
            
            results_df = results_df.sort_values('period_end', ascending=False)
            records = self._format_results(results_df, include_labels)
            
            logger.info(f"Retrieved {len(records)} fundamentals records for ticker: {ticker}")
            return records
            
        except Exception as e:
            logger.error(f"Error retrieving by ticker '{ticker}': {e}")
            return []
    
    def retrieve_by_date_range(
        self,
        start_date: str,
        end_date: str,
        limit: int = None,
        report_type: Optional[str] = None,
        include_labels: bool = True
    ) -> List[Dict[str, Any]]:
        """Retrieve fundamentals records within date range"""
        limit = limit or self.default_limit
        
        try:
            conditions = [
                f"period_end >= '{start_date}'",
                f"period_end <= '{end_date}'"
            ]
            
            if report_type:
                conditions.append(f"report_type = '{report_type}'")
            
            where_clause = " AND ".join(conditions)
            results_df = self.table.search().where(where_clause).limit(limit).to_pandas()
            
            if results_df.empty:
                logger.warning(f"No fundamentals records found for date range: {start_date} to {end_date}")
                return []
            
            results_df = results_df.sort_values('period_end', ascending=False)
            records = self._format_results(results_df, include_labels)
            
            logger.info(f"Retrieved {len(records)} fundamentals records for date range: {start_date} to {end_date}")
            return records
            
        except Exception as e:
            logger.error(f"Error retrieving by date range '{start_date}' to '{end_date}': {e}")
            return []
    
    def semantic_search(
        self,
        query: str,
        limit: int = None,
        filter_conditions: Optional[str] = None,
        include_labels: bool = True
    ) -> List[Dict[str, Any]]:
        """Perform semantic search on financial summaries"""
        limit = limit or self.default_limit
        
        try:
            query_vector = self.embedding_model.encode(query)
            search = self.table.search(query_vector).limit(limit)
            
            if filter_conditions:
                search = search.where(filter_conditions)
            
            results_df = search.to_pandas()
            
            if results_df.empty:
                logger.warning(f"No fundamentals records found for query: {query}")
                return []
            
            records = self._format_results(results_df, include_labels, include_similarity=True)
            
            logger.info(f"Retrieved {len(records)} fundamentals records for semantic query: '{query}'")
            return records
            
        except Exception as e:
            logger.error(f"Error in semantic search for query '{query}': {e}")
            return []
    
    def find_similar_companies(
        self,
        reference_ticker: str,
        reference_period: Optional[str] = None,
        limit: int = None,
        exclude_same_company: bool = True,
        include_labels: bool = True
    ) -> List[Dict[str, Any]]:
        """Find companies with similar financial profiles"""
        limit = limit or self.default_limit
        
        try:
            ref_conditions = f"ticker = '{reference_ticker}'"
            if reference_period:
                ref_conditions += f" AND period_end = '{reference_period}'"
            
            ref_df = self.table.search().where(ref_conditions).limit(1).to_pandas()
            
            if ref_df.empty:
                logger.warning(f"Reference company not found: {reference_ticker}")
                return []
            
            ref_summary = ref_df.iloc[0]['financial_summary']
            search = self.table.search(self.embedding_model.encode(ref_summary)).limit(limit + 1)
            
            if exclude_same_company:
                search = search.where(f"ticker != '{reference_ticker}'")
            
            results_df = search.to_pandas()
            
            if results_df.empty:
                logger.warning(f"No similar companies found for: {reference_ticker}")
                return []
            
            records = self._format_results(results_df, include_labels, include_similarity=True)
            
            logger.info(f"Found {len(records)} similar companies to {reference_ticker}")
            return records
            
        except Exception as e:
            logger.error(f"Error finding similar companies to '{reference_ticker}': {e}")
            return []
    
    def analyze_performance_by_fundamentals(
        self,
        metric_filter: str,
        limit: int = None,
        sort_by_performance: bool = True
    ) -> List[Dict[str, Any]]:
        """Analyze companies based on fundamental metrics and performance"""
        limit = limit or self.default_limit
        
        try:
            where_clause = f"has_performance_labels = true AND {metric_filter}"
            results_df = self.table.search().where(where_clause).limit(limit).to_pandas()
            
            if results_df.empty:
                logger.warning(f"No companies found matching criteria: {metric_filter}")
                return []
            
            if sort_by_performance and 'outperformance_10d' in results_df.columns:
                results_df = results_df.sort_values('outperformance_10d', ascending=False)
            
            records = self._format_results(results_df, include_labels=True)
            
            logger.info(f"Analyzed {len(records)} companies with criteria: {metric_filter}")
            return records
            
        except Exception as e:
            logger.error(f"Error analyzing performance by fundamentals '{metric_filter}': {e}")
            return []
    
    def _format_results(
        self, 
        results_df: pd.DataFrame, 
        include_labels: bool = True,
        include_similarity: bool = False
    ) -> List[Dict[str, Any]]:
        """Format query results into standardized output"""
        records = []
        
        for idx, row in results_df.iterrows():
            record = {
                'id': row.get('id', ''),
                'ticker': row.get('ticker', ''),
                'report_type': row.get('report_type', ''),
                'period_end': row.get('period_end', ''),
                'financial_summary': row.get('financial_summary', ''),
                
                # Core financial metrics
                'revenue': float(row.get('revenue', 0)),
                'net_income': float(row.get('net_income', 0)),
                'total_assets': float(row.get('total_assets', 0)),
                
                # Key ratios
                'roe': float(row.get('roe', 0)),
                'roa': float(row.get('roa', 0)),
                'debt_to_equity': float(row.get('debt_to_equity', 0)),
                'current_ratio': float(row.get('current_ratio', 0)),
                'pe_ratio': float(row.get('pe_ratio', 0)),
                
                'embedded_at': row.get('embedded_at', ''),
                'has_performance_labels': bool(row.get('has_performance_labels', False))
            }
            
            # Include performance labels if available
            if include_labels and record['has_performance_labels']:
                record.update({
                    'setup_id': row.get('setup_id', ''),
                    'stock_return_10d': float(row.get('stock_return_10d', 0)),
                    'outperformance_10d': float(row.get('outperformance_10d', 0)),
                    'days_outperformed_10d': int(row.get('days_outperformed_10d', 0))
                })
            
            # Include similarity score if available
            if include_similarity and '_distance' in row:
                record['similarity_score'] = 1.0 - float(row['_distance'])
            
            records.append(record)
        
        return records
    
    def get_table_stats(self) -> Dict[str, Any]:
        """Get statistics about the fundamentals table"""
        try:
            full_df = self.table.to_pandas()
            
            stats = {
                'total_records': len(full_df),
                'unique_tickers': full_df['ticker'].nunique(),
                'tickers': list(full_df['ticker'].unique()),
                'report_types': list(full_df['report_type'].unique()),
                'date_range': {
                    'earliest': full_df['period_end'].min(),
                    'latest': full_df['period_end'].max()
                },
                'records_with_labels': len(full_df[full_df['has_performance_labels'] == True]),
                'label_percentage': len(full_df[full_df['has_performance_labels'] == True]) / len(full_df) * 100,
                'records_with_revenue': len(full_df[full_df['revenue'] > 0]),
                'records_with_roe': len(full_df[full_df['roe'] != 0])
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting table stats: {e}")
            return {}


def run_fundamentals_agent_tests():
    """Run comprehensive tests of the Fundamentals Agent"""
    
    print("=" * 60)
    print("FUNDAMENTALS AGENT UNIT TESTS")
    print("=" * 60)
    
    # Initialize agent
    agent = FundamentalsAgent()
    
    # Test 1: Table stats
    print("\nTest 1: Table Connection and Statistics")
    stats = agent.get_table_stats()
    print(f"  Total records: {stats.get('total_records', 0)}")
    print(f"  Unique tickers: {stats.get('unique_tickers', 0)}")
    print(f"  Records with labels: {stats.get('records_with_labels', 0)}")
    
    # Test 2: Retrieve by ticker
    print("\nTest 2: Retrieve by Ticker")
    ticker_results = agent.retrieve_by_ticker("LLOY.L", limit=3)
    
    if ticker_results:
        print(f"  Found {len(ticker_results)} records for LLOY.L")
        for i, record in enumerate(ticker_results[:2]):
            print(f"    {record['ticker']} {record['period_end']}: ROE {record['roe']:.2%}")
    
    # Test 3: Semantic search
    print("\nTest 3: Semantic Search - Profitable Companies")
    profitable_results = agent.semantic_search(
        "profitable companies with strong returns and good margins",
        limit=3
    )
    
    if profitable_results:
        print(f"  Found {len(profitable_results)} profitable companies")
        for record in profitable_results:
            print(f"    {record['ticker']}: ROE {record['roe']:.2%}")
    
    # Test 4: Date range
    print("\nTest 4: Date Range Query")
    date_results = agent.retrieve_by_date_range("2024-01-01", "2024-12-31", limit=3)
    
    if date_results:
        print(f"  Found {len(date_results)} records for 2024")
        for record in date_results:
            print(f"    {record['ticker']} {record['period_end']}")
    
    # Test 5: Performance analysis
    print("\nTest 5: Performance Analysis")
    performance_results = agent.analyze_performance_by_fundamentals("roe > 0.10", limit=3)
    
    if performance_results:
        print(f"  Found {len(performance_results)} companies with ROE > 10%")
        for record in performance_results:
            if record['has_performance_labels']:
                print(f"    {record['ticker']}: ROE {record['roe']:.2%}, Outperformance {record['outperformance_10d']:.2f}%")
    
    print("\n" + "=" * 60)
    print("FUNDAMENTALS AGENT TESTS COMPLETED ✅")
    print("=" * 60)


if __name__ == "__main__":
    run_fundamentals_agent_tests() 