#!/usr/bin/env python3
"""
agent_news.py - News Domain RAG Agent

Provides retrieval and query functions for the News domain.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import logging

import lancedb
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsAgent:
    """News Domain RAG Agent for intelligent news retrieval"""
    
    def __init__(
        self,
        lancedb_dir: str = "lancedb_store",
        table_name: str = "news_embeddings",
        embedding_model: str = "all-MiniLM-L6-v2",
        default_limit: int = 10
    ):
        self.lancedb_dir = Path(lancedb_dir)
        self.table_name = table_name
        self.default_limit = default_limit
        
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        self._connect_to_database()
        
    def _connect_to_database(self) -> None:
        """Connect to LanceDB and open the news table"""
        try:
            self.db = lancedb.connect(str(self.lancedb_dir))
            self.table = self.db.open_table(self.table_name)
            logger.info(f"Connected to table '{self.table_name}' with {len(self.table)} records")
        except Exception as e:
            logger.error(f"Failed to connect to LanceDB: {e}")
            raise
    
    def retrieve_by_setup_id(
        self, 
        setup_id: str, 
        limit: Optional[int] = None,
        include_labels: bool = True
    ) -> List[Dict[str, Any]]:
        """Retrieve news records by setup_id"""
        limit = limit or self.default_limit
        
        try:
            df = self.table.to_pandas()
            filtered_df = df[df['setup_id'] == setup_id].head(limit)
            
            if len(filtered_df) == 0:
                logger.warning(f"No records found for setup_id: {setup_id}")
                return []
            
            results = []
            for _, row in filtered_df.iterrows():
                result = self._format_result(row, include_labels=include_labels)
                result['query_type'] = 'setup_id'
                result['query_value'] = setup_id
                results.append(result)
            
            logger.info(f"Retrieved {len(results)} records for setup_id: {setup_id}")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving by setup_id '{setup_id}': {e}")
            return []
    
    def retrieve_by_ticker(
        self, 
        ticker: str, 
        limit: Optional[int] = None,
        source_type: Optional[str] = None,
        include_labels: bool = True
    ) -> List[Dict[str, Any]]:
        """Retrieve news records by ticker symbol"""
        limit = limit or self.default_limit
        
        try:
            df = self.table.to_pandas()
            filtered_df = df[df['ticker'] == ticker]
            
            if source_type:
                filtered_df = filtered_df[filtered_df['source_type'] == source_type]
            
            if 'rns_date' in filtered_df.columns:
                filtered_df = filtered_df.sort_values('rns_date', ascending=False)
            
            filtered_df = filtered_df.head(limit)
            
            if len(filtered_df) == 0:
                logger.warning(f"No records found for ticker: {ticker}")
                return []
            
            results = []
            for _, row in filtered_df.iterrows():
                result = self._format_result(row, include_labels=include_labels)
                result['query_type'] = 'ticker'
                result['query_value'] = ticker
                results.append(result)
            
            logger.info(f"Retrieved {len(results)} records for ticker: {ticker}")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving by ticker '{ticker}': {e}")
            return []
    
    def retrieve_by_text_query(
        self, 
        query: str, 
        limit: Optional[int] = None,
        ticker_filter: Optional[str] = None,
        include_labels: bool = True
    ) -> List[Dict[str, Any]]:
        """Perform semantic text search on news content"""
        limit = limit or self.default_limit
        
        try:
            query_vector = self.embedding_model.encode(query)
            search_results = self.table.search(query_vector).limit(limit * 2)
            df = search_results.to_pandas()
            
            if ticker_filter:
                df = df[df['ticker'] == ticker_filter]
            
            if '_distance' in df.columns:
                df['similarity_score'] = 1 / (1 + df['_distance'])
            else:
                df['similarity_score'] = 1.0
            
            df = df.head(limit)
            
            if len(df) == 0:
                logger.warning(f"No records found for query: '{query}'")
                return []
            
            results = []
            for _, row in df.iterrows():
                result = self._format_result(row, include_labels=include_labels)
                result['query_type'] = 'semantic_search'
                result['query_value'] = query
                result['similarity_score'] = float(row.get('similarity_score', 1.0))
                results.append(result)
            
            logger.info(f"Retrieved {len(results)} records for query: '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic search for query '{query}': {e}")
            return []
    
    def _format_result(self, row: pd.Series, include_labels: bool = True) -> Dict[str, Any]:
        """Format a database row into a structured result"""
        result = {
            'id': str(row.get('id', '')),
            'source_type': str(row.get('source_type', '')),
            'setup_id': str(row.get('setup_id', '')),
            'ticker': str(row.get('ticker', '')),
            'headline': str(row.get('headline', '')),
            'chunk_text': str(row.get('chunk_text', '')),
            'chunk_type': str(row.get('chunk_type', '')),
            'text_length': int(row.get('text_length', 0)),
        }
        
        # Source-specific metadata
        if row.get('source_type') == 'rns_announcement':
            result.update({
                'rns_date': str(row.get('rns_date', '')),
                'rns_time': str(row.get('rns_time', '')),
                'url': str(row.get('url', ''))
            })
        elif row.get('source_type') == 'enhanced_news':
            result.update({
                'publisher': str(row.get('publisher', '')),
                'sentiment_score': float(row.get('sentiment_score', 0.0)),
                'article_type': str(row.get('article_type', ''))
            })
        
        # Performance labels
        if include_labels and row.get('outperformance_10d', 0) != 0:
            result['performance_labels'] = {
                'stock_return_10d': float(row.get('stock_return_10d', 0.0)),
                'outperformance_10d': float(row.get('outperformance_10d', 0.0)),
                'days_outperformed_10d': int(row.get('days_outperformed_10d', 0))
            }
        
        result['retrieved_at'] = datetime.now().isoformat()
        return result


def run_unit_tests():
    """Unit tests for the News Agent"""
    print("=" * 60)
    print("NEWS AGENT UNIT TESTS")
    print("=" * 60)
    
    try:
        agent = NewsAgent()
        
        print("\nTEST 1: Retrieve by setup_id")
        print("-" * 40)
        setup_results = agent.retrieve_by_setup_id("LGEN_2025-05-29", limit=3)
        print(f"Results: {len(setup_results)}")
        if setup_results:
            print(f"First result ticker: {setup_results[0]['ticker']}")
        
        print("\nTEST 2: Retrieve by ticker")
        print("-" * 40)
        ticker_results = agent.retrieve_by_ticker("CNA", limit=5)
        print(f"Results: {len(ticker_results)}")
        if ticker_results:
            print(f"First headline: {ticker_results[0]['headline'][:50]}...")
        
        print("\nTEST 3: Semantic text query")
        print("-" * 40)
        semantic_results = agent.retrieve_by_text_query("earnings financial results", limit=3)
        print(f"Results: {len(semantic_results)}")
        if semantic_results:
            print(f"Top result similarity: {semantic_results[0].get('similarity_score', 'N/A')}")
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED ✅")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    run_unit_tests() 