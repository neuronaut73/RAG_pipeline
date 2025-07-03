#!/usr/bin/env python3
"""
test_news_embeddings.py - Test News Embeddings in LanceDB

Simple script to demonstrate querying the news embeddings table
and performing semantic searches.
"""

import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer

def test_news_embeddings():
    """Test and demonstrate news embeddings functionality"""
    
    # Connect to LanceDB
    db = lancedb.connect("lancedb_store")
    table = db.open_table("news_embeddings")
    
    print("=" * 60)
    print("NEWS EMBEDDINGS TABLE TEST")
    print("=" * 60)
    
    # Basic table info
    print(f"Table size: {len(table)}")
    print(f"Schema columns: {len(table.schema)}")
    
    # Sample some records
    print("\nSample records:")
    sample_df = table.head(3).to_pandas()
    for idx, row in sample_df.iterrows():
        print(f"\nRecord {idx + 1}:")
        print(f"  ID: {row['id']}")
        print(f"  Ticker: {row['ticker']}")
        print(f"  Source: {row['source_type']}")
        print(f"  Headline: {row['headline'][:100]}...")
        print(f"  Text length: {row['text_length']}")
        print(f"  Chunk type: {row['chunk_type']}")
    
    # Test semantic search
    print("\n" + "=" * 60)
    print("SEMANTIC SEARCH TEST")
    print("=" * 60)
    
    # Load embedding model for query
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Example queries
    queries = [
        "financial results and earnings",
        "director trading and insider transactions", 
        "AGM annual general meeting",
        "share buyback and treasury shares"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 40)
        
        # Create query embedding
        query_vector = model.encode(query)
        
        # Search for similar content
        results = table.search(query_vector).limit(3).to_pandas()
        
        for i, row in results.iterrows():
            print(f"  Result {i+1}:")
            print(f"    Ticker: {row['ticker']}")
            print(f"    Headline: {row['headline'][:80]}...")
            print(f"    Source: {row['source_type']}")
            print(f"    Text: {row['chunk_text'][:100]}...")
            if 'stock_return_10d' in row and row['stock_return_10d'] != 0:
                print(f"    10d Return: {row['stock_return_10d']:.2f}%")
            print()
    
    # Statistics by source type
    print("=" * 60)
    print("DATA STATISTICS")
    print("=" * 60)
    
    full_df = table.to_pandas()
    
    print("\nRecords by source type:")
    print(full_df['source_type'].value_counts())
    
    print("\nRecords by ticker:")
    print(full_df['ticker'].value_counts().head(10))
    
    print("\nChunk types:")
    print(full_df['chunk_type'].value_counts())
    
    # Sentiment analysis (for enhanced news)
    enhanced_news = full_df[full_df['source_type'] == 'enhanced_news']
    if len(enhanced_news) > 0:
        print(f"\nEnhanced news sentiment scores:")
        print(f"  Mean: {enhanced_news['sentiment_score'].mean():.3f}")
        print(f"  Min: {enhanced_news['sentiment_score'].min():.3f}")
        print(f"  Max: {enhanced_news['sentiment_score'].max():.3f}")
    
    # Label enrichment statistics
    labeled_records = full_df[full_df['outperformance_10d'] != 0]
    if len(labeled_records) > 0:
        print(f"\nLabel enrichment:")
        print(f"  Records with labels: {len(labeled_records)}")
        print(f"  Mean 10d outperformance: {labeled_records['outperformance_10d'].mean():.3f}%")
        print(f"  Positive outperformance: {len(labeled_records[labeled_records['outperformance_10d'] > 0])}")
        print(f"  Negative outperformance: {len(labeled_records[labeled_records['outperformance_10d'] < 0])}")


if __name__ == "__main__":
    test_news_embeddings() 