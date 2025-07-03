#!/usr/bin/env python3
"""
test_fundamentals_embeddings.py - Test Fundamentals Embeddings

Demonstrates functionality of the fundamentals embeddings LanceDB table
with sample queries and analysis.
"""

import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer

def test_fundamentals_embeddings():
    """Test and demonstrate fundamentals embeddings functionality"""
    
    # Connect to LanceDB
    db = lancedb.connect("lancedb_store")
    table = db.open_table("fundamentals_embeddings")
    
    print("=" * 60)
    print("FUNDAMENTALS EMBEDDINGS TABLE TEST")
    print("=" * 60)
    
    # Basic table info
    print(f"Table size: {len(table)}")
    
    # Sample some records
    print("\nSample records:")
    sample_df = table.head(3).to_pandas()
    for idx, row in sample_df.iterrows():
        print(f"\nRecord {idx + 1}:")
        print(f"  Ticker: {row['ticker']}")
        print(f"  Report Type: {row['report_type']}")
        print(f"  Period: {row['period_end']}")
        print(f"  Revenue: ${row['revenue']:,.0f}")
        print(f"  ROE: {row['roe']:.2%}")
        print(f"  Has Performance Labels: {row['has_performance_labels']}")
        if row['has_performance_labels']:
            print(f"  Outperformance: {row['outperformance_10d']:.2f}%")
        print(f"  Summary (first 100 chars): {row['financial_summary'][:100]}...")
    
    print("\n" + "=" * 40)
    print("SEMANTIC SEARCH EXAMPLES")
    print("=" * 40)
    
    # Initialize embedding model for queries
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Test 1: Search for profitable companies
    print("\nTest 1: Search for 'profitable companies with strong margins'")
    query1 = "profitable companies with strong margins and good returns"
    query1_vector = embedding_model.encode(query1)
    results1 = table.search(query1_vector).limit(3).to_pandas()
    
    for idx, row in results1.iterrows():
        print(f"  Result {idx + 1}: {row['ticker']} ({row['period_end']})")
        print(f"    ROE: {row['roe']:.2%}, Revenue: ${row['revenue']:,.0f}")
        print(f"    Summary: {row['financial_summary'][:100]}...")
        print()
    
    # Test 2: Search for companies with debt concerns
    print("\nTest 2: Search for 'companies with high debt levels'")
    query2 = "companies with high debt levels and leverage concerns"
    query2_vector = embedding_model.encode(query2)
    results2 = table.search(query2_vector).limit(3).to_pandas()
    
    for idx, row in results2.iterrows():
        print(f"  Result {idx + 1}: {row['ticker']} ({row['period_end']})")
        print(f"    D/E Ratio: {row['debt_to_equity']:.2f}, Current Ratio: {row['current_ratio']:.2f}")
        print(f"    Summary: {row['financial_summary'][:100]}...")
        print()
    
    # Test 3: Search for undervalued companies
    print("\nTest 3: Search for 'undervalued companies trading below book value'")
    query3 = "undervalued companies trading below book value with attractive valuation"
    query3_vector = embedding_model.encode(query3)
    results3 = table.search(query3_vector).limit(3).to_pandas()
    
    for idx, row in results3.iterrows():
        print(f"  Result {idx + 1}: {row['ticker']} ({row['period_end']})")
        print(f"    PE Ratio: {row['pe_ratio']:.1f}, Revenue: ${row['revenue']:,.0f}")
        print(f"    Summary: {row['financial_summary'][:100]}...")
        print()
    
    print("\n" + "=" * 40)
    print("PERFORMANCE ANALYSIS")
    print("=" * 40)
    
    # Analyze performance labels
    full_df = table.to_pandas()
    labeled_df = full_df[full_df['has_performance_labels'] == True]
    
    print(f"\nRecords with performance labels: {len(labeled_df)}")
    print(f"Average outperformance: {labeled_df['outperformance_10d'].mean():.2f}%")
    print(f"Best performer: {labeled_df.loc[labeled_df['outperformance_10d'].idxmax(), 'ticker']} "
          f"({labeled_df['outperformance_10d'].max():.2f}%)")
    print(f"Worst performer: {labeled_df.loc[labeled_df['outperformance_10d'].idxmin(), 'ticker']} "
          f"({labeled_df['outperformance_10d'].min():.2f}%)")
    
    # ROE vs Performance correlation
    if len(labeled_df) > 5:
        correlation = labeled_df['roe'].corr(labeled_df['outperformance_10d'])
        print(f"\nROE vs Outperformance correlation: {correlation:.3f}")
    
    print("\n" + "=" * 40)
    print("TICKER-BASED FILTERING")
    print("=" * 40)
    
    # Show records for a specific ticker
    sample_ticker = full_df['ticker'].value_counts().index[0]
    ticker_records = full_df[full_df['ticker'] == sample_ticker]
    
    print(f"\nRecords for {sample_ticker}: {len(ticker_records)}")
    for idx, row in ticker_records.iterrows():
        print(f"  {row['report_type']} {row['period_end']}: "
              f"Revenue ${row['revenue']:,.0f}, ROE {row['roe']:.2%}")
    
    print("\n" + "=" * 40)
    print("DATA QUALITY SUMMARY")
    print("=" * 40)
    
    print(f"Total records: {len(full_df)}")
    print(f"Unique tickers: {full_df['ticker'].nunique()}")
    print(f"Report types: {list(full_df['report_type'].unique())}")
    print(f"Records with revenue data: {(full_df['revenue'] > 0).sum()}")
    print(f"Records with ROE data: {(full_df['roe'] != 0).sum()}")
    print(f"Records with performance labels: {len(labeled_df)}")
    
    print("\n" + "=" * 60)
    print("FUNDAMENTALS EMBEDDINGS TEST COMPLETED ✅")
    print("=" * 60)


if __name__ == "__main__":
    test_fundamentals_embeddings() 