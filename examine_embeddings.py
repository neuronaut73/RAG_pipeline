#!/usr/bin/env python3
"""
Examine sample embeddings from LanceDB to verify content and text conversion
"""

import lancedb
import pandas as pd

def main():
    # Connect to LanceDB
    db = lancedb.connect('lancedb_store')
    
    print('=== LANCEDB TABLES SUMMARY ===')
    tables = db.table_names()
    for table_name in tables:
        table = db.open_table(table_name)
        count = table.count_rows()
        print(f'{table_name}: {count} records')
    
    print('\n=== FUNDAMENTALS EMBEDDINGS SAMPLE ===')
    fundamentals_table = db.open_table('fundamentals_embeddings')
    fundamentals_sample = fundamentals_table.search().limit(2).to_pandas()
    
    print('Fundamentals columns:', list(fundamentals_sample.columns))
    
    for i, row in fundamentals_sample.iterrows():
        print(f'\nFundamentals Record {i+1}:')
        print(f'Ticker: {row["ticker"]}')
        print(f'Setup ID: {row["setup_id"]}')
        print(f'Report Type: {row["report_type"]}')
        print(f'Financial Summary (first 600 chars):')
        print(row['financial_summary'][:600] + '...')
        print(f'Raw Revenue: {row["revenue"]:,.0f}')
        print(f'Raw ROE: {row["roe"]:.4f}')
        print(f'Raw Current Ratio: {row["current_ratio"]:.2f}')
        break
    
    print('\n=== NEWS EMBEDDINGS SAMPLE ===')
    news_table = db.open_table('news_embeddings')
    news_sample = news_table.search().limit(2).to_pandas()
    
    print('News columns:', list(news_sample.columns))
    
    for i, row in news_sample.iterrows():
        print(f'\nNews Record {i+1}:')
        print(f'Setup ID: {row["setup_id"]}')
        print(f'Source Type: {row["source_type"]}')
        text_col = 'chunk_text' if 'chunk_text' in news_sample.columns else 'text'
        print(f'Text (first 400 chars):')
        print(row[text_col][:400] + '...')
        break
    
    print('\n=== USER POSTS EMBEDDINGS SAMPLE ===')
    userposts_table = db.open_table('userposts_embeddings')
    userposts_sample = userposts_table.search().limit(2).to_pandas()
    
    for i, row in userposts_sample.iterrows():
        print(f'\nUser Post Record {i+1}:')
        print(f'Setup ID: {row["setup_id"]}')
        print(f'Ticker: {row["ticker"]}')
        print(f'Post Content (first 300 chars):')
        print(row["chunk_text"][:300] + '...')
        print(f'Sentiment Score: {row["sentiment_score"]:.3f}')
        break

if __name__ == "__main__":
    main() 