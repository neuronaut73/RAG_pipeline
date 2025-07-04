#!/usr/bin/env python3
"""
embed_userposts.py - UserPosts Domain Embedding Pipeline

Processes user posts from stock discussion forums, creates embeddings of post content,
and stores in LanceDB with metadata and performance labels for RAG retrieval.

Now includes proper setup validation to ensure only confirmed setups are used.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import hashlib
import re
import sys

# LanceDB and embeddings
import lancedb
from sentence_transformers import SentenceTransformer

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))
from setup_validator import SetupValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_post_content(content: str) -> str:
    """Clean and normalize post content"""
    if not content or pd.isna(content):
        return ""
    
    # Convert to string and strip whitespace
    content = str(content).strip()
    
    # Remove excessive whitespace
    content = re.sub(r'\s+', ' ', content)
    
    # Remove very short posts that are not meaningful
    if len(content) < 10:
        return ""
    
    return content


def extract_sentiment_indicators(content: str) -> Dict[str, Any]:
    """Extract sentiment indicators from post content"""
    content_lower = content.lower()
    
    # Positive sentiment indicators
    positive_words = ['buy', 'bullish', 'strong', 'good', 'great', 'excellent', 'positive', 
                     'optimistic', 'growth', 'profit', 'successful', 'upgrade', 'target',
                     'outperform', 'recommend', 'love', 'like', 'fantastic', 'brilliant']
    
    # Negative sentiment indicators
    negative_words = ['sell', 'bearish', 'weak', 'bad', 'terrible', 'awful', 'negative',
                     'pessimistic', 'loss', 'decline', 'downgrade', 'avoid', 'hate',
                     'disappointed', 'worried', 'concerning', 'risky', 'dangerous']
    
    # Uncertainty indicators
    uncertainty_words = ['maybe', 'perhaps', 'might', 'could', 'uncertain', 'confused',
                        'not sure', 'wondering', 'question', 'doubt', 'unclear']
    
    positive_count = sum(1 for word in positive_words if word in content_lower)
    negative_count = sum(1 for word in negative_words if word in content_lower)
    uncertainty_count = sum(1 for word in uncertainty_words if word in content_lower)
    
    # Calculate sentiment score
    total_words = len(content.split())
    sentiment_score = (positive_count - negative_count) / max(total_words, 1)
    
    return {
        'positive_indicators': positive_count,
        'negative_indicators': negative_count,
        'uncertainty_indicators': uncertainty_count,
        'sentiment_score': sentiment_score,
        'post_length': total_words
    }


def chunk_post_content(content: str, max_chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """
    Chunk post content for embedding. Most posts are short, but some may be long.
    
    Args:
        content: Post content to chunk
        max_chunk_size: Maximum words per chunk
        overlap: Number of words to overlap between chunks
    
    Returns:
        List of text chunks
    """
    if not content:
        return []
    
    words = content.split()
    
    # If post is short enough, return as single chunk
    if len(words) <= max_chunk_size:
        return [content]
    
    chunks = []
    start = 0
    
    while start < len(words):
        end = min(start + max_chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append(' '.join(chunk_words))
        
        # Move start position with overlap
        start = end - overlap
        if start >= len(words):
            break
    
    return chunks


def process_user_posts(
    posts_file: str = "data/user_posts.csv",
    labels_file: str = "data/labels.csv"
) -> pd.DataFrame:
    """
    Load and process user posts data, merging with performance labels
    Now includes setup validation to filter by confirmed setups only.
    
    Args:
        posts_file: Path to user posts CSV
        labels_file: Path to labels CSV
        
    Returns:
        Processed DataFrame with posts and labels for confirmed setups only
    """
    logger.info(f"Loading user posts from {posts_file}")
    
    # Initialize setup validator
    data_dir = str(Path(posts_file).parent)
    setup_validator = SetupValidator(data_dir=data_dir)
    logger.info(f"Setup validator initialized with {len(setup_validator.confirmed_setup_ids)} confirmed setups")
    
    # Load posts data
    posts_df_raw = pd.read_csv(posts_file)
    logger.info(f"Loaded {len(posts_df_raw)} user posts")
    
    # Load labels data and filter by confirmed setups
    labels_df_raw = pd.read_csv(labels_file)
    logger.info(f"Loaded {len(labels_df_raw)} raw performance labels")
    
    # Filter labels to only include confirmed setups
    labels_df = setup_validator.filter_labels_by_confirmed_setups(labels_df_raw)
    logger.info(f"Filtered to {len(labels_df)} confirmed setup labels")
    
    # Basic data validation
    required_posts_cols = ['post_id', 'setup_id', 'ticker', 'post_date', 'user_handle', 'post_content']
    missing_cols = [col for col in required_posts_cols if col not in posts_df_raw.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in posts data: {missing_cols}")
    
    # Filter posts to only include confirmed setups
    initial_posts_count = len(posts_df_raw)
    posts_df = posts_df_raw[posts_df_raw['setup_id'].isin(setup_validator.confirmed_setup_ids)]
    filtered_posts_count = len(posts_df)
    logger.info(f"Filtered posts by confirmed setups: {initial_posts_count} -> {filtered_posts_count} posts "
               f"(removed {initial_posts_count - filtered_posts_count} posts from non-confirmed setups)")
    
    # Clean post content
    posts_df['post_content_clean'] = posts_df['post_content'].apply(clean_post_content)
    
    # Filter out empty posts after cleaning
    initial_count = len(posts_df)
    posts_df = posts_df[posts_df['post_content_clean'] != '']
    filtered_count = len(posts_df)
    logger.info(f"Filtered out {initial_count - filtered_count} empty posts, {filtered_count} remaining")
    
    # Extract sentiment indicators
    sentiment_data = posts_df['post_content_clean'].apply(extract_sentiment_indicators)
    sentiment_df = pd.DataFrame(sentiment_data.tolist())
    posts_df = pd.concat([posts_df, sentiment_df], axis=1)
    
    # Convert post_date to datetime with mixed format handling
    posts_df['post_date'] = pd.to_datetime(posts_df['post_date'], format='mixed')
    
    # Merge with performance labels
    merged_df = posts_df.merge(
        labels_df[['setup_id', 'stock_return_10d', 'benchmark_return_10d', 'outperformance_10d', 
                  'days_outperformed_10d', 'setup_date']],
        on='setup_id',
        how='left'
    )
    
    # Add performance label indicators
    merged_df['has_performance_labels'] = ~merged_df['stock_return_10d'].isna()
    
    # Fill NaN values for performance metrics
    performance_cols = ['stock_return_10d', 'benchmark_return_10d', 'outperformance_10d', 'days_outperformed_10d']
    for col in performance_cols:
        merged_df[col] = merged_df[col].fillna(0.0)
    
    logger.info(f"Merged data: {len(merged_df)} posts, {merged_df['has_performance_labels'].sum()} with performance labels")
    
    return merged_df


def create_embeddings_dataset(
    processed_df: pd.DataFrame,
    embedding_model: str = "all-MiniLM-L6-v2",
    max_chunk_size: int = 300,
    overlap: int = 50
) -> List[Dict[str, Any]]:
    """
    Create embeddings dataset from processed posts
    
    Args:
        processed_df: Processed posts DataFrame
        embedding_model: Name of sentence transformer model
        max_chunk_size: Maximum words per chunk
        overlap: Overlap between chunks
        
    Returns:
        List of embedding records
    """
    logger.info(f"Loading embedding model: {embedding_model}")
    model = SentenceTransformer(embedding_model)
    
    records = []
    total_chunks = 0
    
    for idx, row in processed_df.iterrows():
        # Chunk the post content
        chunks = chunk_post_content(row['post_content_clean'], max_chunk_size, overlap)
        
        if not chunks:
            continue
            
        for chunk_idx, chunk in enumerate(chunks):
            # Create unique ID for each chunk
            chunk_id = f"{row['post_id']}_chunk_{chunk_idx}"
            
            # Generate embedding
            embedding = model.encode(chunk)
            
            # Create record
            record = {
                # Identifiers
                'id': chunk_id,
                'post_id': row['post_id'],
                'setup_id': row['setup_id'],
                'ticker': row['ticker'],
                'chunk_index': chunk_idx,
                'total_chunks': len(chunks),
                
                # Content
                'post_content': chunk,
                'full_post_content': row['post_content_clean'],
                
                # User and timing
                'user_handle': row['user_handle'],
                'post_date': row['post_date'].isoformat(),
                'post_url': row.get('post_url', ''),
                'scraping_timestamp': row.get('scraping_timestamp', ''),
                
                # Sentiment analysis
                'positive_indicators': int(row['positive_indicators']),
                'negative_indicators': int(row['negative_indicators']),
                'uncertainty_indicators': int(row['uncertainty_indicators']),
                'sentiment_score': float(row['sentiment_score']),
                'post_length': int(row['post_length']),
                
                # Performance labels
                'has_performance_labels': bool(row['has_performance_labels']),
                'stock_return_10d': float(row['stock_return_10d']),
                'benchmark_return_10d': float(row['benchmark_return_10d']),
                'outperformance_10d': float(row['outperformance_10d']),
                'days_outperformed_10d': int(row['days_outperformed_10d']),
                'setup_date': row.get('setup_date', ''),
                
                # Embedding
                'vector': embedding.tolist(),
                'embedded_at': datetime.now().isoformat()
            }
            
            records.append(record)
            total_chunks += 1
            
            if total_chunks % 50 == 0:
                logger.info(f"Processed {total_chunks} chunks...")
    
    logger.info(f"Created embeddings for {total_chunks} chunks from {len(processed_df)} posts")
    return records


def store_in_lancedb(
    records: List[Dict[str, Any]],
    lancedb_dir: str = "lancedb_store",
    table_name: str = "userposts_embeddings"
) -> None:
    """
    Store embeddings in LanceDB
    
    Args:
        records: List of embedding records
        lancedb_dir: Directory for LanceDB storage
        table_name: Name of the table to create
    """
    logger.info(f"Storing {len(records)} records in LanceDB")
    
    # Ensure directory exists
    Path(lancedb_dir).mkdir(parents=True, exist_ok=True)
    
    # Connect to LanceDB
    db = lancedb.connect(lancedb_dir)
    
    # Drop existing table if it exists
    try:
        db.drop_table(table_name)
        logger.info(f"Dropped existing table: {table_name}")
    except Exception:
        pass
    
    # Create table
    df = pd.DataFrame(records)
    table = db.create_table(table_name, df)
    
    logger.info(f"Created LanceDB table '{table_name}' with {len(table)} records")
    
    # Verify the table
    sample_query = table.search(records[0]['vector']).limit(3).to_pandas()
    logger.info(f"Table verification: Retrieved {len(sample_query)} sample records")
    
    return table


def generate_summary_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate summary statistics about the embedded data"""
    
    df = pd.DataFrame(records)
    
    stats = {
        'total_chunks': len(records),
        'unique_posts': df['post_id'].nunique(),
        'unique_users': df['user_handle'].nunique(),
        'unique_tickers': df['ticker'].nunique(),
        'unique_setups': df['setup_id'].nunique(),
        
        'date_range': {
            'earliest': df['post_date'].min(),
            'latest': df['post_date'].max()
        },
        
        'chunks_with_labels': len(df[df['has_performance_labels'] == True]),
        'label_percentage': len(df[df['has_performance_labels'] == True]) / len(df) * 100,
        
        'sentiment_stats': {
            'avg_sentiment_score': df['sentiment_score'].mean(),
            'positive_posts': len(df[df['sentiment_score'] > 0.02]),
            'negative_posts': len(df[df['sentiment_score'] < -0.02]),
            'neutral_posts': len(df[df['sentiment_score'].between(-0.02, 0.02)])
        },
        
        'post_length_stats': {
            'avg_length': df['post_length'].mean(),
            'min_length': df['post_length'].min(),
            'max_length': df['post_length'].max()
        },
        
        'performance_stats': {
            'avg_outperformance': df[df['has_performance_labels']]['outperformance_10d'].mean(),
            'best_performer_outperformance': df[df['has_performance_labels']]['outperformance_10d'].max(),
            'worst_performer_outperformance': df[df['has_performance_labels']]['outperformance_10d'].min()
        } if len(df[df['has_performance_labels']]) > 0 else {},
        
        'top_tickers': df['ticker'].value_counts().head(10).to_dict(),
        'top_users': df['user_handle'].value_counts().head(10).to_dict()
    }
    
    return stats


def main():
    """Main embedding pipeline execution"""
    
    print("=" * 60)
    print("USERPOSTS DOMAIN EMBEDDING PIPELINE")
    print("=" * 60)
    
    try:
        # Step 1: Process user posts data
        print("\n📊 Step 1: Processing User Posts Data")
        processed_df = process_user_posts()
        
        # Step 2: Create embeddings
        print("\n🤖 Step 2: Creating Embeddings")
        records = create_embeddings_dataset(processed_df)
        
        if not records:
            print("❌ No valid records created. Exiting.")
            return
        
        # Step 3: Store in LanceDB
        print("\n💾 Step 3: Storing in LanceDB")
        table = store_in_lancedb(records)
        
        # Step 4: Generate and display summary statistics
        print("\n📈 Step 4: Summary Statistics")
        stats = generate_summary_stats(records)
        
        print(f"\n📋 USERPOSTS EMBEDDING SUMMARY:")
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"  Unique posts: {stats['unique_posts']}")
        print(f"  Unique users: {stats['unique_users']}")
        print(f"  Unique tickers: {stats['unique_tickers']}")
        print(f"  Date range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}")
        print(f"  Chunks with performance labels: {stats['chunks_with_labels']} ({stats['label_percentage']:.1f}%)")
        
        print(f"\n💭 SENTIMENT ANALYSIS:")
        sentiment = stats['sentiment_stats']
        print(f"  Average sentiment score: {sentiment['avg_sentiment_score']:.3f}")
        print(f"  Positive posts: {sentiment['positive_posts']}")
        print(f"  Negative posts: {sentiment['negative_posts']}")
        print(f"  Neutral posts: {sentiment['neutral_posts']}")
        
        if stats['performance_stats']:
            print(f"\n📈 PERFORMANCE CORRELATION:")
            perf = stats['performance_stats']
            print(f"  Average outperformance: {perf['avg_outperformance']:.2f}%")
            print(f"  Best performer: {perf['best_performer_outperformance']:.2f}%")
            print(f"  Worst performer: {perf['worst_performer_outperformance']:.2f}%")
        
        print(f"\n🏢 TOP COMPANIES (by post count):")
        for ticker, count in list(stats['top_tickers'].items())[:5]:
            print(f"  {ticker}: {count} posts")
        
        print(f"\n👥 TOP USERS (by post count):")
        for user, count in list(stats['top_users'].items())[:5]:
            print(f"  {user}: {count} posts")
        
        print("\n" + "=" * 60)
        print("✅ USERPOSTS EMBEDDING PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"📁 Data stored in: lancedb_store/userposts_embeddings")
        print(f"🎯 Ready for UserPosts RAG Agent development")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"\n❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main() 