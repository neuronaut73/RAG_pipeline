#!/usr/bin/env python3
"""
embed_news.py - News Domain Embedding & LanceDB Pipeline

Processes news data from:
- rns_announcements.csv
- stock_news_enhanced.csv

Chunks and embeds content fields (text, headline, content_summary), 
attaches metadata including labels, and stores in LanceDB.

Now includes proper setup validation to ensure only confirmed setups are used.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import re
from pathlib import Path

# LanceDB and embeddings
import lancedb
from sentence_transformers import SentenceTransformer

# Text processing
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))
from setup_validator import SetupValidator

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

print("News Domain Embedding Pipeline initialized")

class NewsEmbeddingPipeline:
    """
    News Domain Embedding Pipeline for RAG System with Setup Validation
    
    Processes RNS announcements and enhanced stock news data,
    creates semantic embeddings, and stores in LanceDB with rich metadata.
    Now includes proper filtering by confirmed setups only.
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        lancedb_dir: str = "lancedb_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        max_chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        """
        Initialize the News Embedding Pipeline
        
        Args:
            data_dir: Directory containing CSV files
            lancedb_dir: Directory for LanceDB storage
            embedding_model: HuggingFace model for embeddings
            max_chunk_size: Maximum tokens per text chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        self.data_dir = Path(data_dir)
        self.lancedb_dir = Path(lancedb_dir)
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize setup validator
        self.setup_validator = SetupValidator(data_dir=str(self.data_dir))
        print(f"Setup validator initialized with {len(self.setup_validator.confirmed_setup_ids)} confirmed setups")
        
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize LanceDB
        self.lancedb_dir.mkdir(exist_ok=True)
        self.db = lancedb.connect(str(self.lancedb_dir))
        
        # Data containers
        self.rns_data = None
        self.news_data = None
        self.labels_data = None
        
    def load_data(self) -> None:
        """Load all news domain data files with setup validation"""
        print("Loading news domain data...")
        
        # Load RNS announcements
        rns_path = self.data_dir / "rns_announcements.csv"
        if rns_path.exists():
            self.rns_data = pd.read_csv(rns_path)
            print(f"Loaded {len(self.rns_data)} RNS announcements")
        else:
            print(f"Warning: {rns_path} not found")
            
        # Load enhanced stock news
        news_path = self.data_dir / "stock_news_enhanced.csv"
        if news_path.exists():
            self.news_data = pd.read_csv(news_path)
            print(f"Loaded {len(self.news_data)} enhanced news articles")
        else:
            print(f"Warning: {news_path} not found")
            
        # Load labels and filter by confirmed setups
        labels_path = self.data_dir / "labels.csv"
        if labels_path.exists():
            labels_data_raw = pd.read_csv(labels_path)
            print(f"Loaded {len(labels_data_raw)} raw labels")
            
            # Filter labels to only include confirmed setups
            self.labels_data = self.setup_validator.filter_labels_by_confirmed_setups(labels_data_raw)
            print(f"Filtered to {len(self.labels_data)} confirmed setup labels")
        else:
            print(f"Warning: {labels_path} not found")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if pd.isna(text) or text == "":
            return ""
            
        # Convert to string and strip
        text = str(text).strip()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove RNS boilerplate (common in announcements)
        boilerplate_patterns = [
            r'This information is provided by RNS.*?END [A-Z]+',
            r'RNS may use your IP address.*?commercial services\.',
            r'For further information about how RNS.*?please see our \.',
            r'Terms and conditions relating to.*?visit www\.rns\.com',
        ]
        
        for pattern in boilerplate_patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean up remaining artifacts
        text = re.sub(r'\|+', ' ', text)  # Remove pipe separators
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def chunk_text(self, text: str, title: str = "") -> List[Dict[str, Any]]:
        """
        Chunk text into manageable pieces for embedding
        
        Args:
            text: Main text content
            title: Optional title/headline to prepend
            
        Returns:
            List of chunks with metadata
        """
        if not text or len(text.strip()) == 0:
            return []
            
        chunks = []
        
        # Combine title and text
        full_text = f"{title}\n\n{text}" if title else text
        
        # Split into sentences
        try:
            sentences = sent_tokenize(full_text)
        except:
            # Fallback to simple splitting
            sentences = [s.strip() for s in full_text.split('.') if s.strip()]
        
        current_chunk = ""
        current_chunk_idx = 0
        
        for sentence in sentences:
            # Estimate token count (rough approximation: 1 token ≈ 4 characters)
            estimated_tokens = len(current_chunk + " " + sentence) // 4
            
            if estimated_tokens > self.max_chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    'chunk_text': current_chunk.strip(),
                    'chunk_idx': current_chunk_idx,
                    'chunk_type': 'content'
                })
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-self.chunk_overlap*4:] if len(current_chunk) > self.chunk_overlap*4 else current_chunk
                current_chunk = overlap_text + " " + sentence
                current_chunk_idx += 1
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'chunk_text': current_chunk.strip(),
                'chunk_idx': current_chunk_idx,
                'chunk_type': 'content'
            })
        
        return chunks
    
    def process_rns_announcements(self) -> List[Dict[str, Any]]:
        """Process RNS announcements data with setup validation"""
        if self.rns_data is None:
            return []
            
        print("Processing RNS announcements...")
        processed_records = []
        skipped_unconfirmed = 0
        
        for idx, row in self.rns_data.iterrows():
            # Check if setup_id is confirmed
            setup_id = row.get('setup_id', '')
            if setup_id and not self.setup_validator.is_confirmed_setup(setup_id):
                skipped_unconfirmed += 1
                continue  # Skip non-confirmed setups
            
            # Skip records without a setup_id (no trading setup associated)
            if not setup_id:
                skipped_unconfirmed += 1
                continue
            
            # Clean content fields
            headline = self.clean_text(row.get('headline', ''))
            text_content = self.clean_text(row.get('text', ''))
            
            if not text_content and not headline:
                continue  # Skip empty records
            
            # Create chunks
            chunks = self.chunk_text(text_content, headline)
            
            if not chunks:
                # If no chunks from text, at least create one for the headline
                chunks = [{
                    'chunk_text': headline,
                    'chunk_idx': 0,
                    'chunk_type': 'headline'
                }]
            
            # Process each chunk
            for chunk in chunks:
                # Base record structure
                record = {
                    'id': f"rns_{row.get('id', idx)}_{chunk['chunk_idx']}",
                    'source_type': 'rns_announcement',
                    'source_id': row.get('id', idx),
                    'setup_id': row.get('setup_id', ''),
                    'ticker': row.get('ticker', ''),
                    'headline': headline,
                    'rns_date': row.get('rns_date', ''),
                    'rns_time': row.get('rns_time', ''),
                    'url': row.get('url', ''),
                    'scraped_at': row.get('scraped_at', ''),
                    'content_length': row.get('content_length', 0),
                    'chunk_text': chunk['chunk_text'],
                    'chunk_idx': chunk['chunk_idx'],
                    'chunk_type': chunk['chunk_type'],
                    'text_length': len(chunk['chunk_text']),
                    'created_at': datetime.now().isoformat(),
                    # Additional fields for compatibility
                    'title': headline,
                    'content_summary': chunk['chunk_text'][:200] + "..." if len(chunk['chunk_text']) > 200 else chunk['chunk_text'],
                    'publisher': 'RNS',
                    'sentiment_score': 0.0,  # Neutral default
                    'relevance_score': 1.0,  # Default high relevance
                    'article_type': 'announcement'
                }
                
                processed_records.append(record)
        
        print(f"Created {len(processed_records)} RNS chunks from confirmed setups")
        print(f"Skipped {skipped_unconfirmed} RNS records from non-confirmed setups")
        return processed_records
    
    def process_enhanced_news(self) -> List[Dict[str, Any]]:
        """Process enhanced stock news data with setup validation"""
        if self.news_data is None:
            return []
            
        print("Processing enhanced stock news...")
        processed_records = []
        skipped_unconfirmed = 0
        
        for idx, row in self.news_data.iterrows():
            # Check if setup_id is confirmed
            setup_id = row.get('setup_id', '')
            if setup_id and not self.setup_validator.is_confirmed_setup(setup_id):
                skipped_unconfirmed += 1
                continue  # Skip non-confirmed setups
            
            # Skip records without a setup_id (no trading setup associated)
            if not setup_id:
                skipped_unconfirmed += 1
                continue
            
            # Clean content fields
            title = self.clean_text(row.get('title', ''))
            content_summary = self.clean_text(row.get('content_summary', ''))
            
            if not content_summary and not title:
                continue  # Skip empty records
            
            # Create chunks
            chunks = self.chunk_text(content_summary, title)
            
            if not chunks:
                # If no chunks from content, at least create one for the title
                chunks = [{
                    'chunk_text': title,
                    'chunk_idx': 0,
                    'chunk_type': 'title'
                }]
            
            # Process each chunk
            for chunk in chunks:
                # Base record structure
                record = {
                    'id': f"news_{row.get('id', idx)}_{chunk['chunk_idx']}",
                    'source_type': 'enhanced_news',
                    'source_id': row.get('id', idx),
                    'setup_id': row.get('setup_id', ''),
                    'ticker': row.get('ticker', ''),
                    'title': title,
                    'headline': title,  # Alias for compatibility
                    'publisher': row.get('publisher', ''),
                    'link': row.get('link', ''),
                    'url': row.get('link', ''),  # Alias for compatibility
                    'provider_publish_time': row.get('provider_publish_time', ''),
                    'rns_date': row.get('provider_publish_time', ''),  # Alias for compatibility
                    'sentiment_score': float(row.get('sentiment_score', 0.0)),
                    'relevance_score': float(row.get('relevance_score', 1.0)),
                    'article_type': row.get('article_type', 'story'),
                    'setup_date_window': row.get('setup_date_window', ''),
                    'days_before_setup': row.get('days_before_setup', ''),
                    'chunk_text': chunk['chunk_text'],
                    'chunk_idx': chunk['chunk_idx'],
                    'chunk_type': chunk['chunk_type'],
                    'text_length': len(chunk['chunk_text']),
                    'created_at': datetime.now().isoformat(),
                    'content_summary': content_summary,
                    'scraped_at': row.get('created_at', ''),
                    'content_length': len(content_summary) if content_summary else 0,
                    'rns_time': ''  # Not applicable for enhanced news
                }
                
                processed_records.append(record)
        
        print(f"Created {len(processed_records)} enhanced news chunks from confirmed setups")
        print(f"Skipped {skipped_unconfirmed} enhanced news records from non-confirmed setups")
        return processed_records
    
    def enrich_with_labels(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich records with label information"""
        if self.labels_data is None:
            print("No labels data available for enrichment")
            return records
            
        print("Enriching records with labels...")
        
        # Create lookup dictionary for labels
        labels_lookup = {}
        for _, label_row in self.labels_data.iterrows():
            setup_id = label_row.get('setup_id', '')
            if setup_id:
                labels_lookup[setup_id] = {
                    'setup_date': label_row.get('setup_date', ''),
                    'stock_return_10d': float(label_row.get('stock_return_10d', 0.0)),
                    'benchmark_return_10d': float(label_row.get('benchmark_return_10d', 0.0)),
                    'outperformance_10d': float(label_row.get('outperformance_10d', 0.0)),
                    'days_outperformed_10d': int(label_row.get('days_outperformed_10d', 0)),
                    'benchmark_ticker': label_row.get('benchmark_ticker', ''),
                    'calculation_date': label_row.get('calculation_date', ''),
                    'actual_days_calculated': label_row.get('actual_days_calculated', 0)
                }
        
        # Enrich records
        enriched_count = 0
        for record in records:
            setup_id = record.get('setup_id', '')
            if setup_id and setup_id in labels_lookup:
                # Add label information
                record.update(labels_lookup[setup_id])
                enriched_count += 1
            else:
                # Add default values for consistency
                record.update({
                    'setup_date': '',
                    'stock_return_10d': 0.0,
                    'benchmark_return_10d': 0.0,
                    'outperformance_10d': 0.0,
                    'days_outperformed_10d': 0,
                    'benchmark_ticker': '',
                    'calculation_date': '',
                    'actual_days_calculated': 0
                })
        
        print(f"Enriched {enriched_count} records with labels")
        return records
    
    def create_embeddings(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create embeddings for all text chunks"""
        if not records:
            return records
            
        print("Creating embeddings...")
        
        # Extract texts for batch embedding
        texts = [record['chunk_text'] for record in records]
        
        # Create embeddings in batches
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch_texts,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            all_embeddings.extend(batch_embeddings)
        
        # Add embeddings to records
        for record, embedding in zip(records, all_embeddings):
            record['vector'] = embedding.tolist()
        
        print(f"Created {len(all_embeddings)} embeddings")
        return records
    
    def store_in_lancedb(self, records: List[Dict[str, Any]], table_name: str = "news_embeddings") -> None:
        """Store records in LanceDB"""
        if not records:
            print("No records to store")
            return
            
        print(f"Storing {len(records)} records in LanceDB table: {table_name}")
        
        # Convert to DataFrame for LanceDB
        df = pd.DataFrame(records)
        
        # Ensure vector column is properly formatted
        df['vector'] = df['vector'].apply(lambda x: np.array(x, dtype=np.float32))
        
        # Fix data types to avoid conversion issues
        string_columns = [
            'id', 'source_type', 'setup_id', 'ticker', 'headline', 'title', 
            'rns_date', 'rns_time', 'url', 'scraped_at', 'chunk_text', 
            'chunk_type', 'created_at', 'content_summary', 'publisher',
            'link', 'provider_publish_time', 'article_type', 'setup_date_window',
            'benchmark_ticker', 'calculation_date'
        ]
        
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        # Ensure numeric columns are properly typed
        numeric_columns = [
            'source_id', 'content_length', 'chunk_idx', 'text_length',
            'sentiment_score', 'relevance_score', 'days_before_setup',
            'stock_return_10d', 'benchmark_return_10d', 'outperformance_10d',
            'days_outperformed_10d', 'actual_days_calculated'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Create or overwrite table
        try:
            # Drop existing table if it exists
            if table_name in self.db.table_names():
                self.db.drop_table(table_name)
                
            # Create new table
            table = self.db.create_table(table_name, df)
            print(f"Successfully created table '{table_name}' with {len(df)} records")
            
            # Display table info
            print(f"Table schema: {table.schema}")
            print(f"Table size: {len(table)}")
            
        except Exception as e:
            print(f"Error storing data in LanceDB: {e}")
            raise
    
    def run_pipeline(self) -> None:
        """Run the complete news embedding pipeline"""
        print("=" * 60)
        print("NEWS DOMAIN EMBEDDING PIPELINE")
        print("=" * 60)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Process data sources
        all_records = []
        
        # Process RNS announcements
        rns_records = self.process_rns_announcements()
        all_records.extend(rns_records)
        
        # Process enhanced news
        news_records = self.process_enhanced_news()
        all_records.extend(news_records)
        
        if not all_records:
            print("No records to process. Exiting.")
            return
        
        print(f"Total records to process: {len(all_records)}")
        
        # Step 3: Enrich with labels
        enriched_records = self.enrich_with_labels(all_records)
        
        # Step 4: Create embeddings
        embedded_records = self.create_embeddings(enriched_records)
        
        # Step 5: Store in LanceDB
        self.store_in_lancedb(embedded_records)
        
        print("=" * 60)
        print("NEWS EMBEDDING PIPELINE COMPLETE")
        print("=" * 60)
        
        # Display summary statistics
        self.display_summary(embedded_records)
    
    def display_summary(self, records: List[Dict[str, Any]]) -> None:
        """Display pipeline summary statistics"""
        if not records:
            return
            
        df = pd.DataFrame(records)
        
        print("\nPIPELINE SUMMARY:")
        print("-" * 40)
        print(f"Total records: {len(df)}")
        print(f"Unique tickers: {df['ticker'].nunique()}")
        print(f"Unique setup_ids: {df['setup_id'].nunique()}")
        print(f"Source types: {df['source_type'].value_counts().to_dict()}")
        print(f"Chunk types: {df['chunk_type'].value_counts().to_dict()}")
        
        # Text length statistics
        text_lengths = df['text_length']
        print(f"Text length stats:")
        print(f"  Mean: {text_lengths.mean():.1f}")
        print(f"  Median: {text_lengths.median():.1f}")
        print(f"  Min: {text_lengths.min()}")
        print(f"  Max: {text_lengths.max()}")
        
        # Sentiment statistics (where available)
        sentiment_scores = df[df['sentiment_score'] != 0.0]['sentiment_score']
        if len(sentiment_scores) > 0:
            print(f"Sentiment scores (non-zero):")
            print(f"  Mean: {sentiment_scores.mean():.3f}")
            print(f"  Min: {sentiment_scores.min():.3f}")
            print(f"  Max: {sentiment_scores.max():.3f}")
        
        # Label enrichment statistics
        enriched_records = df[df['outperformance_10d'] != 0.0]
        print(f"Records enriched with labels: {len(enriched_records)} ({len(enriched_records)/len(df)*100:.1f}%)")


def main():
    """Main execution function"""
    pipeline = NewsEmbeddingPipeline(
        data_dir="data",
        lancedb_dir="lancedb_store",
        embedding_model="all-MiniLM-L6-v2",
        max_chunk_size=512,
        chunk_overlap=50
    )
    
    pipeline.run_pipeline()


if __name__ == "__main__":
    main() 