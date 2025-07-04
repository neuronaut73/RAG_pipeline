# UserPosts Domain Embedding Pipeline

## Overview
The UserPosts domain embedding pipeline processes social media posts from stock discussion forums, creates semantic embeddings, performs sentiment analysis, and stores the results in LanceDB for intelligent retrieval and analysis.

## Data Pipeline Summary

### Input Data
- **Source**: `data/user_posts.csv` - Social media posts from stock forums
- **Labels**: `data/labels.csv` - Performance labels for trading setups
- **Records**: 200 user posts processed
- **Date Range**: September 2024 - May 2025
- **Coverage**: 3 companies (BGO, AML, HWDN) from 68 unique users

### Key Features

#### 1. Content Processing
- **Text Cleaning**: Removes excessive whitespace, filters very short posts
- **Chunking**: Intelligent chunking for long posts (300 words max, 50 word overlap)
- **Data Validation**: Ensures all required fields are present

#### 2. Sentiment Analysis
- **Positive Indicators**: buy, bullish, strong, good, excellent, optimistic, etc.
- **Negative Indicators**: sell, bearish, weak, bad, pessimistic, decline, etc.
- **Uncertainty Indicators**: maybe, perhaps, might, confused, not sure, etc.
- **Sentiment Score**: Calculated as (positive - negative) / total_words

#### 3. Performance Integration
- Links user posts with stock performance data via `setup_id`
- Includes 10-day stock returns, benchmark comparisons, outperformance metrics
- Tracks days outperformed over benchmark

#### 4. Embedding Generation
- **Model**: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- **Output**: 200 embedding vectors stored in LanceDB
- **Unique Features**: Each post preserved as single chunk due to typical forum post length

## Data Schema

### LanceDB Table: `userposts_embeddings`

#### Core Identifiers
```python
'id': str                    # Unique chunk identifier
'post_id': str              # Original post ID
'setup_id': str             # Trading setup identifier
'ticker': str               # Company ticker symbol
'chunk_index': int          # Chunk position (usually 0 for posts)
'total_chunks': int         # Total chunks per post
```

#### Content Fields
```python
'post_content': str         # Processed post content
'full_post_content': str    # Complete original post
'user_handle': str          # Username who posted
'post_date': str           # Timestamp of post (ISO format)
'post_url': str            # Source URL
'scraping_timestamp': str  # When data was collected
```

#### Sentiment Analysis
```python
'positive_indicators': int     # Count of positive sentiment words
'negative_indicators': int     # Count of negative sentiment words
'uncertainty_indicators': int  # Count of uncertainty words
'sentiment_score': float      # Overall sentiment (-1 to 1)
'post_length': int           # Number of words in post
```

#### Performance Labels
```python
'has_performance_labels': bool   # Whether linked to trading results
'stock_return_10d': float       # 10-day stock return %
'benchmark_return_10d': float   # 10-day benchmark return %
'outperformance_10d': float     # Stock outperformance vs benchmark %
'days_outperformed_10d': int    # Days stock beat benchmark
'setup_date': str              # Trading setup date
```

#### Embedding Data
```python
'vector': List[float]      # 384-dimensional embedding vector
'embedded_at': str         # Processing timestamp
```

## Processing Results

### Data Statistics
- **Total Embeddings**: 200 chunks
- **Unique Posts**: 200 (1:1 ratio - posts fit in single chunks)
- **Unique Users**: 68 forum participants
- **Unique Companies**: 3 (BGO, AML, HWDN)
- **Performance Links**: 0% (setup_ids don't match between posts and labels)

### Sentiment Distribution
- **Average Sentiment**: +0.029 (slightly positive)
- **Positive Posts**: 55 (27.5%)
- **Negative Posts**: 7 (3.5%)
- **Neutral Posts**: 138 (69%)

### Company Coverage
1. **BGO (Bango)**: 88 posts (44%)
2. **AML (Aston Martin)**: 82 posts (41%)
3. **HWDN (Howdens)**: 30 posts (15%)

### Top Contributors
1. **iWantThatOne**: 10 posts
2. **Thor111**: 8 posts
3. **c2645sg**: 8 posts
4. **casualinvestor11**: 7 posts
5. **Spacerat**: 7 posts

## Implementation Details

### Core Functions

#### 1. Data Processing
```python
def process_user_posts(posts_file, labels_file) -> pd.DataFrame:
    # Load and validate data
    # Clean post content
    # Extract sentiment indicators
    # Merge with performance labels
    # Return processed DataFrame
```

#### 2. Sentiment Analysis
```python
def extract_sentiment_indicators(content: str) -> Dict[str, Any]:
    # Count positive/negative/uncertainty words
    # Calculate sentiment score
    # Return sentiment metrics
```

#### 3. Content Chunking
```python
def chunk_post_content(content: str, max_chunk_size: int, overlap: int) -> List[str]:
    # Split long posts into manageable chunks
    # Handle overlap for context preservation
    # Return list of text chunks
```

#### 4. Embedding Creation
```python
def create_embeddings_dataset(processed_df, embedding_model) -> List[Dict[str, Any]]:
    # Load sentence transformer model
    # Generate embeddings for each chunk
    # Create structured records with metadata
    # Return embedding dataset
```

### Performance Considerations

#### Storage Efficiency
- **Vector Size**: 384 dimensions (efficient for retrieval)
- **Metadata Rich**: Comprehensive fields for advanced filtering
- **Indexed**: LanceDB provides automatic vector indexing

#### Query Optimization
- **Semantic Search**: Vector similarity for content matching
- **Metadata Filtering**: SQL-like filtering on sentiment, dates, users
- **Hybrid Queries**: Combine semantic and structured search

## Use Cases

### 1. Market Sentiment Analysis
```python
# Find positive sentiment about specific company
results = table.search("positive outlook growth").where("ticker = 'BGO' AND sentiment_score > 0.05")
```

### 2. User Opinion Mining
```python
# Get posts from influential users
results = table.search("investment recommendation").where("user_handle IN ['iWantThatOne', 'Thor111']")
```

### 3. Temporal Sentiment Tracking
```python
# Track sentiment over time periods
results = table.search("market outlook").where("post_date >= '2025-01-01' AND sentiment_score > 0")
```

### 4. Thematic Content Discovery
```python
# Find posts about specific topics
results = table.search("financial results earnings").where("post_length > 50")
```

## Integration Points

### With Other Domains
- **News Domain**: Cross-reference social sentiment with news events
- **Fundamentals Domain**: Compare user opinions with financial metrics
- **Trading Signals**: Use sentiment as input for investment decisions

### RAG Agent Development
- **Query Processing**: Natural language queries about user sentiment
- **Context Retrieval**: Relevant posts for investment analysis
- **Trend Analysis**: Sentiment evolution over time
- **User Insights**: Track opinions from key market participants

## Technical Specifications

### Dependencies
```python
pandas >= 1.5.0
numpy >= 1.21.0
lancedb >= 0.3.0
sentence-transformers >= 2.2.0
```

### Model Details
- **Embedding Model**: all-MiniLM-L6-v2
- **Vector Dimensions**: 384
- **Processing Speed**: ~200 posts/minute
- **Memory Usage**: ~100MB for model + data

### Storage Layout
```
lancedb_store/
├── userposts_embeddings.lance/
│   ├── _transactions/
│   ├── _versions/
│   └── data/
```

## Usage Examples

### Running the Pipeline
```bash
# Activate environment
conda activate sts

# Run embedding pipeline
python embed_userposts.py
```

### Connecting to Data
```python
import lancedb

# Connect to database
db = lancedb.connect("lancedb_store")
table = db.open_table("userposts_embeddings")

# Basic search
results = table.search("bullish market outlook").limit(10).to_pandas()
```

### Advanced Filtering
```python
# Complex query with metadata filters
results = table.search("investment advice") \
    .where("sentiment_score > 0.05") \
    .where("post_length > 30") \
    .where("ticker = 'BGO'") \
    .limit(5) \
    .to_pandas()
```

## Quality Metrics

### Data Quality
- **Completeness**: 100% posts processed successfully
- **Validation**: All required fields validated
- **Consistency**: Standardized data formats throughout

### Embedding Quality
- **Coverage**: Full semantic representation of post content
- **Diversity**: Wide range of topics and sentiment captured
- **Accuracy**: Sentiment analysis aligned with manual review samples

### Performance Metrics
- **Processing Time**: ~2 minutes for 200 posts
- **Storage Size**: ~2MB for embeddings + metadata
- **Query Speed**: <100ms for typical semantic searches

## Troubleshooting

### Common Issues

#### 1. DateTime Parsing Errors
**Problem**: Mixed datetime formats in source data
**Solution**: Use `format='mixed'` in pandas.to_datetime()

#### 2. Empty Post Content
**Problem**: Some posts may be very short or empty
**Solution**: Content validation filters posts < 10 characters

#### 3. Memory Issues with Large Datasets
**Problem**: Large datasets may exceed memory
**Solution**: Process in batches using chunk processing

#### 4. Sentiment Analysis Accuracy
**Problem**: Simple word-based sentiment may miss context
**Solution**: Consider upgrading to transformer-based sentiment models

## Future Enhancements

### 1. Advanced Sentiment Analysis
- Replace keyword-based sentiment with transformer models
- Add emotion detection (fear, greed, uncertainty)
- Context-aware sentiment scoring

### 2. User Influence Scoring
- Track user credibility and influence metrics
- Weight posts by user reputation
- Identify thought leaders in the community

### 3. Real-time Processing
- Stream processing for live social media data
- Incremental embedding updates
- Real-time sentiment dashboards

### 4. Enhanced Performance Linking
- Improve setup_id matching with posts
- Add manual labeling workflows
- Cross-reference with actual trading outcomes

### 5. Multi-modal Analysis
- Process images and links shared in posts
- Analyze emoji sentiment
- Extract structured data from post formats

## Conclusion

The UserPosts embedding pipeline provides a robust foundation for analyzing social media sentiment in financial markets. With 200 posts processed across 3 companies, the system captures the voice of retail investors and forum participants. The combination of semantic embeddings and structured sentiment analysis enables sophisticated queries for market research, investment analysis, and trend identification.

The pipeline is ready for integration with other domain agents and can serve as a key component in a comprehensive RAG-based financial analysis system. 