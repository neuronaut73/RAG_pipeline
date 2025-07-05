# DuckDB-Based RAG Pipeline Guide

## Overview

The DuckDB-based RAG pipeline processes 72 confirmed setups from the `sentiment_system.duckdb` database, creating semantic embeddings across three domains: fundamentals, news, and user posts. This represents a 9x improvement over the CSV-based approach.

## Quick Start

### Running the Complete Pipeline

```bash
# Activate conda environment
conda activate sts

# Run the complete pipeline
python scripts/run_complete_pipeline_duckdb.py
```

### Individual Components

```bash
# Run individual pipelines
python scripts/embed_fundamentals_duckdb.py
python scripts/embed_userposts_duckdb.py
```

## Pipeline Components

### 1. Setup Validator (`setup_validator_duckdb.py`)
- **Purpose**: Centralized validation for confirmed setups
- **Data Source**: `setups` table in DuckDB
- **Key Features**:
  - Loads 72 confirmed setups
  - Provides date validation for historical filtering
  - Filters data by confirmed setup IDs only

### 2. Fundamentals Pipeline (`embed_fundamentals_duckdb.py`)
- **Input**: `fundamentals` and `financial_ratios` tables
- **Output**: `fundamentals_embeddings` (329 records)
- **Features**:
  - Historical data filtering (only data before setup dates)
  - Rich financial summaries with ratios
  - Performance labels (10-day returns)

### 3. User Posts Pipeline (`embed_userposts_duckdb.py`)
- **Input**: `user_posts` table
- **Output**: `userposts_embeddings` (790 records)
- **Features**:
  - Sentiment analysis with indicators
  - Content chunking for long posts
  - User engagement metrics

### 4. News Pipeline (`embed_news_duckdb.py`)
- **Input**: `rns_announcements` and `stock_news_enhanced` tables
- **Output**: `news_embeddings` (266 records)
- **Features**:
  - RNS regulatory announcements
  - Enhanced stock news articles
  - Content chunking and metadata

## Data Structure

### Confirmed Setups
- **Count**: 72 setups
- **Tickers**: 10 unique companies
- **Date Range**: 2024-06-06 to 2025-05-07
- **Format**: `{TICKER}_{YYYY-MM-DD}`

### LanceDB Tables Created

| Table | Records | Description |
|-------|---------|-------------|
| `fundamentals_embeddings` | 329 | Financial analysis with ratios |
| `news_embeddings` | 266 | News & RNS announcements |
| `userposts_embeddings` | 790 | User sentiment & posts |

## Key Features

### Data Quality
- ✅ **Historical Filtering**: Only includes data before setup dates
- ✅ **Performance Labels**: 10-day stock performance metrics
- ✅ **Confirmed Setups**: Only processes validated setup IDs
- ✅ **Rich Metadata**: Comprehensive context for each embedding

### Semantic Search
- ✅ **Vector Embeddings**: Using `all-MiniLM-L6-v2` model
- ✅ **Multi-domain**: Search across fundamentals, news, and sentiment
- ✅ **Contextual**: Rich metadata enables precise retrieval
- ✅ **Performance**: Fast similarity search with LanceDB

## Usage Examples

### Basic Retrieval
```python
import lancedb
from sentence_transformers import SentenceTransformer

# Connect to LanceDB
db = lancedb.connect('lancedb_store')
model = SentenceTransformer('all-MiniLM-L6-v2')

# Query fundamentals
query = "Strong ROE and revenue growth"
query_vector = model.encode(query)

fundamentals_table = db.open_table('fundamentals_embeddings')
results = fundamentals_table.search(query_vector).limit(5).to_pandas()

print(results[['setup_id', 'ticker', 'financial_summary']])
```

### Advanced Filtering
```python
# Filter by performance labels
high_performers = fundamentals_table.search(query_vector) \
    .where("outperformance_10d > 5.0") \
    .limit(3).to_pandas()

# Filter by sentiment
positive_sentiment = db.open_table('userposts_embeddings') \
    .search(query_vector) \
    .where("sentiment_score > 0.5") \
    .limit(5).to_pandas()
```

## Configuration

### Database Path
- Default: `data/sentiment_system.duckdb`
- Override: Set `db_path` parameter in pipeline classes

### LanceDB Storage
- Default: `lancedb_store/`
- Override: Set `lancedb_dir` parameter in pipeline classes

### Embedding Model
- Default: `all-MiniLM-L6-v2`
- Override: Set `embedding_model` parameter in pipeline classes

## Performance Metrics

### Pipeline Execution Times
- **Fundamentals**: ~8 seconds
- **User Posts**: ~10 seconds
- **News**: ~5 seconds
- **Total**: ~20 seconds

### Data Processing
- **Setup Validation**: 72 confirmed setups
- **Historical Filtering**: Applied to all fundamentals
- **Performance Labeling**: 236+ records with performance metrics
- **Sentiment Analysis**: Applied to all user posts

## Troubleshooting

### Common Issues

1. **Database Not Found**
   ```
   Error: DuckDB file not found at data/sentiment_system.duckdb
   ```
   - Ensure the database file exists in the correct location
   - Check file permissions

2. **Empty Results**
   ```
   Warning: No records found for confirmed setups
   ```
   - Verify setup IDs exist in the `setups` table
   - Check data filtering criteria

3. **Memory Issues**
   ```
   Error: OutOfMemoryError during embedding generation
   ```
   - Reduce batch size in embedding creation
   - Process data in smaller chunks

### Validation
```python
# Quick validation script
from scripts.setup_validator_duckdb import SetupValidatorDuckDB

validator = SetupValidatorDuckDB()
stats = validator.get_summary_stats()
print(f"Confirmed setups: {stats['total_confirmed_setups']}")
print(f"Data availability: {stats['data_counts']}")
```

## Migration from CSV

The DuckDB pipeline replaces the CSV-based approach with several improvements:

| Aspect | CSV Pipeline | DuckDB Pipeline |
|--------|-------------|-----------------|
| Setups | 8 confirmed | 72 confirmed |
| Data Source | Static CSV files | Live database |
| Performance | Manual filtering | Automated validation |
| Scalability | Limited | High |
| Maintenance | Manual updates | Automatic |

## Next Steps

1. **Custom Queries**: Implement domain-specific search interfaces
2. **API Integration**: Create REST API for retrieval services
3. **Performance Monitoring**: Add pipeline execution metrics
4. **Data Expansion**: Include additional data sources as needed
5. **Advanced Analytics**: Implement cross-domain correlation analysis

## Support

For issues or questions:
1. Check the pipeline logs in `pipeline_run_*.log`
2. Validate data availability with `setup_validator_duckdb.py`
3. Review LanceDB table statistics for data quality
4. Test individual pipeline components before running complete pipeline 