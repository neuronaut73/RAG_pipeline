# Fundamentals Agent Documentation

## Overview
The Fundamentals Agent (`agent_fundamentals.py`) provides intelligent retrieval and query capabilities for financial fundamentals and ratios data. It enables semantic search, structured queries, and performance analysis across company financial reports stored in LanceDB.

## Dependencies
```bash
conda activate sts
pip install lancedb sentence-transformers pandas numpy
```

## Architecture

### Data Flow
```
[fundamentals.csv] + [financial_ratios.csv] → [embed_fundamentals.py] → [LanceDB] → [agent_fundamentals.py]
```

### Agent Capabilities
- **Setup-based Retrieval**: Find fundamentals by trading setup ID
- **Company Retrieval**: Get financial data by ticker symbol with filtering
- **Date Range Queries**: Filter by report periods and dates
- **Semantic Search**: Natural language queries about financial characteristics
- **Company Similarity**: Find companies with similar financial profiles
- **Performance Analysis**: Correlate fundamentals with stock performance

## Core Features

### 1. FundamentalsAgent Class
Main interface for all fundamentals retrieval operations.

```python
from agent_fundamentals import FundamentalsAgent

# Initialize agent
agent = FundamentalsAgent(
    lancedb_dir="lancedb_store",
    table_name="fundamentals_embeddings", 
    embedding_model="all-MiniLM-L6-v2",
    default_limit=10
)
```

### 2. Retrieval Functions

#### retrieve_by_setup_id()
Links financial data to specific trading setups.

```python
# Get fundamentals for a trading setup
setup_records = agent.retrieve_by_setup_id(
    setup_id="SETUP_001",
    limit=5,
    include_labels=True
)

for record in setup_records:
    print(f"{record['ticker']}: ROE {record['roe']:.2%}")
    if record['has_performance_labels']:
        print(f"  Outperformance: {record['outperformance_10d']:.2f}%")
```

#### retrieve_by_ticker()
Company-specific financial data retrieval with filtering.

```python
# Get latest reports for a company
company_data = agent.retrieve_by_ticker(
    ticker="LLOY.L",
    limit=5,
    report_type="annual",  # or "quarterly"
    period_start="2020-01-01",
    period_end="2024-12-31",
    include_labels=True
)

for record in company_data:
    print(f"{record['period_end']}: Revenue ${record['revenue']:,.0f}")
    print(f"  ROE: {record['roe']:.2%}, ROA: {record['roa']:.2%}")
    print(f"  D/E Ratio: {record['debt_to_equity']:.2f}")
```

#### retrieve_by_date_range()
Time-based queries across all companies.

```python
# Get all reports from 2024
recent_data = agent.retrieve_by_date_range(
    start_date="2024-01-01",
    end_date="2024-12-31",
    limit=20,
    report_type="quarterly",
    include_labels=True
)

print(f"Found {len(recent_data)} quarterly reports from 2024")
```

#### semantic_search()
Natural language queries about financial characteristics.

```python
# Find profitable companies
profitable = agent.semantic_search(
    query="profitable companies with strong margins and returns",
    limit=10,
    filter_conditions="roe > 0.1",  # Additional SQL filter
    include_labels=True
)

# Find distressed companies
distressed = agent.semantic_search(
    query="companies with financial difficulties and high debt",
    limit=5
)

# Find growth companies
growth = agent.semantic_search(
    query="rapidly growing companies with expanding revenues",
    limit=8
)
```

#### find_similar_companies()
Discover companies with comparable financial profiles.

```python
# Find companies similar to LLOY.L
similar_companies = agent.find_similar_companies(
    reference_ticker="LLOY.L",
    reference_period="2023-12-31",  # Specific period (optional)
    limit=5,
    exclude_same_company=True,
    include_labels=True
)

for company in similar_companies:
    similarity = company.get('similarity_score', 0)
    print(f"{company['ticker']}: Similarity {similarity:.3f}")
    print(f"  ROE: {company['roe']:.2%}, D/E: {company['debt_to_equity']:.2f}")
```

#### analyze_performance_by_fundamentals()
Correlate financial metrics with stock performance.

```python
# High ROE companies performance
high_roe = agent.analyze_performance_by_fundamentals(
    metric_filter="roe > 0.15",  # ROE > 15%
    limit=10,
    sort_by_performance=True
)

# Low debt companies performance  
low_debt = agent.analyze_performance_by_fundamentals(
    metric_filter="debt_to_equity < 1.0",
    limit=10
)

# Profitable growth companies
profitable_growth = agent.analyze_performance_by_fundamentals(
    metric_filter="roe > 0.1 AND revenue > 1000000000",  # ROE > 10% and Revenue > £1B
    limit=15
)
```

## Data Schema

### Record Structure
Each fundamentals record contains:

```python
{
    # Identifiers
    'id': 'unique_record_id',
    'ticker': 'LLOY.L',
    'report_type': 'annual',
    'period_end': '2023-12-31',
    
    # Financial Summary
    'financial_summary': 'Comprehensive narrative analysis of company financials...',
    
    # Core Metrics
    'revenue': 12500000000.0,
    'net_income': 2100000000.0, 
    'total_assets': 456000000000.0,
    
    # Key Ratios
    'roe': 0.0876,  # 8.76%
    'roa': 0.0052,  # 0.52%
    'debt_to_equity': 5.09,
    'current_ratio': 1.23,
    'pe_ratio': 8.45,
    
    # Metadata
    'embedded_at': '2024-12-28 10:30:00',
    'has_performance_labels': True,
    
    # Performance Data (if available)
    'setup_id': 'SETUP_LLOY_2023Q4',
    'stock_return_10d': 0.0234,  # 2.34%
    'outperformance_10d': 0.0089,  # 0.89% vs benchmark
    'days_outperformed_10d': 6,
    
    # Similarity Score (for semantic searches)
    'similarity_score': 0.857
}
```

## Advanced Usage

### Complex Filtering
Combine semantic search with SQL-like filters:

```python
# Profitable tech companies
tech_profitable = agent.semantic_search(
    query="technology companies with software revenue",
    filter_conditions="roe > 0.12 AND revenue > 500000000",
    limit=10
)

# Banks with specific characteristics
banks = agent.semantic_search(
    query="major banks with retail banking operations",
    filter_conditions="ticker LIKE '%.L' AND total_assets > 100000000000",
    limit=8
)
```

### Performance Analysis Patterns
```python
# Value vs Growth analysis
value_stocks = agent.analyze_performance_by_fundamentals(
    metric_filter="pe_ratio < 15 AND roe > 0.08",
    sort_by_performance=True
)

growth_stocks = agent.analyze_performance_by_fundamentals(
    metric_filter="revenue > 1000000000 AND roe > 0.15",
    sort_by_performance=True
)

# Quality companies (Buffett-style)
quality = agent.analyze_performance_by_fundamentals(
    metric_filter="roe > 0.15 AND debt_to_equity < 2.0 AND current_ratio > 1.2"
)
```

### Comparative Analysis
```python
# Compare sectors
banking_companies = agent.semantic_search(
    query="banking and financial services companies",
    limit=20
)

retail_companies = agent.semantic_search(
    query="retail and consumer discretionary companies", 
    limit=20
)

# Calculate sector averages
def calculate_sector_metrics(companies):
    if not companies:
        return {}
    
    return {
        'avg_roe': sum(c['roe'] for c in companies) / len(companies),
        'avg_debt_ratio': sum(c['debt_to_equity'] for c in companies) / len(companies),
        'avg_current_ratio': sum(c['current_ratio'] for c in companies) / len(companies)
    }

banking_metrics = calculate_sector_metrics(banking_companies)
retail_metrics = calculate_sector_metrics(retail_companies)
```

## Error Handling

The agent includes comprehensive error handling:

```python
try:
    results = agent.retrieve_by_ticker("INVALID_TICKER")
    if not results:
        print("No records found for ticker")
except Exception as e:
    print(f"Error retrieving data: {e}")
```

## Performance Considerations

### Query Optimization
- Use specific filters to reduce search space
- Limit results appropriately for your use case
- Cache frequently accessed company data

### Memory Management
```python
# For large datasets, process in batches
def process_large_query(agent, tickers, batch_size=50):
    results = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        for ticker in batch:
            ticker_data = agent.retrieve_by_ticker(ticker, limit=5)
            results.extend(ticker_data)
    return results
```

## Integration with Other Components

### With News Agent
```python
from agent_news import NewsAgent
from agent_fundamentals import FundamentalsAgent

news_agent = NewsAgent()
fundamentals_agent = FundamentalsAgent()

# Comprehensive company analysis
def analyze_company(ticker):
    # Get latest fundamentals
    fundamentals = fundamentals_agent.retrieve_by_ticker(ticker, limit=3)
    
    # Get recent news
    news = news_agent.retrieve_by_ticker(ticker, limit=10)
    
    return {
        'fundamentals': fundamentals,
        'news': news,
        'combined_analysis': f"Company {ticker} analysis based on latest data"
    }
```

### With Trading Setups
```python
def get_setup_context(setup_id):
    # Get fundamentals for the setup
    fundamentals = fundamentals_agent.retrieve_by_setup_id(setup_id)
    
    # Get related news
    if fundamentals:
        ticker = fundamentals[0]['ticker']
        news = news_agent.retrieve_by_ticker(ticker, limit=5)
        
        return {
            'setup_id': setup_id,
            'fundamentals': fundamentals,
            'news': news
        }
```

## Table Statistics

Get insights about your data:

```python
stats = agent.get_table_stats()

print(f"Total records: {stats['total_records']}")
print(f"Unique companies: {stats['unique_tickers']}")
print(f"Date range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}")
print(f"Records with performance labels: {stats['records_with_labels']} ({stats['label_percentage']:.1f}%)")
print(f"Companies covered: {', '.join(stats['tickers'][:10])}...")
```

## Testing and Validation

Run the included test suite:

```bash
conda activate sts
python agent_fundamentals.py
```

This validates:
- ✅ Database connectivity
- ✅ Ticker-based retrieval
- ✅ Semantic search functionality  
- ✅ Date range queries
- ✅ Performance analysis
- ✅ Data quality and completeness

## Production Deployment

### Configuration
```python
# Production configuration
agent = FundamentalsAgent(
    lancedb_dir="/path/to/production/lancedb",
    table_name="fundamentals_embeddings",
    embedding_model="all-MiniLM-L6-v2",  # Or larger model for better accuracy
    default_limit=50  # Adjust based on use case
)
```

### Monitoring
```python
import logging

# Enhanced logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fundamentals_agent.log'),
        logging.StreamHandler()
    ]
)
```

## Troubleshooting

### Common Issues

**Empty Results**
```python
# Check if table exists and has data
stats = agent.get_table_stats()
if stats['total_records'] == 0:
    print("No data in fundamentals table. Run embed_fundamentals.py first.")
```

**Performance Issues**
```python
# Reduce query scope
results = agent.semantic_search(
    query="your query",
    limit=10,  # Start small
    filter_conditions="ticker IN ('LLOY.L', 'JDW.L')"  # Specific tickers
)
```

**Model Loading Issues**
```python
# Verify model availability
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model loaded successfully")
except Exception as e:
    print(f"Model loading failed: {e}")
```

---

The Fundamentals Agent provides powerful capabilities for analyzing financial data and correlating it with stock performance. Use it to build sophisticated trading systems, research platforms, and investment analysis tools. 