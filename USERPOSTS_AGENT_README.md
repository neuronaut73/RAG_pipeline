# UserPosts RAG Agent Documentation

## Overview

The UserPosts RAG Agent provides intelligent retrieval and analysis of social media posts from stock discussion forums. It combines semantic search capabilities with structured query functions to enable comprehensive analysis of user sentiment, trending topics, and social signals around trading opportunities.

## Key Features

### 🎯 **Core Retrieval Functions**
- **Setup-based Queries**: Link posts to specific trading setups
- **Ticker-based Queries**: Company-focused discussions and sentiment
- **Date Range Queries**: Temporal analysis of social sentiment
- **Semantic Search**: Natural language queries on post content
- **User Analysis**: Individual user behavior and posting patterns

### 📊 **Advanced Analytics**
- **Sentiment Analysis**: Comprehensive sentiment trends by ticker and time
- **Influential User Identification**: Discover key opinion leaders
- **Trending Topic Discovery**: Identify emerging themes and discussions
- **Social Signal Analysis**: Correlate user activity with market movements

## Quick Start

```python
from agent_userposts import UserPostsAgent

# Initialize the agent
agent = UserPostsAgent()

# Basic ticker search
bgo_posts = agent.retrieve_by_ticker("BGO", limit=10)

# Semantic search
bullish_posts = agent.semantic_search("bullish outlook strong buy")

# Sentiment analysis
sentiment = agent.analyze_sentiment_by_ticker("AML")
```

## API Reference

### Core Retrieval Methods

#### `retrieve_by_setup_id(setup_id, include_sentiment=True, sort_by="post_date")`

Retrieve all posts related to a specific trading setup.

**Parameters:**
- `setup_id` (str): Trading setup identifier
- `include_sentiment` (bool): Include sentiment summary in results
- `sort_by` (str): Field to sort results by

**Returns:** DataFrame with posts related to the setup

**Example:**
```python
# Get posts related to a specific setup
setup_posts = agent.retrieve_by_setup_id("BGO_20250529_001")
print(f"Found {len(setup_posts)} posts for this setup")

# Access sentiment summary
if hasattr(setup_posts, 'attrs') and 'sentiment_summary' in setup_posts.attrs:
    sentiment = setup_posts.attrs['sentiment_summary']
    print(f"Overall sentiment: {sentiment['average_sentiment']:.3f}")
```

#### `retrieve_by_ticker(ticker, start_date=None, end_date=None, sentiment_filter=None, user_filter=None, min_post_length=None, limit=None)`

Retrieve posts discussing a specific company.

**Parameters:**
- `ticker` (str): Company ticker symbol
- `start_date` (str, optional): Start date filter (YYYY-MM-DD)
- `end_date` (str, optional): End date filter (YYYY-MM-DD)
- `sentiment_filter` (str, optional): 'positive', 'negative', 'neutral'
- `user_filter` (List[str], optional): Specific users to include
- `min_post_length` (int, optional): Minimum post length in words
- `limit` (int, optional): Maximum results to return

**Returns:** DataFrame with filtered posts

**Example:**
```python
# Get positive sentiment posts about BGO from specific users
positive_bgo = agent.retrieve_by_ticker(
    ticker="BGO",
    sentiment_filter="positive",
    user_filter=["iWantThatOne", "Stockcompass"],
    start_date="2025-01-01",
    limit=20
)

# Get longer, more detailed posts
detailed_posts = agent.retrieve_by_ticker(
    ticker="AML",
    min_post_length=50,  # At least 50 words
    limit=10
)
```

#### `retrieve_by_date_range(start_date, end_date, ticker=None, sentiment_filter=None, sort_by="post_date", limit=None)`

Retrieve posts within a specific time period.

**Parameters:**
- `start_date` (str): Start date (YYYY-MM-DD)
- `end_date` (str): End date (YYYY-MM-DD)
- `ticker` (str, optional): Filter by specific ticker
- `sentiment_filter` (str, optional): Sentiment filter
- `sort_by` (str): Sorting field
- `limit` (int, optional): Maximum results

**Returns:** DataFrame with posts in date range

**Example:**
```python
# Get all posts from last week
from datetime import datetime, timedelta

end_date = datetime.now()
start_date = end_date - timedelta(days=7)

recent_posts = agent.retrieve_by_date_range(
    start_date=start_date.strftime('%Y-%m-%d'),
    end_date=end_date.strftime('%Y-%m-%d'),
    sentiment_filter="positive"
)

# Analyze activity by day
daily_activity = recent_posts.groupby(recent_posts['post_date'].dt.date).size()
```

#### `semantic_search(query, ticker=None, sentiment_filter=None, date_range=None, user_filter=None, limit=None)`

Perform natural language search on post content.

**Parameters:**
- `query` (str): Natural language search query
- `ticker` (str, optional): Filter by ticker
- `sentiment_filter` (str, optional): Sentiment filter
- `date_range` (Tuple[str, str], optional): (start_date, end_date)
- `user_filter` (List[str], optional): Specific users
- `limit` (int, optional): Maximum results

**Returns:** DataFrame with semantically similar posts

**Example:**
```python
# Find posts about growth potential
growth_posts = agent.semantic_search("strong growth potential revenue increase")

# Search for bearish sentiment with filters
bearish_filtered = agent.semantic_search(
    query="bearish outlook negative concerns",
    ticker="BGO",
    sentiment_filter="negative",
    date_range=("2025-01-01", "2025-06-01")
)

# Find posts about technical analysis
technical_posts = agent.semantic_search("chart pattern breakout support resistance")
```

### Advanced Analytics Methods

#### `retrieve_by_user(user_handle, ticker=None, start_date=None, end_date=None, sentiment_filter=None, limit=None)`

Analyze posting patterns of a specific user.

**Example:**
```python
# Get all posts from a specific user
user_activity = agent.retrieve_by_user("iWantThatOne")

# User's sentiment on a specific ticker
user_bgo_sentiment = agent.retrieve_by_user(
    user_handle="iWantThatOne",
    ticker="BGO",
    sentiment_filter="positive"
)

print(f"User posted {len(user_activity)} times")
print(f"Average sentiment: {user_activity['sentiment_score'].mean():.3f}")
```

#### `analyze_sentiment_by_ticker(ticker, start_date=None, end_date=None, group_by="day")`

Comprehensive sentiment analysis for a specific ticker.

**Parameters:**
- `ticker` (str): Company ticker
- `start_date` (str, optional): Analysis start date
- `end_date` (str, optional): Analysis end date  
- `group_by` (str): Grouping period ('day', 'week', 'month')

**Returns:** Dictionary with detailed sentiment analysis

**Example:**
```python
# Comprehensive sentiment analysis
analysis = agent.analyze_sentiment_by_ticker("AML")

print(f"Total posts: {analysis['total_posts']}")
print(f"Overall sentiment: {analysis['overall_sentiment']['average_score']:.3f}")
print(f"Positive posts: {analysis['overall_sentiment']['positive_percentage']}%")

# Most positive and negative posts
for post in analysis['top_positive_posts'][:2]:
    print(f"Positive: {post['user_handle']} ({post['sentiment_score']:.3f})")

# Most active users
for user, count in list(analysis['most_active_users'].items())[:3]:
    print(f"Active user: {user} ({count} posts)")
```

#### `find_influential_users(ticker=None, min_posts=5, limit=10)`

Identify influential users based on activity and engagement.

**Returns:** DataFrame with user influence metrics

**Example:**
```python
# Find most influential users overall
influential = agent.find_influential_users(limit=5)

for idx, user in influential.iterrows():
    print(f"{user['user_handle']}: {user['total_posts']} posts, "
          f"influence score: {user['influence_score']:.1f}")

# Find influential users for specific ticker
bgo_influencers = agent.find_influential_users(ticker="BGO", min_posts=3)
```

#### `get_trending_topics(days_back=7, ticker=None, min_mentions=3)`

Discover trending topics and themes in recent discussions.

**Returns:** Dictionary with trending topic analysis

**Example:**
```python
# Get trending topics from last week
trending = agent.get_trending_topics(days_back=7)

print(f"Period: {trending['period']}")
print(f"Total posts: {trending['total_posts']}")

# Show trending words
for word, count in trending['trending_words'][:5]:
    print(f"Trending: {word} ({count} mentions)")

# Sentiment by ticker
for ticker, metrics in trending['sentiment_by_ticker'].items():
    print(f"{ticker}: {metrics['avg_sentiment']:.3f} sentiment, {metrics['post_count']} posts")
```

## Data Schema

### Post Records
Each record contains:

```python
{
    'post_id': 'unique_identifier',
    'setup_id': 'trading_setup_id', 
    'ticker': 'company_symbol',
    'user_handle': 'username',
    'post_date': 'timestamp',
    'post_content': 'full_post_text',
    'post_length': 'word_count',
    'sentiment_score': 'float_-1_to_1',
    'positive_indicators': 'count',
    'negative_indicators': 'count',
    'has_performance_labels': 'boolean',
    'stock_return_10d': 'float_percentage',
    'outperformance_10d': 'float_percentage'
}
```

### Key Metrics
- **Sentiment Score**: Range -1.0 (very negative) to +1.0 (very positive)
- **Positive/Negative Indicators**: Count of sentiment words
- **Post Length**: Word count for content analysis
- **Performance Labels**: When available, actual stock performance outcomes

## Integration Patterns

### With Trading Systems
```python
# Get social sentiment for active positions
active_positions = ["BGO", "AML", "HWDN"]

for ticker in active_positions:
    sentiment = agent.analyze_sentiment_by_ticker(ticker)
    current_sentiment = sentiment['overall_sentiment']['average_score']
    
    if current_sentiment > 0.05:
        print(f"{ticker}: Positive social sentiment ({current_sentiment:.3f})")
    elif current_sentiment < -0.05:
        print(f"{ticker}: Negative social sentiment ({current_sentiment:.3f})")
```

### With News Analysis
```python
# Combine with news agent for comprehensive analysis
from agent_news import NewsAgent

news_agent = NewsAgent()
posts_agent = UserPostsAgent()

ticker = "BGO"

# Get both news and social sentiment
news_articles = news_agent.retrieve_by_ticker(ticker, limit=5)
social_posts = posts_agent.retrieve_by_ticker(ticker, limit=10)

# Compare sentiment sources
news_sentiment = news_articles['sentiment_score'].mean()
social_sentiment = social_posts['sentiment_score'].mean()

print(f"News sentiment: {news_sentiment:.3f}")
print(f"Social sentiment: {social_sentiment:.3f}")
```

### Investment Strategy Applications

#### Sentiment Momentum Strategy
```python
def find_sentiment_momentum(agent, min_posts=5):
    """Find stocks with strong positive sentiment momentum"""
    
    # Get all tickers with sufficient posts
    all_data = agent.table.to_pandas()
    ticker_counts = all_data['ticker'].value_counts()
    active_tickers = ticker_counts[ticker_counts >= min_posts].index
    
    momentum_scores = {}
    
    for ticker in active_tickers:
        analysis = agent.analyze_sentiment_by_ticker(ticker)
        if 'error' not in analysis:
            sentiment = analysis['overall_sentiment']
            
            # Calculate momentum score
            positive_pct = sentiment['positive_percentage']
            avg_sentiment = sentiment['average_score']
            post_count = analysis['total_posts']
            
            momentum_score = (positive_pct * 0.4 + (avg_sentiment + 1) * 50 * 0.4 + min(post_count/10, 1) * 0.2)
            momentum_scores[ticker] = momentum_score
    
    # Sort by momentum
    sorted_momentum = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_momentum[:5]

# Find top momentum stocks
momentum_stocks = find_sentiment_momentum(agent)
for ticker, score in momentum_stocks:
    print(f"{ticker}: Momentum score {score:.1f}")
```

#### Contrarian Signal Detection
```python
def find_contrarian_opportunities(agent):
    """Find stocks with negative sentiment but potential value"""
    
    # Get stocks with negative sentiment
    negative_sentiment_results = []
    
    all_data = agent.table.to_pandas()
    for ticker in all_data['ticker'].unique():
        analysis = agent.analyze_sentiment_by_ticker(ticker)
        if 'error' not in analysis:
            overall = analysis['overall_sentiment']
            
            # Look for negative sentiment with high engagement
            if (overall['average_score'] < -0.02 and 
                overall['negative_percentage'] > 40 and
                analysis['total_posts'] >= 5):
                
                negative_sentiment_results.append({
                    'ticker': ticker,
                    'sentiment_score': overall['average_score'],
                    'negative_percentage': overall['negative_percentage'],
                    'post_count': analysis['total_posts']
                })
    
    return sorted(negative_sentiment_results, key=lambda x: x['sentiment_score'])

contrarian_opportunities = find_contrarian_opportunities(agent)
```

## Performance Considerations

### Optimization Tips
1. **Use Filters**: Apply ticker, date, and sentiment filters to reduce data volume
2. **Limit Results**: Set appropriate limits for large queries
3. **Batch Processing**: For multiple tickers, process in batches
4. **Cache Results**: Store frequently accessed sentiment analyses

### Memory Management
```python
# For large-scale analysis, process tickers individually
tickers = ["BGO", "AML", "HWDN"]
sentiment_cache = {}

for ticker in tickers:
    analysis = agent.analyze_sentiment_by_ticker(ticker)
    sentiment_cache[ticker] = analysis['overall_sentiment']['average_score']
    # Process results before moving to next ticker
```

## Troubleshooting

### Common Issues

1. **No Results Found**
   - Check ticker symbols are correct
   - Verify date ranges match data availability
   - Ensure filters aren't too restrictive

2. **Sentiment Analysis Errors**
   - Check for sufficient post volume (minimum 3-5 posts recommended)
   - Verify ticker exists in dataset

3. **Performance Issues**
   - Use appropriate limits on query results
   - Apply filters to reduce search space
   - Consider date range limitations

### Debug Mode
```python
import logging
logging.getLogger('agent_userposts').setLevel(logging.DEBUG)

# Enable detailed logging for troubleshooting
agent = UserPostsAgent()
```

## Production Deployment

### Configuration
```python
# Production configuration
agent = UserPostsAgent(
    lancedb_dir="/path/to/production/lancedb",
    default_limit=50,  # Reasonable limit for production
    embedding_model="all-MiniLM-L6-v2"  # Fast, efficient model
)
```

### Error Handling
```python
try:
    results = agent.semantic_search("investment opportunity")
    if len(results) == 0:
        print("No results found for query")
    else:
        # Process results
        pass
except Exception as e:
    logger.error(f"Query failed: {e}")
    # Implement fallback logic
```

### Integration Monitoring
```python
# Monitor query performance
import time

start_time = time.time()
results = agent.retrieve_by_ticker("BGO", limit=100)
query_time = time.time() - start_time

print(f"Query completed in {query_time:.2f} seconds")
print(f"Retrieved {len(results)} results")
```

## Summary

The UserPosts RAG Agent provides comprehensive social media analysis capabilities for stock discussions. It combines semantic search with structured queries to enable:

- **Social Sentiment Analysis**: Track public opinion and mood
- **User Behavior Analysis**: Identify influential voices and patterns  
- **Trending Topic Discovery**: Spot emerging themes and discussions
- **Integration Ready**: Seamless integration with other domain agents

**Key Benefits:**
- Fast semantic search with 384-dimensional embeddings
- Comprehensive sentiment analysis with temporal trends
- User influence scoring and activity analysis
- Flexible filtering and query options
- Production-ready error handling and optimization

The agent is designed to be a core component of the RAG pipeline, providing social signal intelligence to complement fundamental and news analysis for comprehensive investment research. 