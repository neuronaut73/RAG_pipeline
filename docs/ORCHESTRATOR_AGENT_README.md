# Knowledge Orchestrator Agent - Complete ✅

## Overview

The **Knowledge Orchestrator Agent** is the central coordination layer of the RAG pipeline system. It intelligently routes user queries across all domain agents (Fundamentals, News, UserPosts), aggregates results, applies advanced cross-ranking, and synthesizes comprehensive responses.

## 🎯 **Key Features**

### **🧠 Intelligent Query Routing**
- **Intent Recognition**: Automatically detects query type (company analysis, sentiment analysis, financial screening, etc.)
- **Multi-Agent Coordination**: Routes sub-queries to relevant domain agents based on query intent
- **Context-Aware**: Extracts tickers, setup IDs, and date ranges from natural language queries

### **🔄 Advanced Result Aggregation**
- **Cross-Domain Integration**: Combines results from fundamentals, news, and social sentiment
- **Cross-Encoder Re-ranking**: Uses transformer models to improve result relevance
- **Standardized Output**: Normalizes results from different agents into unified format

### **📊 Comprehensive Synthesis**
- **Multi-Source Insights**: Extracts key patterns across financial, news, and social data
- **Natural Language Summaries**: Generates human-readable analysis summaries
- **Rich Metadata**: Provides relevance scores, source attribution, and confidence metrics

## 🚀 **Quick Start**

```python
from orchestrator_agent import KnowledgeOrchestrator

# Initialize orchestrator
orchestrator = KnowledgeOrchestrator()

# Company analysis query
response = orchestrator.query("Analyze AML sentiment and financial performance")

# Access results
print(f"Summary: {response['summary']}")
print(f"Total results: {response['total_results']}")
print(f"Companies found: {response['insights']['companies_mentioned']}")

# Get sentiment insights
if response['insights']['sentiment_summary']:
    sentiment = response['insights']['sentiment_summary']
    print(f"Social sentiment: {sentiment['average_sentiment']:.3f}")
```

## 📋 **Query Types & Routing**

### **1. Company Analysis**
```python
# Comprehensive company analysis - routes to ALL agents
response = orchestrator.query("Analyze BGO company performance and news")
# → Fundamentals + News + UserPosts
```

### **2. Financial Analysis**
```python
# Financial screening - routes to fundamentals + news
response = orchestrator.query("Show me profitable companies with strong ROE")
# → Fundamentals + News
```

### **3. Sentiment Analysis**
```python
# Social sentiment - routes to user posts
response = orchestrator.query("What is the market sentiment for AML?")
# → UserPosts
```

### **4. News Analysis**
```python
# News focus - routes to news + sentiment
response = orchestrator.query("Recent earnings announcements")
# → News + UserPosts
```

### **5. Semantic Search**
```python
# Intelligent content search - routes to ALL agents
response = orchestrator.query("bullish investment opportunities")
# → Fundamentals + News + UserPosts
```

## 🔧 **Core Methods**

### **Main Query Interface**

#### `query(user_query, query_context=None, max_results=10, include_cross_ranking=True)`

**Parameters:**
- `user_query` (str): Natural language query
- `query_context` (dict, optional): Additional context (ticker, dates, etc.)
- `max_results` (int): Maximum results to return
- `include_cross_ranking` (bool): Whether to apply cross-encoder re-ranking

**Returns:** Comprehensive response with results from all relevant agents

**Example:**
```python
# Basic query
response = orchestrator.query("BGO financial analysis")

# Query with context
response = orchestrator.query(
    "Recent sentiment analysis",
    query_context={
        'ticker': 'AML',
        'start_date': '2025-01-01',
        'end_date': '2025-01-31'
    }
)

# High-volume query with cross-ranking
response = orchestrator.query(
    "profitable companies in technology sector",
    max_results=20,
    include_cross_ranking=True
)
```

### **System Status**

#### `get_system_status()`

Returns status of all domain agents and orchestrator capabilities.

```python
status = orchestrator.get_system_status()
print(f"Available agents: {len(status['agents_available'])}")
print(f"Cross-encoder enabled: {status['cross_encoder_enabled']}")
```

## 📊 **Response Structure**

```python
{
    'success': True,
    'query': 'Original user query',
    'query_type': 'company_analysis',
    'total_results': 15,
    'sources_queried': ['fundamentals', 'news', 'userposts'],
    
    # Natural language summary
    'summary': 'Found comprehensive analysis for BGO. Retrieved 5 financial records, 8 news articles, 12 social posts...',
    
    # Cross-domain insights
    'insights': {
        'companies_mentioned': ['BGO', 'AML'],
        'time_range': {
            'earliest': '2024-09-01',
            'latest': '2025-01-20'
        },
        'sentiment_summary': {
            'average_sentiment': 0.045,
            'total_posts': 12,
            'positive_ratio': 0.33,
            'negative_ratio': 0.08
        },
        'financial_summary': {
            'companies_analyzed': 5,
            'average_roe': 0.18,
            'profitable_companies': 4
        },
        'news_summary': {
            'articles_found': 8,
            'recent_announcements': 3
        }
    },
    
    # Results organized by source
    'results_by_source': {
        'fundamentals': [...],
        'news': [...],
        'userposts': [...]
    },
    
    # Top-ranked results across all sources
    'top_results': [
        {
            'source': 'fundamentals',
            'content': {...},
            'relevance_score': 0.94,
            'result_type': 'financial_data',
            'ticker': 'BGO',
            'timestamp': '2024-12-31'
        },
        ...
    ],
    
    'execution_time_seconds': 0.45,
    'generated_at': '2025-01-21T15:30:00'
}
```

## 🎯 **Use Cases**

### **1. Investment Research Workflow**

```python
# Step 1: Screen for profitable companies
screening = orchestrator.query("profitable companies with ROE > 15%")

# Step 2: Analyze specific company
for company in screening['insights']['companies_mentioned'][:3]:
    analysis = orchestrator.query(f"comprehensive analysis of {company}")
    
    sentiment = analysis['insights']['sentiment_summary']
    financial = analysis['insights']['financial_summary']
    
    print(f"{company}: Sentiment {sentiment['average_sentiment']:.3f}, "
          f"ROE {financial['average_roe']:.2%}")
```

### **2. Market Sentiment Monitoring**

```python
# Monitor sentiment across portfolio
portfolio = ['BGO', 'AML', 'HWDN']

for ticker in portfolio:
    sentiment_analysis = orchestrator.query(f"{ticker} social sentiment analysis")
    
    if sentiment_analysis['insights']['sentiment_summary']:
        sentiment = sentiment_analysis['insights']['sentiment_summary']
        
        if sentiment['average_sentiment'] < -0.05:
            print(f"⚠️  {ticker}: Negative sentiment alert ({sentiment['average_sentiment']:.3f})")
        elif sentiment['average_sentiment'] > 0.05:
            print(f"📈 {ticker}: Positive sentiment trend ({sentiment['average_sentiment']:.3f})")
```

### **3. News Impact Analysis**

```python
# Analyze impact of recent news on sentiment
news_query = orchestrator.query("recent earnings announcements and market reaction")

results_by_source = news_query['results_by_source']
news_results = results_by_source.get('news', [])
sentiment_results = results_by_source.get('userposts', [])

print(f"Found {len(news_results)} news articles and {len(sentiment_results)} social reactions")
```

### **4. Comparative Company Analysis**

```python
# Compare multiple companies across all dimensions
comparison = orchestrator.query("compare BGO vs AML financial performance and sentiment")

companies = comparison['insights']['companies_mentioned']
for company in companies:
    company_results = [r for r in comparison['top_results'] if r['ticker'] == company]
    print(f"\n{company}: {len(company_results)} relevant results")
    
    # Show top insight for each company
    if company_results:
        top_result = company_results[0]
        print(f"  Top insight: {top_result['source']} (score: {top_result['relevance_score']:.2f})")
```

## 🔍 **Advanced Features**

### **Cross-Encoder Re-ranking**

The orchestrator uses a cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to improve result relevance:

```python
# Enable advanced re-ranking (default: True)
response = orchestrator.query(
    "investment opportunities",
    include_cross_ranking=True  # Uses cross-encoder for better relevance
)

# Access relevance scores
for result in response['top_results']:
    print(f"Score: {result['relevance_score']:.3f} - {result['source']}")
```

### **Context-Aware Queries**

```python
# Provide explicit context
response = orchestrator.query(
    "How has sentiment changed recently?",
    query_context={
        'ticker': 'BGO',
        'start_date': '2025-01-01',
        'end_date': '2025-01-31'
    }
)

# The orchestrator will:
# 1. Focus on BGO specifically
# 2. Filter results to the date range
# 3. Emphasize temporal analysis
```

### **Query Plan Inspection**

For debugging and optimization, you can inspect the query execution plan:

```python
# The orchestrator creates execution plans that determine:
# - Which agents to query
# - How to transform the query for each agent  
# - How to synthesize results

# Example internal flow for "BGO sentiment analysis":
# 1. Intent: USER_SENTIMENT
# 2. Target agents: ['userposts']  
# 3. Agent query: retrieve_by_ticker('BGO')
# 4. Synthesis: sentiment-focused summary
```

## ⚡ **Performance Considerations**

### **Query Optimization**
- Specific queries (with tickers) are faster than semantic searches
- Cross-encoder re-ranking adds ~0.1-0.3s but improves relevance significantly
- Limit results appropriately to reduce processing time

### **Memory Management**
```python
# For production use, configure reasonable limits
orchestrator = KnowledgeOrchestrator(
    default_limit=10,           # Reasonable default
    enable_cross_encoding=True  # Better relevance
)

# Use appropriate result limits
response = orchestrator.query("broad search", max_results=50)  # May be slow
response = orchestrator.query("specific query", max_results=10)  # Faster
```

### **Caching Strategies**
```python
# Cache frequently accessed company data
company_cache = {}

def get_company_analysis(ticker):
    if ticker not in company_cache:
        company_cache[ticker] = orchestrator.query(f"{ticker} comprehensive analysis")
    return company_cache[ticker]
```

## 🔧 **Configuration Options**

```python
orchestrator = KnowledgeOrchestrator(
    lancedb_dir="lancedb_store",                              # LanceDB location
    default_limit=10,                                         # Default result count
    enable_cross_encoding=True,                               # Use cross-encoder
    cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2"  # Re-ranking model
)
```

## 🎯 **Integration Examples**

### **With Investment Pipeline**

```python
class InvestmentAnalyzer:
    def __init__(self):
        self.orchestrator = KnowledgeOrchestrator()
    
    def analyze_opportunity(self, ticker):
        # Get comprehensive analysis
        analysis = self.orchestrator.query(f"{ticker} comprehensive investment analysis")
        
        insights = analysis['insights']
        
        # Extract key metrics
        financial_strength = insights.get('financial_summary', {}).get('average_roe', 0)
        social_sentiment = insights.get('sentiment_summary', {}).get('average_sentiment', 0)
        news_coverage = len(analysis['results_by_source'].get('news', []))
        
        # Calculate investment score
        score = (
            min(financial_strength * 100, 50) +  # ROE contribution (max 50 points)
            min(max(social_sentiment * 100, -25), 25) +  # Sentiment (-25 to +25)
            min(news_coverage * 2, 25)  # News coverage (max 25)
        )
        
        return {
            'ticker': ticker,
            'investment_score': score,
            'financial_strength': financial_strength,
            'social_sentiment': social_sentiment,
            'news_coverage': news_coverage,
            'recommendation': 'BUY' if score > 60 else 'HOLD' if score > 40 else 'SELL'
        }

# Usage
analyzer = InvestmentAnalyzer()
result = analyzer.analyze_opportunity('BGO')
print(f"{result['ticker']}: {result['recommendation']} (Score: {result['investment_score']:.1f})")
```

### **With Dashboard/API**

```python
from flask import Flask, jsonify, request

app = Flask(__name__)
orchestrator = KnowledgeOrchestrator()

@app.route('/api/query', methods=['POST'])
def api_query():
    data = request.json
    
    response = orchestrator.query(
        user_query=data['query'],
        query_context=data.get('context'),
        max_results=data.get('max_results', 10)
    )
    
    # Return JSON response
    return jsonify(response)

@app.route('/api/companies/<ticker>/analysis')
def company_analysis(ticker):
    response = orchestrator.query(f"{ticker} comprehensive analysis")
    return jsonify(response)

# Start API server
# app.run(host='0.0.0.0', port=5000)
```

## 📊 **System Requirements**

### **Dependencies**
```bash
conda activate sts
pip install lancedb sentence-transformers pandas numpy
```

### **Hardware Recommendations**
- **Memory**: 8GB+ RAM (embedding models + data)
- **Storage**: 1GB+ for LanceDB tables and models
- **CPU**: Multi-core recommended for cross-encoder processing

### **Performance Metrics**
- **Query latency**: 0.1-0.5s (depending on complexity and cross-ranking)
- **Throughput**: ~10-50 queries/second (depending on result size)
- **Memory usage**: ~2-4GB (with all models loaded)

## 🔧 **Troubleshooting**

### **Common Issues**

**No results found:**
```python
# Check if agents are properly initialized
status = orchestrator.get_system_status()
print(f"Available agents: {[a['name'] for a in status['agents_available']]}")

# Verify ticker is in the dataset
response = orchestrator.query("BGO")  # Should find user posts
```

**Cross-encoder errors:**
```python
# Disable cross-encoding if having issues
orchestrator = KnowledgeOrchestrator(enable_cross_encoding=False)
```

**Query routing issues:**
```python
# Provide explicit context for better routing
response = orchestrator.query(
    "sentiment analysis",
    query_context={'ticker': 'AML'}  # Explicit ticker
)
```

## 🚀 **Future Enhancements**

- **LLM Integration**: Add GPT/Claude for natural language response generation
- **Real-time Data**: Connect to live market data feeds
- **Advanced Analytics**: ML-based trend detection and forecasting
- **Caching Layer**: Redis/Memcached for improved performance
- **API Gateway**: Production-ready REST/GraphQL API

---

The Knowledge Orchestrator Agent provides powerful coordination capabilities for multi-domain RAG queries. Use it to build sophisticated trading systems, research platforms, and investment analysis tools that leverage the full breadth of your financial data ecosystem. 