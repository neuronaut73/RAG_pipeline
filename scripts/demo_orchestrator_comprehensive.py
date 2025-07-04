#!/usr/bin/env python3
"""
demo_orchestrator_comprehensive.py - Comprehensive Knowledge Orchestrator Demonstration

Showcases the full capabilities of the Knowledge Orchestrator Agent with real-world
investment analysis use cases.
"""

from orchestrator_agent import KnowledgeOrchestrator
import time
import json


def print_section_header(title):
    """Print a formatted section header"""
    print(f"\n{'='*70}")
    print(f"🎯 {title}")
    print("="*70)


def print_subsection(title):
    """Print a formatted subsection header"""
    print(f"\n{'-'*50}")
    print(f"📊 {title}")
    print("-"*50)


def format_results_summary(response):
    """Format and display results summary"""
    print(f"✅ {response['summary']}")
    print(f"📊 Results: {response['total_results']} from {len(response['sources_queried'])} sources")
    print(f"⏱️  Execution time: {response['execution_time_seconds']:.2f}s")
    
    # Show source breakdown
    for source, results in response['results_by_source'].items():
        if results:
            print(f"   📁 {source}: {len(results)} results")


def demo_investment_workflow():
    """Demonstrate a complete investment research workflow"""
    
    print_section_header("INVESTMENT RESEARCH WORKFLOW")
    
    orchestrator = KnowledgeOrchestrator()
    
    print_subsection("Step 1: Market Screening - Find Profitable Companies")
    
    # Screen for profitable companies
    screening_query = "profitable companies with strong ROE and returns"
    screening_response = orchestrator.query(screening_query, max_results=8)
    
    format_results_summary(screening_response)
    
    # Extract companies for deeper analysis
    companies_found = screening_response['insights'].get('companies_mentioned', [])
    if companies_found:
        print(f"🏢 Companies identified: {', '.join(companies_found[:5])}")
        
        # Show financial insights
        if screening_response['insights'].get('financial_summary'):
            financial = screening_response['insights']['financial_summary']
            print(f"💰 Average ROE: {financial['average_roe']:.2%}")
            print(f"📈 Profitable companies: {financial['profitable_companies']}/{financial['companies_analyzed']}")
    
    print_subsection("Step 2: Individual Company Deep Dive")
    
    # Analyze top companies individually
    target_companies = ['AML', 'BGO', 'HWDN']  # Companies we know have data
    
    for company in target_companies:
        print(f"\n🔍 Analyzing {company}...")
        
        # Comprehensive company analysis
        company_analysis = orchestrator.query(f"{company} comprehensive analysis sentiment and performance")
        
        if company_analysis['success'] and company_analysis['total_results'] > 0:
            insights = company_analysis['insights']
            
            # Show key metrics
            metrics = []
            
            if insights.get('sentiment_summary'):
                sentiment = insights['sentiment_summary']
                sentiment_text = "Positive" if sentiment['average_sentiment'] > 0.02 else "Negative" if sentiment['average_sentiment'] < -0.02 else "Neutral"
                metrics.append(f"Sentiment: {sentiment_text} ({sentiment['average_sentiment']:.3f})")
            
            if insights.get('financial_summary'):
                financial = insights['financial_summary']
                metrics.append(f"Avg ROE: {financial['average_roe']:.2%}")
            
            news_count = len(company_analysis['results_by_source'].get('news', []))
            posts_count = len(company_analysis['results_by_source'].get('userposts', []))
            
            metrics.append(f"News: {news_count}, Posts: {posts_count}")
            
            print(f"   📊 {' | '.join(metrics)}")
        else:
            print(f"   ⚠️  Limited data available for {company}")


def demo_sentiment_monitoring():
    """Demonstrate real-time sentiment monitoring capabilities"""
    
    print_section_header("SOCIAL SENTIMENT MONITORING")
    
    orchestrator = KnowledgeOrchestrator()
    
    print_subsection("Portfolio Sentiment Dashboard")
    
    # Monitor sentiment across our available companies
    portfolio = ['BGO', 'AML', 'HWDN']
    sentiment_summary = {}
    
    for ticker in portfolio:
        print(f"\n📱 Monitoring {ticker} sentiment...")
        
        sentiment_response = orchestrator.query(f"{ticker} social sentiment analysis recent posts")
        
        if sentiment_response['success'] and sentiment_response['total_results'] > 0:
            insights = sentiment_response['insights']
            
            if insights.get('sentiment_summary'):
                sentiment = insights['sentiment_summary']
                sentiment_summary[ticker] = sentiment
                
                # Classify sentiment
                avg_sentiment = sentiment['average_sentiment']
                if avg_sentiment > 0.05:
                    status = "🟢 BULLISH"
                elif avg_sentiment < -0.05:
                    status = "🔴 BEARISH"
                else:
                    status = "🟡 NEUTRAL"
                
                print(f"   {status} | Score: {avg_sentiment:.3f} | Posts: {sentiment['total_posts']}")
                print(f"   Positive: {sentiment['positive_ratio']:.1%} | Negative: {sentiment['negative_ratio']:.1%}")
        else:
            print(f"   ⚠️  No sentiment data available")
    
    # Sentiment alerts
    print_subsection("Sentiment Alerts")
    
    for ticker, sentiment in sentiment_summary.items():
        avg_sentiment = sentiment['average_sentiment']
        
        if avg_sentiment > 0.08:
            print(f"🚨 HIGH BULLISH ALERT: {ticker} showing strong positive sentiment ({avg_sentiment:.3f})")
        elif avg_sentiment < -0.08:
            print(f"🚨 HIGH BEARISH ALERT: {ticker} showing strong negative sentiment ({avg_sentiment:.3f})")


def demo_news_impact_analysis():
    """Demonstrate news impact analysis"""
    
    print_section_header("NEWS IMPACT ANALYSIS")
    
    orchestrator = KnowledgeOrchestrator()
    
    print_subsection("Recent News & Market Reaction")
    
    # Look for recent news and correlate with sentiment
    news_query = "recent earnings announcements financial results news"
    news_response = orchestrator.query(news_query, max_results=10)
    
    format_results_summary(news_response)
    
    if news_response['total_results'] > 0:
        news_results = news_response['results_by_source'].get('news', [])
        sentiment_results = news_response['results_by_source'].get('userposts', [])
        
        print(f"\n📰 News Analysis:")
        print(f"   Articles found: {len(news_results)}")
        print(f"   Social reactions: {len(sentiment_results)}")
        
        # Show sample news headlines
        print(f"\n📋 Sample Headlines:")
        for i, news in enumerate(news_results[:3], 1):
            headline = news['content'].get('headline', 'No headline')
            ticker = news['content'].get('ticker', 'N/A')
            print(f"   {i}. {ticker}: {headline[:60]}...")


def demo_comparative_analysis():
    """Demonstrate comparative analysis between companies"""
    
    print_section_header("COMPARATIVE COMPANY ANALYSIS")
    
    orchestrator = KnowledgeOrchestrator()
    
    print_subsection("Multi-Company Comparison")
    
    # Compare companies across all dimensions
    comparison_query = "compare AML vs BGO financial performance sentiment and news coverage"
    comparison_response = orchestrator.query(comparison_query, max_results=15)
    
    format_results_summary(comparison_response)
    
    if comparison_response['total_results'] > 0:
        companies = comparison_response['insights'].get('companies_mentioned', [])
        
        print(f"\n🏢 Companies compared: {', '.join(companies)}")
        
        # Analyze results by company
        company_metrics = {}
        
        for company in companies:
            company_results = [r for r in comparison_response['top_results'] if r.get('ticker') == company]
            
            # Count results by source
            fundamentals_count = len([r for r in company_results if r['source'] == 'fundamentals'])
            news_count = len([r for r in company_results if r['source'] == 'news'])  
            posts_count = len([r for r in company_results if r['source'] == 'userposts'])
            
            company_metrics[company] = {
                'total_results': len(company_results),
                'fundamentals': fundamentals_count,
                'news': news_count,
                'posts': posts_count,
                'avg_relevance': sum(r['relevance_score'] for r in company_results) / len(company_results) if company_results else 0
            }
        
        # Display comparison table
        print(f"\n📊 Comparison Summary:")
        print(f"{'Company':<8} {'Total':<6} {'Fund':<5} {'News':<5} {'Posts':<6} {'Avg Score':<9}")
        print("-" * 50)
        
        for company, metrics in company_metrics.items():
            print(f"{company:<8} {metrics['total_results']:<6} {metrics['fundamentals']:<5} {metrics['news']:<5} {metrics['posts']:<6} {metrics['avg_relevance']:.3f}")


def demo_advanced_features():
    """Demonstrate advanced orchestrator features"""
    
    print_section_header("ADVANCED FEATURES DEMONSTRATION")
    
    orchestrator = KnowledgeOrchestrator()
    
    print_subsection("Cross-Encoder Re-ranking Impact")
    
    # Test with and without cross-encoder
    test_query = "bullish investment opportunities strong growth potential"
    
    print("🔄 Testing without cross-encoder re-ranking...")
    start_time = time.time()
    response_without = orchestrator.query(test_query, include_cross_ranking=False, max_results=5)
    time_without = time.time() - start_time
    
    print("🎯 Testing with cross-encoder re-ranking...")
    start_time = time.time()
    response_with = orchestrator.query(test_query, include_cross_ranking=True, max_results=5)
    time_with = time.time() - start_time
    
    print(f"\n📊 Performance Comparison:")
    print(f"   Without cross-encoder: {time_without:.3f}s, {response_without['total_results']} results")
    print(f"   With cross-encoder: {time_with:.3f}s, {response_with['total_results']} results")
    print(f"   Time overhead: +{time_with - time_without:.3f}s for improved relevance")
    
    print_subsection("Context-Aware Queries")
    
    # Demonstrate context-aware querying
    context_query = "recent sentiment analysis"
    
    # Without context
    print("🔍 Query without context:")
    response_no_context = orchestrator.query(context_query, max_results=3)
    print(f"   Results: {response_no_context['total_results']} (general sentiment analysis)")
    
    # With context
    print("\n🎯 Query with context (BGO focus):")
    response_with_context = orchestrator.query(
        context_query,
        query_context={'ticker': 'BGO'},
        max_results=3
    )
    print(f"   Results: {response_with_context['total_results']} (BGO-specific sentiment)")
    
    companies_found = response_with_context['insights'].get('companies_mentioned', [])
    if 'BGO' in companies_found:
        print("   ✅ Successfully focused on BGO")


def demo_system_status():
    """Demonstrate system status and capabilities"""
    
    print_section_header("SYSTEM STATUS & CAPABILITIES")
    
    orchestrator = KnowledgeOrchestrator()
    
    # Get system status
    status = orchestrator.get_system_status()
    
    print(f"🤖 Orchestrator Version: {status['orchestrator_version']}")
    print(f"🎯 Total Capabilities: {status['total_capabilities']}")
    print(f"🔀 Cross-encoder Enabled: {status['cross_encoder_enabled']}")
    
    print(f"\n📊 Available Agents:")
    for agent in status['agents_available']:
        if isinstance(agent, dict):
            if 'records' in agent:
                print(f"   ✅ {agent['name']}: {agent['records']} records, {agent['companies']} companies")
            else:
                print(f"   ✅ {agent['name']}: {agent['status']}")
        else:
            print(f"   ✅ {agent}")
    
    if status['agents_failed']:
        print(f"\n⚠️  Failed Agents: {', '.join(status['agents_failed'])}")


def main():
    """Run comprehensive orchestrator demonstration"""
    
    print("🤖 KNOWLEDGE ORCHESTRATOR - COMPREHENSIVE DEMONSTRATION")
    print("=" * 80)
    print("This demo showcases the full capabilities of the Knowledge Orchestrator Agent")
    print("for real-world investment analysis and research workflows.")
    
    try:
        # Run all demonstration modules
        demo_system_status()
        demo_investment_workflow()
        demo_sentiment_monitoring()
        demo_news_impact_analysis()
        demo_comparative_analysis()
        demo_advanced_features()
        
        print_section_header("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("🎉 The Knowledge Orchestrator Agent is ready for production use!")
        print("\n📚 Next steps:")
        print("   • Review the ORCHESTRATOR_AGENT_README.md for detailed API documentation")
        print("   • Integrate with your investment pipeline or dashboard")
        print("   • Customize query routing and synthesis for your specific use cases")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 