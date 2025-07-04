#!/usr/bin/env python3
"""
demo_userposts_agent.py - UserPosts RAG Agent Comprehensive Demonstration

Showcases the full capabilities of the UserPosts domain RAG agent including
sentiment analysis, user influence scoring, and social signal discovery.
"""

import pandas as pd
from datetime import datetime, timedelta
from agent_userposts import UserPostsAgent

def print_section_header(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"🎯 {title}")
    print("="*70)

def print_subsection(title):
    """Print a formatted subsection header"""
    print(f"\n📊 {title}")
    print("-" * 50)

def demo_userposts_agent():
    """Comprehensive demonstration of UserPosts RAG agent capabilities"""
    
    print("🚀 USERPOSTS RAG AGENT - COMPREHENSIVE DEMONSTRATION")
    print("=" * 70)
    print("This demo showcases social sentiment analysis, user behavior insights,")
    print("and trending topic discovery for stock discussion forums.")
    
    # Initialize agent
    print("\n🤖 Initializing UserPosts Agent...")
    agent = UserPostsAgent()
    
    # Get overview statistics
    all_data = agent.table.to_pandas()
    total_posts = len(all_data)
    unique_users = all_data['user_handle'].nunique()
    unique_tickers = all_data['ticker'].nunique()
    avg_sentiment = all_data['sentiment_score'].mean()
    
    print(f"📈 Dataset Overview:")
    print(f"   • Total Posts: {total_posts}")
    print(f"   • Unique Users: {unique_users}")
    print(f"   • Companies: {unique_tickers}")
    print(f"   • Average Sentiment: {avg_sentiment:.3f}")
    
    # Section 1: Ticker-Based Analysis
    print_section_header("TICKER-BASED SOCIAL SENTIMENT ANALYSIS")
    
    for ticker in ["BGO", "AML", "HWDN"]:
        print_subsection(f"{ticker} Social Sentiment Analysis")
        
        # Get comprehensive sentiment analysis
        sentiment_analysis = agent.analyze_sentiment_by_ticker(ticker)
        
        if 'error' not in sentiment_analysis:
            overall = sentiment_analysis['overall_sentiment']
            
            print(f"   📊 Posts analyzed: {sentiment_analysis['total_posts']}")
            print(f"   💭 Overall sentiment: {overall['average_score']:.3f}")
            print(f"   ✅ Positive posts: {overall['positive_posts']} ({overall['positive_percentage']:.1f}%)")
            print(f"   ❌ Negative posts: {overall['negative_posts']} ({overall['negative_percentage']:.1f}%)")
            print(f"   ⚪ Neutral posts: {overall['neutral_posts']}")
            
            # Most active users for this ticker
            if sentiment_analysis['most_active_users']:
                top_users = list(sentiment_analysis['most_active_users'].items())[:3]
                print(f"   👤 Most active users: {', '.join([f'{user}({count})' for user, count in top_users])}")
            
            # Show sample positive post if available
            if sentiment_analysis['top_positive_posts']:
                top_positive = sentiment_analysis['top_positive_posts'][0]
                print(f"   🌟 Most positive post: {top_positive['user_handle']} ({top_positive['sentiment_score']:.3f})")
                print(f"      \"{top_positive['post_content'][:80]}...\"")
        else:
            print(f"   ⚠️ No posts found for {ticker}")
    
    # Section 2: Semantic Search Demonstrations
    print_section_header("SEMANTIC SEARCH CAPABILITIES")
    
    search_queries = [
        ("bullish outlook strong buy", "Investment Opportunities"),
        ("bearish negative concerns", "Risk Signals"),
        ("dividend growth income", "Income Strategy"),
        ("technical analysis chart pattern", "Technical Discussion")
    ]
    
    for query, description in search_queries:
        print_subsection(f"{description} - '{query}'")
        
        results = agent.semantic_search(query, limit=3)
        
        if len(results) > 0:
            print(f"   🔍 Found {len(results)} relevant posts")
            
            for idx, post in results.iterrows():
                sentiment_emoji = "🟢" if post['sentiment_score'] > 0.02 else "🔴" if post['sentiment_score'] < -0.02 else "⚪"
                print(f"   {sentiment_emoji} {post['ticker']} | {post['user_handle']} ({post['sentiment_score']:.3f})")
                print(f"      \"{post['post_content'][:90]}...\"")
        else:
            print(f"   ⚠️ No results found for '{query}'")
    
    # Section 3: User Influence Analysis
    print_section_header("USER INFLUENCE & BEHAVIOR ANALYSIS")
    
    print_subsection("Most Influential Users")
    influential_users = agent.find_influential_users(limit=8)
    
    if len(influential_users) > 0:
        print("   🌟 Top influential users by activity and engagement:")
        
        for idx, user in influential_users.iterrows():
            print(f"   {idx+1:2d}. {user['user_handle']:15s} | "
                  f"Posts: {user['total_posts']:2d} | "
                  f"Sentiment: {user['avg_sentiment']:6.3f} | "
                  f"Influence: {user['influence_score']:5.1f}")
    
    # Deep dive on top user
    if len(influential_users) > 0:
        top_user = influential_users.iloc[0]['user_handle']
        print_subsection(f"Deep Dive: {top_user}")
        
        user_posts = agent.retrieve_by_user(top_user)
        
        if len(user_posts) > 0:
            user_posts['post_date'] = pd.to_datetime(user_posts['post_date'], format='mixed')
            
            print(f"   📊 Total posts: {len(user_posts)}")
            print(f"   💭 Average sentiment: {user_posts['sentiment_score'].mean():.3f}")
            print(f"   📝 Average post length: {user_posts['post_length'].mean():.1f} words")
            print(f"   🏢 Companies discussed: {', '.join(user_posts['ticker'].unique())}")
            print(f"   📅 Date range: {user_posts['post_date'].min().strftime('%Y-%m-%d')} to {user_posts['post_date'].max().strftime('%Y-%m-%d')}")
            
            # Show sentiment distribution
            positive_posts = len(user_posts[user_posts['sentiment_score'] > 0.02])
            negative_posts = len(user_posts[user_posts['sentiment_score'] < -0.02])
            neutral_posts = len(user_posts) - positive_posts - negative_posts
            
            print(f"   📈 Sentiment distribution: {positive_posts} positive, {negative_posts} negative, {neutral_posts} neutral")
    
    # Section 4: Temporal Analysis
    print_section_header("TEMPORAL SENTIMENT TRENDS")
    
    print_subsection("Recent Activity Analysis")
    
    # Analyze posts by month/week if data allows
    all_data['post_date'] = pd.to_datetime(all_data['post_date'], format='mixed')
    
    # Group by month
    monthly_activity = all_data.groupby(all_data['post_date'].dt.to_period('M')).agg({
        'post_id': 'count',
        'sentiment_score': 'mean',
        'ticker': lambda x: x.nunique()
    }).round(3)
    
    monthly_activity.columns = ['posts', 'avg_sentiment', 'unique_tickers']
    
    print("   📅 Monthly Activity Summary:")
    for period, data in monthly_activity.iterrows():
        print(f"   {period}: {data['posts']} posts, {data['avg_sentiment']:.3f} sentiment, {data['unique_tickers']} companies")
    
    # Section 5: Cross-Company Sentiment Comparison
    print_section_header("CROSS-COMPANY SENTIMENT COMPARISON")
    
    print_subsection("Sentiment Ranking by Company")
    
    company_sentiment = {}
    for ticker in all_data['ticker'].unique():
        ticker_posts = all_data[all_data['ticker'] == ticker]
        if len(ticker_posts) >= 3:  # Minimum posts for reliable sentiment
            avg_sentiment = ticker_posts['sentiment_score'].mean()
            post_count = len(ticker_posts)
            positive_pct = len(ticker_posts[ticker_posts['sentiment_score'] > 0.02]) / post_count * 100
            
            company_sentiment[ticker] = {
                'avg_sentiment': avg_sentiment,
                'post_count': post_count,
                'positive_percentage': positive_pct
            }
    
    # Sort by sentiment
    sorted_sentiment = sorted(company_sentiment.items(), key=lambda x: x[1]['avg_sentiment'], reverse=True)
    
    print("   🏆 Companies ranked by social sentiment:")
    for rank, (ticker, metrics) in enumerate(sorted_sentiment, 1):
        print(f"   {rank}. {ticker}: {metrics['avg_sentiment']:6.3f} sentiment | "
              f"{metrics['post_count']:2d} posts | {metrics['positive_percentage']:4.1f}% positive")
    
    # Section 6: Content Analysis & Trending Topics
    print_section_header("CONTENT ANALYSIS & TRENDING TOPICS")
    
    print_subsection("Word Frequency Analysis")
    
    # Simple word frequency analysis
    all_text = ' '.join(all_data['post_content'].fillna('').astype(str))
    
    import re
    from collections import Counter
    
    # Extract words (simple approach)
    words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text.lower())
    
    # Filter common words
    stop_words = {
        'this', 'that', 'with', 'have', 'will', 'been', 'from', 'they', 'know', 
        'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when', 'come',
        'here', 'just', 'like', 'long', 'make', 'many', 'over', 'such', 'take',
        'than', 'them', 'well', 'were', 'what', 'your', 'also', 'back', 'call',
        'came', 'each', 'even', 'find', 'give', 'hand', 'high', 'keep', 'kind',
        'last', 'left', 'life', 'live', 'look', 'made', 'most', 'move', 'must',
        'name', 'need', 'next', 'only', 'open', 'part', 'play', 'right', 'said',
        'same', 'seem', 'show', 'side', 'tell', 'turn', 'used', 'want', 'ways',
        'work', 'year', 'years', 'think', 'stock', 'price'
    }
    
    filtered_words = [word for word in words if word not in stop_words and len(word) > 4]
    word_counts = Counter(filtered_words)
    
    print("   📝 Most frequently mentioned terms:")
    for word, count in word_counts.most_common(10):
        print(f"   • {word}: {count} mentions")
    
    # Section 7: Setup Integration Analysis
    print_section_header("TRADING SETUP INTEGRATION")
    
    print_subsection("Posts with Performance Labels")
    
    labeled_posts = all_data[all_data['has_performance_labels'] == True]
    
    if len(labeled_posts) > 0:
        print(f"   📊 Posts with performance data: {len(labeled_posts)} ({len(labeled_posts)/len(all_data)*100:.1f}%)")
        
        # Sentiment vs Performance correlation
        positive_sentiment_posts = labeled_posts[labeled_posts['sentiment_score'] > 0.02]
        negative_sentiment_posts = labeled_posts[labeled_posts['sentiment_score'] < -0.02]
        
        if len(positive_sentiment_posts) > 0:
            avg_return_positive = positive_sentiment_posts['stock_return_10d'].mean()
            print(f"   📈 Avg 10-day return for positive sentiment posts: {avg_return_positive:.2f}%")
        
        if len(negative_sentiment_posts) > 0:
            avg_return_negative = negative_sentiment_posts['stock_return_10d'].mean()
            print(f"   📉 Avg 10-day return for negative sentiment posts: {avg_return_negative:.2f}%")
        
        # Setup correlation
        setup_sentiment = labeled_posts.groupby('setup_id').agg({
            'sentiment_score': 'mean',
            'stock_return_10d': 'first',
            'post_id': 'count'
        }).round(3)
        
        setup_sentiment.columns = ['avg_sentiment', 'return_10d', 'post_count']
        setup_sentiment = setup_sentiment[setup_sentiment['post_count'] >= 2]  # Minimum 2 posts
        
        if len(setup_sentiment) > 0:
            print(f"\n   🎯 Setup-level analysis ({len(setup_sentiment)} setups with 2+ posts):")
            
            # Sort by return
            setup_sentiment_sorted = setup_sentiment.sort_values('return_10d', ascending=False)
            
            for setup_id, data in setup_sentiment_sorted.head(5).iterrows():
                print(f"   • {setup_id}: {data['avg_sentiment']:.3f} sentiment → {data['return_10d']:.2f}% return")
    else:
        print("   ⚠️ No posts with performance labels found")
    
    # Section 8: Integration Examples
    print_section_header("INTEGRATION & STRATEGY EXAMPLES")
    
    print_subsection("Social Sentiment Momentum Strategy")
    
    # Find stocks with positive momentum
    momentum_candidates = []
    
    for ticker in all_data['ticker'].unique():
        ticker_posts = all_data[all_data['ticker'] == ticker]
        
        if len(ticker_posts) >= 5:  # Minimum posts for analysis
            recent_posts = ticker_posts.tail(3)  # Last 3 posts
            overall_sentiment = ticker_posts['sentiment_score'].mean()
            recent_sentiment = recent_posts['sentiment_score'].mean()
            positive_ratio = len(ticker_posts[ticker_posts['sentiment_score'] > 0.02]) / len(ticker_posts)
            
            # Momentum score
            momentum_score = (recent_sentiment * 0.4 + overall_sentiment * 0.3 + positive_ratio * 0.3)
            
            momentum_candidates.append({
                'ticker': ticker,
                'momentum_score': momentum_score,
                'post_count': len(ticker_posts),
                'recent_sentiment': recent_sentiment,
                'overall_sentiment': overall_sentiment,
                'positive_ratio': positive_ratio
            })
    
    momentum_candidates.sort(key=lambda x: x['momentum_score'], reverse=True)
    
    print("   🚀 Top momentum candidates by social sentiment:")
    for candidate in momentum_candidates[:3]:
        print(f"   • {candidate['ticker']}: Score {candidate['momentum_score']:.3f} | "
              f"Recent: {candidate['recent_sentiment']:.3f} | "
              f"Overall: {candidate['overall_sentiment']:.3f} | "
              f"{candidate['positive_ratio']:.1%} positive")
    
    print_subsection("Risk Alert System")
    
    # Find negative sentiment spikes
    risk_alerts = []
    
    for ticker in all_data['ticker'].unique():
        ticker_posts = all_data[all_data['ticker'] == ticker]
        
        if len(ticker_posts) >= 3:
            recent_posts = ticker_posts.tail(2)  # Last 2 posts
            recent_sentiment = recent_posts['sentiment_score'].mean()
            negative_ratio = len(recent_posts[recent_posts['sentiment_score'] < -0.02]) / len(recent_posts)
            
            if recent_sentiment < -0.05 or negative_ratio > 0.5:
                risk_alerts.append({
                    'ticker': ticker,
                    'recent_sentiment': recent_sentiment,
                    'negative_ratio': negative_ratio,
                    'alert_score': abs(recent_sentiment) + negative_ratio
                })
    
    risk_alerts.sort(key=lambda x: x['alert_score'], reverse=True)
    
    if risk_alerts:
        print("   ⚠️ Social sentiment risk alerts:")
        for alert in risk_alerts[:3]:
            print(f"   • {alert['ticker']}: {alert['recent_sentiment']:.3f} sentiment | "
                  f"{alert['negative_ratio']:.1%} negative posts | "
                  f"Alert score: {alert['alert_score']:.3f}")
    else:
        print("   ✅ No significant negative sentiment alerts")
    
    # Final Summary
    print_section_header("DEMONSTRATION SUMMARY")
    
    print("🎯 UserPosts RAG Agent Capabilities Demonstrated:")
    print("   ✅ Ticker-based sentiment analysis with comprehensive metrics")
    print("   ✅ Semantic search for thematic content discovery")
    print("   ✅ User influence scoring and behavior analysis")
    print("   ✅ Temporal sentiment trend analysis")
    print("   ✅ Cross-company sentiment comparison")
    print("   ✅ Content analysis and trending topic identification")
    print("   ✅ Trading setup integration and performance correlation")
    print("   ✅ Investment strategy applications (momentum & risk)")
    
    print(f"\n📊 Dataset Insights:")
    print(f"   • {total_posts} social media posts analyzed")
    print(f"   • {unique_users} unique forum participants")
    print(f"   • {unique_tickers} companies under discussion")
    print(f"   • Sentiment distribution enables momentum and risk strategies")
    
    print(f"\n🔧 Integration Ready:")
    print(f"   • Compatible with Fundamentals and News agents")
    print(f"   • Structured query support for trading systems")
    print(f"   • Real-time sentiment monitoring capabilities")
    print(f"   • Performance correlation analysis ready")
    
    print("\n" + "="*70)
    print("✅ USERPOSTS RAG AGENT DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("="*70)

if __name__ == "__main__":
    demo_userposts_agent() 