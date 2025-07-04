#!/usr/bin/env python3
"""
agent_userposts.py - UserPosts Domain RAG Agent

Provides intelligent retrieval and analysis of user posts from stock discussion forums.
Supports queries by setup_id, ticker, date, semantic search, and social media analytics.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from pathlib import Path

# LanceDB and embeddings
import lancedb
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserPostsAgent:
    """
    RAG Agent for UserPosts domain - intelligent retrieval and analysis of social media discussions
    about stocks and trading setups.
    """
    
    def __init__(
        self,
        lancedb_dir: str = "lancedb_store",
        table_name: str = "userposts_embeddings",
        embedding_model: str = "all-MiniLM-L6-v2",
        default_limit: int = 10
    ):
        """
        Initialize UserPosts RAG Agent
        
        Args:
            lancedb_dir: Directory containing LanceDB tables
            table_name: Name of UserPosts embeddings table
            embedding_model: Sentence transformer model name
            default_limit: Default number of results to return
        """
        self.lancedb_dir = lancedb_dir
        self.table_name = table_name
        self.default_limit = default_limit
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)
        
        # Connect to LanceDB
        self._connect_to_database()
        
        logger.info("UserPostsAgent initialized successfully")
    
    def _connect_to_database(self) -> None:
        """Connect to LanceDB and load the table"""
        try:
            self.db = lancedb.connect(self.lancedb_dir)
            self.table = self.db.open_table(self.table_name)
            
            # Get basic table info
            sample_data = self.table.to_pandas()
            total_records = len(sample_data)
            unique_tickers = sample_data['ticker'].nunique()
            unique_users = sample_data['user_handle'].nunique()
            
            logger.info(f"Connected to {self.table_name}: {total_records} posts, {unique_tickers} companies, {unique_users} users")
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def retrieve_by_setup_id(
        self,
        setup_id: str,
        include_sentiment: bool = True,
        sort_by: str = "post_date"
    ) -> pd.DataFrame:
        """
        Retrieve posts related to a specific trading setup
        
        Args:
            setup_id: Trading setup identifier
            include_sentiment: Whether to include sentiment analysis
            sort_by: Field to sort results by
            
        Returns:
            DataFrame with posts related to the setup
        """
        try:
            # Get all data and filter by setup_id
            all_data = self.table.to_pandas()
            results = all_data[all_data['setup_id'] == setup_id].copy()
            
            if len(results) == 0:
                logger.warning(f"No posts found for setup_id: {setup_id}")
                return pd.DataFrame()
            
            # Sort results
            if sort_by in results.columns:
                if sort_by == 'post_date':
                    results['post_date'] = pd.to_datetime(results['post_date'], format='mixed')
                    results = results.sort_values('post_date', ascending=False)
                else:
                    results = results.sort_values(sort_by, ascending=False)
            
            # Add sentiment summary if requested
            if include_sentiment and len(results) > 0:
                sentiment_summary = self._generate_sentiment_summary(results)
                results.attrs['sentiment_summary'] = sentiment_summary
            
            logger.info(f"Retrieved {len(results)} posts for setup {setup_id}")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving posts for setup {setup_id}: {e}")
            return pd.DataFrame()
    
    def retrieve_by_ticker(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        sentiment_filter: Optional[str] = None,
        user_filter: Optional[List[str]] = None,
        min_post_length: Optional[int] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Retrieve posts about a specific company/ticker
        
        Args:
            ticker: Company ticker symbol
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD) 
            sentiment_filter: 'positive', 'negative', 'neutral', or None
            user_filter: List of specific users to include
            min_post_length: Minimum post length in words
            limit: Maximum number of results
            
        Returns:
            DataFrame with posts about the ticker
        """
        try:
            # Get all data and filter by ticker
            all_data = self.table.to_pandas()
            results = all_data[all_data['ticker'] == ticker].copy()
            
            if len(results) == 0:
                logger.warning(f"No posts found for ticker: {ticker}")
                return pd.DataFrame()
            
            # Convert post_date for filtering
            results['post_date'] = pd.to_datetime(results['post_date'], format='mixed')
            
            # Apply date filters
            if start_date:
                start_dt = pd.to_datetime(start_date)
                results = results[results['post_date'] >= start_dt]
            
            if end_date:
                end_dt = pd.to_datetime(end_date)
                results = results[results['post_date'] <= end_dt]
            
            # Apply sentiment filter
            if sentiment_filter:
                if sentiment_filter.lower() == 'positive':
                    results = results[results['sentiment_score'] > 0.02]
                elif sentiment_filter.lower() == 'negative':
                    results = results[results['sentiment_score'] < -0.02]
                elif sentiment_filter.lower() == 'neutral':
                    results = results[results['sentiment_score'].between(-0.02, 0.02)]
            
            # Apply user filter
            if user_filter:
                results = results[results['user_handle'].isin(user_filter)]
            
            # Apply minimum post length filter
            if min_post_length:
                results = results[results['post_length'] >= min_post_length]
            
            # Sort by date (most recent first)
            results = results.sort_values('post_date', ascending=False)
            
            # Apply limit
            if limit:
                results = results.head(limit)
            elif self.default_limit:
                results = results.head(self.default_limit)
            
            logger.info(f"Retrieved {len(results)} posts for ticker {ticker}")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving posts for ticker {ticker}: {e}")
            return pd.DataFrame()
    
    def retrieve_by_date_range(
        self,
        start_date: str,
        end_date: str,
        ticker: Optional[str] = None,
        sentiment_filter: Optional[str] = None,
        sort_by: str = "post_date",
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Retrieve posts within a date range
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            ticker: Optional ticker filter
            sentiment_filter: 'positive', 'negative', 'neutral', or None
            sort_by: Field to sort by
            limit: Maximum number of results
            
        Returns:
            DataFrame with posts in the date range
        """
        try:
            # Get all data
            all_data = self.table.to_pandas()
            all_data['post_date'] = pd.to_datetime(all_data['post_date'], format='mixed')
            
            # Apply date filter
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            results = all_data[
                (all_data['post_date'] >= start_dt) & 
                (all_data['post_date'] <= end_dt)
            ].copy()
            
            # Apply ticker filter
            if ticker:
                results = results[results['ticker'] == ticker]
            
            # Apply sentiment filter
            if sentiment_filter:
                if sentiment_filter.lower() == 'positive':
                    results = results[results['sentiment_score'] > 0.02]
                elif sentiment_filter.lower() == 'negative':
                    results = results[results['sentiment_score'] < -0.02]
                elif sentiment_filter.lower() == 'neutral':
                    results = results[results['sentiment_score'].between(-0.02, 0.02)]
            
            # Sort results
            if sort_by in results.columns:
                ascending = True if sort_by in ['post_date'] else False
                results = results.sort_values(sort_by, ascending=ascending)
            
            # Apply limit
            if limit:
                results = results.head(limit)
            elif self.default_limit:
                results = results.head(self.default_limit)
            
            logger.info(f"Retrieved {len(results)} posts from {start_date} to {end_date}")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving posts by date range: {e}")
            return pd.DataFrame()
    
    def semantic_search(
        self,
        query: str,
        ticker: Optional[str] = None,
        sentiment_filter: Optional[str] = None,
        date_range: Optional[Tuple[str, str]] = None,
        user_filter: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Perform semantic search on user posts
        
        Args:
            query: Natural language search query
            ticker: Optional ticker filter
            sentiment_filter: Optional sentiment filter
            date_range: Optional (start_date, end_date) tuple
            user_filter: Optional list of users to include
            limit: Maximum number of results
            
        Returns:
            DataFrame with semantically similar posts
        """
        try:
            # Generate query embedding
            query_vector = self.model.encode(query)
            
            # Start with vector search
            search_query = self.table.search(query_vector)
            
            # Build WHERE clause for filters
            where_conditions = []
            
            if ticker:
                where_conditions.append(f"ticker = '{ticker}'")
            
            if sentiment_filter:
                if sentiment_filter.lower() == 'positive':
                    where_conditions.append("sentiment_score > 0.02")
                elif sentiment_filter.lower() == 'negative':
                    where_conditions.append("sentiment_score < -0.02")
                elif sentiment_filter.lower() == 'neutral':
                    where_conditions.append("sentiment_score >= -0.02 AND sentiment_score <= 0.02")
            
            if user_filter:
                user_list = "', '".join(user_filter)
                where_conditions.append(f"user_handle IN ('{user_list}')")
            
            # Apply WHERE conditions
            if where_conditions:
                where_clause = " AND ".join(where_conditions)
                search_query = search_query.where(where_clause)
            
            # Set limit
            search_limit = limit or self.default_limit
            results = search_query.limit(search_limit).to_pandas()
            
            # Apply date range filter if specified (post-processing)
            if date_range and len(results) > 0:
                results['post_date'] = pd.to_datetime(results['post_date'], format='mixed')
                start_date, end_date = date_range
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                results = results[
                    (results['post_date'] >= start_dt) & 
                    (results['post_date'] <= end_dt)
                ]
            
            logger.info(f"Semantic search for '{query}': {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic search for '{query}': {e}")
            return pd.DataFrame()
    
    def retrieve_by_user(
        self,
        user_handle: str,
        ticker: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        sentiment_filter: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Retrieve posts from a specific user
        
        Args:
            user_handle: Username to search for
            ticker: Optional ticker filter  
            start_date: Optional start date
            end_date: Optional end date
            sentiment_filter: Optional sentiment filter
            limit: Maximum number of results
            
        Returns:
            DataFrame with posts from the user
        """
        try:
            # Get all data and filter by user
            all_data = self.table.to_pandas()
            results = all_data[all_data['user_handle'] == user_handle].copy()
            
            if len(results) == 0:
                logger.warning(f"No posts found for user: {user_handle}")
                return pd.DataFrame()
            
            # Apply additional filters
            if ticker:
                results = results[results['ticker'] == ticker]
            
            if start_date or end_date:
                results['post_date'] = pd.to_datetime(results['post_date'], format='mixed')
                
                if start_date:
                    start_dt = pd.to_datetime(start_date)
                    results = results[results['post_date'] >= start_dt]
                
                if end_date:
                    end_dt = pd.to_datetime(end_date)
                    results = results[results['post_date'] <= end_dt]
            
            if sentiment_filter:
                if sentiment_filter.lower() == 'positive':
                    results = results[results['sentiment_score'] > 0.02]
                elif sentiment_filter.lower() == 'negative':
                    results = results[results['sentiment_score'] < -0.02]
                elif sentiment_filter.lower() == 'neutral':
                    results = results[results['sentiment_score'].between(-0.02, 0.02)]
            
            # Sort by date (most recent first)
            if 'post_date' in results.columns:
                results = results.sort_values('post_date', ascending=False)
            
            # Apply limit
            if limit:
                results = results.head(limit)
            elif self.default_limit:
                results = results.head(self.default_limit)
            
            logger.info(f"Retrieved {len(results)} posts from user {user_handle}")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving posts for user {user_handle}: {e}")
            return pd.DataFrame()
    
    def analyze_sentiment_by_ticker(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        group_by: str = "day"
    ) -> Dict[str, Any]:
        """
        Analyze sentiment trends for a specific ticker
        
        Args:
            ticker: Company ticker symbol
            start_date: Optional start date
            end_date: Optional end date  
            group_by: Grouping period ('day', 'week', 'month')
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            # Get posts for ticker
            posts = self.retrieve_by_ticker(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                limit=None
            )
            
            if len(posts) == 0:
                return {"error": f"No posts found for ticker {ticker}"}
            
            # Calculate overall sentiment metrics
            avg_sentiment = posts['sentiment_score'].mean()
            positive_posts = len(posts[posts['sentiment_score'] > 0.02])
            negative_posts = len(posts[posts['sentiment_score'] < -0.02])
            neutral_posts = len(posts[posts['sentiment_score'].between(-0.02, 0.02)])
            
            # Time-based analysis
            posts['post_date'] = pd.to_datetime(posts['post_date'], format='mixed')
            
            if group_by == "day":
                posts['period'] = posts['post_date'].dt.date
            elif group_by == "week":
                posts['period'] = posts['post_date'].dt.to_period('W')
            elif group_by == "month":
                posts['period'] = posts['post_date'].dt.to_period('M')
            
            # Group by period and calculate sentiment
            period_sentiment = posts.groupby('period').agg({
                'sentiment_score': ['mean', 'count'],
                'positive_indicators': 'sum',
                'negative_indicators': 'sum'
            }).round(3)
            
            # Top positive and negative posts
            top_positive = posts.nlargest(3, 'sentiment_score')[['user_handle', 'post_date', 'sentiment_score', 'post_content']]
            top_negative = posts.nsmallest(3, 'sentiment_score')[['user_handle', 'post_date', 'sentiment_score', 'post_content']]
            
            # Most active users
            user_activity = posts['user_handle'].value_counts().head(5)
            
            analysis = {
                'ticker': ticker,
                'total_posts': len(posts),
                'date_range': f"{posts['post_date'].min().strftime('%Y-%m-%d')} to {posts['post_date'].max().strftime('%Y-%m-%d')}",
                'overall_sentiment': {
                    'average_score': round(avg_sentiment, 3),
                    'positive_posts': positive_posts,
                    'negative_posts': negative_posts,
                    'neutral_posts': neutral_posts,
                    'positive_percentage': round(positive_posts / len(posts) * 100, 1),
                    'negative_percentage': round(negative_posts / len(posts) * 100, 1)
                },
                'period_sentiment': period_sentiment.to_dict(),
                'top_positive_posts': top_positive.to_dict('records'),
                'top_negative_posts': top_negative.to_dict('records'),
                'most_active_users': user_activity.to_dict()
            }
            
            logger.info(f"Sentiment analysis completed for {ticker}: {len(posts)} posts analyzed")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {ticker}: {e}")
            return {"error": str(e)}
    
    def find_influential_users(
        self,
        ticker: Optional[str] = None,
        min_posts: int = 5,
        limit: int = 10
    ) -> pd.DataFrame:
        """
        Identify influential users based on post frequency and engagement
        
        Args:
            ticker: Optional ticker filter
            min_posts: Minimum number of posts required
            limit: Maximum number of users to return
            
        Returns:
            DataFrame with influential user metrics
        """
        try:
            # Get all data
            all_data = self.table.to_pandas()
            
            # Apply ticker filter if specified
            if ticker:
                all_data = all_data[all_data['ticker'] == ticker]
            
            # Calculate user metrics
            user_metrics = all_data.groupby('user_handle').agg({
                'post_id': 'count',  # Total posts
                'sentiment_score': ['mean', 'std'],  # Sentiment metrics
                'post_length': 'mean',  # Average post length
                'positive_indicators': 'sum',  # Total positive sentiment
                'negative_indicators': 'sum',  # Total negative sentiment
                'ticker': 'nunique'  # Number of different tickers discussed
            }).round(3)
            
            # Flatten column names
            user_metrics.columns = [
                'total_posts', 'avg_sentiment', 'sentiment_volatility',
                'avg_post_length', 'total_positive_indicators', 'total_negative_indicators',
                'tickers_discussed'
            ]
            
            # Filter by minimum posts
            user_metrics = user_metrics[user_metrics['total_posts'] >= min_posts]
            
            # Calculate influence score (combination of activity and engagement)
            user_metrics['influence_score'] = (
                user_metrics['total_posts'] * 0.4 +
                user_metrics['avg_post_length'] * 0.2 +
                user_metrics['tickers_discussed'] * 0.2 +
                (user_metrics['total_positive_indicators'] + user_metrics['total_negative_indicators']) * 0.2
            ).round(2)
            
            # Sort by influence score
            user_metrics = user_metrics.sort_values('influence_score', ascending=False)
            
            # Add recent activity
            all_data['post_date'] = pd.to_datetime(all_data['post_date'], format='mixed')
            recent_cutoff = all_data['post_date'].max() - timedelta(days=30)
            recent_activity = all_data[all_data['post_date'] >= recent_cutoff].groupby('user_handle').size()
            user_metrics['recent_posts_30d'] = user_metrics.index.map(recent_activity).fillna(0).astype(int)
            
            # Apply limit
            result = user_metrics.head(limit).reset_index()
            
            logger.info(f"Identified {len(result)} influential users")
            return result
            
        except Exception as e:
            logger.error(f"Error finding influential users: {e}")
            return pd.DataFrame()
    
    def get_trending_topics(
        self,
        days_back: int = 7,
        ticker: Optional[str] = None,
        min_mentions: int = 3
    ) -> Dict[str, Any]:
        """
        Identify trending topics and themes in recent posts
        
        Args:
            days_back: Number of days to look back
            ticker: Optional ticker filter
            min_mentions: Minimum mentions required for a topic
            
        Returns:
            Dictionary with trending topic analysis
        """
        try:
            # Get recent posts
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            recent_posts = self.retrieve_by_date_range(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                ticker=ticker,
                limit=None
            )
            
            if len(recent_posts) == 0:
                return {"error": "No recent posts found"}
            
            # Extract common words and phrases
            all_text = ' '.join(recent_posts['post_content'].fillna('').astype(str))
            
            # Simple keyword extraction (can be enhanced with NLP libraries)
            import re
            words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
            
            # Filter out common stop words
            stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
            
            filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
            
            # Count word frequency
            from collections import Counter
            word_counts = Counter(filtered_words)
            trending_words = [(word, count) for word, count in word_counts.most_common(20) if count >= min_mentions]
            
            # Sentiment by ticker (if no ticker filter)
            sentiment_by_ticker = {}
            if not ticker:
                for tick in recent_posts['ticker'].unique():
                    tick_posts = recent_posts[recent_posts['ticker'] == tick]
                    sentiment_by_ticker[tick] = {
                        'post_count': len(tick_posts),
                        'avg_sentiment': round(tick_posts['sentiment_score'].mean(), 3),
                        'positive_posts': len(tick_posts[tick_posts['sentiment_score'] > 0.02]),
                        'negative_posts': len(tick_posts[tick_posts['sentiment_score'] < -0.02])
                    }
            
            # Most active discussions (by post count)
            user_activity = recent_posts['user_handle'].value_counts().head(10)
            
            analysis = {
                'period': f"Last {days_back} days",
                'total_posts': len(recent_posts),
                'trending_words': trending_words,
                'sentiment_by_ticker': sentiment_by_ticker,
                'most_active_users': user_activity.to_dict(),
                'avg_sentiment_overall': round(recent_posts['sentiment_score'].mean(), 3)
            }
            
            logger.info(f"Trending topics analysis completed for last {days_back} days")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing trending topics: {e}")
            return {"error": str(e)}
    
    def _generate_sentiment_summary(self, posts_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate a summary of sentiment metrics for a set of posts"""
        if len(posts_df) == 0:
            return {}
        
        total_posts = len(posts_df)
        avg_sentiment = posts_df['sentiment_score'].mean()
        positive_posts = len(posts_df[posts_df['sentiment_score'] > 0.02])
        negative_posts = len(posts_df[posts_df['sentiment_score'] < -0.02])
        neutral_posts = total_posts - positive_posts - negative_posts
        
        return {
            'total_posts': total_posts,
            'average_sentiment': round(avg_sentiment, 3),
            'positive_posts': positive_posts,
            'negative_posts': negative_posts,
            'neutral_posts': neutral_posts,
            'positive_percentage': round(positive_posts / total_posts * 100, 1),
            'negative_percentage': round(negative_posts / total_posts * 100, 1)
        }


def main():
    """Demo and testing of UserPosts RAG Agent"""
    
    print("=" * 70)
    print("USERPOSTS RAG AGENT DEMONSTRATION")
    print("=" * 70)
    
    try:
        # Initialize agent
        print("\n🤖 Initializing UserPosts Agent...")
        agent = UserPostsAgent()
        
        # Test 1: Retrieve by ticker
        print("\n" + "="*50)
        print("📊 TEST 1: Retrieve by Ticker (BGO)")
        print("="*50)
        
        bgo_posts = agent.retrieve_by_ticker("BGO", limit=5)
        if len(bgo_posts) > 0:
            print(f"Found {len(bgo_posts)} BGO posts")
            print(f"Date range: {bgo_posts['post_date'].min()} to {bgo_posts['post_date'].max()}")
            print(f"Users: {', '.join(bgo_posts['user_handle'].unique()[:5])}")
            print(f"Average sentiment: {bgo_posts['sentiment_score'].mean():.3f}")
            
            print("\nSample posts:")
            for idx, post in bgo_posts.head(3).iterrows():
                print(f"  • {post['user_handle']} ({post['sentiment_score']:.3f}): {post['post_content'][:80]}...")
        
        # Test 2: Semantic search
        print("\n" + "="*50)
        print("🔍 TEST 2: Semantic Search - 'bullish outlook investment'")
        print("="*50)
        
        search_results = agent.semantic_search("bullish outlook investment", limit=5)
        if len(search_results) > 0:
            print(f"Found {len(search_results)} semantically similar posts")
            for idx, post in search_results.head(3).iterrows():
                print(f"  • {post['ticker']} by {post['user_handle']} ({post['sentiment_score']:.3f})")
                print(f"    '{post['post_content'][:100]}...'")
        
        # Test 3: Date range analysis
        print("\n" + "="*50)
        print("📅 TEST 3: Recent Posts (Last 30 days)")
        print("="*50)
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        recent_posts = agent.retrieve_by_date_range(start_date, end_date, limit=10)
        if len(recent_posts) > 0:
            print(f"Found {len(recent_posts)} recent posts")
            ticker_counts = recent_posts['ticker'].value_counts()
            print(f"By ticker: {dict(ticker_counts)}")
            
            sentiment_dist = agent._generate_sentiment_summary(recent_posts)
            print(f"Sentiment: {sentiment_dist['positive_percentage']}% positive, {sentiment_dist['negative_percentage']}% negative")
        
        # Test 4: User analysis
        print("\n" + "="*50)
        print("👤 TEST 4: User Analysis - 'iWantThatOne'")
        print("="*50)
        
        user_posts = agent.retrieve_by_user("iWantThatOne", limit=5)
        if len(user_posts) > 0:
            print(f"Found {len(user_posts)} posts from iWantThatOne")
            print(f"Average sentiment: {user_posts['sentiment_score'].mean():.3f}")
            print(f"Companies discussed: {', '.join(user_posts['ticker'].unique())}")
            print(f"Average post length: {user_posts['post_length'].mean():.1f} words")
        
        # Test 5: Sentiment analysis
        print("\n" + "="*50)
        print("💭 TEST 5: Sentiment Analysis for AML")
        print("="*50)
        
        sentiment_analysis = agent.analyze_sentiment_by_ticker("AML")
        if 'error' not in sentiment_analysis:
            overall = sentiment_analysis['overall_sentiment']
            print(f"Total AML posts: {sentiment_analysis['total_posts']}")
            print(f"Overall sentiment: {overall['average_score']:.3f}")
            print(f"Distribution: {overall['positive_percentage']}% positive, {overall['negative_percentage']}% negative")
            
            if sentiment_analysis['most_active_users']:
                top_users = list(sentiment_analysis['most_active_users'].items())[:3]
                print(f"Most active users: {', '.join([f'{user}({count})' for user, count in top_users])}")
        
        # Test 6: Influential users
        print("\n" + "="*50)
        print("🌟 TEST 6: Most Influential Users")
        print("="*50)
        
        influential_users = agent.find_influential_users(limit=5)
        if len(influential_users) > 0:
            print("Top influential users:")
            for idx, user in influential_users.iterrows():
                print(f"  {idx+1}. {user['user_handle']}: {user['total_posts']} posts, "
                      f"influence score: {user['influence_score']:.1f}")
        
        # Test 7: Trending topics
        print("\n" + "="*50)
        print("📈 TEST 7: Trending Topics (Last 7 days)")
        print("="*50)
        
        trending = agent.get_trending_topics(days_back=7)
        if 'error' not in trending:
            print(f"Analysis period: {trending['period']}")
            print(f"Total posts: {trending['total_posts']}")
            print(f"Overall sentiment: {trending['avg_sentiment_overall']:.3f}")
            
            if trending['trending_words']:
                top_words = trending['trending_words'][:5]
                print(f"Trending words: {', '.join([f'{word}({count})' for word, count in top_words])}")
        
        print("\n" + "="*70)
        print("✅ USERPOSTS RAG AGENT DEMONSTRATION COMPLETED")
        print("="*70)
        print("🎯 Key Capabilities Demonstrated:")
        print("  • Ticker-based post retrieval with filtering")
        print("  • Semantic search with natural language queries")
        print("  • Date range analysis and temporal filtering")
        print("  • User-specific post analysis")
        print("  • Comprehensive sentiment analysis")
        print("  • Influential user identification")
        print("  • Trending topic discovery")
        print("\n🔧 Ready for integration with other domain agents!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.error(f"Demo failed: {e}")


if __name__ == "__main__":
    main() 