#!/usr/bin/env python3
"""
orchestrator_agent.py - Knowledge Orchestrator Agent

Coordinates between all domain agents (fundamentals, news, user posts) to provide 
comprehensive, multi-source responses to user queries. Implements query routing,
result aggregation, cross-ranking, and synthesis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import logging
import json
import re
from dataclasses import dataclass, asdict
from enum import Enum

# Import domain agents
from agent_fundamentals import FundamentalsAgent
from agent_news import NewsAgent
from agent_userposts import UserPostsAgent

# Optional: Cross-encoder for re-ranking (install with: pip install sentence-transformers)
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logging.warning("CrossEncoder not available - falling back to simple ranking")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries the orchestrator can handle"""
    COMPANY_ANALYSIS = "company_analysis"
    SETUP_ANALYSIS = "setup_analysis"
    MARKET_SENTIMENT = "market_sentiment"
    FINANCIAL_ANALYSIS = "financial_analysis"
    NEWS_ANALYSIS = "news_analysis"
    USER_SENTIMENT = "user_sentiment"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    GENERAL_SEARCH = "general_search"


@dataclass
class OrchestratorResult:
    """Standardized result format for orchestrator responses"""
    source: str  # 'fundamentals', 'news', 'userposts'
    content: Dict[str, Any]
    relevance_score: float
    result_type: str
    ticker: Optional[str] = None
    timestamp: Optional[str] = None
    setup_id: Optional[str] = None


@dataclass
class QueryPlan:
    """Plan for executing a query across domain agents"""
    query_type: QueryType
    target_agents: List[str]
    agent_queries: Dict[str, Dict[str, Any]]
    synthesis_strategy: str
    max_results_per_agent: int = 10


class KnowledgeOrchestrator:
    """
    Knowledge Orchestrator Agent that coordinates between domain agents
    to provide comprehensive, multi-source responses to user queries.
    """
    
    def __init__(
        self,
        lancedb_dir: str = "../lancedb_store",
        default_limit: int = 10,
        enable_cross_encoding: bool = True,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        Initialize the Knowledge Orchestrator
        
        Args:
            lancedb_dir: Directory containing LanceDB tables
            default_limit: Default number of results to return
            enable_cross_encoding: Whether to use cross-encoder for re-ranking
            cross_encoder_model: Cross-encoder model for result re-ranking
        """
        self.lancedb_dir = Path(lancedb_dir)
        self.default_limit = default_limit
        self.enable_cross_encoding = enable_cross_encoding and CROSS_ENCODER_AVAILABLE
        
        # Initialize domain agents
        logger.info("Initializing domain agents...")
        try:
            self.fundamentals_agent = FundamentalsAgent(lancedb_dir=str(lancedb_dir))
            logger.info("✅ Fundamentals agent initialized")
        except Exception as e:
            logger.warning(f"⚠️  Fundamentals agent initialization failed: {e}")
            self.fundamentals_agent = None
            
        try:
            self.news_agent = NewsAgent(lancedb_dir=str(lancedb_dir))
            logger.info("✅ News agent initialized")
        except Exception as e:
            logger.warning(f"⚠️  News agent initialization failed: {e}")
            self.news_agent = None
            
        try:
            self.userposts_agent = UserPostsAgent(lancedb_dir=str(lancedb_dir))
            logger.info("✅ User posts agent initialized")
        except Exception as e:
            logger.warning(f"⚠️  User posts agent initialization failed: {e}")
            self.userposts_agent = None
        
        # Initialize cross-encoder for re-ranking
        if self.enable_cross_encoding:
            try:
                self.cross_encoder = CrossEncoder(cross_encoder_model)
                logger.info(f"✅ Cross-encoder initialized: {cross_encoder_model}")
            except Exception as e:
                logger.warning(f"⚠️  Cross-encoder initialization failed: {e}")
                self.enable_cross_encoding = False
        
        # Query intent patterns for routing
        self.intent_patterns = {
            QueryType.COMPANY_ANALYSIS: [
                r'\b(company|ticker|stock)\s+analysis\b',
                r'\banalyze\s+(company|ticker|stock)\b',
                r'\b(company|ticker|stock)\s+overview\b',
                r'\bcomprehensive\s+analysis\b'
            ],
            QueryType.SETUP_ANALYSIS: [
                r'\bsetup\s+analysis\b',
                r'\btrading\s+setup\b',
                r'\bsetup_id\b',
                r'\bsetup\s+[A-Z]+_\d{4}-\d{2}-\d{2}\b'
            ],
            QueryType.FINANCIAL_ANALYSIS: [
                r'\b(financial|fundamentals|ratios|metrics)\b',
                r'\b(revenue|profit|roe|debt|ratio)\b',
                r'\b(balance sheet|income statement|cash flow)\b',
                r'\b(profitable|valuation|earnings)\b'
            ],
            QueryType.NEWS_ANALYSIS: [
                r'\b(news|announcement|press release|rns)\b',
                r'\b(headline|article|announcement)\b',
                r'\b(regulatory|earnings|results)\b'
            ],
            QueryType.USER_SENTIMENT: [
                r'\b(sentiment|opinion|social)\b',
                r'\b(forum|post|user|discussion)\b',
                r'\b(bullish|bearish|mood)\b',
                r'\b(community|social media)\b'
            ],
            QueryType.MARKET_SENTIMENT: [
                r'\bmarket\s+sentiment\b',
                r'\binvestor\s+(sentiment|mood)\b',
                r'\boverall\s+(sentiment|opinion)\b'
            ],
            QueryType.COMPARATIVE_ANALYSIS: [
                r'\bcompare\b',
                r'\b(vs|versus|compared to)\b',
                r'\b(similar|comparison|relative)\b'
            ]
        }
        
        logger.info("🤖 Knowledge Orchestrator initialized successfully")
    
    def query(
        self,
        user_query: str,
        query_context: Optional[Dict[str, Any]] = None,
        max_results: Optional[int] = None,
        include_cross_ranking: bool = True
    ) -> Dict[str, Any]:
        """
        Main query interface - processes user query and returns comprehensive results
        
        Args:
            user_query: Natural language query from user
            query_context: Optional context (ticker, date range, etc.)
            max_results: Maximum results to return
            include_cross_ranking: Whether to apply cross-encoder re-ranking
            
        Returns:
            Comprehensive response with results from all relevant agents
        """
        max_results = max_results or self.default_limit
        start_time = datetime.now()
        
        logger.info(f"🔍 Processing query: '{user_query}'")
        
        try:
            # Step 1: Analyze query intent and create execution plan
            query_plan = self._analyze_query_intent(user_query, query_context)
            logger.info(f"📋 Query plan: {query_plan.query_type.value}, agents: {query_plan.target_agents}")
            
            # Step 2: Execute queries across relevant domain agents
            agent_results = self._execute_agent_queries(query_plan)
            
            # Step 3: Standardize and aggregate results
            standardized_results = self._standardize_results(agent_results, user_query)
            
            # Step 4: Cross-rank results (optional)
            if include_cross_ranking and self.enable_cross_encoding:
                ranked_results = self._cross_rank_results(user_query, standardized_results)
            else:
                ranked_results = self._simple_rank_results(standardized_results)
            
            # Step 5: Synthesize final response
            final_response = self._synthesize_response(
                user_query, query_plan, ranked_results[:max_results], query_context
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            final_response['execution_time_seconds'] = execution_time
            
            logger.info(f"✅ Query completed in {execution_time:.2f}s, {len(ranked_results)} total results")
            return final_response
            
        except Exception as e:
            logger.error(f"❌ Query execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'query': user_query,
                'results': [],
                'execution_time_seconds': (datetime.now() - start_time).total_seconds()
            }
    
    def _analyze_query_intent(
        self, 
        user_query: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> QueryPlan:
        """Analyze query to determine intent and create execution plan"""
        
        query_lower = user_query.lower()
        context = context or {}
        
        # Detect query type based on patterns
        detected_types = []
        for query_type, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    detected_types.append(query_type)
                    break
        
        # Default to general search if no specific intent detected
        if not detected_types:
            detected_types = [QueryType.GENERAL_SEARCH]
        
        # Primary query type is the first detected or most specific
        primary_type = detected_types[0]
        
        # Determine target agents based on query type and context
        target_agents = self._determine_target_agents(primary_type, query_lower, context)
        
        # Create agent-specific queries
        agent_queries = self._create_agent_queries(user_query, primary_type, context)
        
        # Determine synthesis strategy
        synthesis_strategy = self._determine_synthesis_strategy(primary_type, target_agents)
        
        return QueryPlan(
            query_type=primary_type,
            target_agents=target_agents,
            agent_queries=agent_queries,
            synthesis_strategy=synthesis_strategy,
            max_results_per_agent=15  # Get extra results for better ranking
        )
    
    def _determine_target_agents(
        self, 
        query_type: QueryType, 
        query_lower: str, 
        context: Dict[str, Any]
    ) -> List[str]:
        """Determine which agents to query based on intent"""
        
        agents = []
        
        # Always include agents that are available
        available_agents = []
        if self.fundamentals_agent:
            available_agents.append('fundamentals')
        if self.news_agent:
            available_agents.append('news')
        if self.userposts_agent:
            available_agents.append('userposts')
        
        if query_type == QueryType.COMPANY_ANALYSIS:
            # Company analysis needs all domains
            agents = available_agents
        elif query_type == QueryType.SETUP_ANALYSIS:
            # Setup analysis needs all domains
            agents = available_agents
        elif query_type == QueryType.FINANCIAL_ANALYSIS:
            # Financial queries focus on fundamentals but include news
            agents = [a for a in ['fundamentals', 'news'] if a in available_agents]
        elif query_type == QueryType.NEWS_ANALYSIS:
            # News queries focus on news but include sentiment
            agents = [a for a in ['news', 'userposts'] if a in available_agents]
        elif query_type == QueryType.USER_SENTIMENT:
            # Sentiment queries focus on user posts
            agents = [a for a in ['userposts'] if a in available_agents]
        elif query_type == QueryType.MARKET_SENTIMENT:
            # Market sentiment includes both news and user posts
            agents = [a for a in ['news', 'userposts'] if a in available_agents]
        elif query_type == QueryType.COMPARATIVE_ANALYSIS:
            # Comparative analysis needs all domains
            agents = available_agents
        else:  # GENERAL_SEARCH
            # General search queries all available agents
            agents = available_agents
        
        return agents
    
    def _create_agent_queries(
        self, 
        user_query: str, 
        query_type: QueryType, 
        context: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Create specific queries for each agent"""
        
        agent_queries = {}
        
        # Extract common parameters from query and context
        ticker = self._extract_ticker(user_query, context)
        setup_id = self._extract_setup_id(user_query, context)
        date_range = self._extract_date_range(user_query, context)
        
        # Fundamentals agent queries
        if self.fundamentals_agent:
            fund_query = {'limit': 15}
            
            if setup_id:
                fund_query.update({
                    'method': 'retrieve_by_setup_id',
                    'setup_id': setup_id
                })
            elif ticker:
                fund_query.update({
                    'method': 'retrieve_by_ticker',
                    'ticker': ticker
                })
            elif 'profitable' in user_query.lower() or 'performance' in user_query.lower():
                fund_query.update({
                    'method': 'analyze_performance_by_fundamentals',
                    'metric_filter': 'roe > 0.05'  # Basic profitability filter
                })
            else:
                fund_query.update({
                    'method': 'semantic_search',
                    'query': user_query
                })
            
            agent_queries['fundamentals'] = fund_query
        
        # News agent queries
        if self.news_agent:
            news_query = {'limit': 15}
            
            if setup_id:
                news_query.update({
                    'method': 'retrieve_by_setup_id',
                    'setup_id': setup_id
                })
            elif ticker:
                news_query.update({
                    'method': 'retrieve_by_ticker',
                    'ticker': ticker
                })
            else:
                news_query.update({
                    'method': 'retrieve_by_text_query',
                    'query': user_query
                })
            
            agent_queries['news'] = news_query
        
        # User posts agent queries
        if self.userposts_agent:
            posts_query = {'limit': 15}
            
            if setup_id:
                posts_query.update({
                    'method': 'retrieve_by_setup_id',
                    'setup_id': setup_id
                })
            elif ticker:
                posts_query.update({
                    'method': 'retrieve_by_ticker',
                    'ticker': ticker
                })
            else:
                posts_query.update({
                    'method': 'semantic_search',
                    'query': user_query
                })
                
            # Add date range if available
            if date_range:
                posts_query['start_date'] = date_range[0]
                posts_query['end_date'] = date_range[1]
            
            agent_queries['userposts'] = posts_query
        
        return agent_queries
    
    def _execute_agent_queries(self, query_plan: QueryPlan) -> Dict[str, Any]:
        """Execute queries across the specified domain agents"""
        
        agent_results = {}
        
        for agent_name in query_plan.target_agents:
            if agent_name not in query_plan.agent_queries:
                continue
                
            try:
                query_params = query_plan.agent_queries[agent_name]
                method_name = query_params.pop('method')
                
                if agent_name == 'fundamentals' and self.fundamentals_agent:
                    method = getattr(self.fundamentals_agent, method_name)
                    results = method(**query_params)
                    agent_results['fundamentals'] = results
                    logger.info(f"📊 Fundamentals: {len(results) if results else 0} results")
                    
                elif agent_name == 'news' and self.news_agent:
                    method = getattr(self.news_agent, method_name)
                    results = method(**query_params)
                    agent_results['news'] = results
                    logger.info(f"📰 News: {len(results) if results else 0} results")
                    
                elif agent_name == 'userposts' and self.userposts_agent:
                    method = getattr(self.userposts_agent, method_name)
                    results = method(**query_params)
                    
                    # Convert DataFrame to list of dicts for consistency
                    if isinstance(results, pd.DataFrame):
                        results = results.to_dict('records')
                    
                    agent_results['userposts'] = results
                    logger.info(f"💬 User posts: {len(results) if results else 0} results")
                    
            except Exception as e:
                logger.warning(f"⚠️  {agent_name} agent query failed: {e}")
                agent_results[agent_name] = []
        
        return agent_results
    
    def _standardize_results(
        self, 
        agent_results: Dict[str, Any], 
        user_query: str
    ) -> List[OrchestratorResult]:
        """Standardize results from different agents into common format"""
        
        standardized = []
        
        # Process fundamentals results
        if 'fundamentals' in agent_results and agent_results['fundamentals']:
            for result in agent_results['fundamentals']:
                relevance = 0.8  # Base relevance for fundamentals
                
                # Boost relevance for queries about financial metrics
                if any(term in user_query.lower() for term in ['financial', 'profit', 'revenue', 'roe', 'debt']):
                    relevance += 0.1
                
                standardized.append(OrchestratorResult(
                    source='fundamentals',
                    content=result,
                    relevance_score=relevance,
                    result_type='financial_data',
                    ticker=result.get('ticker'),
                    timestamp=result.get('period_end'),
                    setup_id=result.get('setup_id')
                ))
        
        # Process news results
        if 'news' in agent_results and agent_results['news']:
            for result in agent_results['news']:
                relevance = 0.7  # Base relevance for news
                
                # Boost relevance for news-related queries
                if any(term in user_query.lower() for term in ['news', 'announcement', 'press', 'article']):
                    relevance += 0.2
                
                # Boost if has similarity score
                if 'similarity_score' in result:
                    relevance = min(1.0, relevance + result['similarity_score'] * 0.3)
                
                standardized.append(OrchestratorResult(
                    source='news',
                    content=result,
                    relevance_score=relevance,
                    result_type='news_article',
                    ticker=result.get('ticker'),
                    timestamp=result.get('rns_date'),
                    setup_id=result.get('setup_id')
                ))
        
        # Process user posts results
        if 'userposts' in agent_results and agent_results['userposts']:
            for result in agent_results['userposts']:
                relevance = 0.6  # Base relevance for user posts
                
                # Boost relevance for sentiment-related queries
                if any(term in user_query.lower() for term in ['sentiment', 'opinion', 'bullish', 'bearish', 'social']):
                    relevance += 0.2
                
                # Boost based on sentiment strength
                sentiment_score = result.get('sentiment_score', 0)
                if abs(sentiment_score) > 0.05:  # Strong sentiment
                    relevance += 0.1
                
                standardized.append(OrchestratorResult(
                    source='userposts',
                    content=result,
                    relevance_score=relevance,
                    result_type='social_post',
                    ticker=result.get('ticker'),
                    timestamp=result.get('post_date'),
                    setup_id=result.get('setup_id')
                ))
        
        logger.info(f"📋 Standardized {len(standardized)} results from {len(agent_results)} agents")
        return standardized
    
    def _cross_rank_results(
        self, 
        user_query: str, 
        results: List[OrchestratorResult]
    ) -> List[OrchestratorResult]:
        """Re-rank results using cross-encoder for better relevance"""
        
        if not self.enable_cross_encoding or len(results) <= 1:
            return self._simple_rank_results(results)
        
        try:
            # Prepare texts for cross-encoder
            query_result_pairs = []
            for result in results:
                # Create a representative text for each result
                if result.source == 'fundamentals':
                    text = f"{result.content.get('ticker', '')} financial data: {result.content.get('financial_summary', '')}"
                elif result.source == 'news':
                    text = f"{result.content.get('headline', '')} {result.content.get('chunk_text', '')}"
                elif result.source == 'userposts':
                    text = f"User post: {result.content.get('post_content', '')}"
                else:
                    text = str(result.content)
                
                query_result_pairs.append([user_query, text[:512]])  # Limit text length
            
            # Get cross-encoder scores
            scores = self.cross_encoder.predict(query_result_pairs)
            
            # Update relevance scores
            for i, result in enumerate(results):
                cross_encoder_score = float(scores[i])
                # Combine original relevance with cross-encoder score
                result.relevance_score = (result.relevance_score * 0.6) + (cross_encoder_score * 0.4)
            
            # Sort by updated relevance score
            ranked_results = sorted(results, key=lambda x: x.relevance_score, reverse=True)
            
            logger.info("🎯 Applied cross-encoder re-ranking")
            return ranked_results
            
        except Exception as e:
            logger.warning(f"⚠️  Cross-encoder ranking failed: {e}")
            return self._simple_rank_results(results)
    
    def _simple_rank_results(self, results: List[OrchestratorResult]) -> List[OrchestratorResult]:
        """Simple ranking based on relevance scores and source priority"""
        
        # Sort by relevance score, with tie-breaking by source priority
        source_priority = {'fundamentals': 3, 'news': 2, 'userposts': 1}
        
        def sort_key(result):
            return (result.relevance_score, source_priority.get(result.source, 0))
        
        ranked = sorted(results, key=sort_key, reverse=True)
        logger.info("📊 Applied simple relevance ranking")
        return ranked
    
    def _synthesize_response(
        self,
        user_query: str,
        query_plan: QueryPlan,
        ranked_results: List[OrchestratorResult],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Synthesize final comprehensive response"""
        
        # Group results by source
        results_by_source = {}
        for result in ranked_results:
            if result.source not in results_by_source:
                results_by_source[result.source] = []
            results_by_source[result.source].append(result)
        
        # Extract key insights
        insights = self._extract_insights(ranked_results, user_query)
        
        # Create summary based on query type and available results
        summary = self._create_summary(user_query, query_plan.query_type, results_by_source, insights)
        
        # Prepare response
        response = {
            'success': True,
            'query': user_query,
            'query_type': query_plan.query_type.value,
            'total_results': len(ranked_results),
            'sources_queried': query_plan.target_agents,
            'summary': summary,
            'insights': insights,
            'results_by_source': {
                source: [asdict(r) for r in results] 
                for source, results in results_by_source.items()
            },
            'top_results': [asdict(r) for r in ranked_results[:5]],  # Top 5 for overview
            'generated_at': datetime.now().isoformat()
        }
        
        return response
    
    def _extract_insights(
        self, 
        results: List[OrchestratorResult], 
        user_query: str
    ) -> Dict[str, Any]:
        """Extract key insights from aggregated results"""
        
        insights = {
            'companies_mentioned': set(),
            'time_range': {'earliest': None, 'latest': None},
            'sentiment_summary': None,
            'financial_summary': None,
            'news_summary': None
        }
        
        # Collect companies and time range
        for result in results:
            if result.ticker:
                insights['companies_mentioned'].add(result.ticker)
            
            if result.timestamp:
                try:
                    timestamp = pd.to_datetime(result.timestamp)
                    if insights['time_range']['earliest'] is None or timestamp < insights['time_range']['earliest']:
                        insights['time_range']['earliest'] = timestamp
                    if insights['time_range']['latest'] is None or timestamp > insights['time_range']['latest']:
                        insights['time_range']['latest'] = timestamp
                except:
                    pass
        
        # Convert companies to list
        insights['companies_mentioned'] = list(insights['companies_mentioned'])
        
        # Sentiment analysis from user posts
        sentiment_scores = []
        for result in results:
            if result.source == 'userposts' and 'sentiment_score' in result.content:
                sentiment_scores.append(result.content['sentiment_score'])
        
        if sentiment_scores:
            insights['sentiment_summary'] = {
                'average_sentiment': np.mean(sentiment_scores),
                'total_posts': len(sentiment_scores),
                'positive_ratio': len([s for s in sentiment_scores if s > 0.02]) / len(sentiment_scores),
                'negative_ratio': len([s for s in sentiment_scores if s < -0.02]) / len(sentiment_scores)
            }
        
        # Financial insights
        fundamentals_results = [r for r in results if r.source == 'fundamentals']
        if fundamentals_results:
            roe_values = [r.content.get('roe', 0) for r in fundamentals_results if r.content.get('roe', 0) != 0]
            if roe_values:
                insights['financial_summary'] = {
                    'companies_analyzed': len(fundamentals_results),
                    'average_roe': np.mean(roe_values),
                    'profitable_companies': len([r for r in roe_values if r > 0.05])
                }
        
        # News insights
        news_results = [r for r in results if r.source == 'news']
        if news_results:
            insights['news_summary'] = {
                'articles_found': len(news_results),
                'recent_announcements': len([r for r in news_results if r.result_type == 'news_article'])
            }
        
        return insights
    
    def _create_summary(
        self,
        user_query: str,
        query_type: QueryType,
        results_by_source: Dict[str, List[OrchestratorResult]],
        insights: Dict[str, Any]
    ) -> str:
        """Create natural language summary of results"""
        
        companies = insights.get('companies_mentioned', [])
        company_text = f" for {', '.join(companies[:3])}" if companies else ""
        
        if query_type == QueryType.COMPANY_ANALYSIS:
            summary = f"Found comprehensive analysis{company_text}. "
        elif query_type == QueryType.FINANCIAL_ANALYSIS:
            summary = f"Retrieved financial data{company_text}. "
        elif query_type == QueryType.NEWS_ANALYSIS:
            summary = f"Found relevant news{company_text}. "
        elif query_type == QueryType.USER_SENTIMENT:
            summary = f"Analyzed social sentiment{company_text}. "
        else:
            summary = f"Search completed{company_text}. "
        
        # Add source-specific details
        source_details = []
        
        if 'fundamentals' in results_by_source:
            count = len(results_by_source['fundamentals'])
            source_details.append(f"{count} financial records")
        
        if 'news' in results_by_source:
            count = len(results_by_source['news'])
            source_details.append(f"{count} news articles")
        
        if 'userposts' in results_by_source:
            count = len(results_by_source['userposts'])
            source_details.append(f"{count} social posts")
        
        if source_details:
            summary += f"Retrieved {', '.join(source_details)}. "
        
        # Add insights
        if insights.get('sentiment_summary'):
            avg_sentiment = insights['sentiment_summary']['average_sentiment']
            sentiment_text = "positive" if avg_sentiment > 0.02 else "negative" if avg_sentiment < -0.02 else "neutral"
            summary += f"Overall social sentiment: {sentiment_text} ({avg_sentiment:.3f}). "
        
        if insights.get('financial_summary'):
            profitable = insights['financial_summary']['profitable_companies']
            total = insights['financial_summary']['companies_analyzed']
            summary += f"Financial analysis: {profitable}/{total} companies showing strong profitability. "
        
        return summary.strip()
    
    def _extract_ticker(self, query: str, context: Dict[str, Any]) -> Optional[str]:
        """Extract ticker symbol from query or context"""
        
        # Check context first
        if context and 'ticker' in context:
            return context['ticker']
        
        # Known tickers in our dataset
        known_tickers = ['BGO', 'AML', 'HWDN', 'LLOY.L', 'CNA', 'SPT', 'LGEN', 'TSCO.L']
        
        # Look for known tickers in the query (case insensitive)
        query_upper = query.upper()
        for ticker in known_tickers:
            if ticker.upper() in query_upper:
                # Check if it's a standalone word (not part of another word)
                if re.search(r'\b' + re.escape(ticker.upper()) + r'\b', query_upper):
                    return ticker
        
        # Fallback: Look for ticker patterns in query
        ticker_patterns = [
            r'\b([A-Z]{2,5}\.L)\b',  # London Exchange tickers
            r'\b([A-Z]{2,5})\b'      # General tickers (3-5 chars)
        ]
        
        for pattern in ticker_patterns:
            matches = re.findall(pattern, query.upper())
            for match in matches:
                # Avoid common English words that look like tickers
                if match not in ['THE', 'AND', 'FOR', 'ARE', 'YOU', 'ALL', 'ANY', 'CAN', 'HAD', 'HAS', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'USE', 'NOW', 'NEW', 'MAY', 'SAY', 'GET', 'WHO', 'OIL', 'ITS', 'TWO', 'WAY', 'SHE', 'MAN', 'DAY', 'GOT', 'HIM', 'OLD', 'SEE', 'BOY', 'DID', 'CAR', 'EAT', 'AGE', 'RED', 'HOT', 'LET', 'PUT', 'END', 'WHY', 'TRY', 'GOD', 'SIX', 'DOG', 'EYE', 'BED', 'OWN', 'BUT', 'TOO', 'FEW', 'FUN', 'SUN', 'SET', 'MEN', 'LOT', 'BAD', 'YES', 'YET', 'ARM', 'BIG', 'BOX', 'BAG', 'JOB', 'LEG', 'RUN', 'TOP', 'WIN', 'CUP', 'CUT', 'LAW', 'SON', 'EAR', 'SEA', 'AIR', 'FAR', 'OFF', 'HOW', 'OWN', 'SAW', 'MOM', 'SIT', 'FIT', 'HIT', 'LOT', 'NET', 'PET', 'PIT', 'POT', 'ROT', 'SIT', 'TAX', 'TIP', 'TOP', 'VET', 'WAR', 'WET', 'WIN', 'WON', 'ZIP']:
                    return match
        
        return None
    
    def _extract_setup_id(self, query: str, context: Dict[str, Any]) -> Optional[str]:
        """Extract setup ID from query or context"""
        
        # Check context first
        if context and 'setup_id' in context:
            return context['setup_id']
        
        # Look for setup ID pattern in query
        setup_pattern = r'\b([A-Z]+_\d{4}-\d{2}-\d{2}(?:_\d+)?)\b'
        matches = re.findall(setup_pattern, query)
        
        return matches[0] if matches else None
    
    def _extract_date_range(self, query: str, context: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        """Extract date range from query or context"""
        
        # Check context first
        if context and 'start_date' in context and 'end_date' in context:
            return (context['start_date'], context['end_date'])
        
        # Look for relative date terms
        today = datetime.now()
        
        if 'last week' in query.lower():
            end_date = today
            start_date = today - timedelta(days=7)
            return (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        if 'last month' in query.lower():
            end_date = today
            start_date = today - timedelta(days=30)
            return (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        if 'recent' in query.lower() or 'latest' in query.lower():
            end_date = today
            start_date = today - timedelta(days=14)
            return (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        return None
    
    def _determine_synthesis_strategy(
        self, 
        query_type: QueryType, 
        target_agents: List[str]
    ) -> str:
        """Determine how to synthesize results based on query type"""
        
        if len(target_agents) == 1:
            return "single_source"
        elif query_type in [QueryType.COMPANY_ANALYSIS, QueryType.SETUP_ANALYSIS]:
            return "comprehensive_analysis"
        elif query_type == QueryType.COMPARATIVE_ANALYSIS:
            return "comparative"
        else:
            return "multi_source_ranking"
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all domain agents and orchestrator capabilities"""
        
        status = {
            'orchestrator_version': '1.0.0',
            'agents_available': [],
            'agents_failed': [],
            'cross_encoder_enabled': self.enable_cross_encoding,
            'total_capabilities': 0
        }
        
        # Check each agent
        if self.fundamentals_agent:
            try:
                stats = self.fundamentals_agent.get_table_stats()
                status['agents_available'].append({
                    'name': 'fundamentals',
                    'records': stats.get('total_records', 0),
                    'companies': stats.get('unique_tickers', 0)
                })
            except:
                status['agents_failed'].append('fundamentals')
        else:
            status['agents_failed'].append('fundamentals')
        
        if self.news_agent:
            status['agents_available'].append({
                'name': 'news',
                'status': 'operational'
            })
        else:
            status['agents_failed'].append('news')
        
        if self.userposts_agent:
            status['agents_available'].append({
                'name': 'userposts',
                'status': 'operational'
            })
        else:
            status['agents_failed'].append('userposts')
        
        status['total_capabilities'] = len(status['agents_available'])
        
        return status


def run_orchestrator_demo():
    """Demonstrate orchestrator capabilities with various query types"""
    
    print("🤖 KNOWLEDGE ORCHESTRATOR DEMONSTRATION")
    print("=" * 70)
    
    try:
        # Initialize orchestrator
        print("\n📡 Initializing Knowledge Orchestrator...")
        orchestrator = KnowledgeOrchestrator()
        
        # Check system status
        status = orchestrator.get_system_status()
        print(f"📊 System Status: {len(status['agents_available'])} agents available")
        for agent in status['agents_available']:
            print(f"   ✅ {agent['name']}: {agent.get('records', 'operational')}")
        
        if status['agents_failed']:
            print(f"   ⚠️  Failed: {', '.join(status['agents_failed'])}")
        
        # Demo queries
        demo_queries = [
            {
                'query': 'Analyze BGO company financial performance and news',
                'description': 'Comprehensive company analysis'
            },
            {
                'query': 'What is the market sentiment for AML?',
                'description': 'Market sentiment analysis'
            },
            {
                'query': 'Show me profitable companies with strong ROE',
                'description': 'Financial screening'
            },
            {
                'query': 'Recent news about Lloyds bank earnings',
                'description': 'News analysis with context'
            }
        ]
        
        for i, demo in enumerate(demo_queries, 1):
            print(f"\n{'='*50}")
            print(f"🔍 DEMO {i}: {demo['description']}")
            print(f"Query: '{demo['query']}'")
            print("="*50)
            
            # Execute query
            response = orchestrator.query(demo['query'], max_results=5)
            
            if response['success']:
                print(f"✅ {response['summary']}")
                print(f"📊 Total results: {response['total_results']} from {len(response['sources_queried'])} sources")
                
                # Show insights
                insights = response['insights']
                if insights.get('companies_mentioned'):
                    print(f"🏢 Companies: {', '.join(insights['companies_mentioned'])}")
                
                if insights.get('sentiment_summary'):
                    sentiment = insights['sentiment_summary']
                    print(f"😊 Sentiment: {sentiment['average_sentiment']:.3f} ({sentiment['total_posts']} posts)")
                
                # Show top results
                print("\n🔝 Top Results:")
                for j, result in enumerate(response['top_results'][:3], 1):
                    source_icon = {'fundamentals': '📊', 'news': '📰', 'userposts': '💬'}.get(result['source'], '📄')
                    print(f"   {j}. {source_icon} {result['source'].title()}: {result['ticker'] or 'N/A'} (score: {result['relevance_score']:.2f})")
                
            else:
                print(f"❌ Query failed: {response.get('error', 'Unknown error')}")
        
        print(f"\n{'='*70}")
        print("🎉 ORCHESTRATOR DEMONSTRATION COMPLETED")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_orchestrator_demo() 