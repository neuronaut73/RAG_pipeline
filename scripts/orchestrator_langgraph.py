#!/usr/bin/env python3
"""
orchestrator_langgraph.py - LangGraph-based Knowledge Orchestrator Agent

Coordinates between all domain agents (fundamentals, news, user posts) using 
LangGraph workflow framework. Implements query routing, result aggregation, 
cross-ranking, and synthesis as a state graph.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple, TypedDict, Annotated
from pathlib import Path
import logging
import json
import re
from dataclasses import dataclass, asdict
from enum import Enum
import operator

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Import domain agents
from agent_fundamentals import FundamentalsAgent
from agent_news import NewsAgent
from agent_userposts import UserPostsAgent

# Optional: Cross-encoder for re-ranking
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


class OrchestratorState(TypedDict):
    """State schema for the LangGraph orchestrator workflow"""
    # Input
    user_query: str
    query_context: Optional[Dict[str, Any]]
    max_results: int
    include_cross_ranking: bool
    
    # Processing states
    query_plan: Optional[QueryPlan]
    agent_results: Optional[Dict[str, Any]]
    standardized_results: Optional[List[OrchestratorResult]]
    ranked_results: Optional[List[OrchestratorResult]]
    
    # Output
    final_response: Optional[Dict[str, Any]]
    execution_metadata: Dict[str, Any]
    
    # Error handling
    errors: Annotated[List[str], operator.add]


class LangGraphOrchestrator:
    """
    LangGraph-based Knowledge Orchestrator that coordinates between domain agents
    using a state graph workflow pattern.
    """
    
    def __init__(
        self,
        lancedb_dir: str = "../lancedb_store",
        default_limit: int = 10,
        enable_cross_encoding: bool = True,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        Initialize the LangGraph Orchestrator
        
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
        self._init_agents(lancedb_dir)
        
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
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        
        logger.info("🤖 LangGraph Knowledge Orchestrator initialized successfully")
    
    def _init_agents(self, lancedb_dir: str):
        """Initialize domain agents with error handling"""
        try:
            self.fundamentals_agent = FundamentalsAgent(lancedb_dir=lancedb_dir)
            logger.info("✅ Fundamentals agent initialized")
        except Exception as e:
            logger.warning(f"⚠️  Fundamentals agent initialization failed: {e}")
            self.fundamentals_agent = None
            
        try:
            self.news_agent = NewsAgent(lancedb_dir=lancedb_dir)
            logger.info("✅ News agent initialized")
        except Exception as e:
            logger.warning(f"⚠️  News agent initialization failed: {e}")
            self.news_agent = None
            
        try:
            self.userposts_agent = UserPostsAgent(lancedb_dir=lancedb_dir)
            logger.info("✅ User posts agent initialized")
        except Exception as e:
            logger.warning(f"⚠️  User posts agent initialization failed: {e}")
            self.userposts_agent = None
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with nodes and conditional edges"""
        
        # Create the state graph
        workflow = StateGraph(OrchestratorState)
        
        # Add nodes for each processing step
        workflow.add_node("analyze_intent", self._analyze_intent_node)
        workflow.add_node("execute_queries", self._execute_queries_node)
        workflow.add_node("standardize_results", self._standardize_results_node)
        workflow.add_node("rank_results", self._rank_results_node)
        workflow.add_node("synthesize_response", self._synthesize_response_node)
        
        # Add edges
        workflow.add_edge(START, "analyze_intent")
        workflow.add_edge("analyze_intent", "execute_queries")
        workflow.add_edge("execute_queries", "standardize_results")
        workflow.add_edge("standardize_results", "rank_results")
        workflow.add_edge("rank_results", "synthesize_response")
        workflow.add_edge("synthesize_response", END)
        
        # Compile the workflow without checkpointing to avoid serialization issues
        compiled_workflow = workflow.compile()
        
        return compiled_workflow
    
    # NODE IMPLEMENTATIONS
    
    def _analyze_intent_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Node: Analyze query intent and create execution plan"""
        try:
            start_time = datetime.now()
            user_query = state["user_query"]
            query_context = state.get("query_context", {}) or {}
            
            logger.info(f"🔍 Analyzing intent for: '{user_query}'")
            
            # Detect query type using regex patterns
            query_lower = user_query.lower()
            detected_type = QueryType.GENERAL_SEARCH
            
            for query_type, patterns in self.intent_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, query_lower):
                        detected_type = query_type
                        break
                if detected_type != QueryType.GENERAL_SEARCH:
                    break
            
            # Extract context from query
            context = query_context.copy()
            context.update({
                'ticker': self._extract_ticker(user_query, context),
                'setup_id': self._extract_setup_id(user_query, context),
                'date_range': self._extract_date_range(user_query, context)
            })
            
            # Determine target agents
            target_agents = self._determine_target_agents(detected_type, query_lower, context)
            
            # Create agent-specific queries
            agent_queries = self._create_agent_queries(user_query, detected_type, context)
            
            # Create query plan
            query_plan = QueryPlan(
                query_type=detected_type,
                target_agents=target_agents,
                agent_queries=agent_queries,
                synthesis_strategy=self._determine_synthesis_strategy(detected_type, target_agents),
                max_results_per_agent=state["max_results"]
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "query_plan": query_plan,
                "execution_metadata": {
                    **state.get("execution_metadata", {}),
                    "intent_analysis_time": execution_time,
                    "detected_query_type": detected_type.value,
                    "target_agents": target_agents,
                    "extracted_context": context
                }
            }
            
        except Exception as e:
            logger.error(f"Intent analysis failed: {e}")
            return {
                "errors": [f"Intent analysis failed: {str(e)}"]
            }
    
    def _execute_queries_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Node: Execute queries across all target agents"""
        try:
            start_time = datetime.now()
            query_plan = state["query_plan"]
            
            if not query_plan:
                return {"errors": ["No query plan available"]}
            
            logger.info(f"🚀 Executing queries on agents: {query_plan.target_agents}")
            
            agent_results = {}
            
            # Execute queries on each target agent using dynamic method calling
            for agent_name in query_plan.target_agents:
                if agent_name not in query_plan.agent_queries:
                    continue
                    
                try:
                    query_params = query_plan.agent_queries[agent_name].copy()
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
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "agent_results": agent_results,
                "execution_metadata": {
                    **state.get("execution_metadata", {}),
                    "query_execution_time": execution_time,
                    "agents_executed": list(agent_results.keys())
                }
            }
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return {
                "errors": [f"Query execution failed: {str(e)}"]
            }
    
    def _standardize_results_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Node: Standardize results from all agents into common format"""
        try:
            start_time = datetime.now()
            agent_results = state["agent_results"]
            user_query = state["user_query"]
            
            if not agent_results:
                return {"errors": ["No agent results available"]}
            
            logger.info("📊 Standardizing results from agents")
            
            standardized_results = []
            
            for source, result_data in agent_results.items():
                if isinstance(result_data, dict) and "error" in result_data:
                    continue
                
                results = result_data.get("results", []) if isinstance(result_data, dict) else result_data
                
                for i, result in enumerate(results):
                    if isinstance(result, dict):
                        # Clean all numpy types from the result
                        clean_result = self._clean_numpy_types(result)
                        
                        standardized_result = OrchestratorResult(
                            source=source,
                            content=clean_result,
                            relevance_score=float(clean_result.get("similarity_score", clean_result.get("score", 0.5))),
                            result_type=self._determine_result_type(clean_result, source),
                            ticker=clean_result.get("ticker") or clean_result.get("Ticker"),
                            timestamp=clean_result.get("timestamp") or clean_result.get("Date") or clean_result.get("created_at"),
                            setup_id=clean_result.get("setup_id")
                        )
                        standardized_results.append(standardized_result)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "standardized_results": standardized_results,
                "execution_metadata": {
                    **state.get("execution_metadata", {}),
                    "standardization_time": execution_time,
                    "total_results": len(standardized_results)
                }
            }
            
        except Exception as e:
            logger.error(f"Result standardization failed: {e}")
            return {
                "errors": [f"Result standardization failed: {str(e)}"]
            }
    
    def _rank_results_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Node: Cross-rank results using cross-encoder or simple ranking"""
        try:
            start_time = datetime.now()
            standardized_results = state["standardized_results"]
            user_query = state["user_query"]
            include_cross_ranking = state.get("include_cross_ranking", True)
            
            if not standardized_results:
                return {"errors": ["No standardized results available"]}
            
            logger.info(f"🎯 Ranking {len(standardized_results)} results")
            
            if include_cross_ranking and self.enable_cross_encoding:
                ranked_results = self._cross_rank_results(user_query, standardized_results)
            else:
                ranked_results = self._simple_rank_results(standardized_results)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "ranked_results": ranked_results,
                "execution_metadata": {
                    **state.get("execution_metadata", {}),
                    "ranking_time": execution_time,
                    "ranking_method": "cross_encoder" if (include_cross_ranking and self.enable_cross_encoding) else "simple"
                }
            }
            
        except Exception as e:
            logger.error(f"Result ranking failed: {e}")
            return {
                "errors": [f"Result ranking failed: {str(e)}"]
            }
    
    def _synthesize_response_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Node: Synthesize final response with insights and summary"""
        try:
            start_time = datetime.now()
            ranked_results = state["ranked_results"]
            user_query = state["user_query"]
            query_plan = state["query_plan"]
            query_context = state.get("query_context")
            
            if not ranked_results:
                return {"errors": ["No ranked results available"]}
            
            logger.info("🔗 Synthesizing comprehensive response")
            
            # Limit results to max_results
            max_results = state["max_results"]
            top_results = ranked_results[:max_results]
            
            # Group results by source
            results_by_source = {}
            for result in top_results:
                if result.source not in results_by_source:
                    results_by_source[result.source] = []
                results_by_source[result.source].append(result)
            
            # Extract insights
            insights = self._extract_insights(top_results, user_query)
            
            # Create summary
            summary = self._create_summary(user_query, query_plan.query_type, results_by_source, insights)
            
            # Build final response
            final_response = {
                "summary": summary,
                "insights": insights,
                "results": {
                    "by_source": {source: [asdict(r) for r in results] for source, results in results_by_source.items()},
                    "top_results": [asdict(r) for r in top_results[:10]]  # Top 10 for quick view
                },
                "metadata": {
                    "query_type": query_plan.query_type.value,
                    "total_results": len(top_results),
                    "sources_used": list(results_by_source.keys()),
                    "synthesis_strategy": query_plan.synthesis_strategy,
                    **state.get("execution_metadata", {})
                }
            }
            
            # Clean all numpy types from the final response
            final_response = self._clean_numpy_types(final_response)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            total_time = sum([
                state.get("execution_metadata", {}).get("intent_analysis_time", 0),
                state.get("execution_metadata", {}).get("query_execution_time", 0),
                state.get("execution_metadata", {}).get("standardization_time", 0),
                state.get("execution_metadata", {}).get("ranking_time", 0),
                execution_time
            ])
            
            final_response["metadata"]["synthesis_time"] = execution_time
            final_response["metadata"]["total_execution_time"] = total_time
            
            return {
                "final_response": final_response,
                "execution_metadata": {
                    **state.get("execution_metadata", {}),
                    "synthesis_time": execution_time,
                    "total_execution_time": total_time
                }
            }
            
        except Exception as e:
            logger.error(f"Response synthesis failed: {e}")
            return {
                "errors": [f"Response synthesis failed: {str(e)}"]
            }
    
    # HELPER METHODS (adapted from original orchestrator)
    
    def _clean_numpy_types(self, obj):
        """Recursively convert numpy types to Python types for serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # numpy scalar types
            return obj.item()
        elif isinstance(obj, dict):
            return {key: self._clean_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._clean_numpy_types(item) for item in obj)
        else:
            return obj
    
    def _determine_target_agents(self, query_type: QueryType, query_lower: str, context: Dict[str, Any]) -> List[str]:
        """Determine which agents to query based on intent and context"""
        agents = []
        
        # Always include available agents for comprehensive analysis
        if self.fundamentals_agent:
            agents.append("fundamentals")
        if self.news_agent:
            agents.append("news") 
        if self.userposts_agent:
            agents.append("userposts")
        
        # Query-type specific logic could filter agents here
        if query_type == QueryType.FINANCIAL_ANALYSIS:
            # Prioritize fundamentals for financial queries
            agents = ["fundamentals"] + [a for a in agents if a != "fundamentals"]
        elif query_type == QueryType.NEWS_ANALYSIS:
            # Prioritize news for news queries
            agents = ["news"] + [a for a in agents if a != "news"]
        elif query_type == QueryType.USER_SENTIMENT:
            # Prioritize userposts for sentiment queries
            agents = ["userposts"] + [a for a in agents if a != "userposts"]
        
        return agents
    
    def _create_agent_queries(self, user_query: str, query_type: QueryType, context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Create agent-specific queries based on intent and context"""
        agent_queries = {}
        

        
        # Fundamentals agent query
        if self.fundamentals_agent:
            fund_query = {'limit': 15}
            ticker = self._extract_ticker(user_query, context)
            setup_id = self._extract_setup_id(user_query, context)
            
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
                    'metric_filter': 'roe > 0.05'
                })
            else:
                fund_query.update({
                    'method': 'semantic_search',
                    'query': user_query
                })
            
            agent_queries["fundamentals"] = fund_query
        
        # News agent query
        if self.news_agent:
            news_query = {'limit': 15}
            ticker = self._extract_ticker(user_query, context)
            setup_id = self._extract_setup_id(user_query, context)
            
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
            
            agent_queries["news"] = news_query
        
        # UserPosts agent query
        if self.userposts_agent:
            posts_query = {'limit': 15}
            ticker = self._extract_ticker(user_query, context)
            setup_id = self._extract_setup_id(user_query, context)
            date_range = self._extract_date_range(user_query, context)
            
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
            
            agent_queries["userposts"] = posts_query
        
        return agent_queries
    
    def _determine_result_type(self, result: Dict[str, Any], source: str) -> str:
        """Determine the type of result based on content and source"""
        if source == "fundamentals":
            if "ratio" in str(result).lower() or "roe" in str(result).lower():
                return "financial_ratio"
            return "fundamental_data"
        elif source == "news":
            if "earnings" in str(result).lower():
                return "earnings_news"
            elif "announcement" in str(result).lower():
                return "corporate_announcement" 
            return "news_article"
        elif source == "userposts":
            if result.get("sentiment"):
                return "sentiment_post"
            return "user_discussion"
        return "general"
    
    def _cross_rank_results(self, user_query: str, results: List[OrchestratorResult]) -> List[OrchestratorResult]:
        """Re-rank results using cross-encoder"""
        if not results:
            return results
        
        try:
            # Prepare query-result pairs for cross-encoder
            pairs = []
            for result in results:
                content = self._extract_text_content(result.content)
                pairs.append([user_query, content])
            
            # Get cross-encoder scores
            scores = self.cross_encoder.predict(pairs)
            
            # Update relevance scores and sort
            for i, result in enumerate(results):
                # Ensure score is a Python float, not numpy type
                score = scores[i]
                if hasattr(score, 'item'):
                    score = score.item()
                result.relevance_score = float(score)
            
            return sorted(results, key=lambda x: x.relevance_score, reverse=True)
            
        except Exception as e:
            logger.warning(f"Cross-encoding failed: {e}, falling back to simple ranking")
            return self._simple_rank_results(results)
    
    def _simple_rank_results(self, results: List[OrchestratorResult]) -> List[OrchestratorResult]:
        """Simple ranking based on relevance scores"""
        def sort_key(result):
            score = result.relevance_score
            # Boost recent results slightly
            if result.timestamp:
                try:
                    date = pd.to_datetime(result.timestamp)
                    days_old = (datetime.now() - date).days
                    recency_boost = max(0, 1 - days_old / 365)  # Boost for recent results
                    score += recency_boost * 0.1
                except:
                    pass
            return score
        
        return sorted(results, key=sort_key, reverse=True)
    
    def _extract_text_content(self, content: Dict[str, Any]) -> str:
        """Extract meaningful text content from result for cross-encoding"""
        text_parts = []
        
        # Common text fields to extract
        text_fields = ['title', 'headline', 'content', 'text', 'summary', 'description', 'message']
        
        for field in text_fields:
            if field in content and content[field]:
                text_parts.append(str(content[field]))
        
        # Add numerical data as text for financial content
        if not text_parts:
            for key, value in content.items():
                if isinstance(value, (str, int, float)) and value is not None:
                    text_parts.append(f"{key}: {value}")
        
        return " ".join(text_parts)[:500]  # Limit length for cross-encoder
    
    def _extract_insights(self, results: List[OrchestratorResult], user_query: str) -> Dict[str, Any]:
        """Extract cross-domain insights from results"""
        insights = {
            "sentiment_summary": {},
            "financial_highlights": {},
            "news_themes": {},
            "cross_correlations": []
        }
        
        # Group results by source
        by_source = {}
        for result in results:
            if result.source not in by_source:
                by_source[result.source] = []
            by_source[result.source].append(result)
        
        # Sentiment analysis
        if "userposts" in by_source:
            sentiment_data = []
            for result in by_source["userposts"]:
                sentiment = result.content.get("sentiment") or result.content.get("Sentiment")
                if sentiment:
                    sentiment_data.append(sentiment.lower())
            
            if sentiment_data:
                positive = sum(1 for s in sentiment_data if s in ["positive", "bullish"])
                negative = sum(1 for s in sentiment_data if s in ["negative", "bearish"])
                total = len(sentiment_data)
                
                insights["sentiment_summary"] = {
                    "overall_sentiment": "positive" if positive > negative else "negative" if negative > positive else "neutral",
                    "positive_ratio": positive / total if total > 0 else 0,
                    "negative_ratio": negative / total if total > 0 else 0,
                    "total_posts": total
                }
        
        # Financial highlights
        if "fundamentals" in by_source:
            financial_metrics = []
            for result in by_source["fundamentals"]:
                content = result.content
                if "ROE" in content:
                    financial_metrics.append(f"ROE: {content['ROE']}")
                if "Revenue" in content:
                    financial_metrics.append(f"Revenue: {content['Revenue']}")
            
            insights["financial_highlights"] = {
                "key_metrics": financial_metrics[:5],
                "companies_analyzed": len(set(r.ticker for r in by_source["fundamentals"] if r.ticker))
            }
        
        # News themes
        if "news" in by_source:
            headlines = []
            for result in by_source["news"]:
                headline = result.content.get("headline") or result.content.get("Headline") or result.content.get("title")
                if headline:
                    headlines.append(headline)
            
            insights["news_themes"] = {
                "recent_headlines": headlines[:3],
                "total_articles": len(by_source["news"])
            }
        
        return insights
    
    def _create_summary(self, user_query: str, query_type: QueryType, results_by_source: Dict[str, List[OrchestratorResult]], insights: Dict[str, Any]) -> str:
        """Create natural language summary of results"""
        summary_parts = []
        
        # Query-specific opening
        if query_type == QueryType.COMPANY_ANALYSIS:
            summary_parts.append(f"Analysis for your query '{user_query}':")
        else:
            summary_parts.append(f"Based on your query '{user_query}', here's what I found:")
        
        # Source-specific summaries
        if "fundamentals" in results_by_source:
            count = len(results_by_source["fundamentals"])
            summary_parts.append(f"Found {count} relevant fundamental data points.")
            
            if insights.get("financial_highlights", {}).get("key_metrics"):
                summary_parts.append(f"Key financial metrics include: {', '.join(insights['financial_highlights']['key_metrics'][:3])}.")
        
        if "news" in results_by_source:
            count = len(results_by_source["news"])
            summary_parts.append(f"Discovered {count} relevant news articles and announcements.")
            
            if insights.get("news_themes", {}).get("recent_headlines"):
                summary_parts.append("Recent headlines focus on market developments and company updates.")
        
        if "userposts" in results_by_source:
            count = len(results_by_source["userposts"])
            summary_parts.append(f"Analyzed {count} user posts and discussions.")
            
            sentiment_info = insights.get("sentiment_summary", {})
            if sentiment_info:
                overall = sentiment_info.get("overall_sentiment", "neutral")
                summary_parts.append(f"Overall social sentiment appears {overall}.")
        
        # Cross-domain insights
        if len(results_by_source) > 1:
            summary_parts.append("This multi-source analysis provides a comprehensive view combining financial data, market news, and social sentiment.")
        
        return " ".join(summary_parts)
    
    def _extract_ticker(self, query: str, context: Dict[str, Any]) -> Optional[str]:
        """Extract ticker symbol from query or context"""
        if context and context.get("ticker"):
            return context["ticker"]
        
        # Look for ticker patterns (2-5 uppercase letters)
        ticker_match = re.search(r'\b([A-Z]{2,5})\b', query)
        if ticker_match:
            return ticker_match.group(1)
        
        return None
    
    def _extract_setup_id(self, query: str, context: Dict[str, Any]) -> Optional[str]:
        """Extract setup ID from query or context"""
        if context and context.get("setup_id"):
            return context["setup_id"]
        
        # Look for setup ID patterns
        setup_match = re.search(r'\b([A-Z]+_\d{4}-\d{2}-\d{2})\b', query)
        if setup_match:
            return setup_match.group(1)
        
        return None
    
    def _extract_date_range(self, query: str, context: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        """Extract date range from query or context"""
        if context and context.get("date_range"):
            return context["date_range"]
        
        # Simple date range extraction
        if "last week" in query.lower():
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            return (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        elif "last month" in query.lower():
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            return (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        
        return None
    
    def _determine_synthesis_strategy(self, query_type: QueryType, target_agents: List[str]) -> str:
        """Determine synthesis strategy based on query type and agents"""
        if len(target_agents) == 1:
            return "single_source"
        elif query_type == QueryType.COMPANY_ANALYSIS:
            return "comprehensive_analysis"
        elif query_type in [QueryType.USER_SENTIMENT, QueryType.MARKET_SENTIMENT]:
            return "sentiment_focused"
        else:
            return "multi_source_synthesis"
    
    # PUBLIC INTERFACE
    
    def query(
        self,
        user_query: str,
        query_context: Optional[Dict[str, Any]] = None,
        max_results: Optional[int] = None,
        include_cross_ranking: bool = True
    ) -> Dict[str, Any]:
        """
        Main query interface using LangGraph workflow
        
        Args:
            user_query: Natural language query from user
            query_context: Optional context (ticker, date range, etc.)
            max_results: Maximum results to return
            include_cross_ranking: Whether to apply cross-encoder re-ranking
            
        Returns:
            Comprehensive response with results from all relevant agents
        """
        max_results = max_results or self.default_limit
        
        # Initialize state
        initial_state = {
            "user_query": user_query,
            "query_context": query_context or {},
            "max_results": max_results,
            "include_cross_ranking": include_cross_ranking,
            "query_plan": None,
            "agent_results": None,
            "standardized_results": None,
            "ranked_results": None,
            "final_response": None,
            "execution_metadata": {"start_time": datetime.now().isoformat()},
            "errors": []
        }
        
        try:
            # Execute the workflow
            result = self.workflow.invoke(
                initial_state,
                config={"configurable": {"thread_id": f"query_{datetime.now().timestamp()}"}}
            )
            
            # Return final response or error details
            if result.get("final_response"):
                return result["final_response"]
            elif result.get("errors"):
                return {
                    "error": "Query execution failed",
                    "details": result["errors"],
                    "metadata": result.get("execution_metadata", {})
                }
            else:
                return {
                    "error": "Unknown execution error",
                    "metadata": result.get("execution_metadata", {})
                }
                
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                "error": "Workflow execution failed",
                "details": [str(e)],
                "metadata": {"execution_time": 0}
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status including agent availability"""
        return {
            "agents": {
                "fundamentals": self.fundamentals_agent is not None,
                "news": self.news_agent is not None,
                "userposts": self.userposts_agent is not None
            },
            "cross_encoder_enabled": self.enable_cross_encoding,
            "workflow_compiled": self.workflow is not None,
            "default_limit": self.default_limit
        }
    
    def visualize_workflow(self) -> None:
        """Print workflow graph structure"""
        try:
            # This would typically use mermaid or graphviz
            # For now, print a simple text representation
            print("\n=== LangGraph Orchestrator Workflow ===")
            print("START → analyze_intent → execute_queries → standardize_results → rank_results → synthesize_response → END")
            print("\nNodes:")
            print("  • analyze_intent: Detect query type and create execution plan")
            print("  • execute_queries: Query all relevant domain agents") 
            print("  • standardize_results: Convert agent results to common format")
            print("  • rank_results: Cross-rank results using cross-encoder")
            print("  • synthesize_response: Create final comprehensive response")
            print("=====================================\n")
        except Exception as e:
            logger.warning(f"Could not visualize workflow: {e}")


def run_langgraph_orchestrator_demo():
    """Demo function to test the LangGraph orchestrator"""
    print("🚀 LangGraph Knowledge Orchestrator Demo")
    print("=" * 50)
    
    # Initialize orchestrator
    orchestrator = LangGraphOrchestrator()
    
    # Show system status
    status = orchestrator.get_system_status()
    print(f"\n📊 System Status:")
    print(f"  Agents available: {sum(status['agents'].values())}/3")
    print(f"  Cross-encoder: {'✅' if status['cross_encoder_enabled'] else '❌'}")
    print(f"  Workflow compiled: {'✅' if status['workflow_compiled'] else '❌'}")
    
    # Visualize workflow
    orchestrator.visualize_workflow()
    
    # Test queries
    test_queries = [
        {
            "query": "AML comprehensive analysis sentiment and performance",
            "description": "Company analysis across all domains"
        },
        {
            "query": "profitable companies with strong ROE",
            "description": "Financial screening query"
        },
        {
            "query": "recent market sentiment",
            "description": "Sentiment analysis query"
        }
    ]
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n🔍 Test Query {i}: {test['description']}")
        print(f"Query: '{test['query']}'")
        print("-" * 40)
        
        # Execute query
        result = orchestrator.query(test['query'], max_results=5)
        
        # Display results
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            if "details" in result:
                for detail in result['details']:
                    print(f"   - {detail}")
        else:
            print(f"✅ Query executed successfully")
            print(f"📝 Summary: {result.get('summary', 'No summary available')}")
            
            metadata = result.get('metadata', {})
            print(f"⏱️  Execution time: {metadata.get('total_execution_time', 0):.3f}s")
            print(f"🎯 Query type: {metadata.get('query_type', 'unknown')}")
            print(f"📊 Total results: {metadata.get('total_results', 0)}")
            print(f"🔧 Sources used: {', '.join(metadata.get('sources_used', []))}")
    
    print("\n✨ Demo completed!")


if __name__ == "__main__":
    run_langgraph_orchestrator_demo() 