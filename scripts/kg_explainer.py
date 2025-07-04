#!/usr/bin/env python3
"""
kg_explainer.py - Knowledge Graph Explainer

LLM-powered explanations for subgraphs, query results, and multi-domain patterns.
Integrates with the LangGraph orchestrator to provide intelligent insights and 
natural language explanations of relationships found in the knowledge graph.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field

# Local imports
from orchestrator_langgraph import LangGraphOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EntityRelationship:
    """Represents a relationship between two entities in the knowledge graph"""
    source_entity: str
    target_entity: str
    relationship_type: str
    relationship_strength: float
    supporting_evidence: List[str]
    metadata: Dict[str, Any]


@dataclass
class PatternAnalysis:
    """Represents a discovered pattern in the knowledge graph"""
    pattern_type: str
    description: str
    confidence_score: float
    supporting_entities: List[str]
    rule_conditions: List[str]
    rule_conclusions: List[str]
    examples: List[Dict[str, Any]]


@dataclass
class ExplanationResult:
    """Complete explanation result for a query or subgraph"""
    query: str
    summary: str
    key_insights: List[str]
    entity_relationships: List[EntityRelationship]
    discovered_patterns: List[PatternAnalysis]
    if_then_rules: List[str]
    natural_language_explanation: str
    confidence_score: float
    metadata: Dict[str, Any]


class LLMExplanationSchema(BaseModel):
    """Pydantic schema for LLM-generated explanations"""
    summary: str = Field(description="Brief summary of the analysis")
    key_insights: List[str] = Field(description="List of key insights discovered")
    relationships: List[Dict[str, Any]] = Field(description="Entity relationships found")
    patterns: List[Dict[str, Any]] = Field(description="Patterns and trends identified")
    if_then_rules: List[str] = Field(description="Logical if-then rules derived")
    confidence_score: float = Field(description="Confidence in the analysis (0-1)")


class KnowledgeGraphExplainer:
    """
    LLM-powered Knowledge Graph Explainer that analyzes multi-domain query results
    and generates intelligent explanations, patterns, and rules.
    """
    
    def __init__(
        self,
        orchestrator: Optional[LangGraphOrchestrator] = None,
        openai_api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 2000,
        enable_caching: bool = True
    ):
        """
        Initialize the Knowledge Graph Explainer
        
        Args:
            orchestrator: LangGraph orchestrator for querying data
            openai_api_key: OpenAI API key (if not set as env var)
            model_name: LLM model to use for explanations
            temperature: Temperature for LLM generation
            max_tokens: Maximum tokens for LLM responses
            enable_caching: Whether to cache LLM responses
        """
        # Initialize LLM
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Initialize orchestrator
        self.orchestrator = orchestrator or LangGraphOrchestrator()
        
        # Configure caching
        self.enable_caching = enable_caching
        self._explanation_cache = {}
        
        # Setup prompt templates
        self._setup_prompt_templates()
        
        logger.info(f"🧠 KnowledgeGraphExplainer initialized with model: {model_name}")
    
    def _setup_prompt_templates(self):
        """Setup LLM prompt templates for different explanation tasks"""
        
        # Main explanation prompt
        self.explanation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert financial analyst and data scientist specializing in 
            knowledge graph analysis. You analyze multi-domain financial data including:
            - Company fundamentals (revenue, profits, ratios)
            - Market sentiment from user discussions
            - News and announcements
            - Trading patterns and setups
            
            Your task is to provide intelligent explanations of relationships, patterns, and 
            derive actionable insights from the data. Focus on:
            1. Clear, concise explanations accessible to investors
            2. Identifying meaningful relationships between entities
            3. Discovering patterns and trends
            4. Creating logical if-then rules
            5. Providing confidence assessments
            
            Always ground your analysis in the provided data and be explicit about 
            uncertainty where evidence is limited."""),
            
            ("user", """Analyze the following query results and provide a comprehensive explanation:

            QUERY: {query}
            
            RESULTS DATA:
            {results_data}
            
            Please provide:
            1. A clear summary of what the data shows
            2. Key insights and relationships discovered
            3. Patterns and trends identified
            4. Logical if-then rules that can be derived
            5. A confidence score for your analysis (0-1)
            
            Format your response as JSON matching this schema:
            {{
                "summary": "Brief summary of the analysis",
                "key_insights": ["Insight 1", "Insight 2", ...],
                "relationships": [
                    {{
                        "source": "Entity A",
                        "target": "Entity B", 
                        "type": "relationship_type",
                        "strength": 0.8,
                        "evidence": ["Supporting fact 1", "Supporting fact 2"]
                    }}
                ],
                "patterns": [
                    {{
                        "type": "pattern_type",
                        "description": "Pattern description",
                        "confidence": 0.9,
                        "examples": ["Example 1", "Example 2"]
                    }}
                ],
                "if_then_rules": ["IF condition THEN conclusion", ...],
                "confidence_score": 0.85
            }}""")
        ])
        
        # Natural language explanation prompt
        self.narrative_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial storyteller who creates engaging, accessible 
            explanations of complex financial data and relationships. Your explanations should:
            - Use clear, jargon-free language
            - Tell a coherent story connecting different data points
            - Highlight the most important insights
            - Provide actionable takeaways for investors
            - Acknowledge limitations and uncertainties"""),
            
            ("user", """Create a natural language explanation based on this analysis:
            
            QUERY: {query}
            ANALYSIS: {analysis_json}
            
            Write a comprehensive but accessible explanation that connects the dots 
            and tells the story behind the data. Focus on what investors should know.""")
        ])
        
        # Pattern discovery prompt
        self.pattern_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a pattern recognition expert specializing in financial markets.
            You identify meaningful patterns, trends, and relationships in multi-domain financial data.
            Look for correlations, causations, and predictive patterns."""),
            
            ("user", """Analyze this data for patterns and relationships:
            
            {data_summary}
            
            Identify:
            1. Correlations between different metrics
            2. Temporal patterns and trends
            3. Sentiment-performance relationships
            4. Risk indicators and warning signs
            5. Opportunity patterns
            
            Focus on actionable patterns that could inform investment decisions.""")
        ])
    
    def explain_query_results(
        self,
        query: str,
        results: Dict[str, Any],
        include_narrative: bool = True,
        include_patterns: bool = True
    ) -> ExplanationResult:
        """
        Generate comprehensive explanation for query results
        
        Args:
            query: Original user query
            results: Results from orchestrator
            include_narrative: Whether to generate natural language narrative
            include_patterns: Whether to perform deep pattern analysis
            
        Returns:
            Complete explanation result
        """
        try:
            logger.info(f"🔍 Analyzing query results for: '{query}'")
            
            # Check cache
            cache_key = self._get_cache_key(query, results)
            if self.enable_caching and cache_key in self._explanation_cache:
                logger.info("📋 Using cached explanation")
                return self._explanation_cache[cache_key]
            
            # Format results for LLM analysis
            formatted_data = self._format_results_for_llm(results)
            
            # Generate main analysis
            analysis = self._generate_llm_analysis(query, formatted_data)
            
            # Generate natural language explanation if requested
            narrative = ""
            if include_narrative:
                narrative = self._generate_narrative_explanation(query, analysis)
            
            # Perform pattern analysis if requested
            patterns = []
            if include_patterns:
                patterns = self._discover_patterns(formatted_data)
            
            # Create explanation result
            explanation = ExplanationResult(
                query=query,
                summary=analysis.get("summary", ""),
                key_insights=analysis.get("key_insights", []),
                entity_relationships=self._parse_relationships(analysis.get("relationships", [])),
                discovered_patterns=patterns,
                if_then_rules=analysis.get("if_then_rules", []),
                natural_language_explanation=narrative,
                confidence_score=analysis.get("confidence_score", 0.5),
                metadata={
                    "generated_at": datetime.now().isoformat(),
                    "llm_model": self.llm.model_name,
                    "results_count": len(results.get("results", {}).get("top_results", [])),
                    "sources_used": results.get("metadata", {}).get("sources_used", [])
                }
            )
            
            # Cache result
            if self.enable_caching:
                self._explanation_cache[cache_key] = explanation
            
            logger.info(f"✅ Generated explanation with {len(explanation.key_insights)} insights")
            return explanation
            
        except Exception as e:
            logger.error(f"❌ Error generating explanation: {e}")
            return self._create_error_explanation(query, str(e))
    
    def explain_subgraph(
        self,
        entities: List[str],
        relationship_types: Optional[List[str]] = None,
        max_depth: int = 2
    ) -> ExplanationResult:
        """
        Explain relationships and patterns within a subgraph
        
        Args:
            entities: List of entity names (tickers, setup_ids, etc.)
            relationship_types: Types of relationships to explore
            max_depth: Maximum traversal depth
            
        Returns:
            Explanation of the subgraph
        """
        try:
            logger.info(f"🕸️ Analyzing subgraph for entities: {entities}")
            
            # Query data for all entities
            subgraph_data = self._build_subgraph_data(entities, relationship_types, max_depth)
            
            # Generate explanation
            query = f"Analyze relationships between: {', '.join(entities)}"
            return self.explain_query_results(query, subgraph_data)
            
        except Exception as e:
            logger.error(f"❌ Error analyzing subgraph: {e}")
            return self._create_error_explanation(f"Subgraph: {entities}", str(e))
    
    def generate_if_then_rules(
        self,
        domain: Optional[str] = None,
        min_confidence: float = 0.7,
        max_rules: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Generate if-then rules from the knowledge graph
        
        Args:
            domain: Specific domain to focus on (fundamentals, sentiment, etc.)
            min_confidence: Minimum confidence threshold for rules
            max_rules: Maximum number of rules to generate
            
        Returns:
            List of if-then rules with metadata
        """
        try:
            logger.info(f"🔧 Generating if-then rules for domain: {domain}")
            
            # Sample data from different domains
            sample_data = self._sample_domain_data(domain)
            
            # Use LLM to derive rules
            rules_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a financial rule extraction expert. Generate logical 
                if-then rules from financial data that could be useful for:
                - Investment decision making
                - Risk assessment
                - Opportunity identification
                - Market timing
                
                Focus on rules that are:
                - Actionable and specific
                - Based on observable patterns
                - Validated by the data provided
                - Useful for investors"""),
                
                ("user", """Extract if-then rules from this financial data:
                
                {data}
                
                Generate rules in this format:
                IF [condition] AND [condition] THEN [conclusion] (confidence: X.XX)
                
                Focus on the most reliable and actionable rules.""")
            ])
            
            chain = rules_prompt | self.llm | StrOutputParser()
            rules_text = chain.invoke({"data": sample_data})
            
            # Parse rules from text
            rules = self._parse_rules_from_text(rules_text, min_confidence)
            
            logger.info(f"✅ Generated {len(rules)} if-then rules")
            return rules[:max_rules]
            
        except Exception as e:
            logger.error(f"❌ Error generating rules: {e}")
            return []
    
    def summarize_market_patterns(
        self,
        time_period: str = "last_30_days",
        include_sentiment: bool = True,
        include_fundamentals: bool = True,
        include_news: bool = True
    ) -> Dict[str, Any]:
        """
        Generate summary of market patterns and trends
        
        Args:
            time_period: Time period to analyze
            include_sentiment: Include sentiment analysis
            include_fundamentals: Include fundamental analysis
            include_news: Include news analysis
            
        Returns:
            Market pattern summary
        """
        try:
            logger.info(f"📊 Summarizing market patterns for: {time_period}")
            
            # Query recent market data
            query = f"recent market patterns and trends {time_period}"
            results = self.orchestrator.query(query, max_results=50)
            
            # Generate explanation with focus on patterns
            explanation = self.explain_query_results(
                query, 
                results, 
                include_patterns=True
            )
            
            # Create market summary
            summary = {
                "period": time_period,
                "overall_sentiment": self._analyze_overall_sentiment(results),
                "key_trends": explanation.key_insights,
                "market_patterns": [asdict(p) for p in explanation.discovered_patterns],
                "risk_indicators": self._extract_risk_indicators(explanation),
                "opportunities": self._extract_opportunities(explanation),
                "confidence": explanation.confidence_score,
                "generated_at": datetime.now().isoformat()
            }
            
            logger.info("✅ Market pattern summary generated")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error summarizing market patterns: {e}")
            return {"error": str(e), "generated_at": datetime.now().isoformat()}
    
    def _format_results_for_llm(self, results: Dict[str, Any]) -> str:
        """Format orchestrator results for LLM consumption"""
        try:
            formatted_sections = []
            
            # Add metadata
            metadata = results.get("metadata", {})
            formatted_sections.append(f"QUERY METADATA:")
            formatted_sections.append(f"- Query Type: {metadata.get('query_type', 'unknown')}")
            formatted_sections.append(f"- Total Results: {metadata.get('total_results', 0)}")
            formatted_sections.append(f"- Sources Used: {', '.join(metadata.get('sources_used', []))}")
            formatted_sections.append(f"- Execution Time: {metadata.get('total_execution_time', 0):.3f}s")
            formatted_sections.append("")
            
            # Add insights if available
            insights = results.get("insights", {})
            if insights:
                formatted_sections.append("INSIGHTS:")
                for category, data in insights.items():
                    formatted_sections.append(f"- {category.title()}: {data}")
                formatted_sections.append("")
            
            # Add top results
            top_results = results.get("results", {}).get("top_results", [])
            if top_results:
                formatted_sections.append("TOP RESULTS:")
                for i, result in enumerate(top_results[:10], 1):  # Limit to top 10
                    content = str(result.get('content', ''))
                    formatted_sections.append(f"{i}. {content[:200]}...")
                    if result.get('metadata'):
                        meta = result['metadata']
                        formatted_sections.append(f"   - Source: {meta.get('source', 'unknown')}")
                        formatted_sections.append(f"   - Ticker: {meta.get('ticker', 'N/A')}")
                        formatted_sections.append(f"   - Relevance: {result.get('relevance_score', 0):.3f}")
                    formatted_sections.append("")
            
            return "\n".join(formatted_sections)
            
        except Exception as e:
            logger.error(f"Error formatting results: {e}")
            return f"Error formatting results: {str(e)}"
    
    def _generate_llm_analysis(self, query: str, formatted_data: str) -> Dict[str, Any]:
        """Generate LLM analysis of the data"""
        try:
            parser = JsonOutputParser(pydantic_object=LLMExplanationSchema)
            chain = self.explanation_prompt | self.llm | parser
            
            result = chain.invoke({
                "query": query,
                "results_data": formatted_data
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            return {
                "summary": f"Analysis failed: {str(e)}",
                "key_insights": [],
                "relationships": [],
                "patterns": [],
                "if_then_rules": [],
                "confidence_score": 0.0
            }
    
    def _generate_narrative_explanation(self, query: str, analysis: Dict[str, Any]) -> str:
        """Generate natural language narrative explanation"""
        try:
            chain = self.narrative_prompt | self.llm | StrOutputParser()
            
            narrative = chain.invoke({
                "query": query,
                "analysis_json": json.dumps(analysis, indent=2)
            })
            
            return narrative.strip()
            
        except Exception as e:
            logger.error(f"Error generating narrative: {e}")
            return f"Unable to generate narrative explanation: {str(e)}"
    
    def _discover_patterns(self, formatted_data: str) -> List[PatternAnalysis]:
        """Discover patterns in the data using LLM"""
        try:
            chain = self.pattern_prompt | self.llm | StrOutputParser()
            
            patterns_text = chain.invoke({"data_summary": formatted_data})
            
            # Parse patterns from text (simplified implementation)
            patterns = []
            lines = patterns_text.split('\n')
            current_pattern = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('Pattern:') or line.startswith('Trend:'):
                    if current_pattern:
                        patterns.append(current_pattern)
                    current_pattern = PatternAnalysis(
                        pattern_type="trend",
                        description=line.replace('Pattern:', '').replace('Trend:', '').strip(),
                        confidence_score=0.7,  # Default confidence
                        supporting_entities=[],
                        rule_conditions=[],
                        rule_conclusions=[],
                        examples=[]
                    )
            
            if current_pattern:
                patterns.append(current_pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error discovering patterns: {e}")
            return []
    
    def _parse_relationships(self, relationships_data: List[Dict[str, Any]]) -> List[EntityRelationship]:
        """Parse relationship data from LLM response"""
        parsed_relationships = []
        
        for rel_data in relationships_data:
            try:
                relationship = EntityRelationship(
                    source_entity=rel_data.get("source", ""),
                    target_entity=rel_data.get("target", ""),
                    relationship_type=rel_data.get("type", ""),
                    relationship_strength=float(rel_data.get("strength", 0.5)),
                    supporting_evidence=rel_data.get("evidence", []),
                    metadata=rel_data
                )
                parsed_relationships.append(relationship)
            except Exception as e:
                logger.warning(f"Error parsing relationship: {e}")
                continue
        
        return parsed_relationships
    
    def _build_subgraph_data(
        self, 
        entities: List[str], 
        relationship_types: Optional[List[str]], 
        max_depth: int
    ) -> Dict[str, Any]:
        """Build subgraph data for entities"""
        # Simplified implementation - query for all entities
        all_results = {"results": {"top_results": []}, "metadata": {}}
        
        for entity in entities:
            try:
                result = self.orchestrator.query(entity, max_results=10)
                if result.get("results"):
                    all_results["results"]["top_results"].extend(
                        result["results"].get("top_results", [])
                    )
            except Exception as e:
                logger.warning(f"Error querying entity {entity}: {e}")
        
        all_results["metadata"] = {
            "query_type": "subgraph_analysis",
            "entities": entities,
            "total_results": len(all_results["results"]["top_results"]),
            "sources_used": ["fundamentals", "news", "userposts"]
        }
        
        return all_results
    
    def _sample_domain_data(self, domain: Optional[str]) -> str:
        """Sample data from specific domain for rule generation"""
        try:
            if domain:
                query = f"sample {domain} data for analysis"
            else:
                query = "sample market data across all domains"
            
            results = self.orchestrator.query(query, max_results=20)
            return self._format_results_for_llm(results)
            
        except Exception as e:
            logger.error(f"Error sampling domain data: {e}")
            return f"Unable to sample data: {str(e)}"
    
    def _parse_rules_from_text(self, rules_text: str, min_confidence: float) -> List[Dict[str, Any]]:
        """Parse if-then rules from LLM generated text"""
        rules = []
        
        for line in rules_text.split('\n'):
            line = line.strip()
            if 'IF' in line and 'THEN' in line:
                # Extract rule parts
                parts = line.split('THEN')
                if len(parts) == 2:
                    condition = parts[0].replace('IF', '').strip()
                    conclusion_part = parts[1].strip()
                    
                    # Extract confidence if present
                    confidence = 0.7  # Default
                    if 'confidence:' in conclusion_part.lower():
                        conf_parts = conclusion_part.split('confidence:')
                        if len(conf_parts) == 2:
                            try:
                                confidence = float(conf_parts[1].strip().replace(')', ''))
                                conclusion_part = conf_parts[0].strip()
                            except:
                                pass
                    
                    if confidence >= min_confidence:
                        rules.append({
                            "condition": condition,
                            "conclusion": conclusion_part,
                            "confidence": confidence,
                            "rule_text": line
                        })
        
        return rules
    
    def _analyze_overall_sentiment(self, results: Dict[str, Any]) -> str:
        """Analyze overall sentiment from results"""
        insights = results.get("insights", {})
        sentiment_data = insights.get("sentiment_summary", {})
        
        if sentiment_data:
            overall = sentiment_data.get("overall_sentiment", "neutral")
            return overall
        
        return "neutral"
    
    def _extract_risk_indicators(self, explanation: ExplanationResult) -> List[str]:
        """Extract risk indicators from explanation"""
        risk_keywords = ["risk", "decline", "loss", "negative", "warning", "concern"]
        risk_indicators = []
        
        for insight in explanation.key_insights:
            if any(keyword in insight.lower() for keyword in risk_keywords):
                risk_indicators.append(insight)
        
        return risk_indicators
    
    def _extract_opportunities(self, explanation: ExplanationResult) -> List[str]:
        """Extract opportunities from explanation"""
        opportunity_keywords = ["opportunity", "growth", "positive", "strong", "increase", "gain"]
        opportunities = []
        
        for insight in explanation.key_insights:
            if any(keyword in insight.lower() for keyword in opportunity_keywords):
                opportunities.append(insight)
        
        return opportunities
    
    def _get_cache_key(self, query: str, results: Dict[str, Any]) -> str:
        """Generate cache key for explanation"""
        # Simple hash of query and result count
        result_count = len(results.get("results", {}).get("top_results", []))
        return f"{hash(query)}_{result_count}"
    
    def _create_error_explanation(self, query: str, error_msg: str) -> ExplanationResult:
        """Create error explanation result"""
        return ExplanationResult(
            query=query,
            summary=f"Analysis failed: {error_msg}",
            key_insights=[],
            entity_relationships=[],
            discovered_patterns=[],
            if_then_rules=[],
            natural_language_explanation=f"Unable to generate explanation due to error: {error_msg}",
            confidence_score=0.0,
            metadata={
                "error": error_msg,
                "generated_at": datetime.now().isoformat()
            }
        )
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get explainer system statistics"""
        return {
            "cache_size": len(self._explanation_cache) if self.enable_caching else 0,
            "llm_model": self.llm.model_name,
            "caching_enabled": self.enable_caching,
            "orchestrator_available": self.orchestrator is not None
        }


def demo_kg_explainer():
    """Demo function to test the Knowledge Graph Explainer"""
    print("🧠 Knowledge Graph Explainer Demo")
    print("=" * 50)
    
    try:
        # Initialize explainer
        explainer = KnowledgeGraphExplainer()
        
        # Test queries
        test_queries = [
            "AML comprehensive analysis sentiment and performance",
            "profitable companies with strong ROE",
            "recent market sentiment"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing Query: '{query}'")
            print("-" * 40)
            
            # Query orchestrator
            results = explainer.orchestrator.query(query, max_results=15)
            
            # Generate explanation
            explanation = explainer.explain_query_results(query, results)
            
            print(f"📊 Summary: {explanation.summary}")
            print(f"🎯 Key Insights ({len(explanation.key_insights)}):")
            for i, insight in enumerate(explanation.key_insights[:3], 1):
                print(f"  {i}. {insight}")
            
            print(f"🔗 Relationships Found: {len(explanation.entity_relationships)}")
            print(f"📈 Patterns Discovered: {len(explanation.discovered_patterns)}")
            print(f"🧠 Confidence Score: {explanation.confidence_score:.2f}")
            
            if explanation.if_then_rules:
                print(f"⚡ Sample Rule: {explanation.if_then_rules[0]}")
        
        # Test rule generation
        print(f"\n🔧 Generating If-Then Rules")
        print("-" * 40)
        rules = explainer.generate_if_then_rules(max_rules=3)
        for i, rule in enumerate(rules, 1):
            print(f"{i}. {rule.get('rule_text', rule.get('condition', '') + ' → ' + rule.get('conclusion', ''))}")
        
        # Test market pattern summary
        print(f"\n📊 Market Pattern Summary")
        print("-" * 40)
        market_summary = explainer.summarize_market_patterns()
        print(f"Overall Sentiment: {market_summary.get('overall_sentiment', 'Unknown')}")
        print(f"Key Trends: {len(market_summary.get('key_trends', []))}")
        print(f"Risk Indicators: {len(market_summary.get('risk_indicators', []))}")
        print(f"Opportunities: {len(market_summary.get('opportunities', []))}")
        
        print(f"\n✅ Knowledge Graph Explainer demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_kg_explainer() 