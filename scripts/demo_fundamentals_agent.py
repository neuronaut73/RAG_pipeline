#!/usr/bin/env python3
"""
demo_fundamentals_agent.py - Fundamentals Agent Demonstration

Showcases the key capabilities of the Fundamentals Domain RAG Agent
with real examples and practical use cases.
"""

from agent_fundamentals import FundamentalsAgent
import pandas as pd


def main():
    """Demonstrate fundamentals agent capabilities"""
    
    print("🏦 FUNDAMENTALS AGENT DEMONSTRATION")
    print("=" * 60)
    
    # Initialize agent
    print("\n📊 Initializing Fundamentals Agent...")
    agent = FundamentalsAgent()
    
    # Get table overview
    stats = agent.get_table_stats()
    print(f"📈 Database: {stats['total_records']} records, {stats['unique_tickers']} companies")
    print(f"📅 Date range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}")
    print(f"🎯 Performance labels: {stats['records_with_labels']} records ({stats['label_percentage']:.1f}%)")
    
    # Demo 1: Company Analysis
    print("\n" + "="*60)
    print("🏢 DEMO 1: COMPREHENSIVE COMPANY ANALYSIS")
    print("="*60)
    
    ticker = "LLOY.L"  # Lloyds Banking Group
    print(f"\n🔍 Analyzing {ticker} (Lloyds Banking Group)...")
    
    company_data = agent.retrieve_by_ticker(ticker, limit=3)
    
    if company_data:
        print(f"📊 Found {len(company_data)} financial reports:")
        for i, record in enumerate(company_data, 1):
            print(f"\n  📋 Report {i}: {record['report_type'].title()} {record['period_end']}")
            print(f"     💰 Revenue: £{record['revenue']:,.0f}")
            print(f"     📈 ROE: {record['roe']:.2%}")
            print(f"     💸 D/E Ratio: {record['debt_to_equity']:.2f}")
            print(f"     💵 Current Ratio: {record['current_ratio']:.2f}")
            if record['has_performance_labels']:
                print(f"     🎯 10-day Outperformance: {record['outperformance_10d']:.2f}%")
    
    # Demo 2: Semantic Search - High Quality Companies
    print("\n" + "="*60)
    print("🌟 DEMO 2: SEMANTIC SEARCH - HIGH QUALITY COMPANIES")
    print("="*60)
    
    print("\n🔍 Searching for: 'Profitable companies with strong returns and low debt'")
    
    quality_companies = agent.semantic_search(
        query="profitable companies with strong returns and low debt",
        limit=5
    )
    
    if quality_companies:
        print(f"🏆 Found {len(quality_companies)} high-quality companies:")
        for i, company in enumerate(quality_companies, 1):
            similarity = company.get('similarity_score', 0)
            print(f"\n  {i}. {company['ticker']} ({company['period_end']})")
            print(f"     🎯 Match Score: {similarity:.3f}")
            print(f"     📈 ROE: {company['roe']:.2%}")
            print(f"     💸 D/E Ratio: {company['debt_to_equity']:.2f}")
            if company['has_performance_labels']:
                print(f"     🚀 Outperformance: {company['outperformance_10d']:.2f}%")
    
    # Demo 3: Performance Analysis
    print("\n" + "="*60)
    print("📊 DEMO 3: PERFORMANCE ANALYSIS BY FUNDAMENTALS")
    print("="*60)
    
    print("\n🔍 Analyzing companies with ROE > 10%...")
    
    high_roe_performers = agent.analyze_performance_by_fundamentals(
        metric_filter="roe > 0.10",
        limit=10,
        sort_by_performance=True
    )
    
    if high_roe_performers:
        print(f"🏆 Found {len(high_roe_performers)} high-ROE companies (sorted by performance):")
        
        total_outperformance = 0
        count_with_labels = 0
        
        for i, company in enumerate(high_roe_performers, 1):
            print(f"\n  {i}. {company['ticker']} ({company['period_end']})")
            print(f"     📈 ROE: {company['roe']:.2%}")
            print(f"     💰 Revenue: £{company['revenue']:,.0f}")
            if company['has_performance_labels']:
                print(f"     🎯 10-day Outperformance: {company['outperformance_10d']:.2f}%")
                total_outperformance += company['outperformance_10d']
                count_with_labels += 1
        
        if count_with_labels > 0:
            avg_outperformance = total_outperformance / count_with_labels
            print(f"\n📊 Average outperformance of high-ROE companies: {avg_outperformance:.2f}%")
    
    # Demo 4: Company Similarity Analysis
    print("\n" + "="*60)
    print("🤝 DEMO 4: FIND SIMILAR COMPANIES")
    print("="*60)
    
    reference_ticker = "JDW.L"  # Wetherspoons
    print(f"\n🔍 Finding companies similar to {reference_ticker} (JD Wetherspoon)...")
    
    similar_companies = agent.find_similar_companies(
        reference_ticker=reference_ticker,
        limit=5,
        exclude_same_company=True
    )
    
    if similar_companies:
        print(f"🏪 Found {len(similar_companies)} companies with similar financial profiles:")
        for i, company in enumerate(similar_companies, 1):
            similarity = company.get('similarity_score', 0)
            print(f"\n  {i}. {company['ticker']} ({company['period_end']})")
            print(f"     🎯 Similarity: {similarity:.3f}")
            print(f"     📈 ROE: {company['roe']:.2%}")
            print(f"     💸 D/E Ratio: {company['debt_to_equity']:.2f}")
            print(f"     📊 Summary: {company['financial_summary'][:80]}...")
    
    # Demo 5: Date Range Analysis
    print("\n" + "="*60)
    print("📅 DEMO 5: RECENT FINANCIAL REPORTS (2024)")
    print("="*60)
    
    print("\n🔍 Getting recent financial reports from 2024...")
    
    recent_reports = agent.retrieve_by_date_range(
        start_date="2024-01-01",
        end_date="2024-12-31",
        limit=8
    )
    
    if recent_reports:
        print(f"📈 Found {len(recent_reports)} reports from 2024:")
        
        quarterly_count = sum(1 for r in recent_reports if r['report_type'] == 'quarterly')
        annual_count = sum(1 for r in recent_reports if r['report_type'] == 'annual')
        
        print(f"   📋 Report types: {quarterly_count} quarterly, {annual_count} annual")
        
        companies_2024 = set(r['ticker'] for r in recent_reports)
        print(f"   🏢 Companies with 2024 data: {', '.join(sorted(companies_2024))}")
        
        for i, report in enumerate(recent_reports[:5], 1):
            print(f"\n  {i}. {report['ticker']} - {report['report_type']} {report['period_end']}")
            if report['revenue'] > 0:
                print(f"     💰 Revenue: £{report['revenue']:,.0f}")
            if report['roe'] != 0:
                print(f"     📈 ROE: {report['roe']:.2%}")
    
    # Demo 6: Value vs Growth Analysis
    print("\n" + "="*60)
    print("💎 DEMO 6: VALUE VS GROWTH ANALYSIS")
    print("="*60)
    
    print("\n🔍 Value stocks (Low P/E, decent ROE)...")
    value_stocks = agent.analyze_performance_by_fundamentals(
        metric_filter="pe_ratio > 0 AND pe_ratio < 15 AND roe > 0.05",
        limit=5
    )
    
    print("\n🔍 Growth stocks (High ROE, large revenue)...")
    growth_stocks = agent.analyze_performance_by_fundamentals(
        metric_filter="roe > 0.15 AND revenue > 1000000000",
        limit=5
    )
    
    print(f"\n📊 Value stocks found: {len(value_stocks)}")
    if value_stocks:
        for stock in value_stocks[:3]:
            print(f"   💎 {stock['ticker']}: P/E {stock['pe_ratio']:.1f}, ROE {stock['roe']:.2%}")
    
    print(f"\n📈 Growth stocks found: {len(growth_stocks)}")
    if growth_stocks:
        for stock in growth_stocks[:3]:
            print(f"   🚀 {stock['ticker']}: ROE {stock['roe']:.2%}, Revenue £{stock['revenue']:,.0f}")
    
    # Summary
    print("\n" + "="*60)
    print("✅ DEMONSTRATION COMPLETE")
    print("="*60)
    
    print("\n🎯 The Fundamentals Agent provides:")
    print("   📊 Comprehensive financial data retrieval")
    print("   🔍 Intelligent semantic search capabilities")  
    print("   📈 Performance correlation analysis")
    print("   🤝 Company similarity matching")
    print("   📅 Flexible date and period filtering")
    print("   💡 Investment strategy support (value, growth, quality)")
    
    print(f"\n📚 Processed {stats['total_records']} financial records across {stats['unique_tickers']} companies")
    print("🏦 Ready for integration with trading systems and research platforms!")


if __name__ == "__main__":
    main() 