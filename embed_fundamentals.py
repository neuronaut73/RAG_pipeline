#!/usr/bin/env python3
"""
embed_fundamentals.py - Fundamentals Domain Embedding Pipeline

Processes financial fundamentals and ratios, converts to text summaries,
and stores embeddings in LanceDB with performance labels.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
import logging

import lancedb
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FundamentalsEmbedder:
    """Fundamentals Domain Embedding Pipeline"""
    
    def __init__(self, data_dir="data", lancedb_dir="lancedb_store", 
                 embedding_model="all-MiniLM-L6-v2"):
        self.data_dir = Path(data_dir)
        self.lancedb_dir = Path(lancedb_dir)
        
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.db = lancedb.connect(str(self.lancedb_dir))
        
    def load_data(self):
        """Load financial data from CSV files"""
        # Load fundamentals
        fundamentals_path = self.data_dir / "fundamentals.csv"
        self.fundamentals_df = pd.read_csv(fundamentals_path)
        logger.info(f"Loaded {len(self.fundamentals_df)} fundamentals records")
        
        # Load ratios
        ratios_path = self.data_dir / "financial_ratios.csv"
        self.ratios_df = pd.read_csv(ratios_path)
        logger.info(f"Loaded {len(self.ratios_df)} ratio records")
        
        # Load labels
        labels_path = self.data_dir / "labels.csv"
        self.labels_df = pd.read_csv(labels_path)
        logger.info(f"Loaded {len(self.labels_df)} label records")
        
    def merge_financial_data(self):
        """Merge fundamentals and ratios data"""
        # Create merge keys
        self.fundamentals_df['merge_key'] = (
            self.fundamentals_df['ticker'].astype(str) + "_" +
            self.fundamentals_df['report_type'].astype(str) + "_" +
            self.fundamentals_df['period_end'].astype(str)
        )
        
        self.ratios_df['merge_key'] = (
            self.ratios_df['ticker'].astype(str) + "_" +
            self.ratios_df['report_type'].astype(str) + "_" +
            self.ratios_df['period_end'].astype(str)
        )
        
        # Merge dataframes
        merged_df = pd.merge(
            self.fundamentals_df, self.ratios_df,
            on='merge_key', how='outer', suffixes=('_fund', '_ratio')
        )
        
        # Clean up columns
        for col in ['ticker', 'report_type', 'period_end']:
            fund_col = f"{col}_fund"
            ratio_col = f"{col}_ratio"
            if fund_col in merged_df.columns and ratio_col in merged_df.columns:
                merged_df[col] = merged_df[fund_col].fillna(merged_df[ratio_col])
                merged_df = merged_df.drop(columns=[fund_col, ratio_col])
        
        merged_df = merged_df.drop(columns=['merge_key'])
        logger.info(f"Merged data contains {len(merged_df)} records")
        return merged_df
        
    def create_financial_summary(self, row):
        """Convert financial data into text summary"""
        ticker = row.get('ticker', 'Unknown')
        period_end = row.get('period_end', 'Unknown')
        report_type = row.get('report_type', 'Unknown')
        
        summary_parts = []
        summary_parts.append(f"Financial Analysis for {ticker} - {report_type.title()} Report ending {period_end}")
        
        # Revenue and Profitability
        revenue = row.get('revenue')
        net_income = row.get('net_income')
        
        if pd.notna(revenue) and revenue != 0:
            revenue_mil = revenue / 1_000_000
            summary_parts.append(f"Revenue of ${revenue_mil:.1f} million")
            
            gross_margin = row.get('gross_margin', 0) * 100 if pd.notna(row.get('gross_margin')) else 0
            operating_margin = row.get('operating_margin', 0) * 100 if pd.notna(row.get('operating_margin')) else 0
            net_margin = row.get('net_margin', 0) * 100 if pd.notna(row.get('net_margin')) else 0
            
            summary_parts.append(f"with gross margin of {gross_margin:.1f}%, operating margin of {operating_margin:.1f}%")
            
            if pd.notna(net_income):
                net_income_mil = net_income / 1_000_000
                if net_income > 0:
                    summary_parts.append(f"delivering net profit of ${net_income_mil:.1f} million ({net_margin:.1f}% margin)")
                else:
                    summary_parts.append(f"reporting net loss of ${abs(net_income_mil):.1f} million")
        
        # Balance Sheet
        total_assets = row.get('total_assets')
        debt_to_equity = row.get('debt_to_equity')
        
        if pd.notna(total_assets) and total_assets != 0:
            assets_bil = total_assets / 1_000_000_000
            summary_parts.append(f"Total assets of ${assets_bil:.1f} billion")
            
            if pd.notna(debt_to_equity):
                if debt_to_equity < 0.3:
                    debt_desc = "conservative debt levels"
                elif debt_to_equity < 0.7:
                    debt_desc = "moderate debt levels"
                else:
                    debt_desc = "high debt levels"
                summary_parts.append(f"with {debt_desc} (D/E ratio: {debt_to_equity:.2f})")
        
        # Profitability Ratios
        roe = row.get('roe')
        roa = row.get('roa')
        
        profitability_parts = []
        if pd.notna(roe) and roe != 0:
            roe_pct = roe * 100
            if roe_pct > 15:
                profitability_parts.append(f"excellent return on equity of {roe_pct:.1f}%")
            elif roe_pct > 10:
                profitability_parts.append(f"good return on equity of {roe_pct:.1f}%")
            elif roe_pct > 0:
                profitability_parts.append(f"modest return on equity of {roe_pct:.1f}%")
            else:
                profitability_parts.append(f"negative return on equity of {roe_pct:.1f}%")
        
        if pd.notna(roa) and roa != 0:
            roa_pct = roa * 100
            if roa_pct > 5:
                profitability_parts.append(f"strong asset efficiency (ROA: {roa_pct:.1f}%)")
            elif roa_pct > 0:
                profitability_parts.append(f"moderate asset efficiency (ROA: {roa_pct:.1f}%)")
            else:
                profitability_parts.append(f"poor asset efficiency (ROA: {roa_pct:.1f}%)")
        
        if profitability_parts:
            summary_parts.append("Profitability metrics show " + " and ".join(profitability_parts))
        
        # Liquidity
        current_ratio = row.get('current_ratio')
        if pd.notna(current_ratio):
            if current_ratio > 2:
                summary_parts.append(f"Strong liquidity position (current ratio: {current_ratio:.2f})")
            elif current_ratio > 1:
                summary_parts.append(f"Adequate liquidity (current ratio: {current_ratio:.2f})")
            else:
                summary_parts.append(f"Tight liquidity (current ratio: {current_ratio:.2f})")
        
        # Valuation
        pe_ratio = row.get('pe_ratio')
        pb_ratio = row.get('pb_ratio')
        
        valuation_parts = []
        if pd.notna(pe_ratio) and pe_ratio > 0:
            if pe_ratio < 10:
                valuation_parts.append(f"attractive valuation (P/E: {pe_ratio:.1f})")
            elif pe_ratio < 20:
                valuation_parts.append(f"reasonable valuation (P/E: {pe_ratio:.1f})")
            else:
                valuation_parts.append(f"premium valuation (P/E: {pe_ratio:.1f})")
        
        if pd.notna(pb_ratio):
            if pb_ratio < 1:
                valuation_parts.append(f"trading below book value (P/B: {pb_ratio:.2f})")
            elif pb_ratio < 3:
                valuation_parts.append(f"reasonable price-to-book ratio ({pb_ratio:.2f})")
            else:
                valuation_parts.append(f"high price-to-book ratio ({pb_ratio:.2f})")
        
        if valuation_parts:
            summary_parts.append("Valuation indicates " + " and ".join(valuation_parts))
        
        return ". ".join(summary_parts) + "."
    
    def create_records_with_labels(self, merged_df):
        """Create embedding records with performance labels"""
        records = []
        
        for idx, row in merged_df.iterrows():
            if pd.isna(row.get('ticker')) or pd.isna(row.get('period_end')):
                continue
            
            financial_summary = self.create_financial_summary(row)
            
            record = {
                'id': f"{row.get('ticker', '')}_{row.get('report_type', '')}_{row.get('period_end', '')}",
                'ticker': str(row.get('ticker', '')),
                'report_type': str(row.get('report_type', '')),
                'period_end': str(row.get('period_end', '')),
                'financial_summary': financial_summary,
                'text_length': len(financial_summary),
                'summary_type': 'comprehensive_financial_analysis',
                
                # Core Metrics
                'revenue': float(row.get('revenue', 0)) if pd.notna(row.get('revenue')) else 0,
                'net_income': float(row.get('net_income', 0)) if pd.notna(row.get('net_income')) else 0,
                'total_assets': float(row.get('total_assets', 0)) if pd.notna(row.get('total_assets')) else 0,
                
                # Key Ratios
                'roe': float(row.get('roe', 0)) if pd.notna(row.get('roe')) else 0,
                'roa': float(row.get('roa', 0)) if pd.notna(row.get('roa')) else 0,
                'debt_to_equity': float(row.get('debt_to_equity', 0)) if pd.notna(row.get('debt_to_equity')) else 0,
                'current_ratio': float(row.get('current_ratio', 0)) if pd.notna(row.get('current_ratio')) else 0,
                'pe_ratio': float(row.get('pe_ratio', 0)) if pd.notna(row.get('pe_ratio')) else 0,
                
                'embedded_at': datetime.now().isoformat()
            }
            
            # Match with performance labels
            ticker = row.get('ticker', '')
            if ticker and self.labels_df is not None:
                ticker_labels = self.labels_df[self.labels_df['stock_ticker'] == ticker]
                
                if not ticker_labels.empty:
                    latest_label = ticker_labels.iloc[-1]
                    record.update({
                        'setup_id': str(latest_label.get('setup_id', '')),
                        'stock_return_10d': float(latest_label.get('stock_return_10d', 0)),
                        'outperformance_10d': float(latest_label.get('outperformance_10d', 0)),
                        'days_outperformed_10d': int(latest_label.get('days_outperformed_10d', 0)),
                        'has_performance_labels': True
                    })
                else:
                    record['has_performance_labels'] = False
            else:
                record['has_performance_labels'] = False
            
            records.append(record)
        
        labeled_records = [r for r in records if r.get('has_performance_labels', False)]
        logger.info(f"Created {len(records)} records, {len(labeled_records)} with performance labels")
        return records
    
    def create_embeddings(self, records):
        """Create embeddings for financial summaries"""
        texts = [record['financial_summary'] for record in records]
        logger.info(f"Generating embeddings for {len(texts)} financial summaries")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        for record, embedding in zip(records, embeddings):
            record['vector'] = embedding.tolist()
        
        logger.info("Embeddings created successfully")
    
    def store_in_lancedb(self, records, table_name="fundamentals_embeddings"):
        """Store records in LanceDB"""
        if not records:
            logger.warning("No records to store")
            return
            
        logger.info(f"Storing {len(records)} records in LanceDB table: {table_name}")
        
        df = pd.DataFrame(records)
        df['vector'] = df['vector'].apply(lambda x: np.array(x, dtype=np.float32))
        
        # Handle data types
        for col in df.columns:
            if df[col].dtype == 'object' and col != 'vector':
                df[col] = df[col].astype(str).replace('nan', '')
        
        try:
            if table_name in self.db.table_names():
                self.db.drop_table(table_name)
                
            table = self.db.create_table(table_name, df)
            logger.info(f"Successfully created table '{table_name}' with {len(df)} records")
            
        except Exception as e:
            logger.error(f"Error storing data in LanceDB: {e}")
            raise
    
    def run_pipeline(self):
        """Execute the complete fundamentals embedding pipeline"""
        logger.info("Starting Fundamentals Domain Embedding Pipeline")
        
        self.load_data()
        merged_df = self.merge_financial_data()
        records = self.create_records_with_labels(merged_df)
        self.create_embeddings(records)
        self.store_in_lancedb(records)
        
        logger.info("Pipeline Summary:")
        logger.info(f"  Total records: {len(records)}")
        logger.info(f"  With performance labels: {len([r for r in records if r.get('has_performance_labels')])}")
        logger.info(f"  Unique tickers: {len(set(r['ticker'] for r in records))}")


def run_fundamentals_pipeline():
    """Run the fundamentals embedding pipeline"""
    embedder = FundamentalsEmbedder()
    embedder.run_pipeline()


if __name__ == "__main__":
    run_fundamentals_pipeline() 