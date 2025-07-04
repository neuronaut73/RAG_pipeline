#!/usr/bin/env python3
"""
ml_model.py - ML Pipeline for Stock Outperformance Prediction

Prepares numeric features from fundamentals/performance tables and builds baseline ML models
to predict the outperformance label. Integrates with the RAG pipeline for enhanced predictions.
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path

# ML Libraries  
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import classification_report, roc_auc_score, f1_score, accuracy_score
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

class MLPipeline:
    """ML Pipeline for Stock Outperformance Prediction"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.models = {}
        self.performance = {}
        
    def load_data(self):
        """Load datasets"""
        print("📊 Loading datasets...")
        
        self.labels_df = pd.read_csv(self.data_dir / "labels.csv")
        self.fundamentals_df = pd.read_csv(self.data_dir / "fundamentals.csv") 
        self.ratios_df = pd.read_csv(self.data_dir / "financial_ratios.csv")
        
        print(f"✅ Loaded labels: {self.labels_df.shape}")
        print(f"✅ Loaded fundamentals: {self.fundamentals_df.shape}")
        print(f"✅ Loaded ratios: {self.ratios_df.shape}")
        
        return True
    
    def engineer_features(self):
        """Engineer features from fundamentals and ratios"""
        print("🔧 Engineering features...")
        
        # Start with fundamentals
        features_df = self.fundamentals_df.copy()
        
        # Merge with ratios
        if not self.ratios_df.empty:
            features_df = features_df.merge(
                self.ratios_df, 
                on=["ticker", "period_end"], 
                how="left",
                suffixes=("", "_ratio")
            )
        
        # Select numeric columns
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ["id", "id_ratio"]]
        
        # Create additional ratios
        if "revenue" in features_df.columns and "total_assets" in features_df.columns:
            features_df["asset_turnover_calc"] = features_df["revenue"] / features_df["total_assets"].replace(0, np.nan)
            
        if "net_income" in features_df.columns and "revenue" in features_df.columns:
            features_df["profit_margin_calc"] = features_df["net_income"] / features_df["revenue"].replace(0, np.nan)
            
        if "total_debt" in features_df.columns and "shareholders_equity" in features_df.columns:
            features_df["debt_to_equity_calc"] = features_df["total_debt"] / features_df["shareholders_equity"].replace(0, np.nan)
        
        # Select final feature columns
        feature_columns = features_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [col for col in feature_columns if col not in ["id", "id_ratio"]]
        
        features_clean = features_df[["ticker", "period_end"] + feature_columns].copy()
        
        print(f"✅ Engineered {len(feature_columns)} features")
        return features_clean
    
    def prepare_dataset(self):
        """Prepare dataset for modeling"""
        print("📋 Preparing dataset...")
        
        # Engineer features
        features_df = self.engineer_features()
        
        # Prepare labels
        labels_df = self.labels_df.copy()
        labels_df["target"] = (labels_df["outperformance_10d"] > 0).astype(int)
        
        # Merge features with labels
        merged_data = []
        
        for _, label_row in labels_df.iterrows():
            ticker = label_row["stock_ticker"]
            setup_date = pd.to_datetime(label_row["setup_date"])
            
            # Find most recent fundamental data before setup date
            ticker_features = features_df[features_df["ticker"] == ticker].copy()
            ticker_features["period_end"] = pd.to_datetime(ticker_features["period_end"])
            ticker_features = ticker_features[ticker_features["period_end"] <= setup_date]
            
            if len(ticker_features) > 0:
                latest_features = ticker_features.sort_values("period_end").iloc[-1]
                
                # Combine with label
                combined_row = latest_features.copy()
                combined_row["setup_id"] = label_row["setup_id"]
                combined_row["target"] = label_row["target"]
                combined_row["outperformance"] = label_row["outperformance_10d"]
                
                merged_data.append(combined_row)
        
        if not merged_data:
            raise ValueError("No matching data found")
            
        # Create final dataset
        final_df = pd.DataFrame(merged_data)
        
        # Select features
        exclude_cols = ["ticker", "period_end", "setup_id", "target", "outperformance"]
        feature_cols = [col for col in final_df.columns if col not in exclude_cols]
        feature_cols = [col for col in feature_cols if final_df[col].dtype in ["float64", "int64"]]
        
        X = final_df[feature_cols].copy()
        y = final_df["target"].copy()
        
        print(f"✅ Dataset prepared: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"   Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
