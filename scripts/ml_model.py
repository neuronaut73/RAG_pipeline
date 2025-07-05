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
    
    def __init__(self, data_dir="../data"):
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
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train multiple baseline models"""
        print("🚀 Training baseline models...")
        
        # Define models
        models = {
            "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "XGBoost": xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0),
            "LightGBM": lgb.LGBMClassifier(n_estimators=100, random_state=42, verbosity=-1)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"   Training {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
                
                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                try:
                    roc_auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5
                except:
                    roc_auc = 0.5
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring="f1")
                
                results[name] = {
                    "model": model,
                    "accuracy": accuracy,
                    "f1_score": f1,
                    "roc_auc": roc_auc,
                    "cv_mean": cv_scores.mean(),
                    "cv_std": cv_scores.std()
                }
                
                print(f"     ✅ {name}: F1={f1:.3f}, ROC-AUC={roc_auc:.3f}")
                
            except Exception as e:
                print(f"     ❌ {name} failed: {e}")
                continue
        
        self.models = {name: result["model"] for name, result in results.items()}
        self.performance = results
        
        print(f"✅ Trained {len(results)} models successfully")
        return results
    
    def get_model_summary(self):
        """Get summary of model performance"""
        if not self.performance:
            return pd.DataFrame()
        
        summary_data = []
        for name, perf in self.performance.items():
            summary_data.append({
                "Model": name,
                "Accuracy": perf["accuracy"],
                "F1-Score": perf["f1_score"],
                "ROC-AUC": perf["roc_auc"],
                "CV Mean": perf["cv_mean"],
                "CV Std": perf["cv_std"]
            })
        
        df = pd.DataFrame(summary_data)
        return df.sort_values("F1-Score", ascending=False)
    
    def predict_outperformance(self, ticker, model_name=None):
        """Predict outperformance for a ticker"""
        if not self.models:
            return {"error": "No trained models available"}
        
        # Get best model if not specified
        if model_name is None:
            best_model = max(self.performance.items(), key=lambda x: x[1]["f1_score"])
            model_name = best_model[0]
        
        if model_name not in self.models:
            return {"error": f"Model {model_name} not found"}
        
        model = self.models[model_name]
        
        # Get latest data for ticker
        ticker_data = self.fundamentals_df[self.fundamentals_df["ticker"] == ticker]
        if ticker_data.empty:
            return {"error": f"No data found for {ticker}"}
        
        # Engineer features for single ticker
        features_df = self.engineer_features()
        ticker_features = features_df[features_df["ticker"] == ticker]
        
        if ticker_features.empty:
            return {"error": f"No features available for {ticker}"}
        
        # Get latest features
        latest_features = ticker_features.sort_values("period_end").iloc[-1]
        
        # Prepare feature vector
        exclude_cols = ["ticker", "period_end"]
        feature_cols = [col for col in latest_features.index if col not in exclude_cols]
        feature_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(latest_features[col])]
        
        X = latest_features[feature_cols].values.reshape(1, -1)
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        try:
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None
            
            result = {
                "ticker": ticker,
                "model": model_name,
                "prediction": int(prediction),
                "prediction_label": "Outperform" if prediction == 1 else "Underperform",
                "probability": {
                    "underperform": float(probability[0]) if probability is not None else None,
                    "outperform": float(probability[1]) if probability is not None else None
                },
                "confidence": float(max(probability)) if probability is not None else None,
                "features_used": len(feature_cols)
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def run_pipeline(self, test_size=0.2):
        """Run the complete ML pipeline"""
        print("🚀 Running complete ML pipeline...")
        
        # Load data
        if not self.load_data():
            return {"error": "Failed to load data"}
        
        # Prepare dataset
        try:
            X, y = self.prepare_dataset()
        except Exception as e:
            return {"error": f"Failed to prepare dataset: {str(e)}"}
        
        if len(X) < 10:
            return {"error": f"Insufficient data: only {len(X)} samples"}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Handle missing values - first remove columns with too many NaNs
        nan_threshold = 0.8  # Remove columns with >80% NaN values
        nan_ratio = X_train.isnull().sum() / len(X_train)
        good_columns = nan_ratio[nan_ratio <= nan_threshold].index.tolist()
        
        X_train_filtered = X_train[good_columns].copy()
        X_test_filtered = X_test[good_columns].copy()
        
        # Now impute remaining missing values
        imputer = SimpleImputer(strategy="median")
        X_train_clean = pd.DataFrame(
            imputer.fit_transform(X_train_filtered),
            columns=X_train_filtered.columns,
            index=X_train_filtered.index
        )
        X_test_clean = pd.DataFrame(
            imputer.transform(X_test_filtered),
            columns=X_test_filtered.columns,
            index=X_test_filtered.index
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train_clean),
            columns=X_train_clean.columns,
            index=X_train_clean.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test_clean),
            columns=X_test_clean.columns,
            index=X_test_clean.index
        )
        
        # Train models
        results = self.train_models(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # Generate summary
        summary = self.get_model_summary()
        
        pipeline_results = {
            "success": True,
            "data_summary": {
                "total_samples": len(X),
                "features": len(X.columns),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "target_distribution": y.value_counts().to_dict()
            },
            "model_performance": summary.to_dict("records") if not summary.empty else [],
            "best_model": summary.iloc[0].to_dict() if not summary.empty else {}
        }
        
        print("✅ ML Pipeline completed successfully!")
        return pipeline_results

def main():
    """Main function for testing the ML pipeline"""
    print("🧪 Testing ML Pipeline for Stock Outperformance Prediction\n")
    
    # Initialize pipeline
    ml_pipeline = MLPipeline()
    
    # Run pipeline
    results = ml_pipeline.run_pipeline(test_size=0.2)
    
    if results.get("success"):
        print("\n📊 Results Summary:")
        data_summary = results["data_summary"]
        print(f"   - Total samples: {data_summary['total_samples']}")
        print(f"   - Features: {data_summary['features']}")
        print(f"   - Models trained: {len(results['model_performance'])}")
        
        if results["model_performance"]:
            best_model = results["best_model"]
            print(f"\n🏆 Best Model: {best_model['Model']}")
            print(f"   - F1-Score: {best_model['F1-Score']:.3f}")
            print(f"   - ROC-AUC: {best_model['ROC-AUC']:.3f}")
            print(f"   - Accuracy: {best_model['Accuracy']:.3f}")
        
        # Test prediction
        sample_tickers = ["LGEN.L", "BLND.L", "DGE.L"]
        for ticker in sample_tickers:
            prediction = ml_pipeline.predict_outperformance(ticker)
            if "error" not in prediction:
                print(f"\n🔮 Prediction for {ticker}:")
                print(f"   - Prediction: {prediction['prediction_label']}")
                if prediction["confidence"]:
                    print(f"   - Confidence: {prediction['confidence']:.3f}")
                break
    else:
        error_msg = results.get("error", "Unknown error")
        print(f"❌ Pipeline failed: {error_msg}")

if __name__ == "__main__":
    main()
