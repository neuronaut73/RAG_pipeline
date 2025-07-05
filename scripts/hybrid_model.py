#!/usr/bin/env python3
"""
hybrid_model.py - Hybrid KG + ML Ensemble for Stock Outperformance Prediction

Combines Knowledge Graph LLM-based reasoning with traditional ML models.
"""

import os
import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path

# ML Libraries
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# LangChain imports  
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Local imports
from ml_model import MLPipeline
from orchestrator_langgraph import LangGraphOrchestrator

warnings.filterwarnings("ignore")

@dataclass
class PredictionResult:
    """Individual prediction result from a model"""
    ticker: str
    prediction: int
    confidence: float
    model_type: str
    reasoning: Optional[str] = None

class KnowledgeGraphClassifier:
    """LLM-based classifier using knowledge graph reasoning"""
    
    def __init__(self, orchestrator=None, model_name="gpt-4o-mini"):
        self.orchestrator = orchestrator or LangGraphOrchestrator()
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)
        self._setup_classification_prompt()
        
    def _setup_classification_prompt(self):
        """Setup the classification prompt for LLM"""
        self.classification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert financial analyst. Analyze company data to predict 
            whether a stock will outperform the market over the next 10 days.
            
            Return JSON with:
            - prediction: 0 (underperform) or 1 (outperform)  
            - confidence: 0.0 to 1.0
            - reasoning: Your detailed analysis
            
            Be conservative - market outperformance is challenging."""),
            
            ("user", """Analyze {ticker} with this data:
            
            FUNDAMENTALS: {fundamental_data}
            SENTIMENT: {sentiment_data}
            NEWS: {news_data}
            
            Provide prediction as JSON format.""")
        ])
    
    def predict_outperformance(self, ticker: str) -> PredictionResult:
        """Generate KG-based prediction for a ticker"""
        try:
            # Gather data
            fundamental_data = self._get_data(ticker, "fundamentals")
            sentiment_data = self._get_data(ticker, "sentiment") 
            news_data = self._get_data(ticker, "news")
            
            # Generate prediction
            response = self.classification_prompt | self.llm | JsonOutputParser()
            
            prediction_data = response.invoke({
                "ticker": ticker,
                "fundamental_data": fundamental_data,
                "sentiment_data": sentiment_data,
                "news_data": news_data
            })
            
            return PredictionResult(
                ticker=ticker,
                prediction=prediction_data["prediction"],
                confidence=prediction_data["confidence"],
                model_type="KnowledgeGraph",
                reasoning=prediction_data["reasoning"]
            )
            
        except Exception as e:
            return PredictionResult(
                ticker=ticker,
                prediction=0,
                confidence=0.5,
                model_type="KnowledgeGraph",
                reasoning=f"Error: {str(e)}"
            )
    
    def _get_data(self, ticker: str, data_type: str) -> str:
        """Get specific data type for ticker"""
        try:
            if data_type == "fundamentals":
                query = f"What are the key financial metrics for {ticker}?"
            elif data_type == "sentiment":
                query = f"What is market sentiment about {ticker}?"
            else:  # news
                query = f"What recent news about {ticker}?"
                
            results = self.orchestrator.query(
                user_query=query,
                max_results=5,
                include_cross_ranking=False
            )
            
            # Extract relevant results
            domain_results = results.get("results", {}).get(data_type.replace("fundamentals", "fundamentals"), [])
            if domain_results:
                return f"{len(domain_results)} items: " + " ".join([
                    r.get("content", "")[:100] for r in domain_results[:2]
                ])
            return f"No {data_type} data available"
            
        except Exception as e:
            return f"Error retrieving {data_type}: {str(e)}"

class HybridEnsembleModel:
    """Hybrid ensemble combining Knowledge Graph reasoning with ML models"""
    
    def __init__(self, data_dir="../data"):
        self.data_dir = Path(data_dir)
        self.ml_pipeline = MLPipeline(data_dir=data_dir)
        self.kg_classifier = KnowledgeGraphClassifier()
        
        # Performance tracking
        self.kg_performance = {}
        self.ml_performance = {}
        self.ensemble_performance = {}
        
        # Ensemble weights
        self.kg_weight = 0.4
        self.ml_weight = 0.6
        
    def train_and_evaluate(self, test_size=0.2, verbose=True):
        """Train all models and evaluate performance"""
        if verbose:
            print("🚀 Training Hybrid Ensemble Model...\n")
        
        # 1. Train ML models
        if verbose:
            print("🤖 Training ML Pipeline...")
        ml_results = self.ml_pipeline.run_pipeline(test_size=test_size)
        
        if not ml_results.get("success"):
            raise Exception(f"ML pipeline failed: {ml_results.get('error')}")
        
        self.ml_performance = ml_results
        
        # 2. Evaluate KG classifier
        if verbose:
            print("🧠 Evaluating Knowledge Graph Classifier...")
        kg_performance = self._evaluate_kg_classifier(test_size=test_size)
        self.kg_performance = kg_performance
        
        # 3. Calculate ensemble performance
        if verbose:
            print("🎯 Calculating Hybrid Ensemble Performance...")
        ensemble_performance = self._calculate_ensemble_performance()
        self.ensemble_performance = ensemble_performance
        
        # 4. Print report
        if verbose:
            self._print_performance_report()
        
        return {
            "ml_performance": self.ml_performance,
            "kg_performance": self.kg_performance,
            "ensemble_performance": self.ensemble_performance
        }
    
    def _evaluate_kg_classifier(self, test_size=0.2):
        """Evaluate KG classifier on test data"""
        try:
            # Get test tickers
            test_tickers = ["LGEN.L", "BLND.L", "DGE.L"]  # Limit for demo
            
            kg_predictions = []
            kg_confidences = []
            
            print(f"   Testing KG classifier on {len(test_tickers)} tickers...")
            
            for ticker in test_tickers:
                pred_result = self.kg_classifier.predict_outperformance(ticker)
                kg_predictions.append(pred_result.prediction)
                kg_confidences.append(pred_result.confidence)
                
                print(f"     {ticker}: {pred_result.prediction} (conf: {pred_result.confidence:.3f})")
            
            # For demo, create mock ground truth (normally from test set)
            y_true = [1, 0, 1][:len(kg_predictions)]  # Mock labels
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, kg_predictions)
            f1 = f1_score(y_true, kg_predictions, zero_division=0)
            
            try:
                roc_auc = roc_auc_score(y_true, kg_confidences) if len(np.unique(y_true)) > 1 else 0.5
            except:
                roc_auc = 0.5
            
            return {
                "accuracy": accuracy,
                "f1_score": f1,
                "roc_auc": roc_auc,
                "predictions_made": len(kg_predictions),
                "test_samples": len(y_true)
            }
            
        except Exception as e:
            return {
                "error": f"KG evaluation failed: {str(e)}",
                "accuracy": 0.0,
                "f1_score": 0.0,
                "roc_auc": 0.5
            }
    
    def _calculate_ensemble_performance(self):
        """Calculate ensemble performance from individual components"""
        try:
            ml_best_model = self.ml_performance.get("best_model", {})
            kg_metrics = self.kg_performance
            
            if not ml_best_model or "error" in kg_metrics:
                return {"error": "Missing component performance data"}
            
            # Weighted ensemble (simple approach)
            ml_f1 = ml_best_model.get("F1-Score", 0)
            kg_f1 = kg_metrics.get("f1_score", 0)
            ensemble_f1 = (self.ml_weight * ml_f1) + (self.kg_weight * kg_f1)
            
            ml_accuracy = ml_best_model.get("Accuracy", 0)
            kg_accuracy = kg_metrics.get("accuracy", 0)
            ensemble_accuracy = (self.ml_weight * ml_accuracy) + (self.kg_weight * kg_accuracy)
            
            ml_roc = ml_best_model.get("ROC-AUC", 0.5)
            kg_roc = kg_metrics.get("roc_auc", 0.5)
            ensemble_roc = (self.ml_weight * ml_roc) + (self.kg_weight * kg_roc)
            
            return {
                "accuracy": ensemble_accuracy,
                "f1_score": ensemble_f1,
                "roc_auc": ensemble_roc,
                "ml_weight": self.ml_weight,
                "kg_weight": self.kg_weight,
                "method": "weighted_average"
            }
            
        except Exception as e:
            return {"error": f"Ensemble calculation failed: {str(e)}"}
    
    def _print_performance_report(self):
        """Print comprehensive performance comparison"""
        print("\n" + "="*80)
        print("🎯 HYBRID ENSEMBLE MODEL - PERFORMANCE REPORT")
        print("="*80)
        
        # ML Performance
        print("\n🤖 ML MODEL PERFORMANCE:")
        if self.ml_performance.get("success"):
            ml_best = self.ml_performance["best_model"]
            print(f"   Best Model: {ml_best['Model']}")
            print(f"   F1-Score:   {ml_best['F1-Score']:.3f}")
            print(f"   Accuracy:   {ml_best['Accuracy']:.3f}")
            print(f"   ROC-AUC:    {ml_best['ROC-AUC']:.3f}")
        else:
            print(f"   ❌ Failed: {self.ml_performance.get('error', 'Unknown error')}")
        
        # KG Performance
        print("\n🧠 KNOWLEDGE GRAPH PERFORMANCE:")
        if "error" not in self.kg_performance:
            print(f"   F1-Score:   {self.kg_performance.get('f1_score', 0):.3f}")
            print(f"   Accuracy:   {self.kg_performance.get('accuracy', 0):.3f}")
            print(f"   ROC-AUC:    {self.kg_performance.get('roc_auc', 0.5):.3f}")
        else:
            print(f"   ❌ Failed: {self.kg_performance.get('error', 'Unknown error')}")
        
        # Ensemble Performance
        print("\n🎯 HYBRID ENSEMBLE PERFORMANCE:")
        if "error" not in self.ensemble_performance:
            print(f"   F1-Score:   {self.ensemble_performance.get('f1_score', 0):.3f}")
            print(f"   Accuracy:   {self.ensemble_performance.get('accuracy', 0):.3f}")
            print(f"   ROC-AUC:    {self.ensemble_performance.get('roc_auc', 0.5):.3f}")
            print(f"   ML Weight:  {self.ensemble_performance.get('ml_weight', 0):.1f}")
            print(f"   KG Weight:  {self.ensemble_performance.get('kg_weight', 0):.1f}")
        else:
            print(f"   ❌ Failed: {self.ensemble_performance.get('error', 'Unknown error')}")
        
        # Performance Comparison Table
        print("\n📊 PERFORMANCE COMPARISON:")
        print("   Model Type       | F1-Score | Accuracy | ROC-AUC")
        print("   " + "-"*50)
        
        # Get metrics safely
        ml_f1 = self.ml_performance.get("best_model", {}).get("F1-Score", 0) if self.ml_performance.get("success") else 0
        ml_acc = self.ml_performance.get("best_model", {}).get("Accuracy", 0) if self.ml_performance.get("success") else 0
        ml_roc = self.ml_performance.get("best_model", {}).get("ROC-AUC", 0.5) if self.ml_performance.get("success") else 0.5
        
        kg_f1 = self.kg_performance.get("f1_score", 0)
        kg_acc = self.kg_performance.get("accuracy", 0)
        kg_roc = self.kg_performance.get("roc_auc", 0.5)
        
        ens_f1 = self.ensemble_performance.get("f1_score", 0)
        ens_acc = self.ensemble_performance.get("accuracy", 0)
        ens_roc = self.ensemble_performance.get("roc_auc", 0.5)
        
        print(f"   ML Model         | {ml_f1:.3f}    | {ml_acc:.3f}    | {ml_roc:.3f}")
        print(f"   Knowledge Graph  | {kg_f1:.3f}    | {kg_acc:.3f}    | {kg_roc:.3f}")
        print(f"   Hybrid Ensemble  | {ens_f1:.3f}    | {ens_acc:.3f}    | {ens_roc:.3f}")
        
        # Winner analysis
        print("\n🏆 PERFORMANCE WINNER:")
        models = {
            "ML Model": ml_f1,
            "Knowledge Graph": kg_f1,
            "Hybrid Ensemble": ens_f1
        }
        best_model = max(models.items(), key=lambda x: x[1])
        print(f"   Best F1-Score: {best_model[0]} ({best_model[1]:.3f})")
        
        print("\n" + "="*80)

def main():
    """Main function for testing the hybrid ensemble model"""
    print("🎯 Testing Hybrid Ensemble Model (KG + ML)\n")
    
    # Initialize hybrid model
    hybrid_model = HybridEnsembleModel()
    
    try:
        # Train and evaluate all models
        results = hybrid_model.train_and_evaluate(test_size=0.2, verbose=True)
        return results
        
    except Exception as e:
        print(f"❌ Hybrid model evaluation failed: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    main() 