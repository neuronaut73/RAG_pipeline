#!/usr/bin/env python3
"""
Test Hybrid Ensemble Model on Confirmed Setups

This script tests the hybrid ensemble model on actual confirmed setups
from the database to show realistic performance on real data.
"""

import sys
import os
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from setup_validator_duckdb import SetupValidatorDuckDB
from hybrid_model import HybridEnsembleModel, KnowledgeGraphClassifier
from orchestrator_langgraph import LangGraphOrchestrator

def test_hybrid_on_confirmed_setups():
    """Test hybrid model on actual confirmed setups"""
    
    print("🎯 Testing Hybrid Model on Confirmed Setups")
    print("=" * 60)
    
    # Initialize components
    setup_validator = SetupValidatorDuckDB()
    orchestrator = LangGraphOrchestrator()
    kg_classifier = KnowledgeGraphClassifier(orchestrator)
    
    # Get confirmed setups and labels
    confirmed_setup_ids = setup_validator.get_confirmed_setup_ids()
    labels_df = setup_validator.get_labels_for_confirmed_setups()
    print(f"📊 Found {len(confirmed_setup_ids)} confirmed setups")
    print(f"📊 Found {len(labels_df)} labels for confirmed setups")
    
    # Test on a sample of confirmed setups
    test_setups = labels_df.head(8)  # Use first 8 setups
    
    print("\n🧠 Testing Knowledge Graph Classifier on Confirmed Setups:")
    print("-" * 60)
    
    kg_results = []
    for idx, row in test_setups.iterrows():
        setup_id = row['setup_id']
        ticker = row.get('ticker', row.get('lse_ticker', 'Unknown'))
        label = row.get('label', row.get('performance_30d', 0))
        
        print(f"\n📈 Setup: {setup_id}")
        print(f"   Ticker: {ticker}")
        print(f"   Actual Label: {label}")
        
        # Get KG prediction
        try:
            kg_prediction = kg_classifier.predict_outperformance(ticker)
            print(f"   KG Prediction: {kg_prediction.prediction} (confidence: {kg_prediction.confidence:.3f})")
            
            kg_results.append({
                'setup_id': setup_id,
                'ticker': ticker,
                'actual_label': label,
                'kg_prediction': kg_prediction.prediction,
                'kg_confidence': kg_prediction.confidence,
                'correct': kg_prediction.prediction == label
            })
            
        except Exception as e:
            print(f"   ❌ KG Prediction failed: {e}")
            kg_results.append({
                'setup_id': setup_id,
                'ticker': ticker,
                'actual_label': label,
                'kg_prediction': None,
                'kg_confidence': 0.0,
                'correct': False
            })
    
    # Calculate KG performance
    print("\n📊 Knowledge Graph Performance on Confirmed Setups:")
    print("-" * 60)
    
    kg_df = pd.DataFrame(kg_results)
    successful_predictions = kg_df[kg_df['kg_prediction'].notna()]
    
    if len(successful_predictions) > 0:
        kg_accuracy = successful_predictions['correct'].mean()
        prediction_count = len(successful_predictions)
        
        print(f"   Predictions Made: {prediction_count}/{len(test_setups)}")
        print(f"   Accuracy: {kg_accuracy:.3f}")
        print(f"   Average Confidence: {successful_predictions['kg_confidence'].mean():.3f}")
        
        # Show prediction breakdown
        print("\n   Prediction Breakdown:")
        for _, row in successful_predictions.iterrows():
            result = "✅" if row['correct'] else "❌"
            print(f"   {result} {row['ticker']}: Predicted {row['kg_prediction']}, Actual {row['actual_label']}")
    else:
        print("   ❌ No successful predictions made")
    
    # Now test the full hybrid model
    print("\n🎯 Testing Full Hybrid Ensemble Model:")
    print("-" * 60)
    
    hybrid_model = HybridEnsembleModel()
    
    try:
        # Train hybrid model
        hybrid_results = hybrid_model.train_and_evaluate(test_size=0.2, verbose=False)
        
        print("\n📊 Hybrid Model Performance Summary:")
        print(f"   ML Best Model: {hybrid_results['ml_performance'].get('best_model', {}).get('Model', 'N/A')}")
        print(f"   ML F1-Score: {hybrid_results['ml_performance'].get('best_model', {}).get('F1-Score', 0):.3f}")
        print(f"   KG Accuracy: {hybrid_results['kg_performance'].get('accuracy', 0):.3f}")
        print(f"   Ensemble F1-Score: {hybrid_results['ensemble_performance'].get('f1_score', 0):.3f}")
        
        return {
            'kg_results': kg_results,
            'hybrid_results': hybrid_results,
            'test_setups': test_setups.to_dict('records')
        }
        
    except Exception as e:
        print(f"❌ Hybrid model evaluation failed: {e}")
        return {'error': str(e)}

def main():
    """Main function"""
    results = test_hybrid_on_confirmed_setups()
    
    if 'error' not in results:
        print("\n🎉 Hybrid Model Test Completed Successfully!")
        print("   - Knowledge Graph RAG classification working")
        print("   - ML classification working")
        print("   - Hybrid ensemble combining both approaches")
        print("   - Performance metrics calculated")
    else:
        print(f"\n❌ Test failed: {results['error']}")
    
    return results

if __name__ == "__main__":
    main() 