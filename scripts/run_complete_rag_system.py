#!/usr/bin/env python3
"""
run_complete_rag_system.py - Complete RAG System Orchestrator

Executes the entire RAG pipeline from data preparation to final ML classification:
1. Data Preparation: Create embeddings (if needed)
2. RAG Orchestration: LangGraph-based agent system  
3. ML Scoring: Hybrid model classification
4. Response Synthesis: Final output generation
"""

import sys
import os
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'complete_rag_system_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CompleteRAGSystemOrchestrator:
    """Complete RAG System Orchestrator implementing the full architecture"""
    
    def __init__(self):
        self.start_time = None
        self.test_queries = [
            "What is the financial health of BGO with strong fundamentals?",
            "Show me negative sentiment analysis for KZG stock", 
            "Analyze recent news impact on HWDN performance",
            "Compare fundamentals and sentiment for AML stock",
            "What are the key risks for DGE based on all data sources?"
        ]
    
    def check_embeddings_ready(self) -> bool:
        """Check if embeddings are available"""
        try:
            import lancedb
            db = lancedb.connect('lancedb_store')
            tables = db.table_names()
            required = ['fundamentals_embeddings', 'userposts_embeddings']
            available = [t for t in required if t in tables]
            
            if len(available) == len(required):
                logger.info(f"✅ All embeddings available: {available}")
                return True
            else:
                logger.warning(f"❌ Missing embeddings: {[t for t in required if t not in tables]}")
                return False
        except Exception as e:
            logger.error(f"Error checking embeddings: {e}")
            return False
    
    def run_embedding_pipeline(self) -> Dict[str, Any]:
        """Run embedding creation if needed"""
        logger.info("🔧 PHASE 1: DATA PREPARATION")
        logger.info("="*50)
        
        try:
            if self.check_embeddings_ready():
                return {'status': 'skipped', 'reason': 'embeddings_exist'}
            
            from run_complete_pipeline_duckdb import CompleteRAGPipelineRunner
            pipeline_runner = CompleteRAGPipelineRunner()
            return pipeline_runner.run_complete_pipeline()
            
        except Exception as e:
            logger.error(f"❌ Embedding pipeline error: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def run_demo_orchestration(self) -> Dict[str, Any]:
        """Run the demo orchestrator to test the complete system"""
        logger.info("🎭 PHASE 2: RAG ORCHESTRATION")
        logger.info("="*50)
        
        try:
            # Try to import and run the demo orchestrator
            from demo_orchestrator_comprehensive import main as demo_main
            
            logger.info("Running comprehensive demo orchestrator...")
            demo_result = demo_main()
            
            logger.info("✅ Demo orchestration completed")
            return {'status': 'success', 'demo_results': demo_result}
            
        except Exception as e:
            logger.error(f"❌ Demo orchestration failed: {e}")
            # Continue with alternative orchestration
            return self._run_alternative_orchestration()
    
    def _run_alternative_orchestration(self) -> Dict[str, Any]:
        """Run alternative orchestration using LangGraph directly"""
        try:
            from orchestrator_langgraph import LangGraphOrchestrator
            
            logger.info("Trying alternative LangGraph orchestration...")
            orchestrator = LangGraphOrchestrator()
            
            # Test with sample queries
            test_results = []
            sample_queries = [
                "BGO financial analysis",
                "AML sentiment analysis",
                "HWDN recent news"
            ]
            
            for query in sample_queries:
                try:
                    result = orchestrator.query(query, max_results=3)
                    test_results.append({'query': query, 'success': True, 'result_count': result.get('total_results', 0)})
                    logger.info(f"  Query '{query}': {result.get('total_results', 0)} results")
                except Exception as e:
                    test_results.append({'query': query, 'success': False, 'error': str(e)})
                    logger.warning(f"  Query '{query}' failed: {e}")
            
            logger.info("✅ Alternative orchestration completed")
            return {'status': 'success', 'orchestration_results': test_results}
            
        except Exception as e:
            logger.error(f"❌ Alternative orchestration also failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def run_ml_scoring(self) -> Dict[str, Any]:
        """Run ML model scoring"""
        logger.info("🧠 PHASE 3: ML MODEL SCORING")
        logger.info("="*50)
        
        ml_results = {}
        
        # Try Hybrid Model
        try:
            from hybrid_model import HybridEnsembleModel
            
            logger.info("Initializing Hybrid Ensemble Model...")
            hybrid_model = HybridEnsembleModel()
            
            logger.info("Training and evaluating hybrid model...")
            hybrid_performance = hybrid_model.train_and_evaluate(test_size=0.2, verbose=False)
            
            ml_results['hybrid_model'] = {
                'status': 'success',
                'performance': hybrid_performance,
                'model_type': 'HybridEnsemble'
            }
            
            logger.info("✅ Hybrid model training completed")
            
        except Exception as e:
            logger.error(f"❌ Hybrid model failed: {e}")
            ml_results['hybrid_model'] = {'status': 'failed', 'error': str(e)}
        
        # Try ML Pipeline
        try:
            from ml_model import MLPipeline
            
            logger.info("Initializing ML Pipeline...")
            ml_pipeline = MLPipeline()
            
            logger.info("Running ML pipeline...")
            ml_performance = ml_pipeline.run_pipeline(test_size=0.2)
            
            ml_results['ml_pipeline'] = {
                'status': 'success',
                'performance': ml_performance,
                'model_type': 'MLPipeline'
            }
            
            logger.info("✅ ML pipeline completed")
            
        except Exception as e:
            logger.error(f"❌ ML pipeline failed: {e}")
            ml_results['ml_pipeline'] = {'status': 'failed', 'error': str(e)}
        
        # Summary
        successful_models = len([r for r in ml_results.values() if r['status'] == 'success'])
        logger.info(f"ML scoring completed: {successful_models}/{len(ml_results)} models successful")
        
        return {'status': 'success', 'ml_results': ml_results}
    
    def run_agent_demos(self) -> Dict[str, Any]:
        """Run individual agent demos"""
        logger.info("🤖 PHASE 4: AGENT DEMONSTRATIONS")
        logger.info("="*50)
        
        agent_results = {}
        
        # Test each agent demo
        demos = [
            ('fundamentals', 'demo_fundamentals_agent'),
            ('userposts', 'demo_userposts_agent')
        ]
        
        for agent_name, demo_module in demos:
            try:
                logger.info(f"Running {agent_name} agent demo...")
                module = __import__(demo_module)
                result = module.main()
                agent_results[agent_name] = {'status': 'success', 'result': result}
                logger.info(f"  ✅ {agent_name} agent demo completed")
            except Exception as e:
                logger.error(f"  ❌ {agent_name} agent demo failed: {e}")
                agent_results[agent_name] = {'status': 'failed', 'error': str(e)}
        
        success_count = len([r for r in agent_results.values() if r['status'] == 'success'])
        logger.info(f"Agent demos completed: {success_count}/{len(demos)} successful")
        
        return {'status': 'success', 'agent_results': agent_results}
    
    def generate_final_report(self, all_results: Dict[str, Any]) -> str:
        """Generate comprehensive system report"""
        logger.info("📋 GENERATING FINAL REPORT")
        logger.info("="*50)
        
        report_lines = [
            "# Complete RAG System Execution Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## System Phases Results"
        ]
        
        # Phase results
        phases = [
            ('Data Preparation', all_results.get('embedding_phase', {})),
            ('RAG Orchestration', all_results.get('orchestration_phase', {})),
            ('ML Model Scoring', all_results.get('ml_phase', {})),
            ('Agent Demonstrations', all_results.get('agent_phase', {}))
        ]
        
        for phase_name, phase_result in phases:
            status = phase_result.get('status', 'unknown')
            emoji = "✅" if status == 'success' else "⏭️" if status == 'skipped' else "❌"
            report_lines.append(f"- {emoji} **{phase_name}**: {status}")
            
            if status == 'failed':
                error = phase_result.get('error', 'Unknown error')
                report_lines.append(f"  - Error: {error}")
        
        # ML Scoring Results
        ml_results = all_results.get('ml_phase', {}).get('ml_results', {})
        if ml_results:
            report_lines.extend([
                "",
                "## ML Model Results",
                ""
            ])
            
            for model_name, model_result in ml_results.items():
                status = model_result.get('status', 'unknown')
                emoji = "✅" if status == 'success' else "❌"
                model_type = model_result.get('model_type', 'Unknown')
                
                if status == 'success':
                    report_lines.append(f"- {emoji} **{model_type}**: Successfully trained")
                    
                    # Show performance metrics if available
                    performance = model_result.get('performance', {})
                    if performance:
                        if 'ensemble_performance' in performance:
                            ensemble_perf = performance['ensemble_performance']
                            report_lines.append(f"  - Ensemble Accuracy: {ensemble_perf.get('accuracy', 0):.3f}")
                        elif 'best_model' in performance:
                            best_model = performance['best_model']
                            report_lines.append(f"  - Best Model: {best_model.get('name', 'Unknown')}")
                            report_lines.append(f"  - F1 Score: {best_model.get('f1_score', 0):.3f}")
                else:
                    error = model_result.get('error', 'Unknown error')
                    report_lines.append(f"- {emoji} **{model_type}**: Failed - {error}")
        
        # Performance Summary
        if self.start_time:
            total_time = (datetime.now() - self.start_time).total_seconds()
            report_lines.extend([
                "",
                "## Performance Summary",
                f"- **Total Execution Time**: {total_time:.1f} seconds",
                f"- **Test Queries**: {len(self.test_queries)}",
                f"- **Phases Completed**: {len([p for _, p in phases if p.get('status') == 'success'])}",
                ""
            ])
        
        report_content = "\n".join(report_lines)
        
        # Save report
        report_file = f"../complete_rag_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"📋 Report saved to: {report_file}")
        return report_content
    
    def run_complete_system(self) -> Dict[str, Any]:
        """Execute the complete RAG system pipeline"""
        self.start_time = datetime.now()
        
        logger.info("🚀 STARTING COMPLETE RAG SYSTEM")
        logger.info("="*60)
        logger.info(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*60)
        
        all_results = {}
        
        try:
            # Phase 1: Data Preparation (Embeddings)
            all_results['embedding_phase'] = self.run_embedding_pipeline()
            
            # Phase 2: RAG Orchestration Demo
            all_results['orchestration_phase'] = self.run_demo_orchestration()
            
            # Phase 3: ML Model Scoring
            all_results['ml_phase'] = self.run_ml_scoring()
            
            # Phase 4: Individual Agent Demos
            all_results['agent_phase'] = self.run_agent_demos()
            
            # Generate Final Report
            report = self.generate_final_report(all_results)
            
            end_time = datetime.now()
            total_duration = (end_time - self.start_time).total_seconds()
            
            logger.info("="*60)
            logger.info("🎉 COMPLETE RAG SYSTEM FINISHED")
            logger.info(f"Total Duration: {total_duration:.1f} seconds")
            logger.info("="*60)
            
            return {
                'status': 'success',
                'duration_seconds': total_duration,
                'phases': all_results,
                'report': report
            }
            
        except Exception as e:
            logger.error(f"💥 COMPLETE RAG SYSTEM FAILED: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return {
                'status': 'failed',
                'error': str(e),
                'phases': all_results
            }


def main():
    """Main execution function"""
    try:
        orchestrator = CompleteRAGSystemOrchestrator()
        results = orchestrator.run_complete_system()
        
        if results['status'] == 'success':
            logger.info("🎉 COMPLETE RAG SYSTEM FINISHED SUCCESSFULLY!")
            sys.exit(0)
        else:
            logger.error("💥 COMPLETE RAG SYSTEM FINISHED WITH ERRORS!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 