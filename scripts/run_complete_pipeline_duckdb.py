#!/usr/bin/env python3
"""
run_complete_pipeline_duckdb.py - Complete RAG Pipeline Runner

Executes the complete DuckDB-based RAG pipeline with all embedding steps:
1. Fundamentals embedding
2. News embedding (RNS + enhanced news)
3. User posts embedding

All data is sourced from the DuckDB database and filtered by confirmed setups.
"""

import sys
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import all embedding pipelines
from setup_validator_duckdb import SetupValidatorDuckDB
from embed_fundamentals_duckdb import FundamentalsEmbedderDuckDB
from embed_userposts_duckdb import UserPostsEmbedderDuckDB

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'pipeline_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CompleteRAGPipelineRunner:
    """
    Complete RAG Pipeline Runner for DuckDB-based data processing
    
    Orchestrates the execution of all embedding pipelines with proper error handling,
    logging, and progress tracking.
    """
    
    def __init__(self, db_path: str = "../data/sentiment_system.duckdb", lancedb_dir: str = "../lancedb_store"):
        """
        Initialize the pipeline runner
        
        Args:
            db_path: Path to DuckDB database file
            lancedb_dir: Directory for LanceDB storage
        """
        self.db_path = Path(db_path)
        self.lancedb_dir = Path(lancedb_dir)
        
        # Pipeline execution tracking
        self.pipeline_results = {}
        self.start_time = None
        self.end_time = None
        
        # Validate database exists
        if not self.db_path.exists():
            raise FileNotFoundError(f"DuckDB file not found at {self.db_path}")
    
    def validate_environment(self) -> bool:
        """
        Validate that the environment is ready for pipeline execution
        
        Returns:
            True if environment is valid, False otherwise
        """
        logger.info("Validating environment...")
        
        try:
            # Test DuckDB connection and data availability
            validator = SetupValidatorDuckDB(db_path=str(self.db_path))
            stats = validator.get_summary_stats()
            
            logger.info(f"✓ DuckDB connection successful")
            logger.info(f"✓ Found {stats['total_confirmed_setups']} confirmed setups")
            logger.info(f"✓ Data counts: {stats['data_counts']}")
            
            # Check for required data
            if stats['total_confirmed_setups'] == 0:
                logger.error("❌ No confirmed setups found in database")
                return False
            
            # Ensure LanceDB directory exists
            self.lancedb_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ LanceDB directory ready: {self.lancedb_dir}")
            
            validator.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Environment validation failed: {e}")
            return False
    
    def run_fundamentals_pipeline(self) -> Dict[str, Any]:
        """
        Execute the fundamentals embedding pipeline
        
        Returns:
            Dictionary with execution results
        """
        logger.info("="*60)
        logger.info("STARTING FUNDAMENTALS EMBEDDING PIPELINE")
        logger.info("="*60)
        
        try:
            start_time = datetime.now()
            
            # Initialize and run fundamentals embedder
            embedder = FundamentalsEmbedderDuckDB(
                db_path=str(self.db_path),
                lancedb_dir=str(self.lancedb_dir)
            )
            
            embedder.run_pipeline()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = {
                'status': 'success',
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'error': None
            }
            
            logger.info(f"✅ Fundamentals pipeline completed successfully in {duration:.1f} seconds")
            return result
            
        except Exception as e:
            error_msg = f"Fundamentals pipeline failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return {
                'status': 'failed',
                'start_time': start_time.isoformat() if 'start_time' in locals() else None,
                'end_time': datetime.now().isoformat(),
                'duration_seconds': 0,
                'error': error_msg
            }
    
    def run_news_pipeline(self) -> Dict[str, Any]:
        """
        Execute the news embedding pipeline
        
        Returns:
            Dictionary with execution results
        """
        logger.info("="*60)
        logger.info("STARTING NEWS EMBEDDING PIPELINE")
        logger.info("="*60)
        
        try:
            start_time = datetime.now()
            
            # Try importing the news pipeline
            try:
                from embed_news_duckdb import NewsEmbeddingPipelineDuckDB
                
                # Initialize and run news embedder
                embedder = NewsEmbeddingPipelineDuckDB(
                    db_path=str(self.db_path),
                    lancedb_dir=str(self.lancedb_dir)
                )
                
                embedder.run_pipeline()
                
            except ImportError:
                logger.warning("News embedding pipeline not available - skipping")
                return {
                    'status': 'skipped',
                    'start_time': start_time.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'duration_seconds': 0,
                    'error': 'Module not found'
                }
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = {
                'status': 'success',
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'error': None
            }
            
            logger.info(f"✅ News pipeline completed successfully in {duration:.1f} seconds")
            return result
            
        except Exception as e:
            error_msg = f"News pipeline failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return {
                'status': 'failed',
                'start_time': start_time.isoformat() if 'start_time' in locals() else None,
                'end_time': datetime.now().isoformat(),
                'duration_seconds': 0,
                'error': error_msg
            }
    
    def run_userposts_pipeline(self) -> Dict[str, Any]:
        """
        Execute the user posts embedding pipeline
        
        Returns:
            Dictionary with execution results
        """
        logger.info("="*60)
        logger.info("STARTING USER POSTS EMBEDDING PIPELINE")
        logger.info("="*60)
        
        try:
            start_time = datetime.now()
            
            # Initialize and run user posts embedder
            embedder = UserPostsEmbedderDuckDB(
                db_path=str(self.db_path),
                lancedb_dir=str(self.lancedb_dir)
            )
            
            embedder.run_pipeline()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = {
                'status': 'success',
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'error': None
            }
            
            logger.info(f"✅ User posts pipeline completed successfully in {duration:.1f} seconds")
            return result
            
        except Exception as e:
            error_msg = f"User posts pipeline failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return {
                'status': 'failed',
                'start_time': start_time.isoformat() if 'start_time' in locals() else None,
                'end_time': datetime.now().isoformat(),
                'duration_seconds': 0,
                'error': error_msg
            }
    
    def generate_pipeline_summary(self) -> None:
        """Generate and display comprehensive pipeline summary"""
        logger.info("\n" + "="*80)
        logger.info("COMPLETE RAG PIPELINE EXECUTION SUMMARY")
        logger.info("="*80)
        
        total_duration = (self.end_time - self.start_time).total_seconds()
        logger.info(f"Total execution time: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
        
        # Pipeline results summary
        successful_pipelines = []
        failed_pipelines = []
        skipped_pipelines = []
        
        for pipeline_name, result in self.pipeline_results.items():
            status = result['status']
            duration = result['duration_seconds']
            
            logger.info(f"\n📊 {pipeline_name.upper()} PIPELINE:")
            logger.info(f"   Status: {status}")
            logger.info(f"   Duration: {duration:.1f} seconds")
            
            if status == 'success':
                successful_pipelines.append(pipeline_name)
                logger.info("   ✅ Completed successfully")
            elif status == 'failed':
                failed_pipelines.append(pipeline_name)
                logger.info(f"   ❌ Failed: {result['error']}")
            elif status == 'skipped':
                skipped_pipelines.append(pipeline_name)
                logger.info(f"   ⏭️  Skipped: {result['error']}")
        
        # Overall summary
        logger.info(f"\n🎯 OVERALL RESULTS:")
        logger.info(f"   ✅ Successful: {len(successful_pipelines)}")
        logger.info(f"   ❌ Failed: {len(failed_pipelines)}")
        logger.info(f"   ⏭️  Skipped: {len(skipped_pipelines)}")
        
        if failed_pipelines:
            logger.info(f"\n⚠️  ATTENTION REQUIRED:")
            for pipeline in failed_pipelines:
                logger.info(f"   - {pipeline} pipeline needs investigation")
        
        # Data summary
        try:
            validator = SetupValidatorDuckDB(db_path=str(self.db_path))
            stats = validator.get_summary_stats()
            
            logger.info(f"\n📈 DATA PROCESSING SUMMARY:")
            logger.info(f"   Confirmed setups processed: {stats['total_confirmed_setups']}")
            logger.info(f"   Unique tickers: {stats['unique_tickers']}")
            logger.info(f"   LanceDB tables created in: {self.lancedb_dir}")
            
            validator.close()
            
        except Exception as e:
            logger.warning(f"Could not generate data summary: {e}")
        
        logger.info("="*80)
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Execute the complete RAG pipeline
        
        Returns:
            Dictionary with overall execution results
        """
        logger.info("🚀 STARTING COMPLETE RAG PIPELINE EXECUTION")
        logger.info(f"Database: {self.db_path}")
        logger.info(f"LanceDB: {self.lancedb_dir}")
        
        self.start_time = datetime.now()
        
        # Validate environment
        if not self.validate_environment():
            return {
                'status': 'failed',
                'error': 'Environment validation failed',
                'pipeline_results': {}
            }
        
        # Execute pipelines in sequence
        pipelines = [
            ('fundamentals', self.run_fundamentals_pipeline),
            ('news', self.run_news_pipeline),
            ('userposts', self.run_userposts_pipeline)
        ]
        
        for pipeline_name, pipeline_func in pipelines:
            logger.info(f"\n⏳ Starting {pipeline_name} pipeline...")
            self.pipeline_results[pipeline_name] = pipeline_func()
        
        self.end_time = datetime.now()
        
        # Generate summary
        self.generate_pipeline_summary()
        
        # Determine overall status
        failed_count = sum(1 for result in self.pipeline_results.values() if result['status'] == 'failed')
        overall_status = 'failed' if failed_count > 0 else 'success'
        
        return {
            'status': overall_status,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'total_duration_seconds': (self.end_time - self.start_time).total_seconds(),
            'pipeline_results': self.pipeline_results,
            'failed_pipelines': failed_count
        }


def main():
    """Main execution function"""
    try:
        # Initialize and run pipeline
        runner = CompleteRAGPipelineRunner()
        results = runner.run_complete_pipeline()
        
        # Exit with appropriate code
        if results['status'] == 'success':
            logger.info("🎉 COMPLETE RAG PIPELINE FINISHED SUCCESSFULLY!")
            sys.exit(0)
        else:
            logger.error("💥 COMPLETE RAG PIPELINE FINISHED WITH ERRORS!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Pipeline execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Pipeline execution failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main() 