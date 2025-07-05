# Complete RAG System Orchestrator

## Overview

The `run_complete_rag_system.py` script is a comprehensive orchestrator that executes the entire RAG (Retrieval-Augmented Generation) pipeline from data preparation to final ML classification scoring. This implements the complete system architecture shown in the flowchart diagram.

## System Architecture

The script executes the following phases in sequence:

### Phase 1: Data Preparation (Embeddings)
- Checks if embeddings already exist in LanceDB
- Creates embeddings from DuckDB data if needed
- Processes fundamentals, news, and user posts data
- Generates vector embeddings for semantic search

### Phase 2: RAG Orchestration
- Runs the LangGraph-based orchestration system
- Tests query routing and intent detection
- Demonstrates multi-agent coordination
- Falls back to alternative orchestration if needed

### Phase 3: ML Model Scoring
- Trains and evaluates Hybrid Ensemble Model
- Runs traditional ML Pipeline
- Provides classification scores and performance metrics
- Combines Knowledge Graph reasoning with ML predictions

### Phase 4: Agent Demonstrations
- Tests individual agent functionality
- Demonstrates fundamentals and user posts agents
- Validates end-to-end retrieval capabilities

## Usage

### Basic Usage

```bash
# Run the complete system
python scripts/run_complete_rag_system.py
```

### What It Does

1. **Data Validation**: Checks if embeddings exist, creates them if needed
2. **System Testing**: Tests all components of the RAG pipeline
3. **ML Training**: Trains and evaluates ML models for stock classification
4. **Report Generation**: Creates comprehensive execution report
5. **Performance Metrics**: Provides detailed performance analysis

## Expected Output

The script generates:

### Console Output
- Real-time progress updates for each phase
- Success/failure status for each component
- Performance metrics and execution times
- ML model training results

### Generated Files
- **Log File**: `complete_rag_system_YYYYMMDD_HHMMSS.log`
- **Report File**: `complete_rag_report_YYYYMMDD_HHMMSS.md`

### Sample Report Structure
```markdown
# Complete RAG System Execution Report

## System Phases Results
- ✅ **Data Preparation**: success
- ✅ **RAG Orchestration**: success  
- ✅ **ML Model Scoring**: success
- ✅ **Agent Demonstrations**: success

## ML Model Results
- ✅ **HybridEnsemble**: Successfully trained
  - Ensemble Accuracy: 0.750
- ✅ **MLPipeline**: Successfully trained
  - Best Model: XGBoost
  - F1 Score: 0.680

## Performance Summary
- **Total Execution Time**: 45.2 seconds
- **Test Queries**: 5
- **Phases Completed**: 4
```

## System Components

### Core Components Tested

1. **Embedding Systems**
   - `embed_fundamentals_duckdb.py`
   - `embed_news_duckdb.py`
   - `embed_userposts_duckdb.py`

2. **RAG Orchestration**
   - `orchestrator_langgraph.py`
   - `demo_orchestrator_comprehensive.py`

3. **ML Models**
   - `hybrid_model.py` - Hybrid KG + ML ensemble
   - `ml_model.py` - Traditional ML pipeline

4. **Domain Agents**
   - `agent_fundamentals.py`
   - `agent_news.py`
   - `agent_userposts.py`

### Data Sources

- **DuckDB Tables**: `sentiment_system.duckdb`
  - `setups` (72 confirmed setups)
  - `fundamentals` (109 records)
  - `user_posts` (1,233 records)
  - `rns_announcements` (554 records)
  - `stock_news_enhanced` (25 records)

- **LanceDB Tables**: `lancedb_store/`
  - `fundamentals_embeddings` (329 records)
  - `news_embeddings` (266 records)
  - `userposts_embeddings` (790 records)

## Error Handling

The script includes comprehensive error handling:

- **Phase Independence**: Failed phases don't stop execution
- **Fallback Mechanisms**: Alternative approaches when components fail
- **Error Logging**: Detailed error messages and stack traces
- **Graceful Degradation**: Continues with available components

## Performance Expectations

### Typical Execution Times
- **Data Preparation**: 5-20 seconds (if embeddings need creation)
- **RAG Orchestration**: 10-30 seconds
- **ML Model Training**: 15-60 seconds
- **Agent Demonstrations**: 5-15 seconds
- **Total**: 35-125 seconds

### Resource Requirements
- **Memory**: 2-4 GB RAM
- **Storage**: 1-2 GB for embeddings and models
- **CPU**: Multi-core recommended for ML training

## Configuration

### Environment Variables
```bash
# OpenAI API Key (required for LLM components)
export OPENAI_API_KEY="your-api-key-here"

# Optional: Adjust model parameters
export RAG_MODEL_NAME="gpt-4o-mini"
export RAG_EMBEDDING_MODEL="text-embedding-3-small"
```

### Customization Options

The script can be customized by modifying:

1. **Test Queries**: Edit `self.test_queries` in the class
2. **ML Parameters**: Adjust model parameters in ML classes
3. **Agent Settings**: Modify agent initialization parameters
4. **Output Format**: Customize report generation methods

## Troubleshooting

### Common Issues

1. **Missing Embeddings**
   - Solution: Script automatically creates them
   - Check: DuckDB file exists and is readable

2. **ML Model Training Fails**
   - Solution: Check data quality and feature availability
   - Verify: Sufficient training data exists

3. **Agent Initialization Errors**
   - Solution: Verify LanceDB tables exist
   - Check: Proper import paths and dependencies

4. **Memory Issues**
   - Solution: Reduce batch sizes or model complexity
   - Monitor: System memory usage during execution

### Debug Mode

For debugging, modify the logging level:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Integration with Existing Pipeline

This script complements the existing pipeline:

- **Builds on**: `run_complete_pipeline_duckdb.py`
- **Extends**: Individual agent and model scripts
- **Integrates**: All system components into single workflow
- **Validates**: End-to-end system functionality

## Future Enhancements

Potential improvements:

1. **Real-time Monitoring**: Add system health monitoring
2. **API Integration**: Expose as REST API endpoints
3. **Batch Processing**: Support for bulk query processing
4. **Model Persistence**: Save and load trained models
5. **Performance Optimization**: Caching and parallel processing

## Support

For issues or questions:
1. Check the generated log files for detailed error messages
2. Verify all dependencies are installed
3. Ensure data files are accessible and properly formatted
4. Review individual component documentation for specific issues 