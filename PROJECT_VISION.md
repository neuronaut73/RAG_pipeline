# Project Vision: Multi-Domain RAG System for Financial Trading

## Overview

- **Goal:** Build a modular, agentic RAG pipeline for trading, news, user sentiment, fundamentals, performance, and company info domains.
- **Approach:** Each domain has its own RAG agent (using LanceDB).
- **Orchestration:** A Knowledge Orchestrator agent enables cross-domain retrieval, traversal, and explanation.
- **ML & KG:** Numeric tables (fundamentals, performance) are used both as ML features and as context in the Knowledge Graph.
- **Labeling:** Labels (from `labels.csv`) are critical for both training and explainability.
- **Relationships:** All tables are linked by `setup_id`, `ticker`, and `date` as available.

## Architecture Highlights

- **Domain-specific RAG agents** for each data source/table.
- **LanceDB** for embeddings and semantic retrieval.
- **Knowledge Orchestrator agent** for cross-domain synthesis.
- **Knowledge Graph** for multi-modal explainability.
- **ML pipeline** for numeric-driven prediction.
- **Meta-Model/Ensembling** for robust signals and explainability.

## Implementation Steps

1. Map tables and fields to RAG domains (see `/RAG/domain_table_map.md`).
2. Chunk/embed each domain’s content, store in LanceDB (see `/RAG/embed_*.md`).
3. Build domain RAG agents (`/RAG/agent_*.md`).
4. Implement the Knowledge Orchestrator agent (`/RAG/orchestrator_agent.md`).
5. Construct and use the Knowledge Graph (`/RAG/kg_builder.md`, `/RAG/kg_explainer.md`).
6. Train ML model(s) (`/RAG/ml_model.md`).
7. Combine RAG and ML outputs in a meta-model (`/RAG/ensemble_model.md`).

---

**Refer to this file at the start of any major code step or when onboarding new contributors.**
