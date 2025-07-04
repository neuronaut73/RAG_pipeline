# Starting Prompt: Multi-Domain RAG System for Financial Trading

You are working on a modular, multi-domain Retrieval-Augmented Generation (RAG) system for a financial trading platform.

**System Architecture Highlights:**
- Each data domain (Trading, News/Sentiment, Fundamentals, Performance, Company Info, User Posts) has its own RAG agent, which stores embeddings and metadata in LanceDB.
- There is a central Knowledge Orchestrator agent that coordinates cross-domain queries, traversals, and explanations.
- Numeric tables (e.g., fundamentals, performance) are used for both machine learning (ML) prediction and as nodes in the Knowledge Graph for explainability.
- Labels (from `labels.csv`) represent outperformance vs. benchmark for 10 consecutive days after a setup, and are essential for supervised tasks and explanations.
- All tables are linked via `setup_id`, `ticker`, and `date` as available.

**Project Guidance:**
- **PROJECT_VISION.md** in the root folder describes the full system context, architecture, and goals. Always refer to this file before starting new modules.
- The `/RAG/` folder contains one markdown instruction file per major development step (table mapping, embedding, agent creation, orchestrator, KG, ML, ensembling).
- All data (DuckDB and CSVs) are in `/data/`.

**Your tasks will proceed in well-defined steps. For each step:**
- Open the corresponding `.md` file in `/RAG/` (e.g., `domain_table_map.md`, `embed_news.md`).
- Copy the instruction into Cursor and execute the coding task.
- At each step, preserve metadata, relationships (`setup_id`, `ticker`, `date`), and label integrity.
- After each step, test and review outputs before proceeding.

---

**First Step:**  
Open and execute the instructions from `/RAG/domain_table_map.md`  
→ This will map each CSV/table to its domain, specify content/metadata/label fields, and document cross-table relationships.

---

_If in doubt or if requirements seem unclear, always refer back to `PROJECT_VISION.md` and ask for clarification!_
