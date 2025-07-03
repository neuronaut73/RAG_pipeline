<!-- orchestrator_agent.md -->

# Step 4: Knowledge Orchestrator Agent

## Reminder: Overarching System Goal
See README / Project Vision above.

### Task for this step:
- Implement orchestration logic that:
    - Accepts user/system queries.
    - Routes subqueries to all relevant domain agents.
    - Aggregates and cross-reranks returned results (e.g. with a cross-encoder).
    - Synthesizes a final answer (summary, top snippets, or LLM answer).
- Output: `orchestrator_agent.py`

---
