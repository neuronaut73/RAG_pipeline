<!-- embed_news.md -->

# Step 2: Embedding & LanceDB Pipeline – News Domain

## Reminder: Overarching System Goal
See README / Project Vision above.

### Task for this step:
- For the News domain (e.g., `rns_announcements.csv`, `stock_news_enhanced.csv`):
    - Chunk and embed the main content field(s) (`text`, `headline`, `content_summary`, etc.).
    - Attach all relevant metadata fields (setup_id, ticker, date, headline/title, sentiment, etc.), including labels from `labels.csv`.
    - Insert into a LanceDB collection/table.
- Output: `embed_news.py`

---
