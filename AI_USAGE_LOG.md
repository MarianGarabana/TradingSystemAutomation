# AI Usage Log

This file documents all use of AI tools (ChatGPT, Claude, Copilot, etc.) during this project.

**Policy:** Every non-trivial AI-assisted contribution must be logged here with the prompt, output summary, and what was kept / changed.

---

## Log Format

```
### [Date] — [Team Member] — [Tool]
**Task:** What you were trying to do
**Prompt (summary):** What you asked
**Output summary:** What the AI produced
**What we used:** Which parts we kept and why
**What we changed:** Edits or rejections and why
```

---

## Entries

### 2026-03-06 — Jorge — Claude (claude-sonnet-4-6)
**Task:** Enrich ETL pipeline with fundamental data, Market_Cap feature, and regression target
**Prompt (summary):** Asked Claude to (1) add quarterly fundamental ratios (income, balance, cashflow) to the ETL with a point-in-time merge to avoid look-ahead bias; (2) add Market_Cap as a dynamic price feature; (3) change the Target variable from binary classification (0/1) to a continuous next-day return for regression; (4) update the data dictionary in the notebook.
**Output summary:** Claude added `fetch_fundamentals()`, `_compute_fundamental_features()`, and `merge_fundamentals()` to `etl/etl.py`; updated `engineer_features()` with Market_Cap and regression Target; updated `etl_exploration.ipynb` with the enriched pipeline, feature inventory, and a full data dictionary (17 features: 12 price + 5 fundamental).
**What we used:** All changes kept — fundamental enrichment logic, Market_Cap feature, regression Target definition, and the data dictionary markdown.
**What we changed:** Verified and re-ran ETL for all 5 tickers to regenerate processed CSVs with the full 17-feature schema.

### 2026-03-03 — Marian Garabana — Claude (claude-sonnet-4-6)
**Task:** Scaffold repository structure
**Prompt (summary):** Asked Claude to generate the full folder/file skeleton for the trading system project including README, .gitignore, requirements.txt, placeholder source files, and .env.example.
**Output summary:** Claude created all files and pushed the initial commit to GitHub.
**What we used:** The full scaffold as a starting point.
**What we changed:** Will update team names, fill in real implementation, and remove placeholder comments as we build.
