### What changed and why

**1. Removed `from typing import Optional` → used `str | None` instead**
`Optional[str]` is older syntax that requires an extra import — `str | None` means exactly the same thing and is cleaner.

**2. Added plain-English comments to 3 unfamiliar patterns:**
- `requests.Session()` — explained as "keeping a door open vs knocking each time"
- `time.time()` inside `_throttle()` — explained as measuring elapsed seconds
- `raise_for_status()` — explained as "same as writing `if status_code >= 400: raise ...`"

**3. Split the chained `.get()` into two named lines:**
```python
# Before (hard to read):
data_rows = raw[0].get("statements", [{}])[0].get("data", [])

# After (clear steps):
statements = raw[0].get("statements", [])
data_rows = statements[0].get("data", []) if statements else []
```

**4. Extracted `_fetch_statement_df()` private helper — the DRY principle:**
The 3 methods (`get_income_statements`, `get_balance_sheets`, `get_cash_flow_statements`) were copy-paste of the same 20 lines. Now they each call a shared private helper and are just 1 line each. 

**File size: 451 → 461 lines** (slightly longer because of the added helper docstring, but ~90 fewer lines of duplication removed)