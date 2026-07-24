# Four SQL/Analytics Patterns — Taught the Same Way
### GROUP BY with CASE WHEN · DISTINCT in Aggregates · Query Optimization · Pareto / 80-20 Analysis

---
---

# Chapter — GROUP BY with CASE WHEN (Conditional Aggregation)

## What is it?

Conditional aggregation is the technique of putting a `CASE WHEN` expression *inside* an aggregate function (`SUM`, `COUNT`, `AVG`, `MAX`) so that a single `GROUP BY` pass can compute multiple, differently-filtered metrics side by side — instead of running several separate queries (or several separate `WHERE`-filtered subqueries) and stitching the results together afterward.

You've actually already used this pattern repeatedly earlier in this series — the pivot step in Cohort Retention (Stage 8) and Conversion Funnel (Stage 8) both used `MAX(CASE WHEN period_number = N THEN retention_rate END)` to turn rows into columns. This chapter isolates that specific technique and teaches it as a general-purpose tool in its own right, beyond just pivoting.

---

## The intuition

Imagine you're asked: "For each store, tell me total revenue, revenue from returning customers only, and revenue from new customers only — in one table, one row per store." The naive approach is three separate queries (one per revenue type) that you then have to manually line up side by side. Conditional aggregation instead says: **for every single row, decide via `CASE WHEN` whether it "counts" toward each metric, converting rows that don't count into `0` or `NULL`, then aggregate as normal** — one pass over the data produces all three columns simultaneously.

---

## Why use this instead of the obvious alternatives?

**Why not just run three separate queries, one per condition?** Because each separate query re-scans the entire table independently — three full table scans instead of one — which is wasteful at any real data volume, and because you then have to manually join or align the three result sets by store, which is extra code and extra room for a mismatch (a store present in one result but accidentally missing from another due to a filter difference).

**Why not use three separate `WHERE`-filtered subqueries joined together?** Same problem as above — multiple passes over the data, plus a join that has to be gotten exactly right (usually a `LEFT JOIN` to avoid silently dropping a store that has zero rows for one of the conditions).

**Why not just filter with `WHERE` before grouping?** Because `WHERE` discards non-matching rows *before* grouping — you'd only ever get one metric per query, since once a row is filtered out it can never contribute to a *different* metric in the same result. Conditional aggregation deliberately keeps every row in play and decides its contribution *per metric, per row*, which is fundamentally different from filtering rows out entirely.

---

## The Sample Data

```sql
SELECT * FROM orders LIMIT 8;
```

| order_id | store_id | customer_type | order_amount |
|----------|----------|----------------|----------------|
| 1 | S1 | new | 40 |
| 2 | S1 | returning | 90 |
| 3 | S1 | returning | 60 |
| 4 | S2 | new | 25 |
| 5 | S2 | new | 35 |
| 6 | S2 | returning | 100 |
| 7 | S1 | new | 15 |
| 8 | S2 | returning | 50 |

---

## Stage 1 — The Naive Multi-Query Approach (What We're Avoiding)

```sql
-- Three separate queries — what we're trying NOT to do
SELECT store_id, SUM(order_amount) AS total_revenue FROM orders GROUP BY store_id;

SELECT store_id, SUM(order_amount) AS new_revenue FROM orders WHERE customer_type = 'new' GROUP BY store_id;

SELECT store_id, SUM(order_amount) AS returning_revenue FROM orders WHERE customer_type = 'returning' GROUP BY store_id;
```

**Why show the "bad" version first?** Because seeing the tedium and duplication makes the payoff of the next stage concrete rather than abstract — you should feel exactly what problem conditional aggregation solves before you see the solution.

---

## Stage 2 — Collapse Into One Pass With CASE WHEN Inside SUM

**Variables:**
- The `CASE WHEN` expression evaluates per-row, returning the row's `order_amount` if the condition matches, or `0`/`NULL` otherwise
- `SUM` then only adds up the rows that "passed," because the failed rows contributed `0` (or `NULL`, which `SUM` ignores)

```sql
-- Stage 2: One pass, three metrics
SELECT
    store_id,
    SUM(order_amount) AS total_revenue,
    SUM(CASE WHEN customer_type = 'new' THEN order_amount ELSE 0 END) AS new_revenue,
    SUM(CASE WHEN customer_type = 'returning' THEN order_amount ELSE 0 END) AS returning_revenue
FROM orders
GROUP BY store_id;
```

**Output:**

| store_id | total_revenue | new_revenue | returning_revenue |
|----------|-----------------|--------------|-----------------------|
| S1 | 205 | 55 | 150 |
| S2 | 210 | 60 | 150 |

**Why does `ELSE 0` matter, and what happens if you use `ELSE NULL` (or omit `ELSE` entirely, which defaults to `NULL`)?** Both work correctly with `SUM`, because `SUM` silently skips `NULL` values — `SUM` of `{40, NULL, NULL}` is `40`, same as `SUM` of `{40, 0, 0}`. But **this equivalence breaks for `AVG` and `COUNT`.** `AVG({40, NULL, NULL})` computes `40/1 = 40` (NULLs excluded from both the sum and the count of rows), while `AVG({40, 0, 0})` computes `40/3 ≈ 13.3` (zeros included in the count) — two very different, and both individually "correct," answers to two different questions ("average among rows that matched" vs. "average across all rows, treating non-matches as zero"). **This is the single most common bug in conditional aggregation** — always ask explicitly which of those two questions you actually want before choosing `ELSE 0` vs `ELSE NULL` (or no `ELSE`).

---

## Stage 3 — Conditional COUNT (a Different Pitfall)

```sql
-- Stage 3: Counting rows that match a condition, per group
SELECT
    store_id,
    COUNT(*) AS total_orders,
    COUNT(CASE WHEN customer_type = 'new' THEN 1 END) AS new_orders,
    COUNT(CASE WHEN customer_type = 'returning' THEN 1 END) AS returning_orders
FROM orders
GROUP BY store_id;
```

**Output:**

| store_id | total_orders | new_orders | returning_orders |
|----------|----------------|--------------|------------------------|
| S1 | 4 | 2 | 2 |
| S2 | 4 | 2 | 2 |

**Why `COUNT(CASE WHEN ... THEN 1 END)` and not `COUNT(CASE WHEN ... THEN 1 ELSE 0 END)`?** This is the exact inverse of the `SUM` trap above, and just as commonly gotten wrong. `COUNT(column)` counts **non-NULL** values, not literal truthy/falsy values — so `COUNT(CASE WHEN condition THEN 1 END)` (implicit `ELSE NULL`) correctly counts only matching rows, because non-matching rows contribute `NULL`, which `COUNT` skips. But `COUNT(CASE WHEN condition THEN 1 ELSE 0 END)` would count **every row**, matching or not, because `0` is a non-NULL value that `COUNT` happily counts — silently producing `total_orders` in every "conditional count" column, a bug that's easy to miss because the query runs without error and just quietly returns the wrong (and suspiciously identical) numbers everywhere.

**Rule of thumb to avoid both traps:** for `SUM`, either `ELSE` is safe as long as you're clear which semantics you want; for `COUNT`, never write `ELSE 0` — always let non-matches fall through to `NULL`.

---

## Stage 4 — Percentage-of-Total Columns (Combining Both Tricks)

```sql
-- Stage 4: Add a computed share/percentage column
SELECT
    store_id,
    SUM(order_amount) AS total_revenue,
    SUM(CASE WHEN customer_type = 'new' THEN order_amount ELSE 0 END) AS new_revenue,
    ROUND(100.0 * SUM(CASE WHEN customer_type = 'new' THEN order_amount ELSE 0 END)
        / SUM(order_amount), 1) AS new_revenue_pct
FROM orders
GROUP BY store_id;
```

**Output:**

| store_id | total_revenue | new_revenue | new_revenue_pct |
|----------|-----------------|--------------|--------------------|
| S1 | 205 | 55 | 26.8% |
| S2 | 210 | 60 | 28.6% |

**Why compute the percentage in SQL rather than in the reporting/BI layer?** Both are valid depending on your stack, but computing it in SQL means the number is defined once, consistently, next to the raw components it's derived from — reducing the risk that two different dashboards compute "new revenue %" with subtly different denominators (e.g., one excluding refunds, one not) because the logic lives in two different places instead of one.

---

## The one thing to remember

Putting `CASE WHEN` inside an aggregate function turns per-row filtering into per-row *contribution decisions*, letting one `GROUP BY` pass compute many differently-conditioned metrics side by side — but `SUM` and `COUNT` behave differently with `NULL` vs `0`, so always match your `ELSE` clause to the aggregate function you're using.

## Interview Q&A

**Q1. Why would you use conditional aggregation instead of just running multiple filtered queries?** A single pass over the data computes all metrics simultaneously, avoiding repeated table scans and avoiding the need to join or align separately-computed result sets — which also removes a common source of bugs (a group present in one result set but missing from another due to a filter mismatch).

**Q2. What's the difference between `SUM(CASE WHEN x THEN y ELSE 0 END)` and `SUM(CASE WHEN x THEN y END)`?** They're equivalent for `SUM` specifically, because `SUM` ignores `NULL`s the same way it treats `0`s as no-ops in an addition — both produce the sum of only the matching rows' values.

**Q3. Why is `COUNT(CASE WHEN x THEN 1 ELSE 0 END)` almost always a bug?** Because `COUNT` counts non-NULL values, and `0` is non-NULL — so this pattern counts every row, matching or not, silently producing the total row count in what's supposed to be a conditional count. The fix is to omit the `ELSE` (or use `ELSE NULL`) so non-matching rows are excluded from the count.

---
---

# Chapter — DISTINCT in Aggregates

## What is it?

`DISTINCT` inside an aggregate function — most commonly `COUNT(DISTINCT column)`, but also valid with `SUM(DISTINCT ...)` and `AVG(DISTINCT ...)` — tells the aggregate to first deduplicate the values it's operating on *before* aggregating, rather than operating on every row as-is.

You've used this constantly throughout this series without pausing to interrogate it — `COUNT(DISTINCT user_id)` appeared in Cohort Retention, Conversion Funnel, Sessionization, and Gaps and Islands. This chapter is the pause: understanding exactly what it does, why it's usually necessary, and where it quietly goes wrong.

---

## The intuition

Imagine a guestbook at an event where people can sign in multiple times throughout the day (once at breakfast, once at lunch). "How many *rows* are in the guestbook" answers "how many total sign-ins happened." "How many *distinct names* are in the guestbook" answers "how many different people came" — a completely different, and usually more useful, question. `COUNT(DISTINCT ...)` is how you ask the second question in SQL.

---

## Why does this matter — why not just use plain COUNT everywhere?

**Why not just `COUNT(*)` or `COUNT(user_id)`?** Because in almost every event-log-style table (exactly the shape used throughout this series), a single user contributes *many* rows — one per action, one per day, one per order. `COUNT(*)` answers "how many events happened," which is a completely different number from "how many distinct users were involved," and conflating the two is one of the most common analytics mistakes: reporting "10,000 users were active" when the true number is "10,000 events from 800 distinct users."

---

## The Sample Data

```sql
SELECT * FROM page_views LIMIT 8;
```

| user_id | page_view_date | page |
|---------|-------------------|--------|
| 501 | 2024-08-01 | home |
| 501 | 2024-08-01 | product |
| 501 | 2024-08-02 | home |
| 502 | 2024-08-01 | home |
| 502 | 2024-08-01 | cart |
| 503 | 2024-08-02 | home |

---

## Stage 1 — COUNT(*) vs. COUNT(DISTINCT ...): See the Gap Directly

```sql
-- Stage 1: The three counts side by side
SELECT
    COUNT(*) AS total_page_views,
    COUNT(DISTINCT user_id) AS distinct_users,
    COUNT(DISTINCT page_view_date) AS distinct_days
FROM page_views;
```

**Output:**

| total_page_views | distinct_users | distinct_days |
|---------------------|-------------------|-------------------|
| 6 | 3 | 2 |

**Why report all three side by side rather than picking one?** Each answers a genuinely different business question ("how much activity happened" vs. "how many people were involved" vs. "how many days had any activity"), and a good analyst reflexively surfaces all three when asked a vague question like "how engaged were users" — because the *ratio* between them (6 views / 3 users = 2 views per user) is often the actually interesting number, and you can't compute that ratio if you only queried one of the three.

---

## Stage 2 — DISTINCT Combined With GROUP BY

```sql
-- Stage 2: Distinct users per page, per day
SELECT
    page,
    COUNT(DISTINCT user_id) AS distinct_users,
    COUNT(*) AS total_views
FROM page_views
GROUP BY page;
```

**Output:**

| page | distinct_users | total_views |
|--------|-------------------|----------------|
| home | 3 | 3 |
| product | 1 | 1 |
| cart | 1 | 1 |

**Why does `home` show `distinct_users = 3` and `total_views = 3` (identical), while a busier page in a larger dataset might show `distinct_users = 3` but `total_views = 10`?** Because the two numbers are only equal when every user visited that page at most once — the moment any single user visits the same page multiple times, `total_views` grows while `distinct_users` stays capped at the actual number of unique people. Watching this gap widen or stay tight is itself a useful diagnostic for "are a few people visiting this page repeatedly, or are many different people visiting it once each."

---

## Stage 3 — The Performance Cost of DISTINCT (Why Not Use It Everywhere "Just to Be Safe")

**Why not just always write `COUNT(DISTINCT ...)` defensively, even when you're fairly sure there are no duplicates?** Because deduplication isn't free — computationally, `COUNT(DISTINCT x)` typically requires the database to either sort or hash every value of `x` within each group to identify duplicates before it can count them, which is meaningfully more expensive than a plain `COUNT(*)` (which just needs a row counter, no comparison against other rows at all). On a small table the difference is invisible; on a billion-row event log, indiscriminate `COUNT(DISTINCT ...)` across many columns in a single query can be a real, measurable performance cost. The right default is: use `COUNT(DISTINCT ...)` when you have an actual reason to believe duplicates exist and matter (e.g., any user-level metric on an event-log table, where duplication is the norm, not the exception) — not as a reflexive habit on every column.

---

## Stage 4 — A Subtle Trap: DISTINCT Applies to the Whole Expression, Not Just One Column

```sql
-- Stage 4: COUNT(DISTINCT ...) with a multi-column expression
SELECT
    COUNT(DISTINCT user_id) AS distinct_users,
    COUNT(DISTINCT page) AS distinct_pages,
    COUNT(DISTINCT CONCAT(user_id, '-', page)) AS distinct_user_page_combos
FROM page_views;
```

**Output:**

| distinct_users | distinct_pages | distinct_user_page_combos |
|--------------------|--------------------|---------------------------------|
| 3 | 3 | 5 |

**Why is `distinct_user_page_combos` (5) different from either `distinct_users` (3) or `distinct_pages` (3) individually?** `COUNT(DISTINCT CONCAT(user_id, '-', page))` deduplicates on the *combination* — user 501 visiting `home` and user 501 visiting `product` are two different combinations even though they share the same `user_id`, so this answers a third, distinct question: "how many unique (user, page) pairs occurred," which is neither "how many users" nor "how many pages" but genuinely the count of unique pairings between them (useful for, e.g., "how many distinct user-page relationships exist," a common precursor to building a recommendation system's interaction matrix).

**Why not use `COUNT(DISTINCT user_id, page)` (multiple arguments) instead of concatenation?** Some databases (MySQL) support multi-column `COUNT(DISTINCT a, b)` directly, treating the pair as the deduplication key without string concatenation — but this syntax is **not standard SQL** and isn't supported everywhere (PostgreSQL and many others reject it), so the `CONCAT`-based (or, better, a `ROW(...)`-based, where supported) approach is the more portable pattern to reach for if you're not certain which database engine you're targeting.

---

## The one thing to remember

`COUNT(DISTINCT ...)` deduplicates *before* counting, answering "how many unique X" rather than "how many rows" — critical on any event-log-shaped table where a single entity naturally produces multiple rows, but not free computationally, and its deduplication key is whatever full expression you put inside the parentheses, not automatically just "one column."

## Interview Q&A

**Q1. When would `COUNT(*)` and `COUNT(DISTINCT user_id)` give the same answer, and when would they diverge?** They're equal only when every user contributes exactly one row to the table (no repeat events per user) — the moment any user has more than one row, `COUNT(*)` exceeds `COUNT(DISTINCT user_id)`, and the size of that gap tells you how much repeat activity exists on average.

**Q2. Is `COUNT(DISTINCT ...)` expensive? When would you avoid it?** Yes, relative to a plain `COUNT(*)` — it typically requires sorting or hashing values to detect duplicates, an extra computational step a simple row count doesn't need. On very large tables, applying it indiscriminately across many columns in one query can meaningfully slow things down; it should be used deliberately, where duplication is expected and the distinction matters, not defensively on every column.

**Q3. How would you count the number of unique (user, product) pairs someone has purchased, and why can't you just report `COUNT(DISTINCT user_id)` and `COUNT(DISTINCT product_id)` separately for this?** You'd deduplicate on the combined key — e.g. `COUNT(DISTINCT CONCAT(user_id, '-', product_id))` or a portable equivalent. Reporting the two counts separately doesn't answer the pairing question at all: knowing there are 500 distinct users and 200 distinct products tells you nothing about how many of the up-to-100,000 possible pairings actually occurred — that requires deduplicating on the pair itself, not on either column independently.

---
---

# Chapter — Query Optimization

## What is it?

Query optimization is the practice of rewriting or restructuring a SQL query — and the schema/indexes it runs against — so it returns the same correct result **using less time and fewer resources**. It sits at the intersection of understanding *what the database engine actually does* when it executes your SQL, and understanding *how to read its own diagnostic output* (the execution/query plan) to find out where it's spending its effort.

This chapter is intentionally broader than the others in this series — it's less "one clever SQL trick" and more "a systematic way of thinking," so it's organized as a diagnostic workflow rather than a single build-up of one query.

---

## The intuition

Imagine asking a librarian to find every book about "gardening" published after 2010. A librarian with no system would walk every single aisle, pick up every book, check its topic and year — technically correct, but painfully slow (this is a **full table scan**). A librarian with a card catalog organized by topic could jump straight to the "gardening" section (this is an **index**), then only check publication years within that much smaller set. Query optimization is teaching the database to behave like the second librarian — and, just as importantly, learning to *notice* when it's accidentally behaving like the first.

---

## Why does this matter — why not just let the database "figure it out"?

**Why not trust the query planner completely and never think about it?** Modern query planners (PostgreSQL, MySQL, etc.) are genuinely sophisticated and often make good choices automatically — but they work from statistics, heuristics, and the exact way you *wrote* the query, and they can be led astray by missing indexes, stale statistics, or query patterns that accidentally defeat their optimizations (e.g., wrapping an indexed column in a function). A senior analyst/engineer needs to be able to verify the planner's choice, not just hope it's right — especially once a query that ran fine on a small dev dataset gets slow in production at real scale.

---

## Stage 1 — Read the Execution Plan First, Always

Before changing anything, see what the database is actually doing.

```sql
-- PostgreSQL / most engines
EXPLAIN ANALYZE
SELECT user_id, COUNT(*)
FROM page_views
WHERE page_view_date >= '2024-08-01'
GROUP BY user_id;
```

**Why `EXPLAIN ANALYZE` and not just `EXPLAIN`?** Plain `EXPLAIN` shows the planner's *estimated* plan and cost — a prediction. `EXPLAIN ANALYZE` actually **runs the query** and shows real elapsed time and real row counts alongside the plan, letting you compare the planner's estimate against reality — a large gap between estimated and actual row counts is itself a diagnostic signal (usually meaning the table's statistics are stale and need to be refreshed, e.g. `ANALYZE` in Postgres or updating stats in other engines).

**Why is this always Stage 1, before trying any fix?** Because optimizing blind — guessing that "adding an index will help" or "this join is probably the slow part" — wastes effort and can even make things worse (an unnecessary index slows down every future write to that table). The plan tells you, concretely, whether the bottleneck is a full table scan, a slow join algorithm, a sort, or something else entirely — fix the thing the plan says is actually slow, not the thing that intuitively *feels* slow.

---

## Stage 2 — Indexes: What They Fix, and What They Don't

**Why an index helps:** an index on `page_view_date` turns "scan every row to find dates >= 2024-08-01" (an O(n) full table scan) into "jump directly to the right starting point in a sorted structure" (roughly O(log n) to find the start, then a fast sequential read from there) — the same "card catalog" idea from the intuition section.

**Why not index every column "just in case":** every index speeds up reads on that column but slows down every `INSERT`/`UPDATE`/`DELETE` on that table (the index itself has to be maintained/updated on every write), and consumes disk space — indexing indiscriminately trades write performance and storage for read performance you may never actually need. The right approach is to index columns that are frequently used in `WHERE`, `JOIN`, and `ORDER BY` clauses on large, read-heavy tables — not every column reflexively.

**A specific trap — functions on indexed columns silently disable the index:**

```sql
-- BAD: wrapping the indexed column in a function prevents the index from being used
SELECT * FROM page_views WHERE YEAR(page_view_date) = 2024;

-- GOOD: rewrite as a range so the index on page_view_date can still be used
SELECT * FROM page_views WHERE page_view_date >= '2024-01-01' AND page_view_date < '2025-01-01';
```

**Why does `YEAR(page_view_date) = 2024` defeat the index even though `page_view_date` is indexed?** The index is built on the *raw values* of `page_view_date`, sorted — it has no way to look up "which raw dates would produce `2024` when passed through the `YEAR()` function" without first computing `YEAR()` on every row, which requires reading every row, which is exactly the full scan the index was supposed to avoid. Rewriting the condition as a plain range on the raw column lets the planner use the index's sorted structure directly.

---

## Stage 3 — Join Order and Join Type

```sql
-- Filter BEFORE joining when possible, not after
-- WORSE: join everything, then filter
SELECT o.order_id, u.user_name
FROM orders o
JOIN users u ON o.user_id = u.user_id
WHERE o.order_date >= '2024-08-01';

-- Often better (though modern planners frequently do this automatically):
-- explicitly reduce one side of the join first
WITH recent_orders AS (
    SELECT * FROM orders WHERE order_date >= '2024-08-01'
)
SELECT ro.order_id, u.user_name
FROM recent_orders ro
JOIN users u ON ro.user_id = u.user_id;
```

**Why might pre-filtering into a CTE help, if the planner "should" figure this out automatically?** Good planners often *do* push the filter down into the join automatically (this is called "predicate pushdown") — so in many modern engines, both versions produce an identical execution plan. The reason to know the manual version anyway: (1) not every engine or every query shape triggers automatic pushdown reliably — complex queries, certain window function combinations, or specific join types can block it — and (2) explicitly filtering first makes your *intent* clear to a future reader of the query, which has value independent of pure performance. **Always verify with `EXPLAIN` whether your manual rewrite actually changed the plan** — if it didn't, keep whichever version is more readable.

---

## Stage 4 — Avoid `SELECT *`

**Why:** `SELECT *` forces the database to read and transmit every column, even ones you don't need — on a wide table (many columns, some large text/blob fields), this can dramatically increase I/O and network transfer for no benefit. It also silently breaks if the table's schema changes (a new column added later suddenly appears in your result set, potentially breaking downstream code that assumed a fixed column order).

**Why not always list every column explicitly, even for quick exploratory queries?** For genuine one-off, throwaway exploration ("let me just glance at this table"), the productivity cost of typing out every column name outweighs the marginal performance cost on a small ad hoc query — the discipline of explicit column lists matters most for queries that will run repeatedly in production or at scale, not for a single interactive glance at the data.

---

## Stage 5 — Aggregate Before Joining, Not After (When Possible)

```sql
-- WORSE: join first (multiplies rows), then aggregate over the bloated result
SELECT u.user_id, SUM(o.order_amount)
FROM users u
JOIN orders o ON u.user_id = o.user_id
GROUP BY u.user_id;

-- BETTER: pre-aggregate the many-side of the join, then join the much smaller result
WITH order_totals AS (
    SELECT user_id, SUM(order_amount) AS total_spent
    FROM orders
    GROUP BY user_id
)
SELECT u.user_id, ot.total_spent
FROM users u
JOIN order_totals ot ON u.user_id = ot.user_id;
```

**Why does joining first and aggregating after risk being slower?** If a user has 500 orders, joining `users` to `orders` first produces 500 duplicated rows for that one user (each order row repeats the user's other columns) *before* the `GROUP BY` collapses them back down — for a large orders table, this "fan-out then collapse" pattern can force the database to churn through vastly more intermediate rows than necessary. Pre-aggregating `orders` down to one row per user *before* joining means the join itself operates on two already-small, one-row-per-user tables. **Why isn't this always faster, and why must you still check the plan?** For small tables, or when the planner already knows how to optimize the fan-out efficiently, both versions can perform identically — the pre-aggregation pattern is a tool to reach for when the plan shows the join itself (not the aggregation) is the expensive step, not a rule to apply blindly to every join.

---

## The one thing to remember

Never optimize blind: read the execution plan first to find out what's actually slow (full scan? bad join order? missing index? function-wrapped predicate defeating an index?), then apply the specific fix that plan justifies — indexing is a read/write tradeoff, not a free win, and pre-aggregating before a fan-out join is a targeted fix for a specific, plan-diagnosable problem, not a universal rule.

## Interview Q&A

**Q1. A query is slow. What's your first step?** Run `EXPLAIN ANALYZE` (or the equivalent in your database engine) before changing anything — it shows whether the bottleneck is a full table scan, a slow join, a sort, or something else, and compares the planner's row-count estimates to what actually happened, which flags stale statistics as a possible root cause on its own.

**Q2. Why doesn't adding an index always speed up a query?** An index only helps if the query's predicates can actually use it — wrapping the indexed column in a function (`YEAR(date_col) = 2024`), using a leading wildcard in a `LIKE '%term'`, or having such low column selectivity that scanning the index is barely better than scanning the table, can all cause the planner to ignore an existing index. Indexes also add overhead to every write, so they're a genuine tradeoff, not a strictly-free performance win.

**Q3. When would pre-aggregating before a join actually matter, versus being unnecessary?** It matters most when one side of the join has a high fan-out ratio (each row on the "one" side matches many rows on the "many" side) and the aggregation afterward would otherwise have to process a hugely inflated intermediate row count. It's unnecessary noise when the planner's automatic optimizations already produce an equivalent plan, or when the tables involved are small enough that the difference is immeasurable — always confirm with `EXPLAIN` rather than assuming.

---
---

# Chapter — Pareto / 80-20 Analysis

## What is it?

Pareto analysis (the "80-20 rule") tests whether a small share of contributors accounts for a disproportionately large share of some outcome — classically, "80% of revenue comes from 20% of customers." It's not really a SQL technique so much as a **specific analytical question** ("how concentrated is this distribution?") that you answer using tools you already have from earlier in this series: ranking, running totals, and cumulative percentages.

---

## The intuition

Imagine a bag of marbles of different sizes. If you line them up from biggest to smallest and start adding up their weights one by one, you'll typically find that the first handful of marbles already accounts for most of the bag's total weight, while the long tail of small marbles barely moves the needle. Pareto analysis is the formal version of that observation — line up your contributors from biggest to smallest, walk down the list adding up their cumulative share, and see how quickly you approach 100%.

---

## Why run this analysis at all?

**Why not just report the mean or median contribution per customer?** Because an average completely hides concentration — a mean revenue-per-customer of $500 is consistent both with "everyone spends about $500" and with "a handful of whales spend $50,000 each while most customers spend $10," and these two scenarios demand completely different business strategies (broad-based retention efforts vs. white-glove account management for a small number of key accounts). Pareto analysis is specifically designed to reveal *which* of those scenarios you're actually in.

**Why not just eyeball a bar chart of revenue per customer?** For a handful of customers, sure — but for thousands or millions of contributors, a bar chart is unreadable, and "is the top 20% really responsible for 80%, or is it actually 40%, or 95%?" is a precise, single-number question a chart can only answer approximately at best.

---

## The Sample Data

```sql
SELECT * FROM customer_revenue ORDER BY revenue DESC;
```

| customer_id | revenue |
|---------------|-----------|
| C1 | 5000 |
| C2 | 3000 |
| C3 | 1200 |
| C4 | 400 |
| C5 | 250 |
| C6 | 150 |
| C7 | 100 |
| C8 | 60 |
| C9 | 30 |
| C10 | 10 |

---

## Stage 1 — Rank Contributors From Largest to Smallest

**Variables:**
- `rank_desc` — a customer's rank position, 1 = largest contributor

```sql
-- Stage 1: Rank customers by revenue, descending
SELECT
    customer_id,
    revenue,
    ROW_NUMBER() OVER (ORDER BY revenue DESC) AS rank_desc
FROM customer_revenue;
```

**Why `ROW_NUMBER()` and not `RANK()` or `DENSE_RANK()`?** For Pareto analysis specifically, you want every contributor to occupy a distinct, sequential position so that "top 20%" maps unambiguously onto "top N rows" — `RANK()` and `DENSE_RANK()` both allow ties to share a position (and `RANK()` additionally skips subsequent numbers after a tie), which can make "the top 20% of customers" an ambiguous or non-round-number boundary when ties exist. `ROW_NUMBER()` guarantees a clean 1-to-N sequence regardless of ties, which is what the percentile math in the next stage assumes.

---

## Stage 2 — Compute Cumulative Revenue and Cumulative Percentage

**Variables:**
- `cumulative_revenue` — running total of revenue from rank 1 down to this row
- `cumulative_pct_revenue` — that running total as a % of total revenue
- `cumulative_pct_customers` — this row's rank as a % of total customer count

```sql
-- Stage 2: Running totals and percentages
SELECT
    customer_id,
    revenue,
    rank_desc,
    SUM(revenue) OVER (ORDER BY revenue DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_revenue,
    ROUND(100.0 * SUM(revenue) OVER (ORDER BY revenue DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
        / SUM(revenue) OVER (), 1) AS cumulative_pct_revenue,
    ROUND(100.0 * rank_desc / COUNT(*) OVER (), 1) AS cumulative_pct_customers
FROM (... Stage 1 query ...);
```

**Output:**

| customer_id | revenue | rank_desc | cumulative_revenue | cumulative_pct_revenue | cumulative_pct_customers |
|---------------|-----------|--------------|------------------------|-----------------------------|--------------------------------|
| C1 | 5000 | 1 | 5000 | 49.5% | 10.0% |
| C2 | 3000 | 2 | 8000 | 79.2% | 20.0% |
| C3 | 1200 | 3 | 9200 | 91.1% | 30.0% |
| C4 | 400 | 4 | 9600 | 95.0% | 40.0% |
| C5 | 250 | 5 | 9850 | 97.5% | 50.0% |
| C6 | 150 | 6 | 10000 | 99.0% | 60.0% |
| C7 | 100 | 7 | 10100 | 100.0% | 70.0% |
| C8 | 60 | 8 | 10160 | ~100.0% | 80.0% |
| C9 | 30 | 9 | 10190 | ~100.0% | 90.0% |
| C10 | 10 | 10 | 10200 | 100.0% | 100.0% |

**Why two separate `SUM() OVER (...)` window functions — one with `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` and one with an empty `OVER ()`?** They compute genuinely different things and both are needed. `SUM(revenue) OVER (ORDER BY ... ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)` is a **running total** — it only sums rows up to and including the current row, which is what makes it "cumulative." `SUM(revenue) OVER ()` — no `ORDER BY`, no frame clause — sums **every row in the entire result set**, giving you the grand total needed as the denominator to turn the running total into a percentage. Using the same window specification for both (accidentally) would either fail to accumulate properly or fail to capture the true grand total — this pairing (a running-total window and a whole-partition-total window) is a very common and very reusable combination beyond just Pareto analysis.

---

## Stage 3 — Answer the Actual Pareto Question

```sql
-- Stage 3: Find the exact point where cumulative % of customers first reaches 20%,
-- and read off the cumulative revenue % at that point
SELECT customer_id, cumulative_pct_customers, cumulative_pct_revenue
FROM (... Stage 2 query ...)
WHERE cumulative_pct_customers <= 20
ORDER BY rank_desc DESC
LIMIT 1;
```

**Output:** `C2, 20.0%, 79.2%` — in this dataset, the top 20% of customers (C1 and C2) account for **79.2%** of total revenue — remarkably close to the textbook "80/20" split, though real data rarely lines up this exactly and the whole point of running the query is to find out your *actual* ratio rather than assuming it's 80/20 by default.

**Why is "assume it's exactly 80/20" itself a mistake worth calling out explicitly?** "80-20" is a catchy mnemonic for the general *pattern* of skewed concentration, not a law that every dataset obeys precisely — some businesses are far more concentrated (top 5% of customers = 90%+ of revenue, common in enterprise B2B), others are far less concentrated (near-uniform distribution, common in low-differentiation consumer subscriptions). Running the actual query and reading off your real ratio is the entire point of the analysis — quoting "80/20" without checking is exactly the kind of unverified assumption a rigorous analyst should avoid.

---

## Stage 4 — Visualize With a Lorenz-Curve-Style Output (and Why This Extends to a Gini Coefficient)

The table from Stage 2, plotted with `cumulative_pct_customers` on the x-axis and `cumulative_pct_revenue` on the y-axis, is exactly a **Lorenz curve** — the standard economics tool for visualizing inequality/concentration. A perfectly equal distribution (every customer contributes identically) would trace a straight diagonal line from (0,0) to (100,100); the more the actual curve bows away from that diagonal, the more concentrated the distribution.

**Why mention the Lorenz curve / Gini coefficient connection at all?** Because it signals that Pareto analysis isn't an ad hoc business trick — it's a specific application of a well-established statistical concept (distributional inequality), and the **Gini coefficient** (twice the area between the diagonal and the Lorenz curve) is the natural next step if you need a single summary number for "how concentrated is this" that you can track and compare *over time* or *across segments*, rather than re-reading a full cumulative table every time. This is a strong thing to mention in an interview to show you understand the theoretical grounding behind a "simple" business heuristic.

---

## The one thing to remember

Pareto analysis is: rank contributors descending, compute a running cumulative total *and* a separate grand total (two different window function specs), express the running total as a percentage of the grand total, then read off the cumulative-revenue-% at the point where cumulative-count-% crosses your threshold (commonly 20%) — and never assume the answer is exactly 80/20 without actually running the query.

## Formulas Used in This Chapter

| Formula | Meaning |
|---------|---------|
| `rank_desc = ROW_NUMBER() OVER (ORDER BY value DESC)` | Contributor's position, largest first |
| `cumulative_value = SUM(value) OVER (ORDER BY value DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)` | Running total down to this row |
| `grand_total = SUM(value) OVER ()` | Total across all rows (no ordering/frame) |
| `cumulative_pct_value = 100 * cumulative_value / grand_total` | Running total as % of grand total |
| `cumulative_pct_contributors = 100 * rank_desc / COUNT(*) OVER ()` | This row's rank as % of all contributors |

## Interview Q&A

**Q1. Walk me through how you'd compute what % of revenue comes from the top 20% of customers.** Rank customers by revenue descending using `ROW_NUMBER()` (not `RANK()`, to avoid tie-related ambiguity in the percentile boundary), compute a running cumulative sum of revenue ordered the same way, compute the grand total revenue separately with an unordered window `SUM() OVER ()`, divide the two to get a cumulative revenue percentage per row, then find the row where the cumulative count of customers first reaches 20% of the total customer count and read off the cumulative revenue percentage at that point.

**Q2. Why might a business specifically care about this number, beyond just being an interesting statistic?** It directly informs resource allocation — a highly concentrated distribution (e.g., top 5% = 90% of revenue) justifies investing heavily in dedicated account management or white-glove retention for a small number of customers, while a much flatter distribution justifies broad-based, scalable retention programs instead, since no small group of customers dominates enough to warrant individual attention. It's also an early-warning signal: if concentration is increasing over time, the business is becoming more dependent on fewer customers, which is a genuine risk worth flagging even if current revenue looks healthy.

**Q3. Is the 80/20 ratio something you'd expect to hold exactly? What would you do if your data showed 95/5 or 60/40 instead?** No — "80/20" is a mnemonic for the general pattern of skewed concentration, not a rule any specific dataset must obey. A 95/5 split indicates extreme concentration (common in enterprise B2B, or long-tail content platforms) and would push toward heavy investment in retaining the tiny top segment; a 60/40 split indicates a much flatter, more broadly distributed base, where losing any single customer matters far less and retention strategy should be broad rather than targeted. Either result is a legitimate, useful finding — the goal is measuring the actual ratio, not confirming a preconceived 80/20 assumption.
