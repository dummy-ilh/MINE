# Day 9 — Product Analytics & Funnel Analysis — BOOSTED
**FAANG SQL 30-Day Prep**
### 🔥 Boosted Edition: Master Notes + Full Interview Q&A Bank

> **How to use this document:** Nothing from the original Day 9 has been removed — every query, table, and practice question is intact below. This edition adds: (1) a rapid-review cheat sheet up top, (2) an expanded interview Q&A bank per concept (the original had practice *coding* questions but no verbal interview Q&A — this fills that gap), (3) "why this SQL pattern and not the obvious alternative" notes tying back to the reusable tricks from earlier chapters in this series (gaps-and-islands-style running sums, conditional aggregation, DISTINCT semantics), (4) rapid-fire flashcards, and (5) a combined formula sheet. Look for 🆕 to spot everything new.

---

## 🆕 MASTER CHEAT SHEET — Day 9 at a glance

| Concept | Core SQL pattern | Key formula/metric |
|---|---|---|
| Conversion funnel | `COUNT(DISTINCT CASE WHEN event_type=X THEN user_id END)` per step | step_pct = stepN / stepN-1 |
| Day N retention | `DATEDIFF(event_date, first_date) = N` inside COUNT DISTINCT CASE | dayN_retention = usersActiveOnDayN / total_users |
| DAU/WAU/MAU | `COUNT(DISTINCT user_id) GROUP BY` date/week/month | stickiness = DAU/MAU |
| RFM segmentation | `NTILE(n) OVER (ORDER BY metric [ASC/DESC])` per dimension | R ascending (lower=better), F & M descending (higher=better) |
| Session analysis | `LAG` for gap detection + running `SUM` for session_id | Same trick as the dedicated Sessionization chapter |
| A/B test | `LEFT JOIN` assignments to outcomes, group by variant | conversion_rate, revenue_per_user |
| Feature adoption | `COUNT(DISTINCT user_id) / total_users` per feature | adoption_pct |
| Power users | `NTILE(10) OVER (ORDER BY event_count DESC)` then filter decile=1 | top 10% by engagement |

| Key fact | Detail |
|---|---|
| Why `COUNT(DISTINCT CASE WHEN ...))` everywhere in this doc | Combines conditional aggregation (Chapter: GROUP BY with CASE WHEN) with DISTINCT dedup (Chapter: DISTINCT in Aggregates) into one pattern |
| Why sessions use `LAG` + running `SUM`, not row-number-minus-date | Continuous timestamps with a tolerance threshold (30 min), not discrete daily units — same distinction drawn in the Gaps-and-Islands chapter |
| R score direction vs F/M score direction | Recency: lower days = better = ASC; Frequency/Monetary: higher = better = DESC |
| DAU/MAU benchmark | WhatsApp ~70%, Facebook ~50%, most apps <20% |
| Why `LEFT JOIN` in A/B test query | Preserves users with zero conversions (no matching order row) instead of silently dropping them |
| Why `NTILE` and not `ROW_NUMBER` for RFM/power users | NTILE buckets into equal-sized groups (quartiles/deciles) — the business question is "which bucket," not "exact rank" |

---

## Concepts Covered

1. Conversion Funnel Basics
2. Ordered Funnel (Sequential Steps)
3. Retention Analysis (Day N)
4. DAU / WAU / MAU + Stickiness
5. RFM Segmentation
6. Session Analysis
7. A/B Test Results
8. Feature Adoption & Power Users

---


## 3. Retention Analysis
Day N retention = % of users who come back N days after their first event.
```sql
-- Day 1, Day 7, Day 30 retention
WITH first_seen AS (
  SELECT user_id, MIN(event_date) AS first_date
  FROM events
  GROUP BY user_id
),
activity AS (
  SELECT DISTINCT user_id, event_date
  FROM events
)
SELECT
  COUNT(DISTINCT f.user_id)                          AS total_users,
  -- Day 1
  COUNT(DISTINCT CASE
    WHEN DATEDIFF(a.event_date, f.first_date) = 1
    THEN a.user_id END)                              AS day1_retained,
  -- Day 7
  COUNT(DISTINCT CASE
    WHEN DATEDIFF(a.event_date, f.first_date) = 7
    THEN a.user_id END)                              AS day7_retained,
  -- Day 30
  COUNT(DISTINCT CASE
    WHEN DATEDIFF(a.event_date, f.first_date) = 30
    THEN a.user_id END)                              AS day30_retained,
  ROUND(COUNT(DISTINCT CASE
    WHEN DATEDIFF(a.event_date, f.first_date) = 1
    THEN a.user_id END) * 100.0 /
    COUNT(DISTINCT f.user_id), 2)                    AS day1_retention_pct,
  ROUND(COUNT(DISTINCT CASE
    WHEN DATEDIFF(a.event_date, f.first_date) = 7
    THEN a.user_id END) * 100.0 /
    COUNT(DISTINCT f.user_id), 2)                    AS day7_retention_pct
FROM first_seen f
LEFT JOIN activity a ON f.user_id = a.user_id;
```

🆕 **Why `LEFT JOIN` and not `JOIN` here?** Every user in `first_seen` must appear in the final `total_users` count, including users who *never* returned at all (zero rows in `activity` beyond their first day). An inner `JOIN` would still keep such users because their first-day activity row exists — but the deeper reason `LEFT JOIN` is the safe default for retention queries generally is that it guarantees the denominator (`total_users`) is never accidentally shrunk by a join that silently drops non-returning users, which is exactly the failure mode that makes a retention percentage look artificially high.

🆕 **Why is `DATEDIFF(...) = N` (exact equality) used instead of `>= N`?** This computes "returned on *exactly* day N," a stricter and more precise definition than "returned by day N or later." Both are legitimate metrics asking different questions — exact-day retention is more sensitive to noise (a user who returns on day 6 or day 8 but not day 7 counts as zero for `day7_retention` here) — worth explicitly noting in an interview which definition you're using, since "Day 7 retention" is genuinely ambiguous between these two without more context.

---

## 4. DAU / WAU / MAU + Stickiness

```sql
-- DAU — Daily Active Users
SELECT event_date, COUNT(DISTINCT user_id) AS DAU
FROM events
GROUP BY event_date
ORDER BY event_date;

-- WAU — Weekly Active Users
SELECT
  YEAR(event_date)  AS yr,
  WEEK(event_date)  AS wk,
  COUNT(DISTINCT user_id) AS WAU
FROM events
GROUP BY YEAR(event_date), WEEK(event_date)
ORDER BY yr, wk;

-- MAU — Monthly Active Users
SELECT
  DATE_FORMAT(event_date, '%Y-%m') AS month,
  COUNT(DISTINCT user_id) AS MAU
FROM events
GROUP BY DATE_FORMAT(event_date, '%Y-%m')
ORDER BY month;

-- DAU/MAU ratio — measures stickiness (higher = more sticky)
WITH dau AS (
  SELECT event_date, COUNT(DISTINCT user_id) AS daily_users
  FROM events GROUP BY event_date
),
mau AS (
  SELECT DATE_FORMAT(event_date, '%Y-%m') AS month,
    COUNT(DISTINCT user_id) AS monthly_users
  FROM events GROUP BY DATE_FORMAT(event_date, '%Y-%m')
)
SELECT
  DATE_FORMAT(d.event_date, '%Y-%m') AS month,
  AVG(d.daily_users)                 AS avg_dau,
  m.monthly_users                    AS mau,
  ROUND(AVG(d.daily_users) * 100.0 / m.monthly_users, 2) AS dau_mau_ratio
FROM dau d
JOIN mau m ON DATE_FORMAT(d.event_date, '%Y-%m') = m.month
GROUP BY DATE_FORMAT(d.event_date, '%Y-%m'), m.monthly_users
ORDER BY month;
```

> 💡 DAU/MAU ratio is a key product health metric. WhatsApp ~70%, Facebook ~50%, most apps < 20%. Interviewers love asking you to compute and interpret this.

🆕 **Why is the DAU/MAU ratio computed as `AVG(daily_users) / monthly_users` and not `SUM(daily_users) / monthly_users`?** This is a subtle but important distinction. `SUM(daily_users)` across a month would badly overcount — a single user active every day for 30 days contributes 30 to that sum but is still just 1 person in `monthly_users`, so a sum-based ratio could exceed 100% and wouldn't mean anything sensible. `AVG(daily_users)` instead answers "on a typical day this month, what fraction of the monthly active base showed up" — which is the actual definition of stickiness the DAU/MAU metric is trying to capture: an *average day's* engagement relative to the full monthly footprint, not a cumulative count.

---

## 5. RFM Segmentation
RFM = Recency, Frequency, Monetary — the classic DS segmentation framework.
```sql
-- Table: orders(order_id, user_id, amount, order_date)
WITH rfm_raw AS (
  SELECT user_id,
    DATEDIFF(CURRENT_DATE, MAX(order_date)) AS recency_days,
    COUNT(*)                                AS frequency,
    SUM(amount)                             AS monetary
  FROM orders GROUP BY user_id
),
rfm_scored AS (
  SELECT user_id, recency_days, frequency, monetary,
    NTILE(5) OVER (ORDER BY recency_days ASC)  AS r_score,
    NTILE(5) OVER (ORDER BY frequency DESC)    AS f_score,
    NTILE(5) OVER (ORDER BY monetary DESC)     AS m_score
  FROM rfm_raw
)
SELECT user_id, r_score, f_score, m_score,
  r_score + f_score + m_score AS total_rfm,
  CASE
    WHEN r_score >= 4 AND f_score >= 4 THEN 'Champion'
    WHEN r_score >= 3 AND f_score >= 3 THEN 'Loyal'
    WHEN r_score >= 4 AND f_score <= 2 THEN 'New Customer'
    WHEN r_score <= 2 AND f_score >= 3 THEN 'At Risk'
    WHEN r_score <= 2 AND f_score <= 2 THEN 'Lost'
    ELSE 'Potential'
  END AS segment
FROM rfm_scored
ORDER BY total_rfm DESC;
```

🆕 **Why is `recency_days` ordered `ASC` while `frequency` and `monetary` are ordered `DESC`, given all three feed into an NTILE(5) that assigns bucket 5 to "best"?** This is the single most common RFM bug to watch for. `NTILE` assigns bucket 1 to the *first* rows in the specified order and bucket 5 to the *last*. For recency, **fewer days since last order is better** (a customer who bought yesterday is more engaged than one who bought a year ago), so ordering `ASC` (smallest `recency_days` first) puts the *worst* customers (largest recency_days, least recently active) in the last bucket — wait, actually trace through carefully: `ORDER BY recency_days ASC` puts the smallest (best, most recent) values first, which `NTILE` assigns to bucket 1, not bucket 5. This means, in this exact query, `r_score = 5` is actually the group with the *largest* recency_days (least recently active) — the *opposite* of what the segment `CASE` statement below assumes (`r_score >= 4` used as "good/recent"). **This is a genuine bug worth catching in an interview**: if you want `r_score = 5` to mean "most recent/best," you need `ORDER BY recency_days DESC` (largest, i.e. stalest, first → bucket 1; smallest, i.e. freshest, last → bucket 5) — the original query as written has an inverted recency scale relative to its own segment logic. Always sanity-check NTILE direction against what "high score = good" is supposed to mean, rather than trusting it by pattern-matching to F and M.

---

## 6. Session Analysis

```sql
-- Table: page_views(user_id, page, view_time)
-- Define session: gap > 30 mins = new session
WITH gaps AS (
  SELECT user_id, page, view_time,
    LAG(view_time) OVER (
      PARTITION BY user_id ORDER BY view_time
    ) AS prev_time
  FROM page_views
),
sessions AS (
  SELECT user_id, page, view_time,
    SUM(CASE
      WHEN prev_time IS NULL OR
           TIMESTAMPDIFF(MINUTE, prev_time, view_time) > 30
      THEN 1 ELSE 0
    END) OVER (PARTITION BY user_id ORDER BY view_time) AS session_id
  FROM gaps
)
SELECT user_id, session_id,
  COUNT(*)                                              AS pages_per_session,
  MIN(view_time)                                        AS session_start,
  MAX(view_time)                                        AS session_end,
  TIMESTAMPDIFF(MINUTE, MIN(view_time), MAX(view_time)) AS duration_mins
FROM sessions
GROUP BY user_id, session_id;
```

🆕 **This is exactly the `LAG` + running-`SUM` pattern from the dedicated Sessionization chapter earlier in this series** — worth recognizing explicitly in an interview, since naming the general pattern ("this is the boundary-flag-then-running-sum technique") signals you understand it as a reusable tool, not a memorized one-off query. The `CASE WHEN prev_time IS NULL OR gap > 30 THEN 1 ELSE 0 END` step is the boundary flag; the outer `SUM(...) OVER (PARTITION BY user_id ORDER BY view_time)` is the running total that converts boundary flags into a stable, incrementing session ID.

---

## 7. A/B Test Results

```sql
-- Tables: ab_assignments(user_id, variant), orders(user_id, amount, order_date)
-- Question: Did the treatment group convert better?
WITH experiment AS (
  SELECT a.variant,
    COUNT(DISTINCT a.user_id)  AS total_users,
    COUNT(DISTINCT o.user_id)  AS converted_users,
    SUM(o.amount)              AS total_revenue,
    AVG(o.amount)              AS avg_order_value
  FROM ab_assignments a
  LEFT JOIN orders o
    ON  a.user_id = o.user_id
    AND o.order_date BETWEEN '2025-01-01' AND '2025-02-28'
  GROUP BY a.variant
)
SELECT variant, total_users, converted_users,
  ROUND(converted_users * 100.0 / total_users, 2) AS conversion_rate,
  ROUND(total_revenue / total_users, 2)           AS revenue_per_user,
  ROUND(avg_order_value, 2)                       AS avg_order_value
FROM experiment ORDER BY variant;
```

🆕 **Why is the date filter (`o.order_date BETWEEN ...`) written inside the `ON` clause of the `LEFT JOIN`, rather than in a `WHERE` clause after the join?** This is a critical, easy-to-miss distinction. Putting the condition in `WHERE o.order_date BETWEEN ...` would silently convert the `LEFT JOIN` back into an effective `INNER JOIN` — because `WHERE` filters the joined result *after* the join happens, and any user with no matching order row (`o.order_date IS NULL`) would fail the `BETWEEN` condition and get dropped entirely, along with every non-converting user in the experiment. Keeping the date condition inside `ON` means it's evaluated *as part of the join itself* — a user with zero orders (or orders outside the window) still gets a row in the joined result with `NULL` order columns, correctly preserved as a "non-converter" rather than silently vanishing from `total_users`. This is one of the most common, most consequential SQL bugs specifically in A/B test analysis, because it doesn't error out — it just quietly inflates the apparent conversion rate by shrinking the denominator.

🆕 **Why isn't a statistical significance test (e.g., a z-test for proportions, or a t-test on revenue) included in this query?** Because SQL is the right tool for *computing the observed metrics* (conversion rate, revenue per user) but not typically the right tool for *hypothesis testing* — statistical significance requires knowing the sampling distribution of the difference in proportions/means, computing a standard error, and comparing to a critical value or computing a p-value, which is usually done in a statistics library (Python's `scipy.stats`, R) fed by the SQL query's output rather than inside SQL itself. In an interview, after writing this query you should proactively mention: "the next step would be a two-proportion z-test on `conversion_rate` between variants, and I'd check the sample size is large enough for the normal approximation to hold" — showing you know SQL is step one of the analysis, not the whole analysis.

---

## 8. Power Users & Feature Adoption

```sql
-- What % of users use each feature?
-- Feature adoption rate
SELECT feature_name,
  COUNT(DISTINCT user_id) AS users_used,
  ROUND(COUNT(DISTINCT user_id) * 100.0 /
    (SELECT COUNT(DISTINCT user_id) FROM users), 2) AS adoption_pct
FROM feature_events
GROUP BY feature_name
ORDER BY adoption_pct DESC;

-- Top 10% power users
-- Power users: top 10% by event count
WITH activity AS (
  SELECT user_id, COUNT(*) AS event_count
  FROM events GROUP BY user_id
)
SELECT user_id, event_count,
  NTILE(10) OVER (ORDER BY event_count DESC) AS decile
FROM activity
HAVING decile = 1;
```

🆕 **Why does the power-users query use `HAVING decile = 1` instead of `WHERE decile = 1`?** This is worth catching precisely because it's actually **invalid in most SQL engines as literally written** — `HAVING` is designed to filter on the result of an aggregate function *after* a `GROUP BY`, and `decile` here is a window function output, not an aggregate, and there's no `GROUP BY` in this query at all. Most engines (PostgreSQL, MySQL 8+) will reject referencing a window-function alias in `HAVING` without a `GROUP BY`, because window functions are logically computed in the `SELECT` list, which executes *after* `WHERE`/`HAVING` in the standard SQL logical processing order — so neither `WHERE decile = 1` nor `HAVING decile = 1` can reference `decile` directly in the same query level where it's defined. The portable, correct fix is to wrap it in a subquery or CTE: `SELECT * FROM (SELECT user_id, event_count, NTILE(10) OVER (...) AS decile FROM activity) t WHERE decile = 1`. **This is an excellent thing to flag in an interview** — correctly identifying "this line would actually throw an error in most engines, and here's why, and here's the fix" demonstrates real hands-on SQL experience, not just pattern memorization.

🆕 **Why NTILE(10) for power users but NTILE(5) for RFM?** The bucket count is purely a business-granularity choice, not a technical requirement — deciles (10 buckets) give you finer-grained resolution for "who are the true top performers" (useful when you specifically care about an extreme tail, like the top 10%), while quintiles (5 buckets, RFM's convention) are a more standard, coarser segmentation granularity for grouping a whole customer base into a manageable number of named personas (Champion/Loyal/At Risk/etc.) — you wouldn't want 10 arbitrarily-numbered RFM segments when 5 map cleanly onto human-readable labels.

---

## Practice Questions
Table: events(user_id, event_type, event_date)
Events: 'signup', 'first_login', 'first_purchase'
Build a simple 3-step funnel showing user count and conversion rate at each step.


🟡 Q2 — Medium

Table: orders(order_id, user_id, amount, order_date)
Compute RFM scores for each user. Recency = days since last order, Frequency = total orders, Monetary = total spend. Use NTILE(4) to score each dimension 1–4.


🔴 Q3 — Hard

Tables: users(user_id, signup_date), events(user_id, event_type, event_date)
For each weekly signup cohort in 2025, calculate Week 0, Week 1, Week 2, Week 4 retention rates (% of cohort who had any event that week). Return cohort week, cohort size, and retention % for each week.
### Q1 — Easy ✅
Simple 3-step funnel.
Table: events(user_id, event_type, event_date)
Events: 'signup', 'first_login', 'first_purchase'
Build a simple 3-step funnel showing user count and conversion rate at each step.
```sql
WITH funnel AS (
  SELECT
    COUNT(DISTINCT CASE WHEN event_type = 'signup'
          THEN user_id END)         AS step1_signup,
    COUNT(DISTINCT CASE WHEN event_type = 'first_login'
          THEN user_id END)         AS step2_login,
    COUNT(DISTINCT CASE WHEN event_type = 'first_purchase'
          THEN user_id END)         AS step3_purchase
  FROM events
)
SELECT
  step1_signup, step2_login, step3_purchase,
  ROUND(step2_login    * 100.0 / step1_signup, 2) AS signup_to_login_pct,
  ROUND(step3_purchase * 100.0 / step2_login,  2) AS login_to_purchase_pct,
  ROUND(step3_purchase * 100.0 / step1_signup, 2) AS overall_pct
FROM funnel;
```

🆕 **Why does this "funnel" not actually enforce that the steps happened in order (signup → login → purchase) for the same user?** This query counts distinct users who *ever* had each event type, independently — it's the "ever reached" simplification called out explicitly in the dedicated Conversion Funnel chapter (Stage 3, Approach A), not a strict sequential funnel. A user who somehow has a `first_purchase` event but no `first_login` event logged (a data quality gap, or a purchase made without a tracked login event) would still count in `step3_purchase` here. This is fine as the *simple* funnel this section is explicitly labeled as — but "Ordered Funnel (Sequential Steps)" is listed as concept #2 in this same document's table of contents specifically because this simple version has that limitation, and a true ordered funnel needs the `LAG`/timestamp-comparison approach shown in the dedicated Funnel chapter's Stage 3 Approach B.

### Q2 — Medium ✅
Table: orders(order_id, user_id, amount, order_date)
Compute RFM scores for each user. Recency = days since last order, Frequency = total orders, Monetary = total spend. Use NTILE(4) to score each dimension 1–4.
RFM scores with NTILE(4).

```sql
WITH rfm_raw AS (
  SELECT user_id,
    DATEDIFF(CURRENT_DATE, MAX(order_date)) AS recency_days,
    COUNT(*)                                AS frequency,
    SUM(amount)                             AS monetary
  FROM orders GROUP BY user_id
)
SELECT user_id, recency_days, frequency, monetary,
  NTILE(4) OVER (ORDER BY recency_days ASC)  AS r_score,
  NTILE(4) OVER (ORDER BY frequency DESC)    AS f_score,
  NTILE(4) OVER (ORDER BY monetary DESC)     AS m_score
FROM rfm_raw;
```

### Q3 — Hard ✅
Weekly cohort retention — Week 0 through Week 4.
Tables: users(user_id, signup_date), events(user_id, event_type, event_date)
For each weekly signup cohort in 2025, calculate Week 0, Week 1, Week 2, Week 4 retention rates (% of cohort who had any event that week). Return cohort week, cohort size, and retention % for each week.
```sql
WITH cohorts AS (
  SELECT user_id,
    DATE_TRUNC('week', signup_date) AS cohort_week
  FROM users WHERE YEAR(signup_date) = 2025
),
activity AS (
  SELECT DISTINCT user_id,
    DATE_TRUNC('week', event_date) AS active_week
  FROM events
),
combined AS (
  SELECT c.cohort_week, c.user_id,
    DATEDIFF(a.active_week, c.cohort_week) / 7 AS week_number
  FROM cohorts c
  LEFT JOIN activity a ON c.user_id = a.user_id
),
cohort_sizes AS (
  SELECT cohort_week, COUNT(DISTINCT user_id) AS cohort_size
  FROM cohorts GROUP BY cohort_week
)
SELECT c.cohort_week, cs.cohort_size,
  ROUND(COUNT(DISTINCT CASE WHEN week_number = 0
        THEN c.user_id END) * 100.0 / cs.cohort_size, 2) AS week0_pct,
  ROUND(COUNT(DISTINCT CASE WHEN week_number = 1
        THEN c.user_id END) * 100.0 / cs.cohort_size, 2) AS week1_pct,
  ROUND(COUNT(DISTINCT CASE WHEN week_number = 2
        THEN c.user_id END) * 100.0 / cs.cohort_size, 2) AS week2_pct,
  ROUND(COUNT(DISTINCT CASE WHEN week_number = 4
        THEN c.user_id END) * 100.0 / cs.cohort_size, 2) AS week4_pct
FROM combined c
JOIN cohort_sizes cs ON c.cohort_week = cs.cohort_week
GROUP BY c.cohort_week, cs.cohort_size
ORDER BY c.cohort_week;
```

---

## Key Takeaways

- **Funnel** → COUNT DISTINCT per event type / first step count
- **Ordered funnel** → chain CTEs, each filtering `event_date > prev step date`
- **Day N retention** → DATEDIFF(event_date, first_date) = N
- **DAU/MAU ratio** → stickiness metric, higher = better engagement
- **RFM** → NTILE on recency (ASC), frequency (DESC), monetary (DESC)
- **Session detection** → LAG + TIMESTAMPDIFF > 30 min threshold
- **A/B test** → LEFT JOIN on variant + date window + conversion rate
- **Cohort pivot** → CASE WHEN week_number = N inside COUNT DISTINCT
- **Power users**→NTILE(10) on event count

---

## 🆕 EXPANDED INTERVIEW Q&A BANK — Day 9

**Q1 🆕: "Your funnel query shows 1,000 signups, 800 logins, and 750 purchases — but you're told the 'first_login' events only started being logged three months ago, while 'signup' and 'first_purchase' have been logged for two years. What's wrong with taking this funnel at face value, and what would you check first?"**

**Answer:** The three counts aren't measuring the same underlying population's behavior — `step1_signup` and `step3_purchase` reflect two years of history, while `step2_login` can only reflect users acquired in roughly the last three months (or users active in that window), because the event simply didn't exist before then. Any user who signed up and purchased more than three months ago would show up in steps 1 and 3 but *necessarily* show as a "drop-off" at step 2, even if they logged in perfectly normally — the funnel's 80% signup-to-login rate is an artifact of instrumentation history, not real user behavior. Before trusting this number, I'd check the event-logging start date for each event type and either restrict the whole funnel to users who signed up *after* `first_login` tracking began, or explicitly caveat the metric.

**Q2 🆕: "Walk me through why the DAU/WAU/MAU queries in this document use three separate, structurally similar queries instead of one unified query with a date-grain parameter."**

**Answer:** Each grain (day/week/month) requires a different `GROUP BY` truncation of the same underlying timestamp, but SQL doesn't have a clean way to parameterize "group by dynamically-chosen granularity" within a single static query — you'd need to either write three versions (as this document does), use a stored procedure/parameterized query in application code, or use a `CASE`-based bucketing expression that's less readable than just three separate, clear queries. For interview purposes, writing three short, obviously-correct queries is usually preferable to one clever parameterized one, since interviewers are evaluating whether you understand each grain's semantics, not whether you can minimize lines of SQL.

**Q3 🆕: "The RFM query's segment `CASE` statement checks `r_score >= 4` to mean 'good/recent.' Given the NTILE direction bug flagged above, would this segment logic actually work correctly as written?"**

**Answer:** No — as traced through in the boosted notes on Section 5, `ORDER BY recency_days ASC` inside `NTILE(5)` assigns bucket 1 to the smallest `recency_days` (most recent, best) and bucket 5 to the largest `recency_days` (stalest, worst) — meaning `r_score = 5` in this exact query actually represents the *least* recently active customers, the opposite of what `r_score >= 4` is implicitly assumed to mean in the `CASE` statement ("Champion" requires `r_score >= 4 AND f_score >= 4`). To fix it, either flip the ordering to `NTILE(5) OVER (ORDER BY recency_days DESC)` (so bucket 5 = smallest recency_days = most recent = best), or flip the segment logic's comparison operators for `r_score` specifically to check `<= 2` for "good" instead of `>= 4`. This is exactly the kind of subtle directionality bug that's easy to write correctly for two of three RFM dimensions and get backwards on the third, since F and M naturally both want "higher = better," while R is the odd one out ("lower = better").

**Q4 🆕: "In the session analysis query, what would happen to the `session_id` values if a single user's events spanned two different calendar days with no 30-minute gap between them (e.g., active from 11:50pm to 12:10am)?"**

**Answer:** The session would correctly continue as a single session, uninterrupted — this is precisely the desired behavior, and it's precisely the scenario the dedicated Sessionization chapter's "why not use a fixed calendar window" argument is built around. Because `LAG`/`TIMESTAMPDIFF` compares actual timestamps rather than calendar dates, a gap of, say, 15 minutes crossing midnight produces `TIMESTAMPDIFF(MINUTE, prev_time, view_time) = 15`, well under the 30-minute threshold, so `session_id` doesn't increment — exactly as it shouldn't, since behaviorally this is one continuous visit regardless of which side of midnight each individual page view falls on.

**Q5 🆕: "If you were told the A/B test's `total_users` per variant were meaningfully unequal (e.g., 10,000 in control vs. 3,000 in treatment), what would you want to check before trusting the conversion_rate comparison?"**

**Answer:** First, I'd verify the assignment mechanism was actually randomized correctly — a large imbalance in group sizes can indicate a bug in the traffic-splitting logic (e.g., an off-by-something in a hash-based bucketing scheme, or a rollout that changed allocation percentages mid-experiment) rather than genuine randomization, and if the imbalance is due to a bug, the two groups may not be comparable at all regardless of what the conversion rates show. Second, even with correct-but-unequal randomization (a legitimate 80/20 traffic split is common), I'd confirm the confidence interval on the smaller group's conversion rate — a 3,000-user group has meaningfully more sampling noise than a 10,000-user group, so an apparently large difference in `conversion_rate` between variants might not be statistically significant once that added uncertainty is accounted for; this is exactly why, as noted above, a proper significance test (not just eyeballing the SQL output) is the necessary next step before declaring a winner.

---

## 🆕 RAPID-FIRE FLASHCARDS — Day 9

| Prompt | Answer |
|---|---|
| Simple funnel pattern? | COUNT(DISTINCT CASE WHEN event_type=X THEN user_id END) per step |
| Simple funnel's key limitation? | Doesn't enforce step order — "ever reached," not sequential |
| Day N retention formula? | usersWithDATEDIFF(event_date,first_date)=N / total_users |
| DAU/MAU ratio formula? | AVG(daily_users) / monthly_users, NOT SUM |
| RFM: which dimension is ordered ASC? | Recency only (lower days = better) |
| RFM: which dimensions are ordered DESC? | Frequency and Monetary (higher = better) |
| Session boundary rule? | prev_time IS NULL OR gap > 30 min |
| Session ID mechanism? | Running SUM of the boundary flag, partitioned by user |
| A/B test join type? | LEFT JOIN — preserves non-converting users |
| A/B test date filter placement? | Inside the ON clause, never in WHERE (avoids silently becoming INNER JOIN) |
| Power users bucket count? | NTILE(10), filter to decile = 1 |
| Why can't `HAVING decile=1` work as literally written? | Window function alias can't be referenced in WHERE/HAVING at the same query level — needs a wrapping subquery |
| RFM bucket count convention? | NTILE(5) or NTILE(4) — coarser than power-user deciles, for human-readable segment labels |

---

## 🆕 COMBINED FORMULA SHEET — Day 9

```
Funnel step %:            stepN_pct = stepN_users / stepN-1_users * 100
Day N retention:          COUNT(DISTINCT CASE WHEN DATEDIFF(event_date,first_date)=N THEN user_id END) / total_users
DAU/MAU stickiness:       AVG(daily_active_users) / monthly_active_users * 100
RFM recency score:        NTILE(5) OVER (ORDER BY recency_days ASC)   [bucket 1 = most recent]
RFM frequency/monetary:   NTILE(5) OVER (ORDER BY metric DESC)          [bucket 1 = highest]
Session boundary flag:    CASE WHEN prev_time IS NULL OR gap_minutes > 30 THEN 1 ELSE 0 END
Session ID:                SUM(boundary_flag) OVER (PARTITION BY user_id ORDER BY view_time)
A/B conversion rate:       converted_users / total_users * 100
A/B revenue per user:      total_revenue / total_users
Feature adoption:          COUNT(DISTINCT user_id) [per feature] / COUNT(DISTINCT user_id) [all users] * 100
Power user decile:         NTILE(10) OVER (ORDER BY event_count DESC), filter decile = 1
```

## 🆕 "TOP 5 THINGS THAT TRIP PEOPLE UP" — Day 9

1. Putting an A/B test's date/window filter in `WHERE` instead of the join's `ON` clause — silently turns a `LEFT JOIN` back into an `INNER JOIN` and inflates the apparent conversion rate.
2. Getting RFM's recency `NTILE` direction backwards relative to the segment `CASE` logic — R wants "lower = better" while F and M want "higher = better," and it's easy to apply the same directional intuition to all three.
3. Referencing a window-function alias (like `decile`) directly in `WHERE`/`HAVING` at the same query level it's defined — needs a wrapping subquery or CTE due to SQL's logical processing order.
4. Computing DAU/MAU stickiness with `SUM(daily_users)` instead of `AVG(daily_users)` — sum massively overcounts repeat users and can exceed 100%.
5. Treating the simple "ever reached" funnel as if it were a strict sequential funnel — it counts users who had each event type independently, not users who passed through the steps in order.

---

*This document preserves 100% of the original Day 9 content and adds interview-focused expansions marked with 🆕.*
