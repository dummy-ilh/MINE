# Gaps and Islands — Finding Consecutive Streaks and Missing Periods

## What is it?

"Gaps and islands" is a classic SQL pattern for two mirror-image problems:

- **Islands** — find maximal runs of *consecutive* dates/values that belong together (e.g., a user's unbroken login streak, consecutive days a server was healthy, consecutive months a subscriber stayed active).
- **Gaps** — find the *missing* dates/values between islands (e.g., days a user didn't log in, a churn window between two subscription periods, missing sensor readings).

They're the same underlying technique applied to opposite questions. Once you can find islands, gaps are just "the space between islands" — and once you can find gaps, islands are just "the space between gaps." You'll usually only need to build one directly; the other falls out almost for free.

This sits one level of abstraction above **Sessionization** (the previous chapter): sessionization groups events by a time *gap threshold* on a continuous timestamp; gaps-and-islands groups events by *consecutive discrete units* (usually calendar days) with **zero tolerance** — a single missing day breaks the streak. Same family of problem, different granularity and different tolerance for "how close is close enough."

---

## The intuition

Imagine a wall calendar where you put a sticker on every day you went to the gym. "Islands" is the question: "what were my longest unbroken sticker streaks?" "Gaps" is the question: "on which stretches of days did I have no sticker at all?" You can answer either by staring at the calendar and looking for breaks in the pattern — the SQL trick below is just a mechanical, exact way of doing that "staring" at any scale, for any user, over any time range, without eyeballing a calendar.

---

## Why build this instead of just counting distinct active days?

**Why not just `COUNT(DISTINCT active_date)` per user and call that "engagement"?** Because a raw count treats 30 scattered active days across a year identically to 30 *consecutive* active days — but a 30-day unbroken streak and 30 random days spread over 12 months represent completely different user behavior (deep habitual engagement vs. occasional, unpredictable usage), and most product questions ("what's our longest login streak leaderboard," "how many users lapsed for 14+ days," "what's a typical churn-and-return gap") are explicitly about *consecutiveness*, which a plain count destroys.

**Why not just eyeball it in a spreadsheet for a few users?** That doesn't scale — you need an answer for millions of users simultaneously, and manual inspection can't be automated into a recurring dashboard or alert.

---

## The Sample Data

```sql
SELECT * FROM logins ORDER BY user_id, login_date;
```

| user_id | login_date |
|---------|------------|
| 401 | 2024-07-01 |
| 401 | 2024-07-02 |
| 401 | 2024-07-03 |
| 401 | 2024-07-06 |
| 401 | 2024-07-07 |
| 401 | 2024-07-10 |
| 402 | 2024-07-01 |
| 402 | 2024-07-02 |
| 402 | 2024-07-03 |
| 402 | 2024-07-04 |
| 402 | 2024-07-08 |

**Why one row per (user, active day) rather than one row per raw event?** Islands and gaps operate on discrete calendar units (days), not continuous timestamps — if your raw data is event-level (multiple logins per day), you first collapse it with `SELECT DISTINCT user_id, DATE(event_time) AS login_date`, exactly the same "roll up to the right grain first" principle used in cohort retention (Stage 2) and sessionization (the midnight-boundary warning in that chapter).

---

## Stage 1 — The Core Trick: Row Number Minus Date

This is the single insight the entire pattern is built on, so it's worth deriving slowly rather than just memorizing it.

**Variables:**
- `rn` — a simple row number per user, ordered by date (1, 2, 3, 4, ...)
- `island_key` — `login_date - rn` (subtracting the row number from the date, in days)

```sql
-- Stage 1: Compute row number and the island key
SELECT
    user_id,
    login_date,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_date) AS rn,
    login_date - ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_date) AS island_key
FROM logins;
```

**Output:**

| user_id | login_date | rn | island_key |
|---------|------------|----|------------|
| 401 | 07-01 | 1 | 06-30 |
| 401 | 07-02 | 2 | 06-30 |
| 401 | 07-03 | 3 | 06-30 |
| 401 | 07-06 | 4 | 07-02 |
| 401 | 07-07 | 5 | 07-02 |
| 401 | 07-10 | 6 | 07-04 |
| 402 | 07-01 | 1 | 06-30 |
| 402 | 07-02 | 2 | 06-30 |
| 402 | 07-03 | 3 | 06-30 |
| 402 | 07-04 | 4 | 06-30 |
| 402 | 07-08 | 5 | 07-03 |

**Why does subtracting the row number from the date produce identical values for consecutive dates?** This is the core mathematical trick — walk through it slowly. If dates are truly consecutive (no gap), each date increases by exactly 1 day, and `rn` *also* increases by exactly 1 for each row. Since both sides of the subtraction increase in lockstep by the same amount, their **difference stays constant** — `07-02 - 2 = 06-30`, exactly the same as `07-01 - 1 = 06-30`. The moment there's a gap (07-06 is 3 days after 07-03, not 1), the date jumps by more than `rn` does, so the difference shifts to a new value (`07-02` instead of `06-30`) — and it stays at *that* new value for the next consecutive run. In short: **`island_key` is constant within a consecutive run and changes exactly when a gap occurs** — which means it's a ready-made group identifier, for free, from one line of arithmetic.

**Why not just compare each date to the previous date with `LAG`, the way Sessionization did?** You could — and for detecting *that* a gap occurred, `LAG` works identically (`login_date - LAG(login_date) > 1` means a gap). But the row-number-minus-date trick has a distinct advantage here: it produces a ready-made **group identifier** in a single pass with no window-function-within-a-window-function nesting, whereas the `LAG`-based approach (like Stage 2–4 of Sessionization) needs a *second* pass — flag the gap, then take a running `SUM` of the flag — to get the same grouping. Both are valid; the row-number trick is simply more compact specifically because gaps-and-islands typically works on evenly-spaced discrete units (days) where "subtract the position" is meaningful, whereas Sessionization's continuous timestamps and variable-length gap thresholds don't have that same clean arithmetic shortcut available.

---

## Stage 2 — Group By the Island Key to Find Islands

Now that every row carries a group identifier that's constant within a streak, finding each streak's start/end/length is a simple aggregation.

**Variables:**
- `streak_start`, `streak_end` — first and last date in the island
- `streak_length` — number of consecutive days in the island

```sql
-- Stage 2: Aggregate by island_key to find each streak
SELECT
    user_id,
    island_key,
    MIN(login_date) AS streak_start,
    MAX(login_date) AS streak_end,
    COUNT(*) AS streak_length
FROM (... Stage 1 query ...)
GROUP BY user_id, island_key
ORDER BY user_id, streak_start;
```

**Output:**

| user_id | island_key | streak_start | streak_end | streak_length |
|---------|------------|----------------|--------------|-----------------|
| 401 | 06-30 | 07-01 | 07-03 | 3 |
| 401 | 07-02 | 07-06 | 07-07 | 2 |
| 401 | 07-04 | 07-10 | 07-10 | 1 |
| 402 | 06-30 | 07-01 | 07-04 | 4 |
| 402 | 07-03 | 07-08 | 07-08 | 1 |

**Why `COUNT(*)` for streak length rather than `streak_end - streak_start + 1`?** Both give the same answer *when the underlying data has exactly one row per calendar day with no duplicates* — but `COUNT(*)` is the more defensive choice because it counts what's actually there rather than assuming a clean date range; if there were ever a duplicate row for the same user/date (a data quality issue), `COUNT(*)` would silently surface that as an inflated count you could catch, whereas the date-arithmetic version would mask it. That said, computing both and asserting they're equal is a good sanity check to build into a production pipeline.

**Why is `island_key` itself (e.g., "06-30") not a meaningful business value?** It's an intentionally arbitrary byproduct of the subtraction — nobody cares that a streak's island_key is "June 30th," because that's not a real date the user did anything on. It only exists to be *grouped by*; the moment you've used it for `GROUP BY`, discard it from any user-facing output and report `streak_start`/`streak_end`/`streak_length` instead.

---

## Stage 3 — Filter to the "Interesting" Islands

Rarely do you want *every* island — usually you want the longest streak per user, or all streaks above some length.

```sql
-- Stage 3a: Longest streak per user
SELECT
    user_id,
    MAX(streak_length) AS longest_streak
FROM (... Stage 2 query ...)
GROUP BY user_id;

-- Stage 3b: All streaks of at least 3 days (a common "meaningfully engaged" cutoff)
SELECT *
FROM (... Stage 2 query ...)
WHERE streak_length >= 3;

-- Stage 3c: The user's CURRENT streak (the one ending on the most recent date in the data)
SELECT user_id, streak_start, streak_end, streak_length
FROM (... Stage 2 query ...) s
WHERE streak_end = (SELECT MAX(login_date) FROM logins WHERE user_id = s.user_id);
```

**Why is "current streak" (3c) meaningfully different from "longest streak ever" (3a)?** A product dashboard showing "your current streak is 12 days" (Duolingo-style) needs *only* the island that's still active as of today — a user's all-time-best 40-day streak from 3 months ago is irrelevant to whether they should feel motivated to log in today. Conflating these two is a common mistake: querying "longest streak" when the product actually needs "current streak" will show stale, demotivating (or confusingly inflated) numbers to users.

---

## Stage 4 — Finding Gaps (The Mirror-Image Problem)

Gaps are the inverse: instead of grouping consecutive dates together, we want to find where a user's *next* date is further away than expected.

**Variables:**
- `next_login_date` — the user's next chronological login after this one (via `LEAD`)
- `gap_days` — how many days between this login and the next one
- `gap_start`, `gap_end` — the first and last day the user was *absent*

```sql
-- Stage 4: Find gaps between consecutive logins
SELECT
    user_id,
    login_date,
    LEAD(login_date) OVER (PARTITION BY user_id ORDER BY login_date) AS next_login_date,
    LEAD(login_date) OVER (PARTITION BY user_id ORDER BY login_date) - login_date AS gap_days,
    login_date + INTERVAL '1 day' AS gap_start,
    LEAD(login_date) OVER (PARTITION BY user_id ORDER BY login_date) - INTERVAL '1 day' AS gap_end
FROM logins;
```

**Output (gap rows only, i.e. `gap_days > 1`):**

| user_id | login_date | next_login_date | gap_days | gap_start | gap_end |
|---------|------------|--------------------|----------|-----------|---------|
| 401 | 07-03 | 07-06 | 3 | 07-04 | 07-05 |
| 401 | 07-07 | 07-10 | 3 | 07-08 | 07-09 |
| 402 | 07-04 | 07-08 | 4 | 07-05 | 07-07 |

**Why `LEAD` here instead of `LAG` (which Sessionization used)?** `LAG` looks *backward* ("what happened before me") — that's what you need to detect "did a gap just occur before this row," which is the right question when you're trying to *start a new group* (Sessionization's use case). `LEAD` looks *forward* ("what happens next") — which is the right question when the thing you actually want to report is the gap *itself* as its own entity with a start and end date, since a gap's boundaries are naturally defined by "the day after I left" through "the day before I came back," both of which require knowing the *next* row, not the previous one. You could solve this with `LAG` too (by shifting your frame of reference by one row), but `LEAD` maps onto the question more directly here.

**Why `gap_days > 1` and not `gap_days > 0`?** Consecutive days (e.g., 07-01 then 07-02) have `gap_days = 1`, which is *not* a gap — it's the expected, no-gap case. A gap only exists when `gap_days > 1`, meaning at least one calendar day was skipped in between. This off-by-one is the single most common bug in gaps-and-islands SQL — always sanity-check it against a hand-verified example before trusting the output.

---

## Stage 5 — Gaps and Islands Are Two Views of the Same Partition

It's worth proving to yourself that these aren't two unrelated techniques — they're complementary views of the exact same underlying sequence.

```sql
-- Combine both into one narrative per user: alternating islands and gaps
SELECT 'island' AS segment_type, user_id, streak_start AS seg_start, streak_end AS seg_end
FROM (... Stage 2 query ...)
UNION ALL
SELECT 'gap' AS segment_type, user_id, gap_start AS seg_start, gap_end AS seg_end
FROM (... Stage 4 query ...)
WHERE gap_days > 1
ORDER BY user_id, seg_start;
```

**Output for user 401:**

| segment_type | user_id | seg_start | seg_end |
|---------------|---------|-----------|---------|
| island | 401 | 07-01 | 07-03 |
| gap | 401 | 07-04 | 07-05 |
| island | 401 | 07-06 | 07-07 |
| gap | 401 | 07-08 | 07-09 |
| island | 401 | 07-10 | 07-10 |

**Why show this combined view at all, rather than keeping islands and gaps as two separate, unrelated query results?** Seeing them interleaved makes it visually obvious that they tile the entire timeline with no overlap and no missing coverage — every single day is accounted for as belonging to exactly one island or exactly one gap, never both and never neither. This is a genuinely useful correctness check: if you sum up all island lengths plus all gap lengths for a user, the total should exactly equal `(last_date - first_date + 1)` — if it doesn't, there's a bug somewhere in the boundary logic (usually an off-by-one on `gap_start`/`gap_end`).

---

## Stage 6 — A Critical Decision: What Counts As "Consecutive"?

Just like Sessionization's threshold (30 minutes, or something else — a judgment call), gaps-and-islands has an analogous decision point: **what discrete unit, and how strict is "no gap allowed"?**

**Why "exactly 1 calendar day apart, zero tolerance" is the default:** for login streaks, the whole point is usually to reward strict daily consistency (Duolingo, fitness apps) — allowing any slack defeats the purpose of a "streak" as a concept.

**Why you might NOT want zero tolerance:**
- **Weekly-cadence products** (e.g., a weekly newsletter or a B2B SaaS dashboard checked once a week) — using calendar-day islands would show *every* user as having zero streaks longer than 1 day, which is meaningless; you'd redefine the unit as *weeks*, not days, and ask "did they engage in consecutive weeks."
- **"Business days only" products** — a user who logs in Friday and again Monday has *not* actually broken a "daily" habit if weekends aren't expected usage days; treating Fri→Mon as a 3-day gap would incorrectly break streaks that are, behaviorally, perfectly consistent. Fix: exclude weekends from the date sequence before running the row-number trick (generate a calendar of only business days, then rank against that calendar instead of raw dates).
- **"Streak freeze" / grace-period products** (Duolingo explicitly sells this) — the product intentionally allows one missed day without breaking the streak. This can't be expressed with vanilla row-number-minus-date; you'd need to first "backfill" grace days as if they were real logins before running Stage 1, or use a more complex tolerance-based version of the `LAG`-and-running-sum approach from Sessionization instead.

**Why does this matter for an interview?** Because reciting the row-number trick alone is table stakes — demonstrating that you know *when it silently breaks* (weekly cadence, business-day-only usage, grace periods) and how to adapt it is what separates a memorized pattern from genuine understanding.

---

## The Full Query (Islands + Gaps Combined)

```sql
WITH
-- Stage 1: Row number and island key
numbered AS (
    SELECT
        user_id,
        login_date,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_date) AS rn,
        login_date - ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_date) AS island_key
    FROM logins
),

-- Stage 2: Islands (consecutive streaks)
islands AS (
    SELECT
        user_id,
        island_key,
        MIN(login_date) AS streak_start,
        MAX(login_date) AS streak_end,
        COUNT(*) AS streak_length
    FROM numbered
    GROUP BY user_id, island_key
),

-- Stage 4: Gaps (missing periods between logins)
gaps AS (
    SELECT
        user_id,
        login_date + INTERVAL '1 day' AS gap_start,
        LEAD(login_date) OVER (PARTITION BY user_id ORDER BY login_date) - INTERVAL '1 day' AS gap_end,
        LEAD(login_date) OVER (PARTITION BY user_id ORDER BY login_date) - login_date AS gap_days
    FROM logins
)

-- Final: both, tagged and unioned
SELECT 'island' AS segment_type, user_id, streak_start AS seg_start, streak_end AS seg_end, streak_length AS length_days
FROM islands
UNION ALL
SELECT 'gap' AS segment_type, user_id, gap_start AS seg_start, gap_end AS seg_end, gap_days - 1 AS length_days
FROM gaps
WHERE gap_days > 1
ORDER BY user_id, seg_start;
```

---

## Key Variables Summary

| Variable | What It Is | Where Used |
|----------|-----------|------------|
| `rn` | Row number per user, ordered by date | Stage 1 |
| `island_key` | `login_date - rn`, constant within a consecutive streak | Stages 1–3 |
| `streak_start` / `streak_end` / `streak_length` | Boundaries and size of an island | Stages 2–3, 5 |
| `next_login_date` | This user's next chronological active date (via `LEAD`) | Stage 4 |
| `gap_days` | Days between this login and the next (1 = no gap) | Stage 4 |
| `gap_start` / `gap_end` | First/last day of an absence period | Stages 4–5 |

---

## Why it works / why it fails

**Why it works:**
- The row-number-minus-date trick is O(n log n) (just a sort + window function) — no self-joins, no procedural loops, scales cleanly to millions of users
- Islands and gaps are provably complementary — they tile the full timeline with no overlap, which gives you a free correctness check
- Generalizes beyond dates to *any* evenly-spaced ordinal sequence — consecutive invoice numbers, consecutive log-in ranks, consecutive integer IDs all use the identical trick

**Why it fails / where it breaks down:**
- Assumes one row per discrete unit (day) with no duplicates — must pre-aggregate event-level data first (same caveat as Sessionization)
- Zero-tolerance consecutiveness doesn't match every product's definition of a "streak" — weekly cadence, business-days-only, and grace-period products all need the underlying unit or tolerance adjusted (Stage 6)
- Doesn't natively handle calendar irregularities (leap years, month-end boundaries) if you're working with month-level rather than day-level islands — date arithmetic libraries generally handle this correctly, but it's worth explicitly testing rather than assuming
- Silently produces wrong islands if the input has gaps in `user_id` coverage due to upstream filtering (e.g., a WHERE clause that excludes certain dates) rather than genuine absence — always confirm the input table represents *true* activity/non-activity, not a pre-filtered subset

---

## The one thing to remember

Subtracting a per-user row number from a date produces a value that's constant across any run of consecutive dates and changes exactly when a gap occurs — turning "find streaks" into a one-line `GROUP BY`. Gaps are simply the inverse view of the same sequence, found by comparing each date to its `LEAD`(next date) and reporting where the difference exceeds one unit.

---

## Formulas Used in This Chapter

| Formula | Meaning |
|---------|---------|
| `island_key = login_date - ROW_NUMBER() OVER (...)` | Constant within a consecutive run, changes at each gap |
| `streak_length = COUNT(*) GROUP BY user_id, island_key` | Size of each island |
| `gap_days = LEAD(login_date) OVER (...) - login_date` | Days until the next active date; `1` = no gap |
| `gap_start = login_date + 1`, `gap_end = next_login_date - 1` | Boundaries of an absence period |
| Coverage check | `SUM(streak_length) + SUM(gap_length) = (last_date - first_date + 1)` |

---

## Interview Q&A

**Q1. Explain the row-number-minus-date trick from first principles — why does it work?**

If you order a user's dates chronologically and assign each one a row number (1, 2, 3, ...), then for any run of truly consecutive dates, both the date and the row number increase by exactly 1 with each step. Subtracting the row number from the date therefore produces a value that doesn't change as long as the dates stay consecutive — the increase on one side of the subtraction exactly cancels the increase on the other. The instant there's a gap, the date jumps by more than 1 while the row number still only increases by 1, so the difference shifts to a new value — and that new value then stays constant for the next consecutive run. The net effect is that this difference is a ready-made group identifier: constant within an island, distinct across islands, computable in a single window-function pass.

**Q2. How would you find each user's *current* login streak, not just their longest one ever?**

After computing islands (grouping by `island_key` to get `streak_start`, `streak_end`, `streak_length` per group), filter to the island whose `streak_end` equals that user's most recent login date in the dataset — that's the streak still "in progress" as of the most recent data point, which is different from and often much shorter than their all-time longest streak. This distinction matters a lot for product surfaces like a streak counter, which should always show the current, still-active streak, not a historical best.

**Q3. Your product allows one "streak freeze" — a single missed day that doesn't break the streak. How would you adapt this pattern?**

The vanilla row-number-minus-date trick assumes zero tolerance for gaps, so it can't directly express "one missed day is forgiven." Two approaches: (1) Pre-process the data to insert a synthetic "freeze" login row on any single-day gap before running Stage 1 — effectively treating a used freeze as if the user had logged in, so the island-key arithmetic proceeds unmodified. This requires tracking how many freezes a user has available and consuming them correctly, which is business logic that belongs upstream of the SQL pattern. (2) Alternatively, adapt the `LAG`-and-running-sum approach from Sessionization instead of the row-number trick: flag a new streak only when the gap exceeds *2* days (not 1), which tolerates exactly one skipped day per gap — though this version can't limit a user to a *specific number* of freezes without additional logic tracking freeze consumption over time.

**Q4. What's the difference between the approach here and the gap-threshold approach used in Sessionization? When would you use each?**

Sessionization operates on continuous timestamps with a *tolerance threshold* in minutes (typically 30) — the gap between events can be anywhere up to that threshold and still count as the same session, and the threshold itself is a tunable, empirically-validated business decision. Gaps-and-islands, as taught here, typically operates on discrete calendar units (days) with *zero tolerance* by default — any missed day breaks the streak, full stop. Use Sessionization-style gap thresholds when the question is about grouping fine-grained, continuous-time activity into visits. Use gaps-and-islands when the question is about coarse-grained, calendar-based consistency — daily/weekly active-day streaks, subscription lapses, missing report dates. If a product needs *both* tolerance and calendar-grain streaks (e.g., "1 missed day forgiven"), you're in Q3's territory — a hybrid of the two techniques.

**Q5. How would you find, for every user, the longest gap (churn window) they've ever had between two logins — and why might a business care about that number specifically?**

Using the `gaps` CTE (Stage 4), compute `gap_days - 1` as the length of each absence, then take `MAX(gap_length) GROUP BY user_id` to get each user's longest-ever absence. A business cares about this because it's a strong churn-risk and win-back signal: a user whose longest historical gap was 3 days behaves very differently from one who once disappeared for 60 days and came back — the latter has demonstrated they *can* return from a long absence, which is directly useful for calibrating a win-back campaign's expected success, or for setting a "declare this user churned" threshold empirically (e.g., "95% of returning users come back within 45 days of their last login, so we call anyone silent past 45 days a likely-churned user") rather than picking an arbitrary cutoff.

---

Ready for your comments — want the next item taught the same way?
