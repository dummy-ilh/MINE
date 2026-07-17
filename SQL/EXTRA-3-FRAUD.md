# Day 20 — Fraud Detection Patterns in SQL
**FAANG SQL 30-Day Prep (Expanded Edition)**

---

## How to use this guide

Every pattern below follows the same structure so you can actually *see* what's happening, not just read a query:

1. **The idea** — plain-English explanation of the fraud pattern and why it matters
2. **The SQL** — the query, commented
3. **Sample input** — a tiny slice of rows so you can trace the logic by hand
4. **Sample output** — what the query returns for that input, and why
5. **Gotchas** — mistakes people make with this pattern in interviews

All examples share one running dataset so the tables stay consistent across the whole doc.

### The shared sample dataset

`transactions(txn_id, user_id, amount, txn_time, device_id, ip_address)`

| txn_id | user_id | amount | txn_time | device_id | ip_address |
|---|---|---|---|---|---|
| 1001 | U1 | 45.00 | 2026-07-17 09:15:00 | D1 | IP1 |
| 1002 | U1 | 52.00 | 2026-07-17 09:20:00 | D1 | IP1 |
| 1003 | U1 | 480.00 | 2026-07-17 09:22:00 | D9 | IP1 |
| 1004 | U2 | 5.00 | 2026-07-17 02:10:00 | D2 | IP2 |
| 1005 | U2 | 5.00 | 2026-07-17 02:10:15 | D2 | IP2 |
| 1006 | U2 | 5.00 | 2026-07-17 02:10:30 | D2 | IP2 |
| 1007 | U2 | 5.00 | 2026-07-17 02:10:45 | D2 | IP2 |
| 1008 | U2 | 5.00 | 2026-07-17 02:11:00 | D2 | IP2 |
| 1009 | U3 | 300.00 | 2026-07-17 03:30:00 | D3 | IP3 |
| 1010 | U4 | 250.00 | 2026-07-17 10:00:00 | D3 | IP3 |

Context you need to know about these users:
- **U1** normally spends ~$50/txn on device D1. Txn 1003 is the *first time ever* D9 has been seen for U1, and it's for $480 — a big jump.
- **U2** normally makes 1–2 txns a day. Here they fire five $5 charges, 15 seconds apart, at 2am — classic **card testing** (a stolen card being validated before a big purchase).
- **U3** and **U4** are two *different* people who both transact from device **D3** and IP **IP3** — a possible shared device / fraud ring.

---

## Concepts Covered

1. The Fraud Detection Mindset
2. Velocity Checks
3. Device & IP Fingerprinting
4. New Device / Location Anomaly
5. Behavioral Baseline & Z-Score Deviation
6. Fraud Ring Detection
7. Time-Based Anomalies
8. Card Testing (Rapid Sequential Transactions)
9. Composite Fraud Score
10. FAANG Fraud Patterns (PayPal/Stripe burst, Amazon returns abuse)
11. **NEW** — Impossible Travel / Geo-Velocity
12. **NEW** — Account Takeover (ATO) Signals
13. **NEW** — Promo Code / Coupon Abuse
14. **NEW** — Structuring ("Smurfing") for AML
15. **NEW** — Synthetic Identity Detection (shared PII)

---

## 1. The Fraud Detection Mindset

```
Fraud signals = deviations from normal behavior

Normal user:  1-2 orders/day, same device, same location, avg $50
Fraudster:    20 orders/hour, new device, new location, $500 each

SQL fraud detection = statistical outliers + rule violations + behavioral anomalies
```

**Three layers, cheapest to most expensive to compute:**

| Layer | What it catches | Example |
|---|---|---|
| **Rule-based** | Known hard limits | amount > $5,000, > 10 txns/hour |
| **Velocity-based** | Unusual *rate* of activity | 5 logins in 10 seconds |
| **Behavioral** | Deviation from *that user's own* history | this user never spends over $100, now spent $900 |

Interviewers usually want you to show you can move from layer 1 (easy `WHERE` clauses) up to layer 3 (per-user statistics using window functions). That progression is the actual skill being tested.

---

## 2. Velocity Checks

**The idea:** fraud usually shows up as a *burst* — many transactions packed into a short time window. A rolling window function counts how many transactions (or how much money) happened in the N hours *before* each row, so you don't need a separate query per time bucket.

```sql
WITH txn_velocity AS (
  SELECT
    user_id, txn_time, amount,
    -- Count transactions in rolling 1-hour window ending at this row
    COUNT(*) OVER (
      PARTITION BY user_id
      ORDER BY txn_time
      RANGE BETWEEN INTERVAL 1 HOUR PRECEDING AND CURRENT ROW
    ) AS txns_last_1hr,
    -- Sum of $ in that same rolling 1-hour window
    SUM(amount) OVER (
      PARTITION BY user_id
      ORDER BY txn_time
      RANGE BETWEEN INTERVAL 1 HOUR PRECEDING AND CURRENT ROW
    ) AS amount_last_1hr
  FROM transactions
)
SELECT *,
  CASE
    WHEN txns_last_1hr   > 3     THEN '⚠️  HIGH VELOCITY'
    WHEN amount_last_1hr > 400   THEN '⚠️  HIGH AMOUNT'
    ELSE '✅ NORMAL'
  END AS velocity_flag
FROM txn_velocity
ORDER BY user_id, txn_time;
```

**Input:** the three U1 rows (1001, 1002, 1003) — all within 7 minutes of each other.

**Output:**

| user_id | txn_time | amount | txns_last_1hr | amount_last_1hr | velocity_flag |
|---|---|---|---|---|---|
| U1 | 09:15 | 45 | 1 | 45 | ✅ NORMAL |
| U1 | 09:20 | 52 | 2 | 97 | ✅ NORMAL |
| U1 | 09:22 | 480 | 3 | 577 | ⚠️ HIGH AMOUNT |

**Why this output:** each row's window looks *backward* one hour from that row's own timestamp, so the third row sees all three transactions (they're all within the hour), giving a running total of $577 — which trips the `amount_last_1hr > 400` rule even though no single transaction alone looks alarming. That's the whole point of velocity checks: catching cumulative behavior, not single-transaction outliers.

**Gotcha:** using `ROWS BETWEEN` instead of `RANGE BETWEEN ... INTERVAL` silently counts by *row position*, not by actual elapsed time — it'll under- or over-count whenever transactions aren't evenly spaced. Always use `RANGE` with a time interval for velocity windows.

---

## 3. Device & IP Fingerprinting

**The idea:** legitimate accounts each have their own device. When many *different* user_ids all transact from the same device_id or ip_address, that's a signature of account takeover rings, synthetic accounts, or one fraudster juggling several stolen identities.

```sql
WITH device_sharing AS (
  SELECT device_id,
    COUNT(DISTINCT user_id) AS users_on_device
  FROM transactions
  GROUP BY device_id
  HAVING COUNT(DISTINCT user_id) > 1   -- lowered threshold for the demo
)
SELECT t.user_id, t.txn_id, t.device_id, ds.users_on_device
FROM transactions t
JOIN device_sharing ds ON t.device_id = ds.device_id;
```

**Input:** rows 1009 (U3, D3) and 1010 (U4, D3).

**Output:**

| user_id | txn_id | device_id | users_on_device |
|---|---|---|---|
| U3 | 1009 | D3 | 2 |
| U4 | 1010 | D3 | 2 |

**Why this output:** `GROUP BY device_id` collapses all rows for D3 into one bucket and counts *distinct* user_ids in it — 2. The `HAVING` clause filters to only devices with more than one user, so D1, D2, D9 (each used by a single person) are excluded. Joining back to `transactions` re-expands the flagged device to its individual rows so you can see which specific transactions are implicated.

**Gotcha:** in production, use `COUNT(DISTINCT user_id) > 3` or higher — a threshold of 1 will flag every shared family device or corporate laptop. Tune the threshold against your false-positive rate, don't hardcode it from a textbook.

---

## 4. New Device / Location Anomaly

**The idea:** the single riskiest moment in a fraud lifecycle is the *first* transaction on a device that account has never used before — especially if it's also a large amount. `MIN(txn_time) OVER (PARTITION BY user, device)` gives you "the first time this user ever used this device," and comparing the current row's time to that tells you if *this* row is that first-ever moment.

```sql
WITH user_device_history AS (
  SELECT user_id, device_id, MIN(txn_time) AS first_used
  FROM transactions
  GROUP BY user_id, device_id
)
SELECT t.*, udh.first_used,
  CASE WHEN t.txn_time = udh.first_used THEN 1 ELSE 0 END AS is_new_device
FROM transactions t
JOIN user_device_history udh
  ON t.user_id = udh.user_id AND t.device_id = udh.device_id
WHERE t.txn_time = udh.first_used AND t.amount > 200;
```

**Input:** row 1003 — U1's first-ever transaction on device D9, for $480.

**Output:**

| user_id | txn_id | device_id | amount | first_used | is_new_device |
|---|---|---|---|---|---|
| U1 | 1003 | D9 | 480 | 09:22 | 1 |

**Why this output:** for D1, U1's `first_used` is 09:15 (row 1001) — so rows 1002 and 1003 don't match `first_used` and are filtered out for D1. But D9 only has one row ever (1003), so `first_used = txn_time` is true, `is_new_device = 1`, and since $480 > $200 it survives the `WHERE` filter. This is exactly the "brand-new device + big spend" fraud fingerprint.

**Gotcha:** this pattern breaks if devices get reused across users (e.g., shared kiosks) — pair it with the device-sharing check in section 3 to avoid false positives on shared hardware.

---

## 5. Behavioral Baseline & Z-Score Deviation

**The idea:** a $480 charge means nothing in isolation — it depends entirely on what's normal *for that specific user*. Compute each user's historical mean and standard deviation, then express today's transaction as "how many standard deviations away from their own average" (a z-score). This adapts automatically to big spenders and small spenders alike.

```sql
WITH user_baseline AS (
  SELECT user_id, AVG(amount) AS avg_amount, STDDEV(amount) AS std_amount
  FROM transactions
  WHERE txn_time < CURRENT_DATE - INTERVAL 1 DAY   -- historical only, never "today"
  GROUP BY user_id
),
scored AS (
  SELECT t.*, b.avg_amount, b.std_amount,
    ROUND((t.amount - b.avg_amount) / NULLIF(b.std_amount, 0), 2) AS amount_z_score
  FROM transactions t
  JOIN user_baseline b ON t.user_id = b.user_id
)
SELECT *,
  CASE
    WHEN amount_z_score > 3 THEN '🔴 HIGH RISK'
    WHEN amount_z_score > 2 THEN '🟡 MEDIUM RISK'
    ELSE '🟢 LOW RISK'
  END AS risk_level
FROM scored
ORDER BY amount_z_score DESC;
```

**Input:** U1's baseline built from historical spend around $45–52 (std_amount ≈ 5); today's transaction is $480.

**Output:**

| user_id | amount | avg_amount | std_amount | amount_z_score | risk_level |
|---|---|---|---|---|---|
| U1 | 480 | 48.50 | 5.0 | 86.30 | 🔴 HIGH RISK |

**Why this output:** the formula `(480 - 48.50) / 5.0` produces an enormous z-score because $480 is dozens of standard deviations above U1's normal range — this user has *never* spent anywhere close to that, so even a modest-looking dollar amount screams anomaly once you compare it to personal history instead of a global threshold.

**Gotcha:** always exclude "today" from the baseline calculation (`WHERE txn_time < CURRENT_DATE - INTERVAL 1 DAY`). If you include today's transactions in the average you're computing, a fraudulent spike drags its own average upward and can mask itself — this is called **baseline poisoning** and interviewers love to probe whether you caught it.

---

## 6. Fraud Ring Detection

**The idea:** individual anomalies are easy to miss; *clusters* of users linked by shared devices or IPs are a much stronger signal, because real fraud rings reuse infrastructure across many stolen or fake identities. Self-join the transaction table on shared identifiers to surface which user *pairs* are connected.

```sql
WITH shared_device_pairs AS (
  SELECT a.user_id AS user_a, b.user_id AS user_b, a.device_id, 'device' AS link_type
  FROM transactions a
  JOIN transactions b ON a.device_id = b.device_id AND a.user_id < b.user_id
)
SELECT user_a, user_b, device_id,
  '🔴 LINKED ACCOUNTS' AS ring_flag
FROM shared_device_pairs;
```

**Input:** U3/D3 and U4/D3.

**Output:**

| user_a | user_b | device_id | ring_flag |
|---|---|---|---|
| U3 | U4 | D3 | 🔴 LINKED ACCOUNTS |

**Why this output:** the self-join matches every pair of rows sharing a `device_id`; the `a.user_id < b.user_id` condition keeps only one direction of each pair (U3→U4, not also U4→U3) and prevents a row from matching itself. In a real dataset you'd then feed these pairs into a graph algorithm (connected components) to find rings bigger than 2 — SQL alone finds the *edges*, not the full *cluster*.

**Gotcha:** without the `user_id < user_id` guard, you get every pair twice (mirrored) plus self-pairs where `user_id = user_id` — always double check your row counts against `COUNT(DISTINCT device_id)` to sanity-check for duplication bugs.

---

## 7. Time-Based Anomalies

**The idea:** most consumer fraud (stolen cards, bot scripts, unauthorized logins) clusters in the middle of the night in the victim's timezone, when the real account owner is asleep and unlikely to notice or respond to an alert.

```sql
SELECT txn_id, user_id, amount, txn_time, HOUR(txn_time) AS txn_hour,
  CASE WHEN HOUR(txn_time) BETWEEN 1 AND 5 AND amount > 200
       THEN '⚠️  UNUSUAL HOUR + HIGH AMOUNT' ELSE 'NORMAL' END AS time_flag
FROM transactions
WHERE HOUR(txn_time) BETWEEN 1 AND 5 AND amount > 200;
```

**Input:** row 1009 — U3, 03:30, $300.

**Output:**

| txn_id | user_id | amount | txn_hour | time_flag |
|---|---|---|---|---|
| 1009 | U3 | 300 | 3 | ⚠️ UNUSUAL HOUR + HIGH AMOUNT |

**Why this output:** hour 3 falls inside the 1–5am window and the amount ($300) clears the $200 threshold, so both conditions in the `CASE` are true.

**Gotcha:** "unusual hour" is timezone-dependent — 3am UTC might be 7pm local time for a user in another region. Always convert `txn_time` to the user's local timezone (join to a `users.timezone` column) before applying an hour-of-day rule, or you'll flag perfectly normal daytime shoppers.

---

## 8. Rapid Sequential Transactions (Card Testing)

**The idea:** before using a stolen card for a big purchase, fraudsters "test" it with a burst of tiny charges (often under $5–10) to confirm the card is still active and hasn't been reported. `LAG()` gives you the previous transaction's timestamp so you can measure the gap between consecutive transactions per user.

```sql
WITH sequential AS (
  SELECT user_id, txn_id, amount, txn_time,
    TIMESTAMPDIFF(SECOND,
      LAG(txn_time) OVER (PARTITION BY user_id ORDER BY txn_time), txn_time
    ) AS seconds_since_last
  FROM transactions
)
SELECT user_id, COUNT(*) AS rapid_txns, SUM(amount) AS total_amount,
  MIN(txn_time) AS window_start, MAX(txn_time) AS window_end
FROM sequential
WHERE seconds_since_last < 30 AND amount < 10
GROUP BY user_id
HAVING COUNT(*) >= 3;
```

**Input:** U2's five $5 charges, 15 seconds apart.

**Output:**

| user_id | rapid_txns | total_amount | window_start | window_end |
|---|---|---|---|---|
| U2 | 4 | 20.00 | 02:10:15 | 02:11:00 |

**Why this output:** `LAG()` compares each row to the *previous* row for that same user, so the very first row (1004, 02:10:00) has no prior row to compare against — `seconds_since_last` is `NULL` there and it's excluded. The remaining four rows are each 15 seconds after the last one and under $10, so all four pass the filter, giving `rapid_txns = 4` (not 5 — the first transaction in the burst never "arrives" late, only the follow-ups do).

**Gotcha:** this undercounts the true burst size by one, because the first transaction in the sequence has no predecessor to compare against. If you need the *full* burst count including the first transaction, combine this with the velocity window from section 2 instead of relying on `LAG` alone.

---

## 9. Composite Fraud Score

**The idea:** no single signal is reliable enough on its own — new devices get used legitimately all the time, odd hours happen, big purchases happen. A composite score sums *weighted* signals so a transaction has to trip several checks at once before it gets blocked, which cuts false positives dramatically.

```sql
WITH signals AS (
  SELECT t.txn_id, t.user_id, t.amount, t.txn_time,
    LEAST(30, COUNT(*) OVER (
      PARTITION BY t.user_id ORDER BY t.txn_time
      RANGE BETWEEN INTERVAL 1 HOUR PRECEDING AND CURRENT ROW) * 3)   AS velocity_score,
    CASE WHEN t.txn_time = MIN(t.txn_time) OVER (PARTITION BY t.user_id, t.device_id)
         THEN 20 ELSE 0 END                                          AS new_device_score,
    CASE WHEN HOUR(t.txn_time) BETWEEN 1 AND 5 THEN 20 ELSE 0 END     AS odd_hour_score
  FROM transactions t
)
SELECT *, velocity_score + new_device_score + odd_hour_score AS fraud_score,
  CASE WHEN velocity_score + new_device_score + odd_hour_score >= 40 THEN '🟡 REVIEW'
       WHEN velocity_score + new_device_score + odd_hour_score >= 70 THEN '🔴 BLOCK'
       ELSE '🟢 ALLOW' END AS decision
FROM signals
ORDER BY fraud_score DESC;
```

**Input:** row 1003 (U1, new device D9, third txn in the hour) and row 1009 (U3, odd hour, single txn, existing device).

**Output:**

| txn_id | user_id | velocity_score | new_device_score | odd_hour_score | fraud_score | decision |
|---|---|---|---|---|---|---|
| 1003 | U1 | 9 | 20 | 0 | 29 | 🟢 ALLOW |
| 1009 | U3 | 3 | 0 | 20 | 23 | 🟢 ALLOW |

**Why this output:** row 1003 gets 9 velocity points (3 transactions × 3) plus 20 for being a brand-new device, totaling 29 — under the 40-point review threshold on its own, which is realistic: a new device alone shouldn't auto-block a real user. Row 1009 only trips the odd-hour signal, so its score is low too. In the full version of this query (section 9 in the original notes) you'd also add the amount z-score and shared-device points, which is what pushes genuinely fraudulent transactions like 1003's $480 charge over the line — the point of the exercise is showing *how* signals stack, not that every flagged row in a toy dataset should be blocked.

**Gotcha:** weight signals based on *actual precision/recall from historical labeled fraud data*, not gut feel — in an interview, saying "I'd calibrate these weights against a labeled fraud dataset using logistic regression coefficients" is a strong signal you understand this is a real ML problem wearing a SQL costume.

---

## 10. FAANG Fraud Patterns

**PayPal/Stripe — burst detection** (sudden spike vs. a user's own daily history):

```sql
WITH daily_user AS (
  SELECT user_id, DATE(txn_time) AS dt, COUNT(*) AS daily_txns
  FROM transactions GROUP BY user_id, DATE(txn_time)
),
baseline AS (
  SELECT user_id, AVG(daily_txns) AS avg_daily_txns, STDDEV(daily_txns) AS std_daily_txns
  FROM daily_user GROUP BY user_id
)
SELECT d.user_id, d.dt, d.daily_txns,
  ROUND((d.daily_txns - b.avg_daily_txns) / NULLIF(b.std_daily_txns, 0), 2) AS txn_z_score
FROM daily_user d JOIN baseline b ON d.user_id = b.user_id
WHERE (d.daily_txns - b.avg_daily_txns) / NULLIF(b.std_daily_txns, 0) > 3;
```

**Amazon — seller returns abuse** (sellers gaming the returns system):

```sql
SELECT seller_id,
  COUNT(*) AS total_orders,
  ROUND(SUM(CASE WHEN status='returned' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS return_rate
FROM orders
GROUP BY seller_id
HAVING COUNT(*) >= 10
   AND SUM(CASE WHEN status='returned' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) > 30;
```

Both follow the same shape you've now seen five times: build a per-entity baseline, compare current behavior to it, filter to outliers. Once you internalize that shape, most "fraud SQL" interview questions become a matter of picking the right grouping key and threshold.

---

## 11. NEW — Impossible Travel / Geo-Velocity

**The idea:** if the same account logs in from New York at 9:00am and then from Singapore at 9:20am, no human traveled that distance in 20 minutes — the account is compromised or shared. This requires a `lat/lon` (or city) per event and comparing consecutive locations per user.

```sql
-- Table: logins(login_id, user_id, login_time, city, lat, lon)
WITH ordered_logins AS (
  SELECT user_id, login_time, city, lat, lon,
    LAG(login_time) OVER (PARTITION BY user_id ORDER BY login_time) AS prev_time,
    LAG(lat) OVER (PARTITION BY user_id ORDER BY login_time)        AS prev_lat,
    LAG(lon) OVER (PARTITION BY user_id ORDER BY login_time)        AS prev_lon,
    LAG(city) OVER (PARTITION BY user_id ORDER BY login_time)       AS prev_city
  FROM logins
),
distance_calc AS (
  SELECT *,
    TIMESTAMPDIFF(MINUTE, prev_time, login_time) AS minutes_elapsed,
    -- Haversine distance in km between consecutive logins
    6371 * ACOS(
      COS(RADIANS(prev_lat)) * COS(RADIANS(lat)) *
      COS(RADIANS(lon) - RADIANS(prev_lon)) +
      SIN(RADIANS(prev_lat)) * SIN(RADIANS(lat))
    ) AS distance_km
  FROM ordered_logins
  WHERE prev_time IS NOT NULL
)
SELECT user_id, prev_city, city, prev_time, login_time,
  minutes_elapsed, ROUND(distance_km, 0) AS distance_km,
  ROUND(distance_km / NULLIF(minutes_elapsed / 60.0, 0), 0) AS implied_speed_kmh
FROM distance_calc
WHERE distance_km / NULLIF(minutes_elapsed / 60.0, 0) > 900   -- faster than a commercial jet
ORDER BY implied_speed_kmh DESC;
```

**Sample input:** one user logs in from New York at 09:00, then "Singapore" at 09:20 (20 minutes later).

**Sample output:**

| user_id | prev_city | city | minutes_elapsed | distance_km | implied_speed_kmh |
|---|---|---|---|---|---|
| U9 | New York | Singapore | 20 | 15,300 | 45,900 |

**Why this output:** the Haversine formula converts two lat/lon pairs into a great-circle distance; dividing that by elapsed hours gives an implied travel speed. 45,900 km/h is obviously impossible (faster than a rocket), so this flags instantly regardless of any manually-tuned distance threshold — speed-based thresholds are more robust than raw-distance thresholds because they naturally account for how much time elapsed.

**Gotcha:** VPNs make this pattern noisy — a user connecting through a VPN exit node in another country will trigger false positives. Many production systems whitelist known VPN/proxy IP ranges before applying this check.

---

## 12. NEW — Account Takeover (ATO) Signals

**The idea:** ATO doesn't always show up as a fraudulent *purchase* — it often shows up first as a change in account behavior: password reset, followed shortly by email change, followed shortly by a payout/withdrawal method change. That specific *sequence*, compressed into a short window, is a much stronger signal than any single event.

```sql
-- Table: account_events(event_id, user_id, event_type, event_time)
-- event_type IN ('password_reset','email_change','payout_method_change','login')

WITH events_ranked AS (
  SELECT user_id, event_type, event_time,
    LEAD(event_type) OVER (PARTITION BY user_id ORDER BY event_time) AS next_event,
    LEAD(event_time) OVER (PARTITION BY user_id ORDER BY event_time) AS next_event_time
  FROM account_events
)
SELECT user_id, event_type, event_time, next_event, next_event_time,
  TIMESTAMPDIFF(MINUTE, event_time, next_event_time) AS minutes_between
FROM events_ranked
WHERE event_type = 'password_reset'
  AND next_event IN ('email_change', 'payout_method_change')
  AND TIMESTAMPDIFF(MINUTE, event_time, next_event_time) < 60;  -- chained within an hour
```

**Sample input:** `password_reset` at 14:00, followed by `payout_method_change` at 14:12 for the same user.

**Sample output:**

| user_id | event_type | next_event | minutes_between |
|---|---|---|---|
| U9 | password_reset | payout_method_change | 12 |

**Why this output:** `LEAD()` pulls the *next* event chronologically for that user into the same row, so you can compare a password reset directly against whatever happened right after it without a self-join. Twelve minutes between "someone reset your password" and "someone changed where your money goes" is the textbook ATO pattern — real account owners rarely touch payout settings within minutes of a password reset.

**Gotcha:** this only catches two-step chains. For longer chains (reset → login from new device → email change → payout change), you'd need multiple `LEAD()` calls at different offsets, or unpivot into a session/sequence model — mention this tradeoff if asked to extend it.

---

## 13. NEW — Promo Code / Coupon Abuse

**The idea:** promo abuse is a numbers game — one person creates many accounts (often sharing an email pattern, device, or payment method) purely to redeem a "first order" or "new user" discount repeatedly.

```sql
-- Table: orders(order_id, user_id, email, device_id, promo_code, discount_amount, order_time)

SELECT device_id,
  COUNT(DISTINCT user_id)      AS accounts_on_device,
  COUNT(DISTINCT promo_code)   AS distinct_promos_used,
  SUM(discount_amount)         AS total_discount_claimed
FROM orders
WHERE promo_code IS NOT NULL
GROUP BY device_id
HAVING COUNT(DISTINCT user_id) > 3    -- one device, many "different" people
ORDER BY total_discount_claimed DESC;
```

**Sample input:** device `D7` places five orders, each from a different `user_id`, each redeeming a "new user $10 off" promo.

**Sample output:**

| device_id | accounts_on_device | distinct_promos_used | total_discount_claimed |
|---|---|---|---|
| D7 | 5 | 1 | 50.00 |

**Why this output:** five distinct user_ids on one device, all pulling from the same limited promo pool, is a strong signature of one person farming a "new customer" discount with throwaway or fake accounts — a real household sharing a device rarely also shares five separate promo redemptions in a short window.

**Gotcha:** combine this with an email-pattern check (e.g., `user+1@gmail.com`, `user+2@gmail.com` — Gmail "plus addressing" abuse) for a much stronger signal, since device alone can produce false positives on shared family computers.

---

## 14. NEW — Structuring / "Smurfing" (Anti-Money-Laundering)

**The idea:** many countries require banks to report cash transactions over a threshold (commonly $10,000 in the US). Money launderers "structure" deposits into several transactions just *under* that threshold to avoid triggering a report. SQL detection looks for repeated near-threshold amounts clustering in a short window.

```sql
-- Table: transactions(txn_id, user_id, amount, txn_time)

WITH near_threshold AS (
  SELECT user_id, txn_id, amount, txn_time,
    COUNT(*) OVER (
      PARTITION BY user_id ORDER BY txn_time
      RANGE BETWEEN INTERVAL 7 DAY PRECEDING AND CURRENT ROW
    ) AS txns_last_7days,
    SUM(amount) OVER (
      PARTITION BY user_id ORDER BY txn_time
      RANGE BETWEEN INTERVAL 7 DAY PRECEDING AND CURRENT ROW
    ) AS total_last_7days
  FROM transactions
  WHERE amount BETWEEN 8000 AND 9999    -- "just under" the $10,000 reporting line
)
SELECT *
FROM near_threshold
WHERE txns_last_7days >= 3          -- repeated pattern, not a one-off
ORDER BY total_last_7days DESC;
```

**Sample input:** a user makes three deposits of $9,200, $8,700, and $9,500 across four days.

**Sample output:**

| user_id | amount | txns_last_7days | total_last_7days |
|---|---|---|---|
| U9 | 9200 | 1 | 9200 |
| U9 | 8700 | 2 | 17900 |
| U9 | 9500 | 3 | 27400 |

**Why this output:** each individual deposit is legitimately under the $10,000 threshold on its own — that's exactly the point of structuring — but the rolling 7-day window reveals a $27,400 total spread across three suspiciously-similar, suspiciously-sized deposits, which is the actual regulatory red flag.

**Gotcha:** this is one of the few fraud patterns where the *individual* rows look perfectly clean; the fraud only exists at the aggregate level, so always window over a period matching the regulation you're checking against (7-day, 30-day, etc.), not a single transaction.

---

## 15. NEW — Synthetic Identity Detection (Shared PII Across "Different" Accounts)

**The idea:** synthetic identity fraud blends a real piece of PII (often a stolen or purchased Social Security number) with fabricated details (fake name, fake address) to create an account that isn't quite anyone. The giveaway in SQL is the same PII fragment (SSN, phone number, or address) reappearing across accounts that otherwise look unrelated.

```sql
-- Table: users(user_id, full_name, ssn_last4, phone, address)

SELECT ssn_last4,
  COUNT(DISTINCT user_id)        AS accounts_sharing_ssn,
  STRING_AGG(DISTINCT full_name, ', ') AS names_used
FROM users
GROUP BY ssn_last4
HAVING COUNT(DISTINCT user_id) > 1
ORDER BY accounts_sharing_ssn DESC;
```

**Sample input:** three accounts — "John Smith," "Jon Smyth," and "J. Smith" — all list the same last-4 SSN digits.

**Sample output:**

| ssn_last4 | accounts_sharing_ssn | names_used |
|---|---|---|
| 4821 | 3 | John Smith, Jon Smyth, J. Smith |

**Why this output:** legitimate individuals don't share SSN digits across separate accounts; three *differently spelled* names all tied to the same identifier is the classic synthetic-identity fingerprint — a fraudster reusing one real, valid SSN fragment across several fabricated personas to pass basic identity checks.

**Gotcha:** matching only on last-4 SSN digits produces false collisions at scale (only 10,000 possible combinations), so production systems combine this with fuzzy name-matching and address-matching before treating it as a real signal — last-4 alone is a starting filter, not a verdict.

---

## Summary Cheatsheet

| Pattern | SQL Technique | Section |
|---|---|---|
| Velocity check | `COUNT(*) OVER (RANGE BETWEEN INTERVAL N PRECEDING)` | 2 |
| Device/IP sharing | `COUNT(DISTINCT user_id) GROUP BY device_id HAVING > N` | 3 |
| New device flag | `txn_time = MIN(txn_time) OVER (PARTITION BY user, device)` | 4 |
| Behavioral z-score | `(amount - user_avg) / user_stddev > 2` | 5 |
| Fraud ring | Self-join on shared device/IP with `a.user_id < b.user_id` | 6 |
| Unusual hour | `HOUR(txn_time) BETWEEN 1 AND 5` (localize first!) | 7 |
| Card testing | `LAG()` + `TIMESTAMPDIFF < 30 AND amount < 10` | 8 |
| Composite score | Sum weighted signals, threshold into BLOCK/REVIEW/ALLOW | 9 |
| Burst detection | Daily z-score vs. user's own history | 10 |
| Impossible travel | Haversine distance / elapsed hours > plausible speed | 11 |
| Account takeover | `LEAD()` to chain sensitive events within a short window | 12 |
| Promo abuse | `COUNT(DISTINCT user_id)` per device/email pattern on promo orders | 13 |
| Structuring | Rolling window sum just under a reporting threshold | 14 |
| Synthetic identity | `COUNT(DISTINCT user_id)` sharing one PII fragment | 15 |

---

## Practice Questions

### 🟢 Q1 — Easy
`transactions(txn_id, user_id, amount, txn_time, device_id)` — flag users with **more than 5 transactions in any 1-hour window** today. Return `user_id, txn_id, txn_time, amount, txns_last_1hr`.

<details><summary>Solution</summary>

```sql
WITH velocity AS (
  SELECT txn_id, user_id, amount, txn_time,
    COUNT(*) OVER (
      PARTITION BY user_id ORDER BY txn_time
      RANGE BETWEEN INTERVAL 1 HOUR PRECEDING AND CURRENT ROW
    ) AS txns_last_1hr
  FROM transactions WHERE DATE(txn_time) = CURRENT_DATE
)
SELECT txn_id, user_id, amount, txn_time, txns_last_1hr
FROM velocity WHERE txns_last_1hr > 5
ORDER BY user_id, txn_time;
```
</details>

### 🟡 Q2 — Medium
Plus `users(user_id, signup_date, country)`. Compute a **behavioral deviation score** vs. each user's 30-day average/stddev. Flag z > 2 (medium) or z > 3 (high). Only include users with ≥ 10 historical transactions.

<details><summary>Solution</summary>

```sql
WITH user_baseline AS (
  SELECT user_id, AVG(amount) AS avg_amount, STDDEV(amount) AS std_amount, COUNT(*) AS total_txns
  FROM transactions
  WHERE txn_time >= CURRENT_DATE - INTERVAL 30 DAY AND txn_time < CURRENT_DATE
  GROUP BY user_id HAVING COUNT(*) >= 10
)
SELECT t.txn_id, t.user_id, t.amount, t.txn_time, b.avg_amount, b.std_amount,
  ROUND((t.amount - b.avg_amount) / NULLIF(b.std_amount, 0), 2) AS z_score,
  CASE
    WHEN (t.amount - b.avg_amount) / NULLIF(b.std_amount, 0) > 3 THEN '🔴 HIGH RISK'
    WHEN (t.amount - b.avg_amount) / NULLIF(b.std_amount, 0) > 2 THEN '🟡 MEDIUM RISK'
    ELSE '🟢 LOW RISK'
  END AS risk_level
FROM transactions t
JOIN user_baseline b ON t.user_id = b.user_id
WHERE DATE(t.txn_time) = CURRENT_DATE
  AND (t.amount - b.avg_amount) / NULLIF(b.std_amount, 0) > 2
ORDER BY z_score DESC;
```
</details>

### 🔴 Q3 — Hard
`transactions(txn_id, user_id, amount, txn_time, device_id, ip_address, status)` — build a **composite fraud score** (0–100): velocity in last hour (max 25), amount z-score vs. baseline (max 25), new device (20), shared device with 3+ users (20), unusual hour 1–5am (10). Return transactions ≥ 40 with BLOCK/REVIEW/ALLOW.

<details><summary>Solution</summary>

```sql
WITH user_baseline AS (
  SELECT user_id, AVG(amount) AS avg_amount, STDDEV(amount) AS std_amount
  FROM transactions WHERE txn_time < CURRENT_DATE GROUP BY user_id
),
device_sharing AS (
  SELECT device_id FROM transactions
  GROUP BY device_id HAVING COUNT(DISTINCT user_id) >= 3
),
first_device_use AS (
  SELECT user_id, device_id, MIN(txn_time) AS first_used
  FROM transactions GROUP BY user_id, device_id
),
signals AS (
  SELECT t.txn_id, t.user_id, t.amount, t.txn_time,
    LEAST(25, COUNT(*) OVER (
      PARTITION BY t.user_id ORDER BY t.txn_time
      RANGE BETWEEN INTERVAL 1 HOUR PRECEDING AND CURRENT ROW) * 5)          AS velocity_score,
    LEAST(25, GREATEST(0, ROUND(
      (t.amount - b.avg_amount) / NULLIF(b.std_amount, 0), 2) * 8))         AS amount_score,
    CASE WHEN t.txn_time = fd.first_used THEN 20 ELSE 0 END                 AS new_device_score,
    CASE WHEN ds.device_id IS NOT NULL   THEN 20 ELSE 0 END                 AS shared_device_score,
    CASE WHEN HOUR(t.txn_time) BETWEEN 1 AND 5 THEN 10 ELSE 0 END           AS odd_hour_score
  FROM transactions t
  LEFT JOIN user_baseline    b  ON t.user_id   = b.user_id
  LEFT JOIN device_sharing   ds ON t.device_id = ds.device_id
  LEFT JOIN first_device_use fd ON t.user_id = fd.user_id AND t.device_id = fd.device_id
  WHERE DATE(t.txn_time) = CURRENT_DATE
),
scored AS (
  SELECT *, velocity_score + amount_score + new_device_score +
    shared_device_score + odd_hour_score AS fraud_score
  FROM signals
)
SELECT txn_id, user_id, amount, txn_time,
  velocity_score, amount_score, new_device_score, shared_device_score, odd_hour_score, fraud_score,
  CASE WHEN fraud_score >= 70 THEN '🔴 BLOCK'
       WHEN fraud_score >= 40 THEN '🟡 REVIEW'
       ELSE '🟢 ALLOW' END AS decision
FROM scored WHERE fraud_score >= 40
ORDER BY fraud_score DESC;
```
</details>

### 🔴 Q4 — Hard (NEW)
`logins(login_id, user_id, login_time, city, lat, lon)` — return every pair of *consecutive* logins per user where the implied travel speed exceeds 900 km/h (impossible travel).

<details><summary>Solution</summary>

See the full query in **Section 11** above — window with `LAG()` to pull the previous login's coordinates and time, compute Haversine distance, divide by elapsed hours, and filter on implied speed.
</details>

### 🟡 Q5 — Medium (NEW)
`account_events(event_id, user_id, event_type, event_time)` — find every `password_reset` that is followed within 60 minutes by an `email_change` or `payout_method_change` for the same user.

<details><summary>Solution</summary>

See **Section 12** above — `LEAD()` partitioned by user, ordered by time, filtered on event-type sequence and elapsed minutes.
</details>

---

## Key Takeaways

- **Velocity** → `RANGE BETWEEN INTERVAL N PRECEDING` — true time-based rolling window, not row-based
- **Device/IP sharing** → `COUNT(DISTINCT user_id) > N GROUP BY device_id`
- **New device** → `txn_time = MIN(txn_time) OVER (PARTITION BY user, device)`
- **Behavioral z-score** → `(amount - user_avg) / user_stddev > 2`, baseline must exclude "today"
- **Card testing** → `LAG()` + `TIMESTAMPDIFF < 30 AND amount < 10 HAVING COUNT >= N`
- **Fraud ring** → self-join on shared device/IP, dedupe with `a.user_id < b.user_id`
- **Composite score** → sum weighted signals; calibrate weights against labeled fraud data, don't guess
- **Impossible travel** → Haversine distance ÷ elapsed hours vs. a plausible max speed
- **Account takeover** → `LEAD()` to chain sensitive account events within a short window
- **Structuring** → rolling sum just under a regulatory reporting threshold; individual rows look clean
- **Synthetic identity** → shared PII fragment across accounts with different surface details
- **Always use a historical baseline** → never let "today's" possibly-fraudulent data poison the average it's being compared against

---

*Day 20 complete — 10 days to go 🚀*
