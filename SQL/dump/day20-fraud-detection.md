# Day 20 — Fraud Detection Patterns in SQL
**FAANG SQL 30-Day Prep**

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
10. FAANG Fraud Patterns

---
# Day 20 — Fraud Detection Patterns in SQL

---

## 1. The Fraud Detection Mindset

```
Fraud signals = deviations from normal behavior

Normal user:  1-2 orders/day, same device, same location, avg $50
Fraudster:    20 orders/hour, new device, new location, $500 each

SQL fraud detection = find statistical outliers + rule violations + behavioral anomalies
```

**Three layers of fraud detection:**
1. **Rule-based** — hard limits (>10 txns/hour, amount > $5000)
2. **Velocity-based** — unusual rate of activity in a time window
3. **Behavioral** — deviation from user's own historical pattern

---

## 2. Velocity Checks

```sql
-- Flag users with too many transactions in a short window
-- Table: transactions(txn_id, user_id, amount, txn_time, device_id, ip_address)

WITH txn_velocity AS (
  SELECT
    user_id, txn_time, amount,
    -- Count transactions in rolling 1-hour window
    COUNT(*) OVER (
      PARTITION BY user_id
      ORDER BY txn_time
      RANGE BETWEEN INTERVAL 1 HOUR PRECEDING AND CURRENT ROW
    ) AS txns_last_1hr,
    -- Sum amount in rolling 1-hour window
    SUM(amount) OVER (
      PARTITION BY user_id
      ORDER BY txn_time
      RANGE BETWEEN INTERVAL 1 HOUR PRECEDING AND CURRENT ROW
    ) AS amount_last_1hr,
    -- Count transactions in rolling 24-hour window
    COUNT(*) OVER (
      PARTITION BY user_id
      ORDER BY txn_time
      RANGE BETWEEN INTERVAL 24 HOUR PRECEDING AND CURRENT ROW
    ) AS txns_last_24hr
  FROM transactions
)
SELECT *,
  CASE
    WHEN txns_last_1hr   > 10     THEN '⚠️  HIGH VELOCITY'
    WHEN amount_last_1hr > 10000  THEN '⚠️  HIGH AMOUNT'
    WHEN txns_last_24hr  > 50     THEN '⚠️  HIGH DAILY VOLUME'
    ELSE '✅ NORMAL'
  END AS velocity_flag
FROM txn_velocity
WHERE txns_last_1hr > 10
   OR amount_last_1hr > 10000
ORDER BY txns_last_1hr DESC;
```

---

## 3. Device & IP Fingerprinting

```sql
-- Flag accounts sharing devices or IPs (account takeover / synthetic fraud)
WITH device_sharing AS (
  SELECT device_id,
    COUNT(DISTINCT user_id) AS users_on_device,
    MIN(txn_time)           AS first_seen,
    MAX(txn_time)           AS last_seen
  FROM transactions
  GROUP BY device_id
  HAVING COUNT(DISTINCT user_id) > 3  -- >3 users on same device = suspicious
),
ip_sharing AS (
  SELECT ip_address,
    COUNT(DISTINCT user_id) AS users_on_ip,
    COUNT(*)                AS total_txns
  FROM transactions
  GROUP BY ip_address
  HAVING COUNT(DISTINCT user_id) > 10  -- >10 users same IP = fraud ring
)
SELECT t.user_id, t.txn_id, t.amount, t.device_id, t.ip_address,
  CASE WHEN ds.device_id IS NOT NULL THEN '⚠️  SHARED DEVICE' END AS device_flag,
  CASE WHEN ip.ip_address IS NOT NULL THEN '⚠️  SHARED IP'    END AS ip_flag,
  ds.users_on_device,
  ip.users_on_ip
FROM transactions t
LEFT JOIN device_sharing ds ON t.device_id  = ds.device_id
LEFT JOIN ip_sharing     ip ON t.ip_address = ip.ip_address
WHERE ds.device_id IS NOT NULL
   OR ip.ip_address IS NOT NULL;
```

---

## 4. New Device / Location Anomaly

```sql
-- Flag transactions from a device the user has never used before
WITH user_device_history AS (
  SELECT user_id, device_id,
    MIN(txn_time) AS first_used
  FROM transactions
  GROUP BY user_id, device_id
),
flagged AS (
  SELECT t.*,
    udh.first_used AS device_first_used,
    -- New device = first time ever seen for this user
    CASE WHEN t.txn_time = udh.first_used THEN 1 ELSE 0 END AS is_new_device,
    -- New device + high amount = very suspicious
    CASE WHEN t.txn_time = udh.first_used
          AND t.amount > 500 THEN 1 ELSE 0 END               AS new_device_high_amount
  FROM transactions t
  JOIN user_device_history udh
    ON  t.user_id   = udh.user_id
    AND t.device_id = udh.device_id
)
SELECT * FROM flagged
WHERE new_device_high_amount = 1
ORDER BY amount DESC;
```

---

## 5. Behavioral Baseline & Deviation Scoring

```sql
-- Compare each transaction to the user's own historical patterns
WITH user_baseline AS (
  SELECT user_id,
    AVG(amount)    AS avg_amount,
    STDDEV(amount) AS std_amount,
    AVG(HOUR(txn_time)) AS avg_hour,   -- typical hour of day
    COUNT(*)       AS total_txns
  FROM transactions
  WHERE txn_time < CURRENT_DATE - INTERVAL 1 DAY  -- use historical only
  GROUP BY user_id
),
scored AS (
  SELECT t.*,
    b.avg_amount, b.std_amount,
    -- Z-score: how many std devs from user's own average?
    ROUND((t.amount - b.avg_amount) /
          NULLIF(b.std_amount, 0), 2)              AS amount_z_score,
    -- Hour deviation
    ABS(HOUR(t.txn_time) - b.avg_hour)             AS hour_deviation,
    b.total_txns
  FROM transactions t
  JOIN user_baseline b ON t.user_id = b.user_id
  WHERE t.txn_time >= CURRENT_DATE - INTERVAL 1 DAY
)
SELECT *,
  CASE
    WHEN amount_z_score > 3  THEN '🔴 HIGH RISK'
    WHEN amount_z_score > 2  THEN '🟡 MEDIUM RISK'
    ELSE                          '🟢 LOW RISK'
  END AS risk_level
FROM scored
ORDER BY amount_z_score DESC;
```

---

## 6. Fraud Ring Detection

```sql
-- Find clusters of users sharing devices + IPs (coordinated fraud rings)
WITH shared_device_pairs AS (
  SELECT a.user_id AS user_a, b.user_id AS user_b,
    a.device_id,
    'device' AS link_type
  FROM transactions a
  JOIN transactions b
    ON  a.device_id = b.device_id
    AND a.user_id   < b.user_id  -- avoid duplicates
),
shared_ip_pairs AS (
  SELECT a.user_id AS user_a, b.user_id AS user_b,
    a.ip_address AS device_id,
    'ip' AS link_type
  FROM transactions a
  JOIN transactions b
    ON  a.ip_address = b.ip_address
    AND a.user_id    < b.user_id
),
all_links AS (
  SELECT * FROM shared_device_pairs
  UNION ALL
  SELECT * FROM shared_ip_pairs
)
SELECT user_a, user_b,
  COUNT(DISTINCT device_id)  AS shared_identifiers,
  COUNT(DISTINCT link_type)  AS link_types,
  CASE
    WHEN COUNT(DISTINCT device_id) >= 2 THEN '🔴 FRAUD RING'
    ELSE '🟡 INVESTIGATE'
  END AS ring_flag
FROM all_links
GROUP BY user_a, user_b
HAVING COUNT(DISTINCT device_id) >= 1
ORDER BY shared_identifiers DESC;
```

---

## 7. Time-Based Anomalies

```sql
-- Transactions at unusual hours for the user
WITH user_hour_profile AS (
  SELECT user_id,
    HOUR(txn_time) AS hr,
    COUNT(*)       AS txns_in_hour
  FROM transactions
  GROUP BY user_id, HOUR(txn_time)
),
user_normal_hours AS (
  SELECT user_id,
    -- Hours where user is normally active (top 80% of activity)
    GROUP_CONCAT(hr ORDER BY txns_in_hour DESC) AS active_hours
  FROM user_hour_profile
  GROUP BY user_id
)
-- Flag transactions happening outside user's normal hours
SELECT t.txn_id, t.user_id, t.amount,
  HOUR(t.txn_time)             AS txn_hour,
  t.txn_time,
  CASE
    WHEN HOUR(t.txn_time) BETWEEN 1 AND 5   -- 1am-5am unusual for most
     AND t.amount > 200 THEN '⚠️  UNUSUAL HOUR + HIGH AMOUNT'
    ELSE 'NORMAL'
  END AS time_flag
FROM transactions t
WHERE HOUR(t.txn_time) BETWEEN 1 AND 5
  AND t.amount > 200
ORDER BY t.amount DESC;
```

---

## 8. Rapid Sequential Transactions (Card Testing)

```sql
-- Card testing: many small txns in quick succession
-- Fraudsters test stolen cards with micro-transactions

WITH sequential AS (
  SELECT
    user_id, txn_id, amount, txn_time,
    LAG(txn_time) OVER (
      PARTITION BY user_id ORDER BY txn_time
    ) AS prev_txn_time,
    TIMESTAMPDIFF(SECOND,
      LAG(txn_time) OVER (PARTITION BY user_id ORDER BY txn_time),
      txn_time
    ) AS seconds_since_last
  FROM transactions
)
SELECT user_id,
  COUNT(*)                         AS rapid_txns,
  SUM(amount)                      AS total_amount,
  MIN(txn_time)                    AS window_start,
  MAX(txn_time)                    AS window_end,
  AVG(seconds_since_last)          AS avg_seconds_between_txns
FROM sequential
WHERE seconds_since_last < 30      -- less than 30 sec apart
  AND amount < 10                  -- micro-transaction
GROUP BY user_id
HAVING COUNT(*) >= 5               -- 5+ rapid micro-transactions
ORDER BY rapid_txns DESC;
```

---

## 9. Composite Fraud Score

```sql
-- Combine multiple signals into a single risk score
WITH signals AS (
  SELECT
    t.txn_id, t.user_id, t.amount, t.txn_time,
    -- Signal 1: velocity (0-30 pts)
    LEAST(30, COUNT(*) OVER (
      PARTITION BY t.user_id
      ORDER BY t.txn_time
      RANGE BETWEEN INTERVAL 1 HOUR PRECEDING AND CURRENT ROW
    ) * 3)                                            AS velocity_score,
    -- Signal 2: amount z-score vs user baseline (0-30 pts)
    LEAST(30, GREATEST(0, ROUND(
      (t.amount - AVG(t.amount) OVER (PARTITION BY t.user_id)) /
      NULLIF(STDDEV(t.amount) OVER (PARTITION BY t.user_id), 0)
    , 0) * 10))                                       AS amount_score,
    -- Signal 3: new device (0-20 pts)
    CASE WHEN t.txn_time = MIN(t.txn_time) OVER (
      PARTITION BY t.user_id, t.device_id)
    THEN 20 ELSE 0 END                                AS new_device_score,
    -- Signal 4: unusual hour (0-20 pts)
    CASE WHEN HOUR(t.txn_time) BETWEEN 1 AND 5
    THEN 20 ELSE 0 END                                AS odd_hour_score
  FROM transactions t
),
scored AS (
  SELECT *,
    velocity_score + amount_score +
    new_device_score + odd_hour_score AS fraud_score
  FROM signals
)
SELECT txn_id, user_id, amount, txn_time,
  velocity_score, amount_score, new_device_score, odd_hour_score,
  fraud_score,
  CASE
    WHEN fraud_score >= 70 THEN '🔴 BLOCK'
    WHEN fraud_score >= 40 THEN '🟡 REVIEW'
    ELSE                        '🟢 ALLOW'
  END AS decision
FROM scored
WHERE fraud_score >= 40
ORDER BY fraud_score DESC;
```

---

## 10. FAANG Fraud Patterns

```sql
-- PayPal/Stripe: Burst detection — sudden spike vs user history
WITH daily_user AS (
  SELECT user_id, DATE(txn_time) AS dt,
    COUNT(*) AS daily_txns, SUM(amount) AS daily_amount
  FROM transactions GROUP BY user_id, DATE(txn_time)
),
baseline AS (
  SELECT user_id,
    AVG(daily_txns)    AS avg_daily_txns,
    STDDEV(daily_txns) AS std_daily_txns,
    AVG(daily_amount)  AS avg_daily_amount
  FROM daily_user GROUP BY user_id
)
SELECT d.user_id, d.dt, d.daily_txns, d.daily_amount,
  ROUND((d.daily_txns - b.avg_daily_txns) /
        NULLIF(b.std_daily_txns, 0), 2)              AS txn_z_score,
  CASE WHEN (d.daily_txns - b.avg_daily_txns) /
        NULLIF(b.std_daily_txns, 0) > 3
       THEN '🔴 BURST DETECTED' ELSE 'NORMAL'
  END AS burst_flag
FROM daily_user d JOIN baseline b ON d.user_id = b.user_id
WHERE (d.daily_txns - b.avg_daily_txns) /
       NULLIF(b.std_daily_txns, 0) > 3
ORDER BY txn_z_score DESC;
```

```sql
-- Amazon: Seller fraud — returns abuse
SELECT seller_id,
  COUNT(*)                                            AS total_orders,
  SUM(CASE WHEN status='returned' THEN 1 ELSE 0 END) AS returns,
  ROUND(SUM(CASE WHEN status='returned'
    THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2)        AS return_rate,
  SUM(CASE WHEN status='returned'
    THEN amount ELSE 0 END)                           AS returned_value,
  CASE WHEN SUM(CASE WHEN status='returned'
    THEN 1 ELSE 0 END) * 100.0 / COUNT(*) > 30
  THEN '⚠️  HIGH RETURN RATE' ELSE '✅ OK' END         AS fraud_flag
FROM orders
GROUP BY seller_id
HAVING COUNT(*) >= 10
ORDER BY return_rate DESC;
```

---

## Summary Cheatsheet

| Pattern | SQL Technique |
|---|---|
| Velocity check | `COUNT(*) OVER (RANGE BETWEEN INTERVAL N PRECEDING)` |
| Device sharing | `COUNT(DISTINCT user_id) GROUP BY device_id HAVING > N` |
| Behavioral Z-score | `(amount - user_avg) / user_stddev` |
| New device flag | `txn_time = MIN(txn_time) OVER (PARTITION BY user, device)` |
| Card testing | `TIMESTAMPDIFF < 30 AND amount < 10` |
| Fraud ring | Self-join on shared device/IP |
| Composite score | Sum weighted signal scores |
| Burst detection | Daily Z-score vs user baseline |

---

### 🟢 Q1 — Easy
> Table: `transactions(txn_id, user_id, amount, txn_time, device_id)`
>
> Flag users who made **more than 5 transactions in any 1-hour window** today. Return `user_id`, `txn_id`, `txn_time`, `amount`, and `txns_last_1hr`.

---

### 🟡 Q2 — Medium
> Same table plus `users(user_id, signup_date, country)`.
>
> For each transaction today, compute a **behavioral deviation score**: compare amount to the user's own 30-day average and stddev. Flag transactions where the z-score > 2 (medium risk) or > 3 (high risk). Include users who have at least 10 historical transactions.

---

### 🔴 Q3 — Hard
> Table: `transactions(txn_id, user_id, amount, txn_time, device_id, ip_address, status)`
>
> Build a **composite fraud score** (0–100) combining: velocity in last hour (max 25 pts), amount z-score vs user baseline (max 25 pts), new device flag (20 pts), shared device with 3+ users (20 pts), and unusual hour 1am–5am (10 pts). Return transactions scoring ≥ 40 with a BLOCK/REVIEW/ALLOW decision.



## Practice Questions

### Q1 — Easy ✅
Flag users with >5 transactions in any 1-hour window today.

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

### Q2 — Medium ✅
Behavioral z-score vs 30-day baseline, flag z > 2.

```sql
WITH user_baseline AS (
  SELECT user_id,
    AVG(amount) AS avg_amount, STDDEV(amount) AS std_amount,
    COUNT(*) AS total_txns
  FROM transactions
  WHERE txn_time >= CURRENT_DATE - INTERVAL 30 DAY
    AND txn_time <  CURRENT_DATE
  GROUP BY user_id HAVING COUNT(*) >= 10
)
SELECT t.txn_id, t.user_id, t.amount, t.txn_time,
  b.avg_amount, b.std_amount,
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

### Q3 — Hard ✅
Composite fraud score (0–100) with BLOCK/REVIEW/ALLOW decision.

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
      RANGE BETWEEN INTERVAL 1 HOUR PRECEDING AND CURRENT ROW
    ) * 5)                                            AS velocity_score,
    LEAST(25, GREATEST(0, ROUND(
      (t.amount - b.avg_amount) / NULLIF(b.std_amount, 0), 2) * 8
    ))                                                AS amount_score,
    CASE WHEN t.txn_time = fd.first_used THEN 20 ELSE 0 END AS new_device_score,
    CASE WHEN ds.device_id IS NOT NULL   THEN 20 ELSE 0 END AS shared_device_score,
    CASE WHEN HOUR(t.txn_time) BETWEEN 1 AND 5 THEN 10 ELSE 0 END AS odd_hour_score
  FROM transactions t
  LEFT JOIN user_baseline    b  ON t.user_id   = b.user_id
  LEFT JOIN device_sharing   ds ON t.device_id = ds.device_id
  LEFT JOIN first_device_use fd ON t.user_id = fd.user_id AND t.device_id = fd.device_id
  WHERE DATE(t.txn_time) = CURRENT_DATE
),
scored AS (
  SELECT *,
    velocity_score + amount_score + new_device_score +
    shared_device_score + odd_hour_score AS fraud_score
  FROM signals
)
SELECT txn_id, user_id, amount, txn_time,
  velocity_score, amount_score, new_device_score,
  shared_device_score, odd_hour_score, fraud_score,
  CASE
    WHEN fraud_score >= 70 THEN '🔴 BLOCK'
    WHEN fraud_score >= 40 THEN '🟡 REVIEW'
    ELSE '🟢 ALLOW'
  END AS decision
FROM scored WHERE fraud_score >= 40
ORDER BY fraud_score DESC;
```

---

## Key Takeaways

- **Velocity** → `RANGE BETWEEN INTERVAL N PRECEDING` — true time-based window
- **Device sharing** → `COUNT(DISTINCT user_id) > N GROUP BY device_id`
- **New device** → `txn_time = MIN(txn_time) OVER (PARTITION BY user, device)`
- **Behavioral Z-score** → `(amount - user_avg) / user_stddev > 2`
- **Card testing** → `TIMESTAMPDIFF < 30 AND amount < 10 HAVING COUNT >= 5`
- **Fraud ring** → self-join on shared device/IP, look for multiple links
- **Composite score** → sum weighted signals, BLOCK ≥ 70, REVIEW ≥ 40
- **Burst detection** → daily Z-score vs user's own history
- **Always use historical baseline** → exclude today when computing avg/stddev

---

*Day 20 complete — 10 days to go 🚀*
