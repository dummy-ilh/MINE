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

## 1. The Fraud Detection Mindset

```
Three layers:
1. Rule-based    → hard limits (>10 txns/hr, amount > $5000)
2. Velocity      → unusual rate in time window
3. Behavioral    → deviation from user's own history

SQL fraud = statistical outliers + rule violations + behavioral anomalies
```

---

## 2. Velocity Checks

```sql
WITH txn_velocity AS (
  SELECT user_id, txn_time, amount,
    COUNT(*) OVER (
      PARTITION BY user_id ORDER BY txn_time
      RANGE BETWEEN INTERVAL 1 HOUR PRECEDING AND CURRENT ROW
    ) AS txns_last_1hr,
    SUM(amount) OVER (
      PARTITION BY user_id ORDER BY txn_time
      RANGE BETWEEN INTERVAL 1 HOUR PRECEDING AND CURRENT ROW
    ) AS amount_last_1hr,
    COUNT(*) OVER (
      PARTITION BY user_id ORDER BY txn_time
      RANGE BETWEEN INTERVAL 24 HOUR PRECEDING AND CURRENT ROW
    ) AS txns_last_24hr
  FROM transactions
)
SELECT *,
  CASE
    WHEN txns_last_1hr   > 10    THEN '⚠️  HIGH VELOCITY'
    WHEN amount_last_1hr > 10000 THEN '⚠️  HIGH AMOUNT'
    WHEN txns_last_24hr  > 50    THEN '⚠️  HIGH DAILY VOLUME'
    ELSE '✅ NORMAL'
  END AS velocity_flag
FROM txn_velocity
WHERE txns_last_1hr > 10 OR amount_last_1hr > 10000;
```

> 💡 Use `RANGE BETWEEN INTERVAL N PRECEDING` for true time-based windows (not row-count based).

---

## 3. Device & IP Fingerprinting

```sql
WITH device_sharing AS (
  SELECT device_id, COUNT(DISTINCT user_id) AS users_on_device
  FROM transactions GROUP BY device_id
  HAVING COUNT(DISTINCT user_id) > 3
),
ip_sharing AS (
  SELECT ip_address, COUNT(DISTINCT user_id) AS users_on_ip
  FROM transactions GROUP BY ip_address
  HAVING COUNT(DISTINCT user_id) > 10
)
SELECT t.user_id, t.txn_id, t.amount, t.device_id, t.ip_address,
  CASE WHEN ds.device_id   IS NOT NULL THEN '⚠️  SHARED DEVICE' END AS device_flag,
  CASE WHEN ip.ip_address  IS NOT NULL THEN '⚠️  SHARED IP'    END AS ip_flag,
  ds.users_on_device, ip.users_on_ip
FROM transactions t
LEFT JOIN device_sharing ds ON t.device_id  = ds.device_id
LEFT JOIN ip_sharing     ip ON t.ip_address = ip.ip_address
WHERE ds.device_id IS NOT NULL OR ip.ip_address IS NOT NULL;
```

---

## 4. New Device Anomaly

```sql
WITH first_device_use AS (
  SELECT user_id, device_id, MIN(txn_time) AS first_used
  FROM transactions GROUP BY user_id, device_id
)
SELECT t.*,
  CASE WHEN t.txn_time = fd.first_used THEN 1 ELSE 0 END AS is_new_device,
  CASE WHEN t.txn_time = fd.first_used
        AND t.amount > 500 THEN 1 ELSE 0 END              AS new_device_high_amount
FROM transactions t
JOIN first_device_use fd
  ON  t.user_id   = fd.user_id
  AND t.device_id = fd.device_id
WHERE t.txn_time = fd.first_used AND t.amount > 500
ORDER BY t.amount DESC;
```

---

## 5. Behavioral Z-Score Deviation

```sql
WITH user_baseline AS (
  SELECT user_id,
    AVG(amount)    AS avg_amount,
    STDDEV(amount) AS std_amount,
    COUNT(*)       AS total_txns
  FROM transactions
  WHERE txn_time < CURRENT_DATE - INTERVAL 1 DAY
  GROUP BY user_id
)
SELECT t.*,
  ROUND((t.amount - b.avg_amount) /
        NULLIF(b.std_amount, 0), 2)          AS amount_z_score,
  CASE
    WHEN (t.amount - b.avg_amount) /
         NULLIF(b.std_amount, 0) > 3 THEN '🔴 HIGH RISK'
    WHEN (t.amount - b.avg_amount) /
         NULLIF(b.std_amount, 0) > 2 THEN '🟡 MEDIUM RISK'
    ELSE '🟢 LOW RISK'
  END AS risk_level
FROM transactions t
JOIN user_baseline b ON t.user_id = b.user_id
WHERE t.txn_time >= CURRENT_DATE - INTERVAL 1 DAY
ORDER BY amount_z_score DESC;
```

---

## 6. Fraud Ring Detection

```sql
WITH shared_device_pairs AS (
  SELECT a.user_id AS user_a, b.user_id AS user_b,
    a.device_id AS identifier, 'device' AS link_type
  FROM transactions a
  JOIN transactions b ON a.device_id = b.device_id AND a.user_id < b.user_id
),
shared_ip_pairs AS (
  SELECT a.user_id, b.user_id,
    a.ip_address, 'ip'
  FROM transactions a
  JOIN transactions b ON a.ip_address = b.ip_address AND a.user_id < b.user_id
),
all_links AS (SELECT * FROM shared_device_pairs UNION ALL SELECT * FROM shared_ip_pairs)
SELECT user_a, user_b,
  COUNT(DISTINCT identifier) AS shared_identifiers,
  COUNT(DISTINCT link_type)  AS link_types,
  CASE WHEN COUNT(DISTINCT identifier) >= 2
       THEN '🔴 FRAUD RING' ELSE '🟡 INVESTIGATE' END AS ring_flag
FROM all_links
GROUP BY user_a, user_b
HAVING COUNT(DISTINCT identifier) >= 1
ORDER BY shared_identifiers DESC;
```

---

## 7. Card Testing Detection

```sql
-- Many micro-transactions in rapid succession
WITH sequential AS (
  SELECT user_id, txn_id, amount, txn_time,
    TIMESTAMPDIFF(SECOND,
      LAG(txn_time) OVER (PARTITION BY user_id ORDER BY txn_time),
      txn_time
    ) AS seconds_since_last
  FROM transactions
)
SELECT user_id,
  COUNT(*)                      AS rapid_txns,
  SUM(amount)                   AS total_amount,
  MIN(txn_time)                 AS window_start,
  MAX(txn_time)                 AS window_end,
  AVG(seconds_since_last)       AS avg_seconds_between
FROM sequential
WHERE seconds_since_last < 30   -- < 30 sec apart
  AND amount < 10               -- micro-transaction
GROUP BY user_id
HAVING COUNT(*) >= 5
ORDER BY rapid_txns DESC;
```

---

## 8. Composite Fraud Score (0–100)

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
    -- Velocity (max 25 pts)
    LEAST(25, COUNT(*) OVER (
      PARTITION BY t.user_id ORDER BY t.txn_time
      RANGE BETWEEN INTERVAL 1 HOUR PRECEDING AND CURRENT ROW
    ) * 5)                                            AS velocity_score,
    -- Amount z-score (max 25 pts)
    LEAST(25, GREATEST(0, ROUND(
      (t.amount - b.avg_amount) / NULLIF(b.std_amount, 0), 2) * 8
    ))                                                AS amount_score,
    -- New device (20 pts)
    CASE WHEN t.txn_time = fd.first_used THEN 20 ELSE 0 END AS new_device_score,
    -- Shared device (20 pts)
    CASE WHEN ds.device_id IS NOT NULL   THEN 20 ELSE 0 END AS shared_device_score,
    -- Odd hour (10 pts)
    CASE WHEN HOUR(t.txn_time) BETWEEN 1 AND 5 THEN 10 ELSE 0 END AS odd_hour_score
  FROM transactions t
  LEFT JOIN user_baseline    b  ON t.user_id   = b.user_id
  LEFT JOIN device_sharing   ds ON t.device_id = ds.device_id
  LEFT JOIN first_device_use fd ON t.user_id   = fd.user_id AND t.device_id = fd.device_id
  WHERE DATE(t.txn_time) = CURRENT_DATE
)
SELECT txn_id, user_id, amount, txn_time,
  velocity_score, amount_score, new_device_score,
  shared_device_score, odd_hour_score,
  velocity_score + amount_score + new_device_score +
  shared_device_score + odd_hour_score AS fraud_score,
  CASE
    WHEN velocity_score + amount_score + new_device_score +
         shared_device_score + odd_hour_score >= 70 THEN '🔴 BLOCK'
    WHEN velocity_score + amount_score + new_device_score +
         shared_device_score + odd_hour_score >= 40 THEN '🟡 REVIEW'
    ELSE '🟢 ALLOW'
  END AS decision
FROM signals
WHERE velocity_score + amount_score + new_device_score +
      shared_device_score + odd_hour_score >= 40
ORDER BY fraud_score DESC;
```

---

## 9. FAANG Patterns

```sql
-- Burst detection: daily Z-score vs user history
WITH daily_user AS (
  SELECT user_id, DATE(txn_time) AS dt,
    COUNT(*) AS daily_txns, SUM(amount) AS daily_amount
  FROM transactions GROUP BY user_id, DATE(txn_time)
),
baseline AS (
  SELECT user_id,
    AVG(daily_txns) AS avg_txns, STDDEV(daily_txns) AS std_txns
  FROM daily_user GROUP BY user_id
)
SELECT d.user_id, d.dt, d.daily_txns,
  ROUND((d.daily_txns - b.avg_txns) / NULLIF(b.std_txns, 0), 2) AS z_score,
  CASE WHEN (d.daily_txns - b.avg_txns) / NULLIF(b.std_txns, 0) > 3
       THEN '🔴 BURST' ELSE 'NORMAL' END AS burst_flag
FROM daily_user d JOIN baseline b ON d.user_id = b.user_id
WHERE (d.daily_txns - b.avg_txns) / NULLIF(b.std_txns, 0) > 3
ORDER BY z_score DESC;

-- Returns abuse (Amazon seller fraud)
SELECT seller_id,
  COUNT(*)                                             AS total_orders,
  ROUND(SUM(CASE WHEN status='returned' THEN 1 ELSE 0 END)*100.0/COUNT(*),2) AS return_rate,
  CASE WHEN SUM(CASE WHEN status='returned' THEN 1 ELSE 0 END)*100.0/COUNT(*) > 30
       THEN '⚠️  HIGH RETURN RATE' ELSE '✅ OK' END AS fraud_flag
FROM orders GROUP BY seller_id HAVING COUNT(*) >= 10
ORDER BY return_rate DESC;
```

---

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
