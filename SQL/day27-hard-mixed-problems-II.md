# Day 27 — Hard Mixed Problems II
**FAANG SQL 30-Day Prep**

---

## Problem 1 — Seat Gap Detection (Islands)

```sql
WITH RECURSIVE s(n) AS (
  SELECT 1 UNION ALL SELECT n+1 FROM s WHERE n < 500
),
available AS (
  SELECT n AS seat_number FROM s
  WHERE n NOT IN (SELECT seat_number FROM reservations WHERE show_id=42 AND status='booked')
),
gaps AS (
  SELECT seat_number,
    seat_number - ROW_NUMBER() OVER (ORDER BY seat_number) AS grp
  FROM available
)
SELECT MIN(seat_number) AS range_start, MAX(seat_number) AS range_end,
  COUNT(*) AS consecutive_seats
FROM gaps GROUP BY grp HAVING COUNT(*) >= 3 ORDER BY range_start;
-- Key: seat - ROW_NUMBER() = constant for consecutive sequences
```

---

## Problem 2 — Rolling 7-Day Unique Users

```sql
WITH dates AS (SELECT DISTINCT event_date FROM events)
SELECT d.event_date,
  COUNT(DISTINCT e.user_id) AS rolling_7d_unique_users
FROM dates d
JOIN events e ON e.event_date <= d.event_date
             AND e.event_date >= d.event_date - INTERVAL 6 DAY
GROUP BY d.event_date
ORDER BY d.event_date;
-- COUNT(DISTINCT) cannot go inside OVER() in MySQL → use self-join
```

---

## Problem 3 — Point-in-Time Price Join

```sql
WITH order_prices AS (
  SELECT o.order_id, o.product_id, o.quantity, o.order_date,
    p.price AS price_at_purchase, cp.current_price
  FROM orders o
  JOIN prices p ON p.product_id = o.product_id
    AND p.effective_date = (
      SELECT MAX(effective_date) FROM prices p2
      WHERE p2.product_id = o.product_id
        AND p2.effective_date <= o.order_date
    )
  JOIN (
    SELECT DISTINCT product_id,
      FIRST_VALUE(price) OVER (PARTITION BY product_id ORDER BY effective_date DESC) AS current_price
    FROM prices
  ) cp ON o.product_id = cp.product_id
)
SELECT product_id, COUNT(*) AS orders, SUM(quantity) AS units,
  ROUND(SUM(quantity*price_at_purchase),2) AS actual_revenue,
  ROUND(SUM(quantity*current_price),2)     AS current_price_revenue,
  ROUND(SUM(quantity*current_price)-SUM(quantity*price_at_purchase),2) AS delta
FROM order_prices GROUP BY product_id ORDER BY delta DESC;
-- Correlated subquery: MAX(effective_date) ≤ order_date per product
```

---

## Problem 4 — Longest Session Gap

```sql
WITH gaps AS (
  SELECT user_id, session_end AS gap_start,
    LEAD(session_start) OVER (PARTITION BY user_id ORDER BY session_start) AS gap_end,
    TIMESTAMPDIFF(HOUR, session_end,
      LEAD(session_start) OVER (PARTITION BY user_id ORDER BY session_start)) AS gap_hours
  FROM sessions
),
ranked AS (
  SELECT *,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY gap_hours DESC) AS rn
  FROM gaps WHERE gap_end IS NOT NULL AND gap_hours > 0
)
SELECT user_id, gap_start, gap_end, gap_hours
FROM ranked WHERE rn = 1 ORDER BY gap_hours DESC;
-- LEAD gives next session start; filter NULL for last session
```

---

## Problem 5 — Churn Prediction Feature Table (Point-in-Time Safe)

```sql
-- Key: define snapshot_date first, compute all features relative to it
-- "last 30d" = [snap-30d, snap), NOT [today-30d, today)
WITH snap AS (SELECT CURRENT_DATE - INTERVAL 90 DAY AS dt),
order_features AS (
  SELECT o.user_id,
    DATEDIFF(s.dt, MAX(o.order_date))           AS days_since_last_order,
    COUNT(CASE WHEN o.order_date >= s.dt - INTERVAL  30 DAY
               AND o.order_date <  s.dt THEN 1 END) AS orders_l30,
    COUNT(CASE WHEN o.order_date >= s.dt - INTERVAL  90 DAY
               AND o.order_date <  s.dt THEN 1 END) AS orders_l90,
    ROUND(AVG(o.amount),2) AS avg_order_value,
    SUM(CASE WHEN o.order_date >= s.dt-INTERVAL 30 DAY
             AND o.order_date <  s.dt THEN o.amount ELSE 0 END) AS spend_l30,
    SUM(CASE WHEN o.order_date >= s.dt-INTERVAL 60 DAY
             AND o.order_date <  s.dt-INTERVAL 30 DAY THEN o.amount ELSE 0 END) AS spend_prior_30
  FROM orders o CROSS JOIN snap s
  WHERE o.order_date < s.dt GROUP BY o.user_id, s.dt
),
session_features AS (
  SELECT se.user_id,
    COUNT(DISTINCT CASE WHEN se.session_date >= s.dt-INTERVAL 30 DAY
      AND se.session_date < s.dt THEN se.session_date END) AS session_days_l30
  FROM sessions se CROSS JOIN snap s
  WHERE se.session_date < s.dt GROUP BY se.user_id, s.dt
)
SELECT u.user_id,
  COALESCE(of.days_since_last_order,9999) AS days_since_last_order,
  COALESCE(of.orders_l30,0) AS orders_l30,
  COALESCE(of.orders_l90,0) AS orders_l90,
  COALESCE(of.avg_order_value,0) AS avg_order_value,
  COALESCE(sf.session_days_l30,0) AS session_days_l30,
  COALESCE(of.spend_l30,0) AS spend_l30,
  COALESCE(of.spend_prior_30,0) AS spend_prior_30,
  ROUND((COALESCE(of.spend_l30,0)-COALESCE(of.spend_prior_30,0))*100.0/
    NULLIF(of.spend_prior_30,0),2) AS spend_trend_pct,
  CASE WHEN COALESCE(of.days_since_last_order,9999) > 60 THEN 1 ELSE 0 END AS churned_label
FROM users u
LEFT JOIN order_features   of ON u.user_id = of.user_id
LEFT JOIN session_features sf ON u.user_id = sf.user_id
ORDER BY churned_label DESC, days_since_last_order DESC;
```

---

## Problem 6 — Cheapest Fulfillable Warehouse

```sql
WITH options AS (
  SELECT o.order_id, o.product_id, o.quantity, i.warehouse_id,
    i.units_on_hand, w.ship_cost_per_unit,
    o.quantity * w.ship_cost_per_unit AS total_cost,
    CASE WHEN i.units_on_hand >= o.quantity THEN 1 ELSE 0 END AS can_fulfill,
    ROW_NUMBER() OVER (
      PARTITION BY o.order_id
      ORDER BY CASE WHEN i.units_on_hand >= o.quantity THEN 0 ELSE 1 END,
               o.quantity * w.ship_cost_per_unit ASC
    ) AS rn
  FROM orders o
  JOIN inventory  i ON o.product_id   = i.product_id
  JOIN warehouses w ON i.warehouse_id = w.warehouse_id
  WHERE o.status = 'unshipped'
)
SELECT order_id, product_id, quantity, warehouse_id,
  units_on_hand, total_cost,
  CASE WHEN can_fulfill=1 THEN '✅ Fulfillable' ELSE '🔴 Insufficient Stock' END AS status
FROM options WHERE rn = 1
ORDER BY status DESC, total_cost;
-- ORDER BY: fulfillable first (0 before 1), then cheapest within
```

---

## Problem 7 — AB Test Contamination Filter

```sql
WITH overlapping AS (
  SELECT DISTINCT a.user_id FROM assignments a
  JOIN assignments b ON a.user_id=b.user_id
    AND a.experiment_id != b.experiment_id
    AND a.assigned_at <= COALESCE(b.next_assigned_at, b.assigned_at + INTERVAL 30 DAY)
    AND b.assigned_at <= COALESCE(a.next_assigned_at, a.assigned_at + INTERVAL 30 DAY)
),
clean AS (SELECT user_id FROM assignments WHERE user_id NOT IN (SELECT user_id FROM overlapping))
SELECT a.experiment_id, a.variant,
  COUNT(DISTINCT a.user_id) AS users,
  ROUND(COUNT(DISTINCT CASE WHEN e.event_type='purchase' THEN a.user_id END)*100.0/
    COUNT(DISTINCT a.user_id),4) AS cvr
FROM assignments a
JOIN clean c ON a.user_id=c.user_id
LEFT JOIN events e ON a.user_id=e.user_id AND e.event_time >= a.assigned_at
GROUP BY a.experiment_id, a.variant ORDER BY experiment_id, variant;
```

---

## Problem 8 — Top N Per Group With Ties

```sql
WITH regional AS (
  SELECT salesperson_id, region, SUM(amount) AS revenue
  FROM sales WHERE sale_date BETWEEN '2025-01-01' AND '2025-03-31'
  GROUP BY salesperson_id, region
)
SELECT salesperson_id, region, revenue,
  ROUND(revenue*100.0/SUM(revenue) OVER (PARTITION BY region),2) AS pct_of_region,
  DENSE_RANK() OVER (PARTITION BY region ORDER BY revenue DESC) AS rank
FROM regional
WHERE DENSE_RANK() OVER (PARTITION BY region ORDER BY revenue DESC) <= 3
ORDER BY region, rank;
-- DENSE_RANK not RANK: no gaps after ties → WHERE <= 3 includes all tied 3rd-placers
```

---

## Practice Questions & Answers

### Q1 ✅ — Point-in-time price join

```sql
WITH order_prices AS (
  SELECT o.order_id, o.product_id, o.quantity, o.order_date,
    p.price AS price_at_time
  FROM orders o
  JOIN prices p ON p.product_id = o.product_id
    AND p.start_date <= o.order_date
    AND (p.end_date >= o.order_date OR p.end_date IS NULL)
)
SELECT product_id, COUNT(*) AS orders, SUM(quantity) AS units_sold,
  ROUND(SUM(quantity * price_at_time),2) AS total_revenue,
  ROUND(AVG(price_at_time),2) AS avg_price_paid
FROM order_prices GROUP BY product_id ORDER BY total_revenue DESC;
```

### Q2 ✅ — Weekly engagement score + peak flag

```sql
WITH weekly AS (
  SELECT user_id,
    DATE_SUB(activity_date, INTERVAL DAYOFWEEK(activity_date)-1 DAY) AS week_start,
    SUM(CASE WHEN activity_type='login'   THEN 1
             WHEN activity_type='post'    THEN 3
             WHEN activity_type='comment' THEN 2
             WHEN activity_type='share'   THEN 4 ELSE 0 END) AS weekly_score
  FROM user_activity GROUP BY user_id, week_start
),
with_peak AS (
  SELECT *,
    MAX(weekly_score) OVER (PARTITION BY user_id) AS peak_score,
    MAX(week_start)   OVER (PARTITION BY user_id) AS latest_week,
    FIRST_VALUE(weekly_score) OVER (PARTITION BY user_id ORDER BY week_start DESC) AS latest_score
  FROM weekly
),
peak_week AS (
  SELECT user_id, week_start AS peak_week
  FROM (
    SELECT user_id, week_start,
      ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY weekly_score DESC) AS rn
    FROM weekly
  ) t WHERE rn = 1
)
SELECT DISTINCT w.user_id, pw.peak_week, w.peak_score,
  w.latest_week, w.latest_score,
  ROUND(w.latest_score*100.0/NULLIF(w.peak_score,0),2) AS pct_of_peak,
  CASE WHEN w.latest_score < w.peak_score*0.5 THEN '🔴 DECLINING' ELSE '✅ ENGAGED' END AS flag
FROM with_peak w JOIN peak_week pw ON w.user_id=pw.user_id
ORDER BY pct_of_peak ASC;
```

### Q3 ✅ — Category gross margin + MoM flag

```sql
WITH monthly AS (
  SELECT p.category, DATE_FORMAT(o.order_date,'%Y-%m-01') AS month,
    SUM(oi.quantity*oi.unit_price) AS revenue,
    SUM(oi.quantity*p.cost_price)  AS total_cost,
    SUM(oi.quantity*(oi.unit_price-p.cost_price)) AS gross_profit
  FROM orders o
  JOIN order_items oi ON o.order_id=oi.order_id
  JOIN products    p  ON oi.product_id=p.product_id
  WHERE o.status NOT IN ('returned','cancelled')
  GROUP BY p.category, DATE_FORMAT(o.order_date,'%Y-%m-01')
),
with_margin AS (
  SELECT *,
    ROUND(gross_profit*100.0/NULLIF(revenue,0),2) AS gross_margin_pct,
    ROUND(gross_profit*100.0/NULLIF(SUM(gross_profit) OVER (PARTITION BY month),0),2) AS contribution_pct
  FROM monthly
),
with_mom AS (
  SELECT *,
    LAG(gross_margin_pct) OVER (PARTITION BY category ORDER BY month) AS prev_margin_pct
  FROM with_margin
)
SELECT category, month, ROUND(revenue,2) AS revenue,
  ROUND(total_cost,2) AS cost, ROUND(gross_profit,2) AS gross_profit,
  gross_margin_pct, prev_margin_pct, contribution_pct,
  ROUND(gross_margin_pct-prev_margin_pct,2) AS margin_pp_change,
  CASE WHEN gross_margin_pct-prev_margin_pct < -5 THEN '🔴 MARGIN DROP' ELSE '✅ STABLE' END AS flag
FROM with_mom ORDER BY month, gross_profit DESC;
-- margin_pp_change is in PERCENTAGE POINTS not relative % — intentional
```

---

## Key Patterns This Day

| Problem | Core Technique |
|---|---|
| Seat gaps | `seat - ROW_NUMBER() = constant` for consecutive groups |
| Rolling 7d unique | Self-join on date range (COUNT DISTINCT ≠ window function) |
| Point-in-time price | Correlated subquery: `MAX(effective_date) ≤ order_date` |
| Longest idle gap | `LEAD(next_start) - session_end`, ROW_NUMBER for max per user |
| Churn features | CROSS JOIN snapshot_date, all windows relative to snap |
| Warehouse routing | ORDER BY can_fulfill first, then cost → ROW_NUMBER rn=1 |
| AB contamination | Self-join assignments on overlapping time windows |
| Top-N with ties | `DENSE_RANK` not `RANK` — no skipped numbers after ties |
| Margin drop | Compare in **percentage points** (pp), not relative % |
| Peak detection | `MAX() OVER` for peak, `FIRST_VALUE ORDER BY DESC` for latest |

---

*Day 27 complete — 3 days to go 🚀*
