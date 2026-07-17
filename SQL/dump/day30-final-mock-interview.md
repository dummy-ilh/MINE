# Day 30 — Final Mock Interview II
**FAANG SQL 30-Day Prep**

---

## Question 1 — Rep Best Month

```sql
WITH monthly AS (
  SELECT rep_id, region,
    DATE_FORMAT(sale_date,'%Y-%m-01') AS month,
    SUM(amount) AS revenue
  FROM sales GROUP BY rep_id, region, DATE_FORMAT(sale_date,'%Y-%m-01')
),
with_prev AS (
  SELECT *,
    LAG(revenue) OVER (PARTITION BY rep_id ORDER BY month) AS prev_revenue,
    RANK() OVER (PARTITION BY rep_id ORDER BY revenue DESC) AS rk
  FROM monthly
),
best AS (SELECT * FROM with_prev WHERE rk = 1),
region_avg AS (
  SELECT region, month, AVG(revenue) AS avg_rev FROM monthly GROUP BY region, month
),
region_rank AS (
  SELECT rep_id, month,
    RANK() OVER (PARTITION BY region, month ORDER BY revenue DESC) AS rank_in_region
  FROM monthly
)
SELECT bm.rep_id, bm.region, bm.month AS best_month,
  ROUND(bm.revenue,2) AS best_month_revenue,
  rr.rank_in_region,
  ROUND((bm.revenue-bm.prev_revenue)*100.0/NULLIF(bm.prev_revenue,0),2) AS mom_growth_pct,
  CASE WHEN bm.revenue > ra.avg_rev THEN 'Above Region Avg' ELSE 'Below Region Avg' END AS vs_region
FROM best bm
JOIN region_avg  ra ON bm.region=ra.region AND bm.month=ra.month
JOIN region_rank rr ON bm.rep_id=rr.rep_id AND bm.month=rr.month
ORDER BY bm.rep_id;
-- RANK not ROW_NUMBER → ties in best month handled correctly
-- MoM NULL when prev_revenue IS NULL (single month) — handled by NULLIF + LAG returning NULL
```

---

## Question 2 — Doctor Performance

```sql
WITH completed AS (
  SELECT a.*, b.amount_charged, b.amount_paid,
    b.amount_charged - b.amount_paid AS outstanding
  FROM appointments a
  LEFT JOIN billing b ON a.appt_id=b.appt_id
  WHERE a.status='completed'
    AND a.appt_date >= CURRENT_DATE - INTERVAL 90 DAY
),
stats AS (
  SELECT doctor_id,
    COUNT(*) AS total_appts,
    COUNT(DISTINCT patient_id) AS unique_patients,
    ROUND(SUM(amount_paid)*100.0/NULLIF(SUM(amount_charged),0),2) AS collection_rate,
    ROUND(SUM(CASE WHEN appointment_type='follow_up' THEN 1 ELSE 0 END)*100.0/COUNT(*),2) AS followup_pct,
    MAX(CASE WHEN outstanding > 500 THEN 1 ELSE 0 END) AS has_large_outstanding
  FROM completed GROUP BY doctor_id HAVING COUNT(*) >= 10
),
busiest AS (
  SELECT doctor_id, DAYNAME(appt_date) AS dow,
    ROW_NUMBER() OVER (PARTITION BY doctor_id ORDER BY COUNT(*) DESC) AS rn
  FROM completed GROUP BY doctor_id, DAYNAME(appt_date)
)
SELECT d.doctor_id, d.name, d.specialty,
  s.total_appts, s.unique_patients, s.collection_rate, s.followup_pct,
  b.dow AS busiest_day,
  CASE WHEN s.has_large_outstanding=1 THEN '⚠️ Yes' ELSE '✅ No' END AS outstanding_flag
FROM stats s
JOIN doctors d ON s.doctor_id=d.doctor_id
JOIN busiest b ON s.doctor_id=b.doctor_id AND b.rn=1
ORDER BY s.total_appts DESC;
-- LEFT JOIN billing → appointments without billing records still included
-- HAVING BEFORE joining doctors → filter early
```

---

## Question 3 — Full User Lifecycle by Channel

```sql
WITH base AS (SELECT user_id, signup_date, acquisition_channel FROM users),
totals AS (SELECT acquisition_channel, COUNT(*) AS total_users FROM base GROUP BY 1),
activation AS (
  SELECT u.acquisition_channel, COUNT(DISTINCT e.user_id) AS activated
  FROM base u JOIN events e ON u.user_id=e.user_id
    AND e.event_type='purchase' AND e.event_time <= u.signup_date+INTERVAL 7 DAY
  GROUP BY u.acquisition_channel
),
d30 AS (
  SELECT u.acquisition_channel, COUNT(DISTINCT e.user_id) AS d30_users
  FROM base u JOIN events e ON u.user_id=e.user_id
    AND e.event_time BETWEEN u.signup_date+INTERVAL 28 DAY AND u.signup_date+INTERVAL 30 DAY
  GROUP BY u.acquisition_channel
),
ltv AS (
  SELECT u.acquisition_channel,
    ROUND(SUM(COALESCE(r.amount,0))/COUNT(DISTINCT u.user_id),2) AS avg_ltv
  FROM base u LEFT JOIN revenue r ON u.user_id=r.user_id GROUP BY u.acquisition_channel
),
payback AS (
  SELECT u.acquisition_channel,
    ROUND(AVG(DATEDIFF(fr.first_rev, u.signup_date)),1) AS avg_payback_days
  FROM base u
  JOIN (SELECT user_id, MIN(revenue_date) AS first_rev FROM revenue GROUP BY user_id) fr
    ON u.user_id=fr.user_id
  GROUP BY u.acquisition_channel
),
top_prod AS (
  SELECT acquisition_channel, product_id FROM (
    SELECT u.acquisition_channel, r.product_id,
      ROW_NUMBER() OVER (PARTITION BY u.acquisition_channel ORDER BY SUM(r.amount) DESC) AS rn
    FROM base u JOIN revenue r ON u.user_id=r.user_id
    GROUP BY u.acquisition_channel, r.product_id
  ) t WHERE rn=1
),
churned AS (
  SELECT u.acquisition_channel, COUNT(DISTINCT u.user_id) AS churned
  FROM base u
  WHERE NOT EXISTS (
    SELECT 1 FROM events e WHERE e.user_id=u.user_id
      AND e.event_time >= CURRENT_DATE - INTERVAL 30 DAY
  )
  GROUP BY u.acquisition_channel
)
SELECT ct.acquisition_channel, ct.total_users,
  ROUND(COALESCE(ac.activated,0)*100.0/ct.total_users,2) AS activation_rate_pct,
  ROUND(COALESCE(d30.d30_users,0)*100.0/ct.total_users,2) AS d30_retention_pct,
  COALESCE(lv.avg_ltv,0) AS avg_ltv,
  pb.avg_payback_days,
  tp.product_id AS top_product,
  ROUND(COALESCE(ch.churned,0)*100.0/ct.total_users,2) AS churn_rate_pct
FROM totals ct
LEFT JOIN activation ac ON ct.acquisition_channel=ac.acquisition_channel
LEFT JOIN d30         ON ct.acquisition_channel=d30.acquisition_channel
LEFT JOIN ltv lv      ON ct.acquisition_channel=lv.acquisition_channel
LEFT JOIN payback pb  ON ct.acquisition_channel=pb.acquisition_channel
LEFT JOIN top_prod tp ON ct.acquisition_channel=tp.acquisition_channel
LEFT JOIN churned ch  ON ct.acquisition_channel=ch.acquisition_channel
ORDER BY avg_ltv DESC;
-- LTV = SUM(revenue) / ALL users (including $0) → correct denominator
-- NOT EXISTS for churn → clean, handles users with zero events naturally
-- ALL JOINs to totals are LEFT → no channel dropped if metric is zero
```

---

## Question 4 — Portfolio P&L Engine

```sql
WITH latest_price AS (
  -- Most recent price per ticker across ALL portfolios
  SELECT t1.ticker, t1.price AS current_price
  FROM trades t1
  WHERE t1.trade_date = (SELECT MAX(t2.trade_date) FROM trades t2 WHERE t2.ticker=t1.ticker)
  GROUP BY t1.ticker, t1.price
),
buys AS (
  SELECT portfolio_id, ticker,
    SUM(quantity) AS buy_qty,
    SUM(quantity*price) AS buy_cost,
    ROUND(SUM(quantity*price)/NULLIF(SUM(quantity),0),4) AS avg_cost_basis
  FROM trades WHERE trade_type='buy' GROUP BY portfolio_id, ticker
),
sells AS (
  SELECT portfolio_id, ticker,
    SUM(quantity) AS sell_qty,
    SUM(quantity*price) AS sell_revenue,
    ROUND(SUM(quantity*price)/NULLIF(SUM(quantity),0),4) AS avg_sell_price
  FROM trades WHERE trade_type='sell' GROUP BY portfolio_id, ticker
),
holdings AS (
  SELECT b.portfolio_id, b.ticker, b.avg_cost_basis,
    b.buy_qty - COALESCE(s.sell_qty,0) AS net_qty,
    b.buy_cost AS total_buy_cost,
    COALESCE(s.sell_qty,0) AS sold_qty,
    COALESCE(s.avg_sell_price,0) AS avg_sell_price
  FROM buys b LEFT JOIN sells s ON b.portfolio_id=s.portfolio_id AND b.ticker=s.ticker
),
pnl AS (
  SELECT h.portfolio_id, h.ticker, h.net_qty, h.avg_cost_basis,
    lp.current_price,
    ROUND(h.net_qty * lp.current_price,2) AS current_value,
    ROUND((lp.current_price - h.avg_cost_basis) * h.net_qty,2) AS unrealized_pnl,
    ROUND((h.avg_sell_price - h.avg_cost_basis) * h.sold_qty,2) AS realized_pnl,
    ROUND(h.avg_cost_basis * h.net_qty,2) AS total_cost_basis
  FROM holdings h JOIN latest_price lp ON h.ticker=lp.ticker
  WHERE h.net_qty > 0 OR h.sold_qty > 0
),
port_totals AS (
  SELECT portfolio_id,
    SUM(total_cost_basis) AS port_cost,
    SUM(unrealized_pnl)   AS port_unrealized_pnl
  FROM pnl GROUP BY portfolio_id
)
SELECT pnl.portfolio_id, po.user_id, pnl.ticker,
  pnl.net_qty, pnl.avg_cost_basis, pnl.current_price,
  pnl.current_value, pnl.unrealized_pnl, pnl.realized_pnl,
  pnl.total_cost_basis,
  ROUND(pnl.unrealized_pnl*100.0/NULLIF(pnl.total_cost_basis,0),2) AS unrealized_pnl_pct,
  CASE WHEN pt.port_unrealized_pnl < -(pt.port_cost*0.20)
       THEN '🔴 LOSS > 20%' ELSE '✅ OK' END AS portfolio_loss_flag
FROM pnl
JOIN portfolios   po ON pnl.portfolio_id=po.portfolio_id
JOIN port_totals  pt ON pnl.portfolio_id=pt.portfolio_id
ORDER BY pnl.unrealized_pnl ASC;
-- Cost basis: buys ONLY (sell_type excluded) → correct weighted avg
-- Current price: latest across ALL portfolios, not just this one
-- Loss flag: evaluated at PORTFOLIO level (sum), not ticker level
```

---

## Common Final-Round Traps

| Q | Trap | Correct |
|---|---|---|
| 1 | ROW_NUMBER for best month | RANK → ties handled correctly |
| 1 | NULL MoM for 1-month reps | LAG returns NULL → NULLIF propagates it |
| 2 | JOIN billing (not LEFT) | LEFT JOIN → unpaid appts still included |
| 3 | LTV = revenue / buyers | LTV = revenue / ALL users incl $0 |
| 3 | INNER JOIN for churn | NOT EXISTS across all channel users |
| 4 | All trades for cost basis | Only `trade_type='buy'` for avg cost |
| 4 | Latest price within portfolio | Latest price across ALL portfolios |
| 4 | Unrealized flag per ticker | Flag at portfolio total level |

---

## 30-Day Readiness Assessment

### Concepts Mastered ✅
- Window functions: ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD, FIRST_VALUE, LAST_VALUE
- Aggregation: ROLLUP, CUBE, GROUPING SETS, GROUPING()
- CTEs: simple, chained, recursive
- Joins: all types, self-join, range joins, point-in-time joins
- Analytics: cohort retention, funnels, A/B testing, fraud detection, recommendations
- ML patterns: feature engineering, target encoding, data leakage prevention
- Graph queries: mutual friends, degree of separation, triangle counting
- Time series: gaps & islands, date spines, MAT, rolling windows
- SaaS metrics: MRR waterfall, NRR, churn, LTV, payback period
- Company-specific: Google (CTR/nDCG/DAU), Meta (PYMK/K-factor), Amazon (RFM/OTD/GMV)

### Interview Readiness: 🟢 READY

---

*Day 30 complete. 30/30. Go get that offer. 🚀*
