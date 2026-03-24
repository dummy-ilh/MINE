# Day 16 — Time Series Analysis & Forecasting in SQL
**FAANG SQL 30-Day Prep**

---

## Concepts Covered

1. Time Series Fundamentals & Date Spine
2. Trend Detection with Moving Averages
3. Seasonality Detection
4. Anomaly Detection (Global & Rolling Z-Score)
5. WoW / YoY Comparisons
6. Forecasting with Weighted Moving Averages
7. Trend Decomposition
8. Event Impact Analysis
9. FAANG Patterns (Spike Detection, MAPE)

---

## 1. Date Spine — Always Start Here

```sql
WITH RECURSIVE dates AS (
  SELECT '2025-01-01' AS dt
  UNION ALL
  SELECT dt + INTERVAL 1 DAY FROM dates WHERE dt < '2025-12-31'
),
daily_revenue AS (
  SELECT DATE(order_date) AS dt, SUM(amount) AS revenue
  FROM orders GROUP BY DATE(order_date)
)
SELECT d.dt, COALESCE(r.revenue, 0) AS revenue
FROM dates d
LEFT JOIN daily_revenue r ON d.dt = r.dt
ORDER BY d.dt;
```

---

## 2. Trend Detection

```sql
WITH daily AS (
  SELECT DATE(order_date) AS dt, SUM(amount) AS revenue
  FROM orders GROUP BY DATE(order_date)
)
SELECT dt, revenue,
  AVG(revenue) OVER (ORDER BY dt ROWS BETWEEN  6 PRECEDING AND CURRENT ROW) AS ma_7d,
  AVG(revenue) OVER (ORDER BY dt ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS ma_30d,
  AVG(revenue) OVER (ORDER BY dt ROWS BETWEEN 89 PRECEDING AND CURRENT ROW) AS ma_90d,
  CASE
    WHEN AVG(revenue) OVER (ORDER BY dt ROWS BETWEEN  6 PRECEDING AND CURRENT ROW)
       > AVG(revenue) OVER (ORDER BY dt ROWS BETWEEN 29 PRECEDING AND CURRENT ROW)
    THEN 'Uptrend' ELSE 'Downtrend'
  END AS trend_signal
FROM daily ORDER BY dt;
```

---

## 3. Seasonality Detection

```sql
-- Day of week seasonality
SELECT DAYNAME(order_date) AS day_of_week,
  AVG(daily_revenue) AS avg_revenue,
  STDDEV(daily_revenue) AS stddev_revenue
FROM (
  SELECT DATE(order_date) AS order_date, SUM(amount) AS daily_revenue
  FROM orders GROUP BY DATE(order_date)
) daily
GROUP BY DAYNAME(order_date), DAYOFWEEK(order_date)
ORDER BY DAYOFWEEK(order_date);

-- Monthly seasonality index (> 1 = above average month)
WITH monthly AS (
  SELECT MONTH(order_date) AS mo, MONTHNAME(order_date) AS month_name,
    AVG(monthly_revenue) AS avg_monthly_revenue
  FROM (
    SELECT DATE_FORMAT(order_date,'%Y-%m') AS ym,
      MONTH(order_date) AS mo, MONTHNAME(order_date) AS month_name,
      SUM(amount) AS monthly_revenue
    FROM orders
    GROUP BY DATE_FORMAT(order_date,'%Y-%m'), MONTH(order_date), MONTHNAME(order_date)
  ) t GROUP BY mo, month_name
),
overall_avg AS (SELECT AVG(avg_monthly_revenue) AS grand_avg FROM monthly)
SELECT m.mo, m.month_name,
  ROUND(m.avg_monthly_revenue, 2)               AS avg_revenue,
  ROUND(m.avg_monthly_revenue / o.grand_avg, 4) AS seasonality_index
FROM monthly m CROSS JOIN overall_avg o ORDER BY m.mo;
```

---

## 4. Anomaly Detection

```sql
-- Global Z-score
WITH daily AS (
  SELECT DATE(order_date) AS dt, SUM(amount) AS revenue
  FROM orders GROUP BY DATE(order_date)
),
stats AS (SELECT AVG(revenue) AS mean_rev, STDDEV(revenue) AS std_rev FROM daily)
SELECT d.dt, d.revenue,
  ROUND((d.revenue - s.mean_rev) / NULLIF(s.std_rev, 0), 2) AS z_score,
  CASE WHEN ABS((d.revenue - s.mean_rev) /
    NULLIF(s.std_rev, 0)) > 2 THEN 'Anomaly' ELSE 'Normal' END AS flag
FROM daily d CROSS JOIN stats s ORDER BY ABS(z_score) DESC;

-- Rolling Z-score (local context — no future data leak)
WITH daily AS (
  SELECT DATE(order_date) AS dt, SUM(amount) AS revenue
  FROM orders GROUP BY DATE(order_date)
),
rolling_stats AS (
  SELECT dt, revenue,
    AVG(revenue)    OVER (ORDER BY dt ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING) AS rolling_mean,
    STDDEV(revenue) OVER (ORDER BY dt ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING) AS rolling_std
  FROM daily
)
SELECT dt, revenue,
  ROUND(rolling_mean, 2) AS rolling_mean,
  ROUND((revenue - rolling_mean) / NULLIF(rolling_std, 0), 2) AS z_score,
  CASE WHEN ABS((revenue - rolling_mean) /
    NULLIF(rolling_std, 0)) > 2 THEN 'Anomaly' ELSE 'Normal' END AS flag
FROM rolling_stats ORDER BY dt;
```

---

## 5. WoW / YoY Comparisons

```sql
WITH daily AS (
  SELECT DATE(order_date) AS dt, SUM(amount) AS revenue
  FROM orders GROUP BY DATE(order_date)
)
SELECT dt, revenue,
  LAG(revenue,   1) OVER (ORDER BY dt) AS prev_day,
  ROUND((revenue - LAG(revenue,1)   OVER (ORDER BY dt)) * 100.0 /
    NULLIF(LAG(revenue,1)   OVER (ORDER BY dt), 0), 2) AS dod_pct,
  LAG(revenue,   7) OVER (ORDER BY dt) AS prev_week,
  ROUND((revenue - LAG(revenue,7)   OVER (ORDER BY dt)) * 100.0 /
    NULLIF(LAG(revenue,7)   OVER (ORDER BY dt), 0), 2) AS wow_pct,
  LAG(revenue, 364) OVER (ORDER BY dt) AS prev_year,
  ROUND((revenue - LAG(revenue,364) OVER (ORDER BY dt)) * 100.0 /
    NULLIF(LAG(revenue,364) OVER (ORDER BY dt), 0), 2) AS yoy_pct
FROM daily ORDER BY dt;
-- LAG(364) preserves same day-of-week vs LAG(365)
```

---

## 6. Forecasting with EWMA

```sql
WITH daily AS (
  SELECT DATE(order_date) AS dt, SUM(amount) AS revenue
  FROM orders GROUP BY DATE(order_date)
)
SELECT dt, revenue,
  AVG(revenue) OVER (ORDER BY dt ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS ma_7d,
  (revenue * 4 +
   LAG(revenue,1) OVER (ORDER BY dt) * 3 +
   LAG(revenue,2) OVER (ORDER BY dt) * 2 +
   LAG(revenue,3) OVER (ORDER BY dt) * 1
  ) / 10.0 AS ewma_forecast
FROM daily ORDER BY dt;
```

---

## 7. Trend Decomposition

```sql
-- signal = trend × seasonal_index × residual
WITH daily AS (
  SELECT DATE(order_date) AS dt, DAYOFWEEK(order_date) AS dow,
    SUM(amount) AS revenue
  FROM orders GROUP BY DATE(order_date), DAYOFWEEK(order_date)
),
trend AS (
  SELECT dt, dow, revenue,
    AVG(revenue) OVER (ORDER BY dt ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING) AS trend_component
  FROM daily
),
seasonality AS (SELECT dow, AVG(revenue) AS avg_dow FROM daily GROUP BY dow),
overall_avg AS (SELECT AVG(revenue) AS grand_avg FROM daily)
SELECT t.dt, t.revenue,
  ROUND(t.trend_component, 2)               AS trend,
  ROUND(s.avg_dow / o.grand_avg, 4)         AS seasonal_index,
  ROUND(t.revenue / NULLIF(t.trend_component *
    (s.avg_dow / o.grand_avg), 0), 4)       AS residual
FROM trend t
JOIN seasonality s ON t.dow = s.dow
CROSS JOIN overall_avg o ORDER BY t.dt;
```

---

## 8. Event Impact Analysis

```sql
WITH event_windows AS (
  SELECT e.event_id, e.event_name, e.event_date, e.category,
    SUM(CASE WHEN o.order_date BETWEEN e.event_date - INTERVAL 14 DAY
      AND e.event_date - INTERVAL 1 DAY THEN o.amount END)     AS pre_revenue,
    COUNT(DISTINCT CASE WHEN o.order_date BETWEEN e.event_date - INTERVAL 14 DAY
      AND e.event_date - INTERVAL 1 DAY THEN o.order_date END) AS pre_days,
    SUM(CASE WHEN o.order_date BETWEEN e.event_date + INTERVAL 1 DAY
      AND e.event_date + INTERVAL 14 DAY THEN o.amount END)    AS post_revenue,
    COUNT(DISTINCT CASE WHEN o.order_date BETWEEN e.event_date + INTERVAL 1 DAY
      AND e.event_date + INTERVAL 14 DAY THEN o.order_date END) AS post_days
  FROM marketing_events e
  JOIN orders o ON o.category = e.category
  GROUP BY e.event_id, e.event_name, e.event_date, e.category
),
impact AS (
  SELECT *,
    ROUND(pre_revenue  / NULLIF(pre_days,  0), 2) AS avg_daily_pre,
    ROUND(post_revenue / NULLIF(post_days, 0), 2) AS avg_daily_post
  FROM event_windows
)
SELECT event_name, event_date, category,
  avg_daily_pre, avg_daily_post,
  ROUND(avg_daily_post - avg_daily_pre, 2)               AS absolute_lift,
  ROUND((avg_daily_post - avg_daily_pre) * 100.0 /
    NULLIF(avg_daily_pre, 0), 2)                         AS pct_lift,
  CASE WHEN (avg_daily_post - avg_daily_pre) * 100.0 /
    NULLIF(avg_daily_pre, 0) > 10 THEN 'High Impact' ELSE 'Normal' END AS impact_flag
FROM impact ORDER BY pct_lift DESC;
```

---

## 9. FAANG Patterns

```sql
-- Spike detection (Google)
WITH hourly AS (
  SELECT DATE(event_time) AS dt, HOUR(event_time) AS hr, COUNT(*) AS searches
  FROM search_events GROUP BY DATE(event_time), HOUR(event_time)
),
baseline AS (SELECT hr, AVG(searches) AS avg_searches FROM hourly GROUP BY hr)
SELECT h.dt, h.hr, h.searches,
  ROUND(h.searches / NULLIF(b.avg_searches, 0), 2) AS vs_avg_ratio,
  CASE WHEN h.searches > 3 * b.avg_searches THEN 'Spike' ELSE 'Normal' END AS status
FROM hourly h JOIN baseline b ON h.hr = b.hr
ORDER BY vs_avg_ratio DESC;

-- Forecast accuracy: MAPE, MAE, RMSE (Amazon)
SELECT product_id,
  ROUND(AVG(ABS(actual - forecasted) / NULLIF(actual, 0)) * 100, 2) AS mape_pct,
  ROUND(AVG(ABS(actual - forecasted)), 2)                           AS mae,
  ROUND(SQRT(AVG(POW(actual - forecasted, 2))), 2)                  AS rmse
FROM demand_forecasts GROUP BY product_id ORDER BY mape_pct;
```

---

## Practice Questions
🟢 Q1 — Easy

Table: daily_sales(sale_date, product_id, revenue)
For each product show daily revenue, 7-day moving average, and a flag above_ma (1 if revenue > 7d MA, else 0). Only include 2025 data.


🟡 Q2 — Medium

Same table.
Detect anomaly days per product using rolling 30-day z-score (using only prior 30 days for mean/std). Flag days where local z-score > 2. Return date, product, revenue, rolling mean, z-score, flag.


🔴 Q3 — Hard

Tables: orders(order_id, order_date, amount, category), marketing_events(event_id, event_date, event_name, category)
For each marketing event, measure revenue impact: compare avg daily revenue in the 14 days before vs 14 days after the event for the relevant category. Show absolute and % lift, and flag events where lift > 10
### Q1 — Easy ✅
```sql
WITH daily AS (
  SELECT sale_date, product_id, revenue,
    AVG(revenue) OVER (
      PARTITION BY product_id ORDER BY sale_date
      ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS ma_7d
  FROM daily_sales WHERE YEAR(sale_date) = 2025
)
SELECT sale_date, product_id, revenue,
  ROUND(ma_7d, 2) AS ma_7d,
  CASE WHEN revenue > ma_7d THEN 1 ELSE 0 END AS above_ma
FROM daily ORDER BY product_id, sale_date;
```

### Q2 — Medium ✅
```sql
WITH daily AS (
  SELECT sale_date, product_id, revenue
  FROM daily_sales WHERE YEAR(sale_date) = 2025
),
rolling_stats AS (
  SELECT sale_date, product_id, revenue,
    AVG(revenue)    OVER (PARTITION BY product_id ORDER BY sale_date
      ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING) AS rolling_mean,
    STDDEV(revenue) OVER (PARTITION BY product_id ORDER BY sale_date
      ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING) AS rolling_std
  FROM daily
)
SELECT sale_date, product_id, revenue,
  ROUND(rolling_mean, 2) AS rolling_mean,
  ROUND(rolling_std,  2) AS rolling_std,
  ROUND((revenue - rolling_mean) / NULLIF(rolling_std, 0), 2) AS z_score,
  CASE WHEN ABS((revenue - rolling_mean) /
    NULLIF(rolling_std, 0)) > 2 THEN 'Anomaly' ELSE 'Normal' END AS anomaly_flag
FROM rolling_stats ORDER BY product_id, sale_date;
```

### Q3 — Hard ✅
```sql
WITH event_windows AS (
  SELECT e.event_id, e.event_name, e.event_date, e.category,
    SUM(CASE WHEN o.order_date BETWEEN e.event_date - INTERVAL 14 DAY
      AND e.event_date - INTERVAL 1 DAY THEN o.amount END)     AS pre_revenue,
    COUNT(DISTINCT CASE WHEN o.order_date BETWEEN e.event_date - INTERVAL 14 DAY
      AND e.event_date - INTERVAL 1 DAY THEN o.order_date END) AS pre_days,
    SUM(CASE WHEN o.order_date BETWEEN e.event_date + INTERVAL 1 DAY
      AND e.event_date + INTERVAL 14 DAY THEN o.amount END)    AS post_revenue,
    COUNT(DISTINCT CASE WHEN o.order_date BETWEEN e.event_date + INTERVAL 1 DAY
      AND e.event_date + INTERVAL 14 DAY THEN o.order_date END) AS post_days
  FROM marketing_events e
  JOIN orders o ON o.category = e.category
  GROUP BY e.event_id, e.event_name, e.event_date, e.category
),
impact AS (
  SELECT *,
    ROUND(pre_revenue  / NULLIF(pre_days,  0), 2) AS avg_daily_pre,
    ROUND(post_revenue / NULLIF(post_days, 0), 2) AS avg_daily_post
  FROM event_windows
)
SELECT event_id, event_name, event_date, category,
  avg_daily_pre, avg_daily_post,
  ROUND(avg_daily_post - avg_daily_pre, 2)               AS absolute_lift,
  ROUND((avg_daily_post - avg_daily_pre) * 100.0 /
    NULLIF(avg_daily_pre, 0), 2)                         AS pct_lift,
  CASE WHEN (avg_daily_post - avg_daily_pre) * 100.0 /
    NULLIF(avg_daily_pre, 0) > 10 THEN 'High Impact' ELSE 'Normal' END AS impact_flag
FROM impact ORDER BY pct_lift DESC;
```

---

## Key Takeaways

- **Date spine first** → zero-fill gaps before any time series analysis
- **Rolling Z-score** → more realistic than global (no future data leak)
- **LAG(364) for YoY** → preserves same weekday
- **Short MA > Long MA** → uptrend signal
- **Seasonality index** → period_avg / grand_avg
- **EWMA** → weight recent more (4,3,2,1)
- **Decomposition** → revenue = trend × seasonal × residual
- **Event impact** → avg daily pre vs post window
- **MAPE** → lower is better forecast accuracy
- **Spike** → actual / hour_baseline > 3x threshold

---

*Day 16 complete — 14 days to go 🚀*
