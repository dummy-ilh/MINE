# Day 25 — Amazon-Style SQL Questions
**FAANG SQL 30-Day Prep**

---

## What Amazon Tests

```
1. Marketplace metrics (GMV, seller health, return rates)
2. Customer LTV & RFM segmentation
3. Prime membership analytics (spend lift, retention)
4. Fulfillment SLA & delivery performance (OTD rate)
5. Inventory turnover & stockout risk
6. Product ratings (Bayesian average)
7. Return rate root cause by category
8. Flash sale performance
```

---

## 1. Seller Health Score

```sql
WITH stats AS (
  SELECT s.seller_id, s.name, s.category,
    COUNT(*) AS total_orders, SUM(o.amount) AS gmv,
    COUNT(DISTINCT o.buyer_id) AS unique_buyers,
    SUM(CASE WHEN o.status='returned'   THEN 1 ELSE 0 END) AS returns,
    SUM(CASE WHEN o.status='cancelled'  THEN 1 ELSE 0 END) AS cancellations
  FROM sellers s JOIN orders o ON s.seller_id = o.seller_id
  WHERE o.order_date >= CURRENT_DATE - INTERVAL 90 DAY
  GROUP BY s.seller_id, s.name, s.category
  HAVING COUNT(*) >= 20
)
SELECT *,
  ROUND(returns*100.0/NULLIF(total_orders,0),2)       AS return_rate,
  ROUND(cancellations*100.0/NULLIF(total_orders,0),2) AS cancel_rate,
  ROUND(100
    - (returns*100.0/NULLIF(total_orders,0)) * 2
    - (cancellations*100.0/NULLIF(total_orders,0)) * 1.5
  ,2) AS health_score,
  RANK() OVER (PARTITION BY category ORDER BY gmv DESC) AS gmv_rank
FROM stats ORDER BY health_score ASC;
-- health_score: 100 = perfect, <75 = at risk
```

---

## 2. Customer LTV

```sql
WITH stats AS (
  SELECT buyer_id,
    MIN(order_date) AS first_order, MAX(order_date) AS last_order,
    COUNT(*) AS total_orders, SUM(amount) AS total_spend,
    AVG(amount) AS avg_order_value,
    DATEDIFF(MAX(order_date), MIN(order_date)) AS lifespan_days
  FROM orders WHERE status NOT IN ('returned','cancelled')
  GROUP BY buyer_id
)
SELECT *,
  ROUND(total_orders*30.0/NULLIF(lifespan_days,0),4) AS orders_per_month,
  ROUND(avg_order_value*(total_orders*30.0/NULLIF(lifespan_days,0))*12,2) AS ltv_12m,
  DATEDIFF(CURRENT_DATE, last_order) AS days_since_last,
  NTILE(5) OVER (ORDER BY total_spend DESC) AS spend_quintile,
  CASE
    WHEN DATEDIFF(CURRENT_DATE, last_order) <= 30  THEN 'Active'
    WHEN DATEDIFF(CURRENT_DATE, last_order) <= 90  THEN 'At Risk'
    WHEN DATEDIFF(CURRENT_DATE, last_order) <= 180 THEN 'Lapsing'
    ELSE 'Churned'
  END AS recency_segment
FROM stats HAVING lifespan_days > 0
ORDER BY ltv_12m DESC;
```

---

## 3. RFM Segmentation

```sql
WITH stats AS (
  SELECT buyer_id,
    DATEDIFF(CURRENT_DATE, MAX(order_date)) AS recency_days,
    COUNT(*) AS frequency, SUM(amount) AS monetary
  FROM orders WHERE status NOT IN ('returned','cancelled')
  GROUP BY buyer_id
),
rfm AS (
  SELECT buyer_id, recency_days, frequency, monetary,
    NTILE(5) OVER (ORDER BY recency_days ASC)  AS r,  -- lower = better
    NTILE(5) OVER (ORDER BY frequency    DESC) AS f,
    NTILE(5) OVER (ORDER BY monetary     DESC) AS m
  FROM stats
)
SELECT buyer_id, recency_days, frequency, monetary, r, f, m,
  CASE
    WHEN r >= 4 AND f >= 4 AND m >= 4 THEN 'Champions'
    WHEN f >= 3 AND m >= 3             THEN 'Loyal'
    WHEN r <= 2 AND f >= 3             THEN 'At Risk'
    WHEN r  = 1 AND f  = 1             THEN 'Lost'
    ELSE 'Others'
  END AS rfm_segment
FROM rfm ORDER BY r DESC, f DESC, m DESC;
```

---

## 4. Prime Membership Spend Lift

```sql
WITH member_spend AS (
  SELECT m.user_id, m.plan_type, m.start_date,
    SUM(CASE WHEN o.order_date < m.start_date  THEN o.amount ELSE 0 END) AS pre_spend,
    SUM(CASE WHEN o.order_date >= m.start_date THEN o.amount ELSE 0 END) AS post_spend,
    COUNT(CASE WHEN o.order_date >= m.start_date THEN 1 END) AS orders_as_member
  FROM memberships m LEFT JOIN orders o ON m.user_id = o.buyer_id
  GROUP BY m.user_id, m.plan_type, m.start_date
)
SELECT plan_type,
  COUNT(*) AS members,
  ROUND(AVG(pre_spend),2) AS avg_pre_spend,
  ROUND(AVG(post_spend),2) AS avg_post_spend,
  ROUND((AVG(post_spend)-AVG(pre_spend))*100.0/NULLIF(AVG(pre_spend),0),2) AS spend_lift_pct,
  ROUND(AVG(orders_as_member),2) AS avg_orders_as_member
FROM member_spend GROUP BY plan_type ORDER BY spend_lift_pct DESC;
```

---

## 5. Fulfillment OTD Rate

```sql
SELECT seller_id, warehouse_id, DATE_FORMAT(order_date,'%Y-%m') AS month,
  COUNT(*) AS shipments,
  ROUND(AVG(DATEDIFF(ship_date,order_date)),2) AS avg_ship_days,
  ROUND(AVG(DATEDIFF(delivery_date,order_date)),2) AS avg_delivery_days,
  ROUND(SUM(CASE WHEN delivery_date<=promised_delivery_date THEN 1 ELSE 0 END)*100.0/COUNT(*),2) AS otd_rate,
  CASE
    WHEN SUM(CASE WHEN delivery_date<=promised_delivery_date THEN 1 ELSE 0 END)*100.0/COUNT(*) >= 95 THEN '🟢 On Track'
    WHEN SUM(CASE WHEN delivery_date<=promised_delivery_date THEN 1 ELSE 0 END)*100.0/COUNT(*) >= 85 THEN '🟡 Watch'
    ELSE '🔴 At Risk'
  END AS sla_status
FROM shipments GROUP BY seller_id, warehouse_id, DATE_FORMAT(order_date,'%Y-%m')
ORDER BY month, otd_rate ASC;
```

---

## 6. Inventory Stockout Risk

```sql
WITH daily AS (
  SELECT product_id, warehouse_id, stock_date,
    units_on_hand,
    ROUND(units_on_hand*1.0/NULLIF(
      AVG(units_sold) OVER (PARTITION BY product_id, warehouse_id
        ORDER BY stock_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW),0),1) AS days_of_supply
  FROM inventory
)
SELECT product_id, warehouse_id, stock_date, units_on_hand, days_of_supply,
  CASE
    WHEN units_on_hand = 0   THEN '🔴 Stockout'
    WHEN days_of_supply <= 3 THEN '🔴 Critical'
    WHEN days_of_supply <= 7 THEN '🟡 Low Stock'
    ELSE '🟢 OK'
  END AS stock_status
FROM daily
WHERE stock_date = CURRENT_DATE AND days_of_supply <= 7
ORDER BY days_of_supply ASC;
```

---

## 7. Bayesian Product Rating

```sql
-- B_avg = (C × global_avg + n × product_avg) / (C + n)
-- C = confidence threshold (e.g. 10 reviews)
WITH stats AS (
  SELECT product_id, COUNT(*) AS reviews,
    ROUND(AVG(rating),2) AS avg_rating,
    ROUND(SUM(helpful_votes)*100.0/NULLIF(SUM(total_votes),0),2) AS helpfulness_rate
  FROM reviews GROUP BY product_id
)
SELECT product_id, reviews, avg_rating, helpfulness_rate,
  ROUND((10*(SELECT AVG(rating) FROM reviews) + reviews*avg_rating)/(10+reviews),4) AS bayesian_avg,
  RANK() OVER (ORDER BY
    (10*(SELECT AVG(rating) FROM reviews) + reviews*avg_rating)/(10+reviews) DESC) AS quality_rank
FROM stats WHERE reviews >= 5
ORDER BY quality_rank;
```

---

## Practice Questions

### Q1 — Medium ✅
Seller GMV, return/cancel rates, health score (20+ orders, last 90d).

```sql
WITH stats AS (
  SELECT s.seller_id, s.name, s.category,
    COUNT(*) AS total_orders, SUM(o.amount) AS gmv,
    COUNT(DISTINCT o.buyer_id) AS unique_buyers,
    SUM(CASE WHEN o.status='returned'  THEN 1 ELSE 0 END) AS returns,
    SUM(CASE WHEN o.status='cancelled' THEN 1 ELSE 0 END) AS cancellations
  FROM sellers s JOIN orders o ON s.seller_id = o.seller_id
  WHERE o.order_date >= CURRENT_DATE - INTERVAL 90 DAY
  GROUP BY s.seller_id, s.name, s.category HAVING COUNT(*) >= 20
),
scored AS (
  SELECT *,
    ROUND(returns*100.0/NULLIF(total_orders,0),2) AS return_rate,
    ROUND(cancellations*100.0/NULLIF(total_orders,0),2) AS cancel_rate,
    ROUND(100-(returns*100.0/NULLIF(total_orders,0))*2
            -(cancellations*100.0/NULLIF(total_orders,0))*1.5,2) AS health_score
  FROM stats
)
SELECT *,
  CASE WHEN health_score >= 90 THEN '🟢 Excellent'
       WHEN health_score >= 75 THEN '🟡 Good'
       ELSE '🔴 At Risk' END AS seller_status,
  RANK() OVER (PARTITION BY category ORDER BY gmv DESC) AS gmv_rank
FROM scored ORDER BY health_score ASC;
```

### Q2 — Hard ✅
RFM scoring with NTILE(5) → Champion / Loyal / At Risk / Lost segments.

```sql
WITH stats AS (
  SELECT buyer_id,
    DATEDIFF(CURRENT_DATE, MAX(order_date)) AS recency_days,
    COUNT(*) AS frequency, SUM(amount) AS monetary
  FROM orders WHERE status NOT IN ('returned','cancelled')
  GROUP BY buyer_id
),
rfm AS (
  SELECT buyer_id, recency_days, frequency, monetary,
    NTILE(5) OVER (ORDER BY recency_days ASC)  AS r_score,
    NTILE(5) OVER (ORDER BY frequency    DESC) AS f_score,
    NTILE(5) OVER (ORDER BY monetary     DESC) AS m_score
  FROM stats
)
SELECT buyer_id, recency_days, frequency, monetary,
  r_score, f_score, m_score,
  CASE
    WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN 'Champions'
    WHEN f_score >= 3 AND m_score >= 3                  THEN 'Loyal'
    WHEN r_score <= 2 AND f_score >= 3                  THEN 'At Risk'
    WHEN r_score  = 1 AND f_score  = 1                  THEN 'Lost'
    ELSE 'Others'
  END AS rfm_segment
FROM rfm ORDER BY r_score DESC, f_score DESC, m_score DESC;
```

### Q3 — Very Hard ✅
Weekly seller scorecard: composite score (GMV 30% + OTD 35% + returns 20% + rating 15%), WoW drop flag, worst metric.

```sql
WITH weekly_orders AS (
  SELECT seller_id,
    DATE_SUB(DATE(order_date), INTERVAL DAYOFWEEK(DATE(order_date))-1 DAY) AS week_start,
    COUNT(*) AS total_orders, SUM(amount) AS gmv,
    SUM(CASE WHEN status='returned' THEN 1 ELSE 0 END) AS returns
  FROM orders GROUP BY seller_id, week_start
),
weekly_sla AS (
  SELECT o.seller_id,
    DATE_SUB(DATE(o.order_date), INTERVAL DAYOFWEEK(DATE(o.order_date))-1 DAY) AS week_start,
    COUNT(*) AS shipments,
    SUM(CASE WHEN s.delivery_date<=s.promised_delivery_date THEN 1 ELSE 0 END) AS on_time
  FROM orders o JOIN shipments s ON o.order_id=s.order_id
  GROUP BY o.seller_id, week_start
),
weekly_ratings AS (
  SELECT p.seller_id,
    DATE_SUB(DATE(r.review_date), INTERVAL DAYOFWEEK(DATE(r.review_date))-1 DAY) AS week_start,
    ROUND(AVG(r.rating),4) AS avg_rating
  FROM reviews r JOIN products p ON r.product_id=p.product_id
  GROUP BY p.seller_id, week_start
),
scored AS (
  SELECT wo.seller_id, wo.week_start, wo.gmv, wo.total_orders,
    ROUND(wo.returns*100.0/NULLIF(wo.total_orders,0),2) AS return_rate,
    ROUND(ws.on_time*100.0/NULLIF(ws.shipments,0),2) AS otd_rate,
    wr.avg_rating,
    ROUND(PERCENT_RANK() OVER (PARTITION BY wo.week_start ORDER BY wo.gmv ASC)*100,2) AS gmv_pct_rank,
    ROUND(COALESCE(ws.on_time*100.0/NULLIF(ws.shipments,0),0),2) AS otd_score,
    ROUND(100-COALESCE(wo.returns*100.0/NULLIF(wo.total_orders,0),0),2) AS return_score,
    ROUND(COALESCE(wr.avg_rating,0)*20,2) AS rating_score
  FROM weekly_orders wo
  LEFT JOIN weekly_sla     ws ON wo.seller_id=ws.seller_id AND wo.week_start=ws.week_start
  LEFT JOIN weekly_ratings wr ON wo.seller_id=wr.seller_id AND wo.week_start=wr.week_start
),
composite AS (
  SELECT *,
    ROUND(gmv_pct_rank*0.30 + otd_score*0.35 + return_score*0.20 + rating_score*0.15,2) AS composite_score
  FROM scored
),
with_wow AS (
  SELECT *,
    LAG(composite_score) OVER (PARTITION BY seller_id ORDER BY week_start) AS prev_score
  FROM composite
)
SELECT seller_id, week_start, gmv, return_rate, otd_rate, avg_rating,
  composite_score, prev_score,
  ROUND(composite_score-prev_score,2) AS score_wow,
  CASE WHEN composite_score-prev_score < -10 THEN '🔴 SCORE DROP' ELSE '✅ STABLE' END AS wow_flag,
  CASE
    WHEN LEAST(gmv_pct_rank,otd_score,return_score,rating_score)=gmv_pct_rank THEN 'GMV'
    WHEN LEAST(gmv_pct_rank,otd_score,return_score,rating_score)=otd_score    THEN 'OTD'
    WHEN LEAST(gmv_pct_rank,otd_score,return_score,rating_score)=return_score THEN 'Returns'
    ELSE 'Ratings'
  END AS worst_metric
FROM with_wow ORDER BY week_start, composite_score ASC;
```

---

## Key Takeaways

- **Seller health** → `100 - return_rate×2 - cancel_rate×1.5` — penalize returns harder
- **RFM** → NTILE(5) on R (ASC days), F (DESC orders), M (DESC spend)
- **LTV** → `avg_order_value × orders_per_month × 12` — exclude returns first
- **Bayesian rating** → `(C×global + n×product) / (C+n)` — shrinks small-n products toward mean
- **OTD rate** → `delivery_date <= promised_delivery_date` — Amazon's #1 seller SLA metric
- **Days of supply** → `units_on_hand / avg_daily_sales_7d` — rolling window for accuracy
- **Composite score** → PERCENT_RANK() to normalize GMV across weeks, then weight
- **Worst metric** → `LEAST(score_a, score_b, score_c)` then CASE to identify which
- **Spend lift** → always compare pre vs post membership, not absolute values

---

*Day 25 complete — 5 days to go 🚀*
