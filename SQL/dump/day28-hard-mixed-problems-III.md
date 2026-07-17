# Day 28 — Hard Mixed Problems III
**FAANG SQL 30-Day Prep**

---

## Problem 1 — Market Share Shift

```sql
WITH totals AS (
  SELECT product_category, sale_month, SUM(revenue) AS cat_total
  FROM sales GROUP BY product_category, sale_month
),
share AS (
  SELECT s.company_id, s.product_category, s.sale_month, s.revenue,
    ROUND(s.revenue*100.0/NULLIF(ct.cat_total,0),4) AS market_share_pct,
    RANK() OVER (PARTITION BY s.product_category, s.sale_month ORDER BY s.revenue DESC) AS mkt_rank
  FROM sales s JOIN totals ct ON s.product_category=ct.product_category AND s.sale_month=ct.sale_month
),
with_prev AS (
  SELECT *,
    LAG(market_share_pct) OVER (PARTITION BY company_id, product_category ORDER BY sale_month) AS prev_share,
    LAG(mkt_rank)         OVER (PARTITION BY company_id, product_category ORDER BY sale_month) AS prev_rank
  FROM share
)
SELECT company_id, product_category, sale_month, market_share_pct, mkt_rank,
  ROUND(market_share_pct-prev_share,4) AS share_pp_change,
  CASE WHEN market_share_pct-prev_share < -5 AND mkt_rank > prev_rank THEN '🔴 SHARE LOSS + RANK DROP'
       WHEN market_share_pct-prev_share < -5 THEN '🟡 SHARE LOSS' ELSE '✅ STABLE' END AS flag
FROM with_prev WHERE prev_share IS NOT NULL ORDER BY share_pp_change ASC;
```

---

## Problem 2 — Funnel Completion Time Percentiles

```sql
WITH funnel AS (
  SELECT user_id,
    MIN(CASE WHEN step=1 THEN step_time END) AS t1,
    MIN(CASE WHEN step=4 THEN step_time END) AS t4
  FROM funnel_events GROUP BY user_id
  HAVING t1 IS NOT NULL AND t4 IS NOT NULL
),
times AS (
  SELECT user_id, TIMESTAMPDIFF(MINUTE,t1,t4) AS total_mins
  FROM funnel WHERE t4 > t1
),
pct AS (
  SELECT *,
    NTILE(100) OVER (ORDER BY total_mins) AS pctile
  FROM times
),
cutoffs AS (
  SELECT MAX(CASE WHEN pctile<=25 THEN total_mins END) AS p25,
         MAX(CASE WHEN pctile<=50 THEN total_mins END) AS p50,
         MAX(CASE WHEN pctile<=75 THEN total_mins END) AS p75,
         MAX(CASE WHEN pctile<=90 THEN total_mins END) AS p90
  FROM pct
)
SELECT p.user_id, p.total_mins, p.pctile,
  c.p25, c.p50, c.p75, c.p90,
  CASE WHEN p.total_mins > c.p90 THEN '🔴 ABOVE P90' ELSE '✅ Normal' END AS flag
FROM pct p CROSS JOIN cutoffs c ORDER BY p.total_mins DESC;
-- NTILE(100) approximates percentiles; for exact use ROW_NUMBER + FLOOR/CEIL
```

---

## Problem 3 — Social Contagion (Purchase Graph)

```sql
WITH first_buys AS (
  SELECT user_id, product_id, MIN(order_date) AS first_date
  FROM orders GROUP BY user_id, product_id
),
pairs AS (
  SELECT a.product_id, a.user_id AS first_mover, b.user_id AS follower,
    DATEDIFF(b.first_date, a.first_date) AS days_gap
  FROM first_buys a JOIN first_buys b
    ON  a.product_id = b.product_id AND a.user_id != b.user_id
    AND a.first_date < b.first_date
    AND DATEDIFF(b.first_date, a.first_date) <= 30
),
contagion AS (
  SELECT product_id, COUNT(*) AS pairs,
    AVG(CASE WHEN rn IN (FLOOR((cnt+1)/2),CEIL((cnt+1)/2)) THEN days_gap END) AS median_gap
  FROM (
    SELECT *, COUNT(*) OVER (PARTITION BY product_id) AS cnt,
      ROW_NUMBER() OVER (PARTITION BY product_id ORDER BY days_gap) AS rn
    FROM pairs
  ) t GROUP BY product_id
)
SELECT product_id, pairs, ROUND(median_gap,1) AS median_gap_days,
  CASE WHEN median_gap < 7 THEN '🔥 Fast' WHEN median_gap < 14 THEN '🟡 Moderate' ELSE '🔵 Slow' END AS label
FROM contagion WHERE pairs >= 10 ORDER BY median_gap ASC;
```

---

## Problem 4 — Multi-Level Quota Attainment

```sql
WITH rev AS (
  SELECT rep_id, DATE_FORMAT(close_date,'%Y-%m-01') AS month, SUM(deal_amount) AS revenue
  FROM deals GROUP BY rep_id, DATE_FORMAT(close_date,'%Y-%m-01')
),
rep AS (
  SELECT q.rep_id, q.month, q.quota_amount,
    COALESCE(r.revenue,0) AS revenue,
    ROUND(COALESCE(r.revenue,0)*100.0/NULLIF(q.quota_amount,0),2) AS attainment_pct
  FROM quotas q LEFT JOIN rev r ON q.rep_id=r.rep_id AND q.month=r.month
),
mgr AS (
  SELECT s.manager_id, r.month,
    ROUND(SUM(r.revenue)*100.0/NULLIF(SUM(r.quota_amount),0),2) AS team_pct
  FROM rep r JOIN salespeople s ON r.rep_id=s.rep_id GROUP BY s.manager_id, r.month
),
rgn AS (
  SELECT s.region_id, r.month,
    ROUND(SUM(r.revenue)*100.0/NULLIF(SUM(r.quota_amount),0),2) AS region_pct
  FROM rep r JOIN salespeople s ON r.rep_id=s.rep_id GROUP BY s.region_id, r.month
)
SELECT ra.rep_id, sp.manager_id, sp.region_id, ra.month,
  ra.attainment_pct, ma.team_pct, rg.region_pct,
  CASE WHEN ra.attainment_pct < 70 THEN '🔴 Rep <70%' ELSE '✅' END AS rep_flag,
  CASE WHEN ma.team_pct < 80 THEN '🔴 Team <80%' ELSE '✅' END AS team_flag,
  CASE WHEN rg.region_pct < 85 THEN '🔴 Region <85%' ELSE '✅' END AS region_flag
FROM rep ra
JOIN salespeople sp ON ra.rep_id=sp.rep_id
JOIN mgr ma ON sp.manager_id=ma.manager_id AND ra.month=ma.month
JOIN rgn rg ON sp.region_id=rg.region_id AND ra.month=rg.month
ORDER BY ra.month, ra.attainment_pct ASC;
```

---

## Problem 5 — Exact Event Sequence Matching

```sql
WITH pivoted AS (
  SELECT user_id,
    MIN(CASE WHEN event_type='search'       THEN event_time END) AS t_search,
    MIN(CASE WHEN event_type='product_view' THEN event_time END) AS t_view,
    MIN(CASE WHEN event_type='add_to_cart'  THEN event_time END) AS t_cart,
    MIN(CASE WHEN event_type='purchase'     THEN event_time END) AS t_purchase
  FROM events
  WHERE event_type IN ('search','product_view','add_to_cart','purchase')
  GROUP BY user_id
)
SELECT user_id, t_search AS seq_start, t_purchase AS seq_end,
  TIMESTAMPDIFF(MINUTE,t_search,t_purchase) AS total_minutes,
  TIMESTAMPDIFF(MINUTE,t_search,t_view)    AS search_to_view,
  TIMESTAMPDIFF(MINUTE,t_view,t_cart)      AS view_to_cart,
  TIMESTAMPDIFF(MINUTE,t_cart,t_purchase)  AS cart_to_purchase
FROM pivoted
WHERE t_search IS NOT NULL AND t_view IS NOT NULL
  AND t_cart IS NOT NULL AND t_purchase IS NOT NULL
  AND t_search < t_view AND t_view < t_cart AND t_cart < t_purchase
  AND TIMESTAMPDIFF(HOUR,t_search,t_purchase) <= 24
ORDER BY total_minutes ASC;
-- Strict ordering via chained WHERE comparisons — simpler than ROW_NUMBER approach
```

---

## Problem 6 — Supply Chain Stockout Risk

```sql
WITH supply AS (
  SELECT i.component_id, i.units_on_hand, i.daily_usage,
    ROUND(i.units_on_hand*1.0/NULLIF(i.daily_usage,0),1) AS days_of_supply,
    s.lead_time_days + c.safety_stock_days AS reorder_threshold,
    DATE_ADD(CURRENT_DATE, INTERVAL FLOOR(i.units_on_hand/NULLIF(i.daily_usage,0)) DAY) AS stockout_date
  FROM inventory i JOIN components c ON i.component_id=c.component_id
  JOIN suppliers s ON c.supplier_id=s.supplier_id
),
product_risk AS (
  SELECT p.product_id,
    MIN(cs.days_of_supply) AS min_days,
    MIN(cs.stockout_date) AS earliest_stockout,
    MAX(CASE WHEN cs.days_of_supply < cs.reorder_threshold THEN 1 ELSE 0 END) AS at_risk,
    GREATEST(0, MAX(cs.reorder_threshold)-MIN(cs.days_of_supply)) AS days_at_risk
  FROM products p JOIN supply cs ON p.component_id=cs.component_id
  GROUP BY p.product_id
)
SELECT product_id, ROUND(min_days,1) AS min_days_supply, earliest_stockout,
  days_at_risk, ROUND(days_at_risk*1000,2) AS revenue_at_risk,
  CASE WHEN at_risk=1 THEN '🔴 AT RISK' ELSE '🟢 OK' END AS status
FROM product_risk ORDER BY earliest_stockout ASC;
```

---

## Problem 7 — Cross-Sell Lift by Channel

```sql
WITH uc AS (SELECT DISTINCT o.user_id, p.category FROM orders o JOIN products p ON o.product_id=p.product_id),
ucc AS (SELECT user_id, COUNT(DISTINCT category) AS cats FROM uc GROUP BY user_id),
channel_base AS (
  SELECT u.acquisition_channel, COUNT(DISTINCT u.user_id) AS total_users,
    COUNT(DISTINCT CASE WHEN ucc.cats >= 2 THEN u.user_id END) AS u2,
    COUNT(DISTINCT CASE WHEN ucc.cats >= 3 THEN u.user_id END) AS u3
  FROM users u LEFT JOIN ucc ON u.user_id=ucc.user_id GROUP BY u.acquisition_channel
),
pairs AS (
  SELECT u.acquisition_channel, a.category AS ca, b.category AS cb,
    COUNT(DISTINCT a.user_id) AS cnt
  FROM uc a JOIN uc b ON a.user_id=b.user_id AND a.category < b.category
  JOIN users u ON a.user_id=u.user_id
  GROUP BY u.acquisition_channel, a.category, b.category
),
top_pair AS (
  SELECT acquisition_channel, ca, cb, cnt
  FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY acquisition_channel ORDER BY cnt DESC) AS rn FROM pairs) t
  WHERE rn=1
)
SELECT cb.acquisition_channel, cb.total_users, cb.u2, ROUND(cb.u2*100.0/NULLIF(cb.total_users,0),2) AS pct_2plus,
  cb.u3, ROUND(cb.u3*100.0/NULLIF(cb.total_users,0),2) AS pct_3plus,
  CONCAT(tp.ca,' + ',tp.cb) AS top_pair, tp.cnt AS top_pair_count
FROM channel_base cb LEFT JOIN top_pair tp ON cb.acquisition_channel=tp.acquisition_channel
ORDER BY pct_2plus DESC;
```

---

## Problem 8 — Moving Annual Total (MAT) + 3-Month Declining Growth Flag

```sql
WITH monthly AS (
  SELECT product_id, DATE_FORMAT(sale_date,'%Y-%m-01') AS month, SUM(revenue) AS rev
  FROM sales GROUP BY product_id, DATE_FORMAT(sale_date,'%Y-%m-01')
),
mat AS (
  SELECT a.product_id, a.month, SUM(b.rev) AS mat_revenue
  FROM monthly a JOIN monthly b ON b.product_id=a.product_id
    AND b.month > DATE_SUB(a.month, INTERVAL 12 MONTH) AND b.month <= a.month
  GROUP BY a.product_id, a.month
),
with_yoy AS (
  SELECT *, LAG(mat_revenue,12) OVER (PARTITION BY product_id ORDER BY month) AS prior_mat
  FROM mat
),
with_growth AS (
  SELECT *,
    ROUND((mat_revenue-prior_mat)*100.0/NULLIF(prior_mat,0),2) AS yoy_pct,
    CASE WHEN (mat_revenue-prior_mat)/NULLIF(prior_mat,0) <
      LAG((mat_revenue-prior_mat)/NULLIF(prior_mat,0)) OVER (PARTITION BY product_id ORDER BY month)
    THEN 1 ELSE 0 END AS declining
  FROM with_yoy WHERE prior_mat IS NOT NULL
),
with_consec AS (
  SELECT *,
    SUM(declining) OVER (PARTITION BY product_id ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS decline_3m
  FROM with_growth
)
SELECT product_id, month, ROUND(mat_revenue,2) AS mat, ROUND(prior_mat,2) AS prior_mat,
  yoy_pct, decline_3m,
  CASE WHEN decline_3m=3 THEN '🔴 3-Month Declining' ELSE '✅ OK' END AS flag
FROM with_consec ORDER BY product_id, month;
```

---

## Practice Questions & Answers

### Q1 ✅ — Dept budget utilization + salary outlier flags

```sql
WITH ds AS (
  SELECT dept_id, COUNT(*) AS headcount,
    SUM(salary) AS total_salary, ROUND(AVG(salary),2) AS avg_salary, MAX(salary) AS max_salary
  FROM employees GROUP BY dept_id
),
hp AS (
  SELECT dept_id, emp_id, name, salary,
    ROW_NUMBER() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS rn
  FROM employees
)
SELECT d.dept_id, d.dept_name, d.budget, ds.headcount,
  ds.total_salary, ds.avg_salary, ds.max_salary,
  ROUND(ds.total_salary*100.0/NULLIF(d.budget,0),2) AS budget_utilization_pct,
  hp.name AS highest_paid,
  CASE WHEN ds.total_salary > d.budget THEN '🔴 OVER BUDGET' ELSE '✅ OK' END AS budget_flag,
  CASE WHEN ds.max_salary > ds.avg_salary*3 THEN '🔴 OUTLIER' ELSE '✅ OK' END AS outlier_flag
FROM departments d JOIN ds ON d.dept_id=ds.dept_id JOIN hp ON d.dept_id=hp.dept_id AND hp.rn=1
ORDER BY budget_utilization_pct DESC;
```

### Q2 ✅ — Page engagement score

```sql
WITH stats AS (
  SELECT page_id,
    COUNT(*) AS total_views,
    COUNT(DISTINCT user_id) AS unique_users,
    ROUND(AVG(LEAST(time_on_page_seconds,300)),2) AS avg_time_capped,
    ROUND(COUNT(DISTINCT CASE WHEN vc >= 2 THEN user_id END)*1.0/
      NULLIF(COUNT(DISTINCT user_id),0),4) AS return_visit_rate,
    ROUND(AVG(LEAST(time_on_page_seconds/60.0,1)),4) AS avg_scroll_depth
  FROM (SELECT *, COUNT(*) OVER (PARTITION BY user_id, page_id) AS vc FROM page_views) t
  GROUP BY page_id
)
SELECT page_id, total_views, unique_users,
  avg_time_capped, return_visit_rate, avg_scroll_depth,
  ROUND(avg_time_capped*0.40 + return_visit_rate*100*0.35 + avg_scroll_depth*100*0.25,4) AS engagement_score,
  RANK() OVER (ORDER BY avg_time_capped*0.40 + return_visit_rate*100*0.35 + avg_scroll_depth*100*0.25 DESC) AS score_rank
FROM stats ORDER BY engagement_score DESC LIMIT 20;
```

### Q3 ✅ — SaaS MRR Waterfall + NRR

```sql
WITH sub_months AS (
  SELECT DISTINCT s.sub_id, s.user_id, s.mrr,
    m.month
  FROM subscriptions s
  JOIN (SELECT DISTINCT DATE_FORMAT(start_date,'%Y-%m-01') AS month FROM subscriptions) m
    ON m.month >= DATE_FORMAT(s.start_date,'%Y-%m-01')
    AND m.month <= COALESCE(DATE_FORMAT(s.end_date,'%Y-%m-01'), DATE_FORMAT(CURRENT_DATE,'%Y-%m-01'))
),
mrr_month AS (SELECT user_id, month, SUM(mrr) AS user_mrr FROM sub_months GROUP BY user_id, month),
classified AS (
  SELECT curr.month, curr.user_id, curr.user_mrr, prev.user_mrr AS prev_mrr,
    CASE WHEN prev.user_mrr IS NULL THEN 'new'
         WHEN curr.user_mrr > prev.user_mrr THEN 'expansion'
         WHEN curr.user_mrr < prev.user_mrr THEN 'contraction'
         ELSE 'retained' END AS mrr_type
  FROM mrr_month curr
  LEFT JOIN mrr_month prev ON curr.user_id=prev.user_id
    AND curr.month=DATE_ADD(prev.month, INTERVAL 1 MONTH)
),
churned AS (
  SELECT DATE_ADD(prev.month, INTERVAL 1 MONTH) AS month, SUM(prev.user_mrr) AS churned_mrr
  FROM mrr_month prev
  WHERE NOT EXISTS (SELECT 1 FROM mrr_month curr WHERE curr.user_id=prev.user_id
    AND curr.month=DATE_ADD(prev.month, INTERVAL 1 MONTH))
  GROUP BY DATE_ADD(prev.month, INTERVAL 1 MONTH)
),
waterfall AS (
  SELECT c.month,
    SUM(CASE WHEN c.mrr_type='retained' THEN c.prev_mrr ELSE 0 END) AS starting_mrr,
    SUM(CASE WHEN c.mrr_type='new' THEN c.user_mrr ELSE 0 END) AS new_mrr,
    SUM(CASE WHEN c.mrr_type='expansion' THEN c.user_mrr-c.prev_mrr ELSE 0 END) AS expansion_mrr,
    SUM(CASE WHEN c.mrr_type='contraction' THEN c.prev_mrr-c.user_mrr ELSE 0 END) AS contraction_mrr,
    COALESCE(ch.churned_mrr,0) AS churned_mrr
  FROM classified c LEFT JOIN churned ch ON c.month=ch.month
  GROUP BY c.month, ch.churned_mrr
)
SELECT month,
  ROUND(starting_mrr,2) AS starting_mrr,
  ROUND(new_mrr,2) AS new_mrr,
  ROUND(expansion_mrr,2) AS expansion_mrr,
  ROUND(-contraction_mrr,2) AS contraction_mrr,
  ROUND(-churned_mrr,2) AS churned_mrr,
  ROUND(starting_mrr+new_mrr+expansion_mrr-contraction_mrr-churned_mrr,2) AS ending_mrr,
  ROUND((starting_mrr+expansion_mrr-contraction_mrr-churned_mrr)*100.0/NULLIF(starting_mrr,0),2) AS nrr_pct,
  CASE WHEN (starting_mrr+expansion_mrr-contraction_mrr-churned_mrr)*100.0/
    NULLIF(starting_mrr,0) < 100 THEN '🔴 NRR < 100%' ELSE '✅ HEALTHY' END AS nrr_flag
FROM waterfall ORDER BY month;
-- NRR excludes new_mrr — measures retention + expansion of existing base only
-- NRR > 100% = expansion > churn (Snowflake ~130%, best-in-class)
```

---

## Key Patterns This Day

| Problem | Core Technique |
|---|---|
| Market share shift | RANK + LAG share_pct, filter pp_change < -5 AND rank dropped |
| Percentile buckets | NTILE(100), cutoffs CTE with MAX(CASE WHEN pctile ≤ N) |
| Social contagion | Self-join first_purchases, median gap via ROW_NUMBER trick |
| Multi-level quotas | Three separate attainment CTEs: rep → manager → region |
| Sequence matching | Pivot MIN per event_type, chain WHERE t1 < t2 < t3 < t4 |
| Supply chain risk | days_of_supply < reorder_threshold, MIN across components |
| Cross-sell by channel | Category pairs + ROW_NUMBER per channel for top pair |
| MAT YoY + declining | Self-join trailing 12m, LAG(12) for prior MAT, SUM decline flags |
| Dept outlier | MAX salary > 3× avg salary — use window or subquery |
| NRR | (starting + expansion - contraction - churn) / starting × 100 |

---

*Day 28 complete — 2 days to go 🚀*
