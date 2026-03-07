# Day 19 — Advanced Aggregations & GROUPING SETS
**FAANG SQL 30-Day Prep**

---

## Concepts Covered

1. The Problem with Multiple GROUP BYs
2. ROLLUP — Hierarchical Subtotals
3. CUBE — All Combinations
4. GROUPING SETS — Precise Control
5. GROUPING() and GROUPING_ID()
6. Running Totals & Cumulative Share
7. Pareto / 80-20 Analysis
8. Pivot with Dynamic Totals
9. Multi-Metric Aggregation Report
10. FAANG Aggregation Patterns

---

## 1. The Problem with Multiple GROUP BYs

```sql
-- ❌ 3 separate queries + UNION ALL
SELECT country, NULL AS category, SUM(amount) FROM orders GROUP BY country
UNION ALL
SELECT NULL, category, SUM(amount) FROM orders GROUP BY category
UNION ALL
SELECT NULL, NULL, SUM(amount) FROM orders;

-- ✅ One query with GROUPING SETS
SELECT country, category, SUM(amount) AS revenue
FROM orders
GROUP BY GROUPING SETS (
  (country),
  (category),
  ()         -- grand total
);
```

---

## 2. ROLLUP — Hierarchical Subtotals

```sql
-- Year → Month with subtotals at each level
SELECT YEAR(order_date) AS yr, MONTH(order_date) AS mo,
  SUM(amount) AS revenue, COUNT(*) AS orders
FROM orders
GROUP BY ROLLUP(YEAR(order_date), MONTH(order_date))
ORDER BY yr, mo;
-- Adds: year subtotals (mo=NULL) + grand total (yr=NULL, mo=NULL)

-- 3-level: region → country → city
SELECT region, country, city,
  SUM(revenue) AS total_revenue,
  GROUPING(region)  AS is_region_subtotal,
  GROUPING(country) AS is_country_subtotal,
  GROUPING(city)    AS is_city_subtotal
FROM sales
GROUP BY ROLLUP(region, country, city);
-- GROUPING() = 1 means that column was aggregated away
```

---

## 3. CUBE — All Combinations

```sql
-- 3 dimensions → 2³ = 8 combinations
SELECT country, category, channel,
  SUM(amount) AS revenue, COUNT(*) AS orders
FROM orders
GROUP BY CUBE(country, category, channel);

-- Label subtotal rows
SELECT
  CASE WHEN GROUPING(country)  = 1 THEN 'ALL' ELSE country  END AS country,
  CASE WHEN GROUPING(category) = 1 THEN 'ALL' ELSE category END AS category,
  SUM(amount) AS revenue,
  GROUPING(country)  AS country_agg,
  GROUPING(category) AS category_agg
FROM orders
GROUP BY CUBE(country, category)
ORDER BY country_agg, category_agg, country, category;
```

---

## 4. GROUPING SETS — Precise Control

```sql
-- Exactly the combinations you need
SELECT country, category, channel, SUM(amount) AS revenue
FROM orders
GROUP BY GROUPING SETS (
  (country, category),  -- combo detail
  (country),            -- country only
  (channel),            -- channel only
  ()                    -- grand total
);
-- More efficient than CUBE — fewer passes

-- Financial report: dept × quarter with all subtotals
SELECT department, QUARTER(sale_date) AS qtr,
  SUM(amount) AS revenue, COUNT(*) AS transactions
FROM sales
GROUP BY GROUPING SETS (
  (department, QUARTER(sale_date)),
  (department),
  (QUARTER(sale_date)),
  ()
)
ORDER BY GROUPING(department), GROUPING(QUARTER(sale_date)),
         department, qtr;
```

---

## 5. GROUPING() and GROUPING_ID()

```sql
SELECT country, category,
  SUM(amount)                     AS revenue,
  GROUPING(country)               AS g_country,   -- 1 = subtotal row
  GROUPING(category)              AS g_category,
  GROUPING_ID(country, category)  AS grouping_id
  -- 0 = detail row
  -- 1 = country subtotal
  -- 2 = category subtotal
  -- 3 = grand total
FROM orders
GROUP BY CUBE(country, category);

-- Clean row labels
SELECT
  CASE
    WHEN GROUPING(country)=1 AND GROUPING(category)=1 THEN '--- GRAND TOTAL ---'
    WHEN GROUPING(category)=1 THEN CONCAT(country, ' SUBTOTAL')
    WHEN GROUPING(country) =1 THEN CONCAT(category, ' SUBTOTAL')
    ELSE CONCAT(country, ' / ', category)
  END AS row_label,
  SUM(amount) AS revenue
FROM orders
GROUP BY CUBE(country, category)
ORDER BY GROUPING_ID(country, category), country, category;
```

---

## 6. Running Totals & Cumulative Share

```sql
WITH revenue_by_category AS (
  SELECT category, SUM(amount) AS revenue
  FROM orders GROUP BY category
)
SELECT category, revenue,
  SUM(revenue) OVER (ORDER BY revenue DESC
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total,
  ROUND(revenue * 100.0 / SUM(revenue) OVER (), 2)   AS pct_of_total,
  ROUND(SUM(revenue) OVER (ORDER BY revenue DESC
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) * 100.0 /
    SUM(revenue) OVER (), 2)                          AS cumulative_pct
FROM revenue_by_category ORDER BY revenue DESC;
```

---

## 7. Pareto / 80-20 Analysis

```sql
WITH product_revenue AS (
  SELECT product_id, SUM(amount) AS revenue
  FROM orders GROUP BY product_id
),
ranked AS (
  SELECT product_id, revenue,
    ROW_NUMBER() OVER (ORDER BY revenue DESC) AS rank_num,
    COUNT(*) OVER ()                          AS total_products,
    SUM(revenue) OVER ()                      AS total_revenue,
    SUM(revenue) OVER (ORDER BY revenue DESC
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_revenue
  FROM product_revenue
)
SELECT product_id, revenue,
  ROUND(revenue * 100.0 / total_revenue, 2)            AS pct_of_total,
  ROUND(cumulative_revenue * 100.0 / total_revenue, 2) AS cumulative_pct,
  ROUND(rank_num * 100.0 / total_products, 2)          AS pct_of_products,
  CASE WHEN cumulative_revenue * 100.0 / total_revenue <= 80
       THEN 'Top 80% revenue' ELSE 'Tail'
  END AS pareto_segment
FROM ranked ORDER BY rank_num;
```

---

## 8. Pivot with ROLLUP Totals

```sql
SELECT
  CASE WHEN GROUPING(country) = 1 THEN 'ALL COUNTRIES' ELSE country END AS country,
  SUM(amount)                                               AS total,
  SUM(CASE WHEN category='Electronics' THEN amount ELSE 0 END) AS electronics,
  SUM(CASE WHEN category='Clothing'    THEN amount ELSE 0 END) AS clothing,
  SUM(CASE WHEN category='Food'        THEN amount ELSE 0 END) AS food,
  COUNT(DISTINCT user_id)                                   AS unique_buyers
FROM orders
GROUP BY ROLLUP(country)
ORDER BY GROUPING(country), total DESC;
```

---

## 9. Multi-Metric Executive Dashboard

```sql
SELECT
  CASE WHEN GROUPING(region)   = 1 THEN 'ALL' ELSE region   END AS region,
  CASE WHEN GROUPING(category) = 1 THEN 'ALL' ELSE category END AS category,
  CASE WHEN GROUPING(channel)  = 1 THEN 'ALL' ELSE channel  END AS channel,
  COUNT(DISTINCT user_id)                                        AS unique_customers,
  COUNT(*)                                                       AS total_orders,
  ROUND(SUM(amount), 2)                                          AS total_revenue,
  ROUND(AVG(amount), 2)                                          AS avg_order_value,
  ROUND(SUM(amount)*100.0 / SUM(SUM(amount)) OVER (), 2)        AS pct_of_grand_total
FROM orders
GROUP BY GROUPING SETS (
  (region, category, channel),
  (region, category),
  (region),
  ()
)
ORDER BY GROUPING(region), GROUPING(category), GROUPING(channel),
         region, category, channel;
```

---

## Practice Questions
🟢 Q1 — Easy

Table: orders(order_id, user_id, amount, status, order_date)
Write a data quality report that checks: NULL counts for all columns, negative amounts, invalid status values (only pending/completed/cancelled are valid), and future order dates. Return one row per check with a pass/fail flag.


🟡 Q2 — Medium

Tables: orders(order_id, user_id, amount, order_date), users(user_id, signup_date)
Find all referential integrity violations: orders with no matching user, orders placed before the user's signup date, and users with more than one order on the same date with the same amount (suspected duplicates). Return a summary count per violation type.


🔴 Q3 — Hard

Table: events(event_id, user_id, event_type, event_date, amount)
Build a daily pipeline health monitor: for each day show total events, NULL rate for amount, duplicate event_id count, % of invalid event_types (valid = click/view/purchase/signup), row count vs 7-day average, and flag days where any metric is out of threshold.
### Q1 — Easy ✅
ROLLUP by year → month with row_type label.

```sql
SELECT
  YEAR(order_date)  AS yr,
  MONTH(order_date) AS mo,
  SUM(amount)       AS revenue,
  COUNT(*)          AS orders,
  CASE
    WHEN GROUPING(YEAR(order_date))  = 1 THEN 'Grand Total'
    WHEN GROUPING(MONTH(order_date)) = 1 THEN 'Year Subtotal'
    ELSE 'Detail'
  END AS row_type
FROM orders
GROUP BY ROLLUP(YEAR(order_date), MONTH(order_date))
ORDER BY yr, mo;
```

### Q2 — Medium ✅
GROUPING SETS: country+category, country, category, grand total — with clean labels.

```sql
SELECT
  CASE WHEN GROUPING(country)  = 1 THEN 'ALL COUNTRIES'  ELSE country  END AS country,
  CASE WHEN GROUPING(category) = 1 THEN 'ALL CATEGORIES' ELSE category END AS category,
  SUM(amount) AS revenue, COUNT(*) AS orders,
  CASE
    WHEN GROUPING(country)=1 AND GROUPING(category)=1 THEN 'Grand Total'
    WHEN GROUPING(category)=1 THEN 'Country Subtotal'
    WHEN GROUPING(country) =1 THEN 'Category Subtotal'
    ELSE 'Detail'
  END AS row_type
FROM orders
GROUP BY GROUPING SETS ((country, category), (country), (category), ())
ORDER BY GROUPING(country), GROUPING(category), country, category;
```

### Q3 — Hard ✅
Pareto analysis + ROLLUP grand total appended.

```sql
WITH category_revenue AS (
  SELECT category, SUM(amount) AS revenue
  FROM orders GROUP BY category
),
pareto AS (
  SELECT category, revenue,
    ROUND(revenue * 100.0 / SUM(revenue) OVER (), 2)     AS pct_of_total,
    ROUND(SUM(revenue) OVER (ORDER BY revenue DESC
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) * 100.0 /
      SUM(revenue) OVER (), 2)                            AS cumulative_pct,
    CASE WHEN SUM(revenue) OVER (ORDER BY revenue DESC
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) * 100.0 /
      SUM(revenue) OVER () <= 80 THEN 'Top 80%' ELSE 'Tail'
    END AS pareto_segment
  FROM category_revenue
),
rollup_total AS (
  SELECT
    CASE WHEN GROUPING(category)=1 THEN '--- GRAND TOTAL ---' ELSE category END AS category,
    SUM(amount)                                            AS revenue,
    ROUND(SUM(amount)*100.0/SUM(SUM(amount)) OVER (), 2)  AS pct_of_total,
    NULL                                                   AS cumulative_pct,
    CASE WHEN GROUPING(category)=1 THEN 'Grand Total' ELSE NULL END AS pareto_segment
  FROM orders
  GROUP BY ROLLUP(category)
  HAVING GROUPING(category) = 1
)
SELECT category, revenue, pct_of_total, cumulative_pct, pareto_segment FROM pareto
UNION ALL
SELECT category, revenue, pct_of_total, cumulative_pct, pareto_segment FROM rollup_total
ORDER BY CASE WHEN pareto_segment='Grand Total' THEN 1 ELSE 0 END, revenue DESC;
```

---

## Key Takeaways

| Feature | Use Case |
|---|---|
| `ROLLUP(a,b,c)` | Hierarchical subtotals: abc → ab → a → total |
| `CUBE(a,b,c)` | All 2³ combinations |
| `GROUPING SETS(...)` | Exactly specified combinations — most efficient |
| `GROUPING(col)` | 1 = col was aggregated (subtotal row) |
| `GROUPING_ID(a,b)` | Bitmap of which cols are aggregated |

- **ROLLUP** → use for date hierarchies (year/month/day) or org hierarchies
- **CUBE** → use for multi-dimensional analysis (all cross-tabs needed)
- **GROUPING SETS** → use when you need specific combos, more efficient than CUBE
- **GROUPING() = 1** → mark subtotal rows, use in CASE for clean labels
- **Pareto** → cumulative SUM OVER + flag where cumulative ≤ 80%
- **GROUPING_ID** → unique integer per grouping combination, useful for sorting

---

*Day 19 complete — 11 days to go 🚀*
