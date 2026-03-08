# Day 21 — Recommendation System Queries in SQL
**FAANG SQL 30-Day Prep**

---

## Concepts Covered

1. The Recommendation Mindset
2. Item-Item Collaborative Filtering (Co-Purchase)
3. User-Based Collaborative Filtering
4. Generating Recommendations for a User
5. Item-Item Recs for a Specific User
6. Popularity-Based Fallback
7. Category-Based Recommendations
8. Session-Based Recommendations
9. Diversity & Re-Ranking
10. FAANG Recommendation Patterns

---

## 1. The Recommendation Mindset

```
Two main approaches:

1. Collaborative Filtering
   - User-based: find similar users → recommend what they liked
   - Item-based: find similar items → co-purchase patterns

2. Content-Based
   - Match item attributes to user preferences

Key metrics:
- Jaccard similarity: co_count / (count_a + count_b - co_count)
- Lift:               (co_count × total) / (count_a × count_b)
- Confidence:         co_count / count_antecedent
- Cosine similarity:  dot_product / (norm_a × norm_b)
```

---

## 2. Item-Item Collaborative Filtering

```sql
WITH co_purchases AS (
  SELECT a.product_id AS product_a, b.product_id AS product_b,
    COUNT(DISTINCT a.order_id) AS co_purchase_count
  FROM orders a
  JOIN orders b
    ON  a.order_id   = b.order_id
    AND a.product_id < b.product_id   -- avoid duplicates
  GROUP BY a.product_id, b.product_id
),
product_counts AS (
  SELECT product_id, COUNT(DISTINCT order_id) AS total_orders
  FROM orders GROUP BY product_id
),
total AS (SELECT COUNT(DISTINCT order_id) AS n FROM orders)
SELECT cp.product_a, cp.product_b, cp.co_purchase_count,
  -- Jaccard similarity
  ROUND(cp.co_purchase_count * 1.0 /
    (pa.total_orders + pb.total_orders - cp.co_purchase_count), 4) AS jaccard,
  -- Lift
  ROUND(cp.co_purchase_count * 1.0 * t.n /
    (pa.total_orders * pb.total_orders), 4) AS lift
FROM co_purchases cp
JOIN product_counts pa ON cp.product_a = pa.product_id
JOIN product_counts pb ON cp.product_b = pb.product_id
CROSS JOIN total t
ORDER BY lift DESC;
```

---

## 3. User-Based Collaborative Filtering

```sql
WITH user_products AS (
  SELECT DISTINCT user_id, product_id FROM orders
),
user_similarity AS (
  SELECT a.user_id AS user_a, b.user_id AS user_b,
    COUNT(*) AS common_products
  FROM user_products a
  JOIN user_products b
    ON  a.product_id = b.product_id AND a.user_id < b.user_id
  GROUP BY a.user_id, b.user_id
),
user_counts AS (
  SELECT user_id, COUNT(DISTINCT product_id) AS total_products
  FROM orders GROUP BY user_id
)
SELECT s.user_a, s.user_b, s.common_products,
  ROUND(s.common_products * 1.0 /
    (pa.total_products + pb.total_products - s.common_products), 4) AS jaccard
FROM user_similarity s
JOIN user_counts pa ON s.user_a = pa.user_id
JOIN user_counts pb ON s.user_b = pb.user_id
ORDER BY jaccard DESC;
```

---

## 4. Personalized Recommendations for a User

```sql
WITH user_products AS (SELECT DISTINCT user_id, product_id FROM orders),
similar_users AS (
  SELECT b.user_id AS similar_user, COUNT(*) AS common_products
  FROM user_products a
  JOIN user_products b ON a.product_id = b.product_id AND b.user_id != 101
  WHERE a.user_id = 101
  GROUP BY b.user_id ORDER BY common_products DESC LIMIT 10
),
already_bought AS (SELECT DISTINCT product_id FROM orders WHERE user_id = 101)
SELECT up.product_id,
  SUM(su.common_products)         AS recommendation_score,
  COUNT(DISTINCT su.similar_user) AS supporting_users
FROM similar_users su
JOIN user_products up ON su.similar_user = up.user_id
WHERE up.product_id NOT IN (SELECT product_id FROM already_bought)
GROUP BY up.product_id
ORDER BY recommendation_score DESC LIMIT 20;
```

---

## 5. Popularity Fallback (New Users)

```sql
SELECT product_id,
  COUNT(DISTINCT order_id)  AS purchase_count,
  COUNT(DISTINCT user_id)   AS unique_buyers,
  SUM(CASE WHEN order_date >= CURRENT_DATE - INTERVAL 7 DAY
           THEN 1 ELSE 0 END) AS purchases_last_7d,
  -- Trending score: recency-weighted
  ROUND(
    SUM(CASE WHEN order_date >= CURRENT_DATE - INTERVAL  7 DAY THEN 1 ELSE 0 END) * 0.5 +
    SUM(CASE WHEN order_date >= CURRENT_DATE - INTERVAL 30 DAY THEN 1 ELSE 0 END) * 0.3 +
    COUNT(DISTINCT user_id) * 0.2, 2
  ) AS trending_score,
  RANK() OVER (ORDER BY COUNT(DISTINCT order_id) DESC) AS all_time_rank
FROM orders GROUP BY product_id
ORDER BY trending_score DESC LIMIT 50;
```

---

## 6. Category Affinity

```sql
SELECT o.user_id, p.category,
  COUNT(*)    AS purchases_in_category,
  ROUND(COUNT(*) * 1.0 /
    SUM(COUNT(*)) OVER (PARTITION BY o.user_id), 4) AS category_affinity
FROM orders o
JOIN products p ON o.product_id = p.product_id
GROUP BY o.user_id, p.category
ORDER BY o.user_id, category_affinity DESC;
```

---

## 7. Session-Based Recs (Co-Views × Conversion)

```sql
WITH session_views AS (
  SELECT DISTINCT session_id, product_id FROM sessions WHERE event_type = 'view'
),
co_views AS (
  SELECT a.product_id AS p_a, b.product_id AS p_b,
    COUNT(DISTINCT a.session_id) AS co_view_sessions
  FROM session_views a
  JOIN session_views b ON a.session_id = b.session_id AND a.product_id < b.product_id
  GROUP BY a.product_id, b.product_id
),
conversion AS (
  SELECT product_id,
    ROUND(COUNT(DISTINCT CASE WHEN event_type='purchase' THEN session_id END)*100.0 /
      NULLIF(COUNT(DISTINCT CASE WHEN event_type='view' THEN session_id END),0),2) AS cvr
  FROM sessions GROUP BY product_id
)
SELECT cv.p_a, cv.p_b, cv.co_view_sessions, ca.cvr,
  ROUND(cv.co_view_sessions * ca.cvr / 100.0, 4) AS recommendation_score
FROM co_views cv JOIN conversion ca ON cv.p_b = ca.product_id
ORDER BY recommendation_score DESC;
```

---

## 8. Diversity Constraint

```sql
-- Max 3 items per category in final recommendations
WITH raw_recs AS (
  SELECT user_id, product_id, category, recommendation_score,
    ROW_NUMBER() OVER (
      PARTITION BY user_id, category
      ORDER BY recommendation_score DESC
    ) AS category_rank
  FROM recommendations_raw
)
SELECT user_id, product_id, category,
  recommendation_score, category_rank
FROM raw_recs WHERE category_rank <= 3
ORDER BY user_id, recommendation_score DESC;
```

---

## 9. Cosine Similarity (Implicit Feedback)

```sql
WITH weighted AS (
  SELECT user_id, product_id,
    SUM(CASE WHEN event_type='purchase'    THEN 10
             WHEN event_type='add_to_cart' THEN 3
             WHEN event_type='view'        THEN 1 ELSE 0 END) AS score
  FROM events GROUP BY user_id, product_id
),
similarity AS (
  SELECT a.user_id AS user_a, b.user_id AS user_b,
    SUM(a.score * b.score)           AS dot_product,
    SQRT(SUM(POW(a.score, 2)))       AS norm_a,
    SQRT(SUM(POW(b.score, 2)))       AS norm_b
  FROM weighted a JOIN weighted b
    ON a.product_id = b.product_id AND a.user_id < b.user_id
  GROUP BY a.user_id, b.user_id
)
SELECT user_a, user_b,
  ROUND(dot_product / NULLIF(norm_a * norm_b, 0), 4) AS cosine_similarity
FROM similarity ORDER BY cosine_similarity DESC;
```

---

## 10. Market Basket (Confidence + Lift)

```sql
WITH basket_pairs AS (
  SELECT a.product_id AS p1, b.product_id AS p2,
    COUNT(DISTINCT a.order_id) AS both_count
  FROM orders a JOIN orders b
    ON a.order_id = b.order_id AND a.product_id < b.product_id
  GROUP BY a.product_id, b.product_id
),
totals AS (SELECT product_id, COUNT(DISTINCT order_id) AS cnt FROM orders GROUP BY product_id),
total_orders AS (SELECT COUNT(DISTINCT order_id) AS n FROM orders)
SELECT bp.p1, bp.p2, bp.both_count,
  ROUND(bp.both_count*1.0/t1.cnt, 4) AS confidence_p1_to_p2,
  ROUND(bp.both_count*1.0/t2.cnt, 4) AS confidence_p2_to_p1,
  ROUND(bp.both_count*1.0*tot.n/(t1.cnt*t2.cnt), 4) AS lift
FROM basket_pairs bp
JOIN totals t1 ON bp.p1 = t1.product_id
JOIN totals t2 ON bp.p2 = t2.product_id
CROSS JOIN total_orders tot
WHERE bp.both_count >= 10
ORDER BY lift DESC;
```

---

## Practice Questions

### Q1 — Easy ✅
Top 10 most co-purchased product pairs.

```sql
SELECT a.product_id AS product_a, b.product_id AS product_b,
  COUNT(DISTINCT a.order_id) AS co_purchase_count
FROM orders a
JOIN orders b ON a.order_id = b.order_id AND a.product_id < b.product_id
GROUP BY a.product_id, b.product_id
ORDER BY co_purchase_count DESC LIMIT 10;
```

### Q2 — Medium ✅
Top 10 personalized recommendations for user 200 based on similar users.

```sql
WITH user_products AS (SELECT DISTINCT user_id, product_id FROM orders),
similar_users AS (
  SELECT b.user_id AS similar_user, COUNT(*) AS common_products
  FROM user_products a
  JOIN user_products b ON a.product_id = b.product_id AND b.user_id != 200
  WHERE a.user_id = 200
  GROUP BY b.user_id ORDER BY common_products DESC LIMIT 10
),
already_bought AS (SELECT DISTINCT product_id FROM orders WHERE user_id = 200)
SELECT up.product_id, p.name, p.category,
  SUM(su.common_products)         AS recommendation_score,
  COUNT(DISTINCT su.similar_user) AS supporting_users
FROM similar_users su
JOIN user_products up ON su.similar_user = up.user_id
JOIN products p       ON up.product_id   = p.product_id
WHERE up.product_id NOT IN (SELECT product_id FROM already_bought)
GROUP BY up.product_id, p.name, p.category
ORDER BY recommendation_score DESC LIMIT 10;
```

### Q3 — Hard ✅
Full pipeline: item-item → popularity fallback → diversity constraint → top 15.

```sql
WITH already_bought AS (
  SELECT DISTINCT product_id FROM orders WHERE user_id = 200
),
item_item AS (
  SELECT b.product_id,
    COUNT(*) AS item_item_score, 'item_item' AS source
  FROM orders a
  JOIN orders b ON a.order_id = b.order_id AND a.product_id != b.product_id
  WHERE a.product_id IN (SELECT product_id FROM already_bought)
    AND b.product_id NOT IN (SELECT product_id FROM already_bought)
  GROUP BY b.product_id
),
popularity AS (
  SELECT o.product_id,
    COUNT(DISTINCT o.order_id) AS pop_score, 'popularity_fallback' AS source
  FROM orders o
  WHERE o.product_id NOT IN (SELECT product_id FROM already_bought)
  GROUP BY o.product_id
),
combined AS (
  SELECT COALESCE(ii.product_id, pop.product_id) AS product_id,
    COALESCE(ii.item_item_score, pop.pop_score * 0.1) AS final_score,
    COALESCE(ii.source, pop.source) AS source
  FROM item_item ii
  FULL OUTER JOIN popularity pop ON ii.product_id = pop.product_id
),
with_category AS (
  SELECT c.product_id, p.name, p.category, c.final_score, c.source,
    ROW_NUMBER() OVER (
      PARTITION BY p.category ORDER BY c.final_score DESC
    ) AS category_rank
  FROM combined c JOIN products p ON c.product_id = p.product_id
)
SELECT product_id, name, category,
  ROUND(final_score, 4) AS score, source, category_rank
FROM with_category
WHERE category_rank <= 3
ORDER BY final_score DESC LIMIT 15;
```

---

## Key Takeaways

- **Jaccard** → `co_count / (count_a + count_b - co_count)` — normalizes by popularity
- **Lift** → `(co_count × total) / (count_a × count_b)` — > 1 means co-purchase beyond chance
- **Confidence** → `co_count / count_antecedent` — "if A, then B" probability
- **Cosine** → `dot_product / (norm_a × norm_b)` — angle between user vectors
- **Popularity fallback** → always needed for cold start (new users)
- **Diversity** → `ROW_NUMBER PARTITION BY user, category ≤ N`
- **Implicit feedback** → weight: purchase (10) > cart (3) > view (1)
- **Item-item > user-user** at scale — pre-computed, stable, no real-time similarity needed
- **Session-based** → co-views × conversion rate = quality signal

---

*Day 21 complete — 9 days to go 🚀*
