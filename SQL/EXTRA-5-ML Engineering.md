# Day 12 — ML Feature Engineering in SQL

## 1. Why SQL for Feature Engineering?

In FAANG companies, ML pipelines begin with SQL. Features are computed directly in BigQuery, Hive, or Spark SQL before being fed into models. Interviewers expect you to write production-grade feature pipelines in SQL.

```
Raw Tables → SQL Feature Engineering → Feature Store → Model Training
```

**Common MLE Interview Questions:**
- Compute lag features for time series
- Rolling aggregates as model inputs
- User behavior features (recency, frequency, diversity)
- Target encoding
- Train/test split logic in SQL

---

## 2. Lag Features

### Problem
You need to know a user's purchase history patterns. Did they buy more or less this time compared to their previous purchases?

### Input Table: `user_purchases`

| user_id | purchase_date | amount | category |
|---------|---------------|--------|----------|
| 101     | 2025-01-15    | 49.99  | Electronics |
| 101     | 2025-01-20    | 24.50  | Books |
| 101     | 2025-01-28    | 199.99 | Electronics |
| 102     | 2025-01-10    | 15.75  | Grocery |
| 102     | 2025-01-18    | 8.50   | Grocery |

### Solution
```sql
SELECT
  user_id, 
  purchase_date, 
  amount,
  category,
  
  -- Previous purchase amounts (lag features)
  LAG(amount, 1) OVER (PARTITION BY user_id ORDER BY purchase_date) AS prev_purchase_1,
  LAG(amount, 2) OVER (PARTITION BY user_id ORDER BY purchase_date) AS prev_purchase_2,
  LAG(amount, 3) OVER (PARTITION BY user_id ORDER BY purchase_date) AS prev_purchase_3,
  
  -- Time between purchases
  DATEDIFF(purchase_date,
    LAG(purchase_date, 1) OVER (PARTITION BY user_id ORDER BY purchase_date)
  ) AS days_since_last_purchase,
  
  -- Category sequence (for pattern recognition)
  LAG(category, 1) OVER (PARTITION BY user_id ORDER BY purchase_date) AS prev_category_1,
  LAG(category, 2) OVER (PARTITION BY user_id ORDER BY purchase_date) AS prev_category_2,
  
  -- Did they buy the same category twice?
  CASE 
    WHEN category = LAG(category, 1) OVER (PARTITION BY user_id ORDER BY purchase_date)
    THEN 1 ELSE 0 
  END AS repeat_category_purchase

FROM user_purchases
ORDER BY user_id, purchase_date;
```

### Output

| user_id | purchase_date | amount | category | prev_purchase_1 | prev_purchase_2 | days_since_last_purchase | repeat_category_purchase |
|---------|---------------|--------|----------|-----------------|-----------------|--------------------------|--------------------------|
| 101     | 2025-01-15    | 49.99  | Electronics | NULL | NULL | NULL | 0 |
| 101     | 2025-01-20    | 24.50  | Books | 49.99 | NULL | 5 | 0 |
| 101     | 2025-01-28    | 199.99 | Electronics | 24.50 | 49.99 | 8 | 0 |
| 102     | 2025-01-10    | 15.75  | Grocery | NULL | NULL | NULL | 0 |
| 102     | 2025-01-18    | 8.50   | Grocery | 15.75 | NULL | 8 | 1 |

### Explanation
- **`LAG(amount, 1)`**: Gets the amount from the immediately previous purchase by the same user
- **`DATEDIFF`**: Calculates days between purchases to measure purchase frequency
- **`repeat_category_purchase`**: Binary feature indicating if user bought same category twice in a row
- **NULL values**: First purchase for a user has no previous data

---

## 3. Rolling Aggregates as Features

### Problem
You need to capture user behavior trends over different time windows. How active was the user in the last week vs. last month?

### Input Table: `daily_user_activity`

| user_id | activity_date | events | revenue |
|---------|---------------|--------|---------|
| 101     | 2025-01-01    | 15     | 450.00  |
| 101     | 2025-01-02    | 22     | 675.50  |
| 101     | 2025-01-03    | 8      | 230.00  |
| 101     | 2025-01-04    | 31     | 920.25  |
| 101     | 2025-01-05    | 12     | 380.00  |
| 102     | 2025-01-01    | 5      | 125.00  |
| 102     | 2025-01-02    | 3      | 75.50   |

### Solution
```sql
SELECT
  user_id, 
  activity_date, 
  events, 
  revenue,
  
  -- 7-day rolling features (last week)
  SUM(events) OVER (
    PARTITION BY user_id ORDER BY activity_date
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  ) AS events_last_7d,
  
  AVG(revenue) OVER (
    PARTITION BY user_id ORDER BY activity_date
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  ) AS avg_revenue_7d,
  
  -- 30-day rolling features (last month)
  SUM(events) OVER (
    PARTITION BY user_id ORDER BY activity_date
    ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
  ) AS events_last_30d,
  
  AVG(revenue) OVER (
    PARTITION BY user_id ORDER BY activity_date
    ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
  ) AS avg_revenue_30d,
  
  -- Trend indicator: are they more active recently?
  ROUND(
    (SUM(events) OVER (
      PARTITION BY user_id ORDER BY activity_date
      ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) * 1.0) /
    NULLIF(SUM(events) OVER (
      PARTITION BY user_id ORDER BY activity_date
      ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ), 0), 2
  ) AS recent_activity_ratio,  -- >1 means more active in last 7 days vs last 30 days
  
  -- Revenue trend
  ROUND(
    (AVG(revenue) OVER (
      PARTITION BY user_id ORDER BY activity_date
      ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) * 1.0) /
    NULLIF(AVG(revenue) OVER (
      PARTITION BY user_id ORDER BY activity_date
      ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ), 0), 2
  ) AS revenue_trend_ratio

FROM daily_user_activity
ORDER BY user_id, activity_date;
```

### Output

| user_id | activity_date | events | revenue | events_last_7d | avg_revenue_7d | recent_activity_ratio |
|---------|---------------|--------|---------|----------------|----------------|----------------------|
| 101     | 2025-01-01    | 15     | 450.00  | 15             | 450.00         | NULL                 |
| 101     | 2025-01-02    | 22     | 675.50  | 37             | 562.75         | NULL                 |
| 101     | 2025-01-03    | 8      | 230.00  | 45             | 451.83         | NULL                 |
| 101     | 2025-01-04    | 31     | 920.25  | 76             | 568.94         | NULL                 |
| 101     | 2025-01-05    | 12     | 380.00  | 88             | 531.15         | NULL                 |
| 101     | 2025-01-06    | 25     | 740.00  | 113            | 566.00         | NULL                 |
| 101     | 2025-01-07    | 18     | 540.00  | 131            | 562.25         | 0.82                 |

### Explanation
- **`ROWS BETWEEN 6 PRECEDING AND CURRENT ROW`**: Includes the current row and the 6 previous rows (7 days total)
- **`recent_activity_ratio`**: Compare 7-day activity vs 30-day activity. If >1, user is becoming more active
- **`NULLIF`**: Prevents division by zero when no activity in the 30-day window
- **`ROUND`**: Makes output more readable for model inputs

---

## 4. User Behavior Features

### Problem
Build a comprehensive user profile for churn prediction or customer segmentation.

### Input Tables

**Orders:**

| user_id | order_date | amount | category |
|---------|------------|--------|----------|
| 101     | 2024-11-01 | 49.99  | Books    |
| 101     | 2024-11-15 | 24.50  | Books    |
| 101     | 2024-12-20 | 199.99 | Electronics |
| 102     | 2024-10-05 | 15.75  | Grocery  |
| 102     | 2024-11-10 | 8.50   | Grocery  |

**Sessions:**

| user_id | session_id | session_date | session_duration_mins | pages_per_session |
|---------|------------|--------------|----------------------|-------------------|
| 101     | s1         | 2024-11-01   | 15.5                 | 8                 |
| 101     | s2         | 2024-11-15   | 22.3                 | 12                |
| 101     | s3         | 2024-12-20   | 35.2                 | 18                |
| 102     | s4         | 2024-10-05   | 5.2                  | 3                 |

### Solution
```sql
WITH order_stats AS (
  SELECT
    user_id,
    COUNT(*) AS total_orders,
    SUM(amount) AS total_revenue,
    AVG(amount) AS avg_order_value,
    MIN(amount) AS min_order_value,
    MAX(amount) AS max_order_value,
    STDDEV(amount) AS stddev_order_value,
    COUNT(DISTINCT category) AS category_diversity,
    COUNT(DISTINCT DATE_FORMAT(order_date, '%Y-%m')) AS active_months,
    DATEDIFF(MAX(order_date), MIN(order_date)) AS customer_lifespan_days,
    DATEDIFF(CURRENT_DATE, MAX(order_date)) AS recency_days,
    -- Order frequency (orders per day of active life)
    ROUND(COUNT(*) / NULLIF(DATEDIFF(MAX(order_date), MIN(order_date)), 0), 4) 
      AS order_frequency_per_day
  FROM orders
  GROUP BY user_id
),

session_stats AS (
  SELECT 
    user_id,
    COUNT(DISTINCT session_id) AS total_sessions,
    AVG(session_duration_mins) AS avg_session_duration,
    AVG(pages_per_session) AS avg_pages_per_session
  FROM sessions
  GROUP BY user_id
),

user_segments AS (
  SELECT
    o.user_id,
    o.total_orders,
    o.total_revenue,
    o.avg_order_value,
    o.category_diversity,
    o.recency_days,
    
    -- RFM-style segmentation
    CASE 
      WHEN o.recency_days <= 7 AND o.total_orders >= 10 THEN 'High Value Active'
      WHEN o.recency_days <= 30 AND o.total_orders >= 5 THEN 'Medium Value Active'
      WHEN o.recency_days <= 30 AND o.total_orders < 5 THEN 'Low Value Active'
      WHEN o.recency_days > 30 AND o.recency_days <= 90 THEN 'At Risk'
      ELSE 'Churned'
    END AS user_segment,
    
    -- Power user indicator
    CASE WHEN o.total_orders >= 10 AND o.category_diversity >= 3 
      THEN 1 ELSE 0 
    END AS is_power_user
    
  FROM order_stats o
)

SELECT
  u.user_id,
  u.total_orders,
  u.total_revenue,
  ROUND(u.avg_order_value, 2) AS avg_order_value,
  u.category_diversity,
  u.recency_days,
  u.customer_lifespan_days,
  u.order_frequency_per_day,
  u.user_segment,
  u.is_power_user,
  
  -- Session engagement metrics
  COALESCE(s.total_sessions, 0) AS total_sessions,
  ROUND(COALESCE(s.avg_session_duration, 0), 1) AS avg_session_duration,
  ROUND(COALESCE(s.avg_pages_per_session, 0), 1) AS avg_pages_per_session,
  
  -- Combined engagement ratio
  ROUND(u.total_orders * 1.0 / NULLIF(s.total_sessions, 0), 2) AS orders_per_session

FROM user_segments u
LEFT JOIN session_stats s ON u.user_id = s.user_id
ORDER BY u.total_revenue DESC;
```

### Output

| user_id | total_orders | total_revenue | avg_order_value | category_diversity | recency_days | user_segment | is_power_user | orders_per_session |
|---------|--------------|---------------|-----------------|-------------------|--------------|--------------|---------------|-------------------|
| 101     | 3            | 274.48        | 91.49           | 2                 | 10           | Medium Value Active | 0 | 1.00 |
| 102     | 2            | 24.25         | 12.13           | 1                 | 25           | Low Value Active | 0 | 0.50 |

### Explanation
- **Recency**: Days since last order (lower = more engaged)
- **Category Diversity**: Number of different categories purchased (higher = more engaged)
- **RFM Segmentation**: Combines Recency, Frequency, Monetary value
- **Power User**: High order count (≥10) and diverse categories (≥3)
- **Orders per Session**: Indicates conversion efficiency

---

## 5. Target Encoding

### Problem
High-cardinality categorical variables (like city, device type, or browser) can't be one-hot encoded. You need to replace them with the average target value for that category.

### Input Table: `orders`

| order_id | user_id | city | amount | product_id |
|----------|---------|------|--------|------------|
| 1        | 101     | NYC  | 49.99  | P1         |
| 2        | 102     | NYC  | 24.50  | P2         |
| 3        | 103     | NYC  | 199.99 | P1         |
| 4        | 104     | BOS  | 15.75  | P3         |
| 5        | 105     | BOS  | 8.50   | P4         |
| 6        | 106     | NYC  | 450.00 | P1         |
| 7        | 107     | NYC  | 75.25  | P2         |
| 8        | 108     | BOS  | 125.50 | P3         |

### Solution

```sql
-- Step 1: Calculate city statistics
WITH city_stats AS (
  SELECT 
    city,
    COUNT(*) AS city_count,
    SUM(amount) AS city_total,
    AVG(amount) AS city_mean
  FROM orders
  GROUP BY city
),

-- Step 2: Calculate global statistics
global_stats AS (
  SELECT 
    AVG(amount) AS global_mean,
    COUNT(*) AS total_records
  FROM orders
),

-- Step 3: Apply different encoding methods
encoded_features AS (
  SELECT 
    o.order_id, 
    o.user_id, 
    o.city, 
    o.amount AS target,
    
    -- Raw city mean (causes data leakage)
    ROUND(cs.city_mean, 2) AS raw_city_encoding,
    
    -- Leave-One-Out encoding (prevents data leakage)
    ROUND(
      (cs.city_total - o.amount) / 
      NULLIF(cs.city_count - 1, 0), 
      2
    ) AS loo_city_encoding,
    
    -- Smoothed encoding (Bayesian smoothing with k=10)
    ROUND(
      (cs.city_total + 10 * gs.global_mean) / 
      (cs.city_count + 10), 
      2
    ) AS smoothed_city_encoding,
    
    -- Weighted encoding (blend city and global)
    ROUND(
      (cs.city_count * cs.city_mean + 5 * gs.global_mean) / 
      (cs.city_count + 5), 
      2
    ) AS weighted_city_encoding
    
  FROM orders o
  JOIN city_stats cs ON o.city = cs.city
  CROSS JOIN global_stats gs
)

SELECT *
FROM encoded_features
ORDER BY city, order_id;
```

### Output

| order_id | user_id | city | target | raw_city_encoding | loo_city_encoding | smoothed_city_encoding |
|----------|---------|------|--------|-------------------|-------------------|----------------------|
| 2        | 102     | NYC  | 24.50  | 159.95            | 174.24            | 145.30               |
| 6        | 106     | NYC  | 450.00 | 159.95            | 87.44             | 145.30               |
| 1        | 101     | NYC  | 49.99  | 159.95            | 181.94            | 145.30               |
| 3        | 103     | NYC  | 199.99 | 159.95            | 151.94            | 145.30               |
| 4        | 104     | BOS  | 15.75  | 49.92              | 60.58             | 53.10                |
| 8        | 108     | BOS  | 125.50 | 49.92              | 12.16             | 53.10                |

### Explanation

**Why target encoding matters:**
- **Raw City Mean**: 159.95 for NYC. But this includes the current row's target value (data leakage!)
- **Leave-One-Out (LOO)**: Removes current row's target from city mean. For NYC order_id=6 with target 450, the LOO mean is (total - 450)/(count - 1) = 87.44
- **Smoothed Encoding**: Blends city mean with global mean using k=10. For NYC: (city_total + 10×global_mean)/(city_count + 10) = 145.30
- **Data Leakage Prevention**: Always use LOO or smoothed encoding when creating training features

### When to Use Each Method

| Method | When to Use | Pros | Cons |
|--------|-------------|------|------|
| **Raw Mean** | Never in training | Simple | Causes data leakage |
| **LOO Encoding** | Training data | Prevents leakage, intuitive | Can be noisy for small categories |
| **Smoothed** | Small categories, high cardinality | Stable, handles rare categories | More complex |
| **Weighted** | Balanced approach | Flexible | Need to tune k parameter |

---

## 6. Train/Test Split in SQL

### Problem
You need to split your data into training and test sets in a reproducible way.

### Method 1: Random Split (NOT Recommended)
```sql
-- ❌ NOT REPRODUCIBLE — will change each run
SELECT user_id, features,
  CASE WHEN RAND() < 0.8 THEN 'train' ELSE 'test' END AS split
FROM feature_table;
```

### Method 2: Deterministic Split (Recommended)

```sql
-- ✅ REPRODUCIBLE — same user always gets same split
SELECT 
  user_id, 
  features,
  MOD(ABS(FARM_FINGERPRINT(CAST(user_id AS STRING))), 10) AS hash_value,
  
  -- 80/20 split
  CASE
    WHEN MOD(ABS(FARM_FINGERPRINT(CAST(user_id AS STRING))), 10) < 8
    THEN 'train' 
    ELSE 'test'
  END AS split
  
FROM feature_table;

-- MySQL equivalent
SELECT 
  user_id, 
  features,
  MOD(CRC32(user_id), 10) AS hash_value,
  CASE
    WHEN MOD(CRC32(user_id), 10) < 8 THEN 'train'
    ELSE 'test'
  END AS split
FROM feature_table;
```

### Method 3: Temporal Split (Best for Time Series)

```sql
-- ✅ Most correct for time-series models
-- Train: before cutoff | Validation: 1 month | Test: after
SELECT 
  user_id, 
  feature_date, 
  features,
  target,
  CASE
    WHEN feature_date < '2025-10-01' THEN 'train'
    WHEN feature_date < '2025-11-01' THEN 'validation'
    ELSE 'test'
  END AS split
FROM feature_table
ORDER BY feature_date;
```

### Output Example

| user_id | feature_date | features | split |
|---------|--------------|----------|-------|
| 101     | 2025-01-15   | {...}    | train |
| 102     | 2025-01-20   | {...}    | train |
| 103     | 2025-10-15   | {...}    | validation |
| 104     | 2025-11-15   | {...}    | test |

### Explanation

| Method | Use Case | Why |
|--------|----------|-----|
| **CRC32 Split** | User-level models | Same user always in same split, deterministic |
| **Temporal Split** | Time series | Prevents future data leaking into training |
| **Random Split** | Cross-sectional data (rarely used in production) | Simple but not reproducible |

> 💡 **Key Insight**: For churn prediction, always use temporal split. A user's behavior in November shouldn't be used to predict churn in October.

---

## 7. Feature Interaction & Derived Features

### Problem
The original features aren't enough. You need to create new features that capture complex patterns.

### Input Table: `user_features`

| user_id | age | country | device_type | amount | recency_days | total_orders |
|---------|-----|---------|-------------|--------|--------------|--------------|
| 101     | 28  | US      | mobile      | 49.99  | 5            | 12           |
| 102     | 45  | UK      | desktop     | 24.50  | 25           | 3            |
| 103     | 19  | FR      | mobile      | 199.99 | 2            | 8            |

### Solution
```sql
SELECT 
  user_id,
  age,
  country,
  device_type,
  amount,
  
  -- 1. PRICE SENSITIVITY: How much does this order deviate from user's average?
  ROUND(
    amount / NULLIF(AVG(amount) OVER (PARTITION BY user_id), 0), 
    2
  ) AS amount_vs_personal_avg,
  
  -- 2. CROSS FEATURES: Combine categorical variables
  CONCAT(device_type, '_', country) AS device_country,
  
  -- 3. BINNED FEATURES: Convert continuous to categorical
  CASE
    WHEN age BETWEEN 18 AND 24 THEN '18-24'
    WHEN age BETWEEN 25 AND 34 THEN '25-34'
    WHEN age BETWEEN 35 AND 44 THEN '35-44'
    WHEN age BETWEEN 45 AND 54 THEN '45-54'
    ELSE '55+'
  END AS age_bucket,
  
  -- 4. BOOLEAN FLAGS: Simple yet powerful
  CASE WHEN recency_days <= 7 THEN 1 ELSE 0 END AS active_last_7d,
  CASE WHEN recency_days <= 30 THEN 1 ELSE 0 END AS active_last_30d,
  CASE WHEN total_orders >= 10 THEN 1 ELSE 0 END AS is_power_user,
  
  -- 5. RATIO FEATURES: Capture relationships
  ROUND(total_orders * 1.0 / NULLIF(recency_days, 0), 2) AS orders_per_day,
  
  -- 6. INTERACTION TERM: Age × Activity
  CASE 
    WHEN age BETWEEN 18 AND 34 AND recency_days <= 7 THEN 'young_active'
    WHEN age BETWEEN 18 AND 34 AND recency_days > 7 THEN 'young_inactive'
    WHEN age >= 35 AND recency_days <= 7 THEN 'older_active'
    ELSE 'older_inactive'
  END AS age_activity_segment
  
FROM user_features;
```

### Output

| user_id | amount_vs_personal_avg | device_country | age_bucket | active_last_7d | is_power_user | age_activity_segment |
|---------|-----------------------|----------------|------------|----------------|---------------|----------------------|
| 101     | 1.00                   | mobile_US      | 25-34      | 1              | 1             | young_active         |
| 102     | 1.00                   | desktop_UK     | 45-54      | 0              | 0             | older_inactive       |
| 103     | 1.00                   | mobile_FR      | 18-24      | 1              | 0             | young_active         |

### Explanation

**Why these features matter:**

| Feature Type | Why It Works | Model Benefit |
|--------------|--------------|---------------|
| **Price Sensitivity** | Users with high deviation might be reacting to discounts or making special purchases | Captures spending patterns |
| **Cross Features** | Mobile users in US might behave differently than mobile users in FR | Captures geographical+device interactions |
| **Binned Features** | Age effect is likely non-linear | Allows non-linear relationships |
| **Boolean Flags** | Simple threshold-based behavior indicators | Model can easily learn simple rules |
| **Ratio Features** | Shows relative behavior (e.g., orders per day) | Normalizes across users |
| **Interaction Terms** | Age × Activity captures different behavioral segments | Explicitly models interactions |

---

## 8. Point-in-Time Correct Features (No Data Leakage)

### Problem
This is the #1 data leakage mistake in ML. You must only use data available **before** the prediction time.

### Input Table: `orders`

| order_id | user_id | amount | order_date |
|----------|---------|--------|------------|
| 1        | 101     | 49.99  | 2024-01-15 |
| 2        | 101     | 24.50  | 2024-02-20 |
| 3        | 101     | 199.99 | 2024-03-28 |
| 4        | 102     | 15.75  | 2024-01-10 |
| 5        | 102     | 8.50   | 2024-02-18 |

### Solution

```sql
-- ❌ WRONG: Uses future data (leakage!)
SELECT 
  o.order_id, 
  o.user_id,
  o.order_date,
  o.amount AS target,
  AVG(o2.amount) AS user_avg_amount  -- Includes future orders! Wrong!
FROM orders o
JOIN orders o2 ON o.user_id = o2.user_id
GROUP BY o.order_id;

-- ✅ CORRECT: Only uses PAST data
SELECT
  o.order_id,
  o.user_id,
  o.order_date,
  o.amount AS target,
  
  -- 1. Prior order count
  COUNT(prev.order_id) AS prior_order_count,
  
  -- 2. Prior average amount
  ROUND(AVG(prev.amount), 2) AS prior_avg_amount,
  
  -- 3. Prior total spend
  ROUND(SUM(prev.amount), 2) AS prior_total_spend,
  
  -- 4. Prior max amount
  ROUND(MAX(prev.amount), 2) AS prior_max_amount,
  
  -- 5. Days since last order
  DATEDIFF(o.order_date, MAX(prev.order_date)) AS days_since_last_order,
  
  -- 6. Prior order frequency (orders per month)
  ROUND(
    COUNT(prev.order_id) * 30.0 / 
    NULLIF(DATEDIFF(o.order_date, MIN(prev.order_date)), 0), 
    2
  ) AS prior_orders_per_month,
  
  -- 7. Prior category diversity
  COUNT(DISTINCT prev.category) AS prior_category_diversity

FROM orders o
LEFT JOIN orders prev
  ON  prev.user_id = o.user_id
  AND prev.order_date < o.order_date   -- ← CRITICAL: ONLY PRIOR ORDERS!
GROUP BY o.order_id, o.user_id, o.order_date, o.amount
ORDER BY o.user_id, o.order_date;
```

### Output

| order_id | user_id | order_date | target | prior_order_count | prior_avg_amount | days_since_last_order |
|----------|---------|------------|--------|-------------------|------------------|----------------------|
| 1        | 101     | 2024-01-15 | 49.99  | 0                 | NULL             | NULL                 |
| 2        | 101     | 2024-02-20 | 24.50  | 1                 | 49.99            | 36                   |
| 3        | 101     | 2024-03-28 | 199.99 | 2                 | 37.25            | 37                   |
| 4        | 102     | 2024-01-10 | 15.75  | 0                 | NULL             | NULL                 |
| 5        | 102     | 2024-02-18 | 8.50   | 1                 | 15.75            | 39                   |

### Explanation

**Why Point-in-Time Matters:**

| Scenario | Leakage Example | Impact |
|----------|----------------|--------|
| **Predicting next purchase** | Using all user orders including future ones | Model learns future patterns → Overly optimistic performance |
| **Churn prediction** | Using data from after the churn date | Model can "see" that user churned → Unrealistic accuracy |
| **Fraud detection** | Using future transactions as features | Model identifies fraud based on future behavior |

**Best Practices:**
- Always use `prev.date < current.date` (strictly less than)
- Never use `<=` because that includes the current transaction
- For date-based features, use `prev.date < current.date` not `<=`
- Always group by the target row's unique identifier

---

## 9. Complete Feature Pipeline Example

### Problem
Build a complete churn prediction feature set for a subscription service.

### Input Tables

**Users:**

| user_id | signup_date | plan_type | country | is_active |
|---------|-------------|-----------|---------|-----------|
| 101     | 2024-01-01  | premium   | US      | 1         |
| 102     | 2024-05-15  | basic     | UK      | 1         |
| 103     | 2024-08-01  | pro       | FR      | 1         |

**Orders:**

| user_id | order_date | amount | category |
|---------|------------|--------|----------|
| 101     | 2024-01-01 | 49.99  | Books    |
| 101     | 2024-02-15 | 24.50  | Books    |
| 101     | 2024-03-20 | 199.99 | Electronics |
| 102     | 2024-05-15 | 15.75  | Grocery  |
| 102     | 2024-06-10 | 8.50   | Grocery  |

**Events:**

| user_id | event_date | event_type |
|---------|------------|------------|
| 101     | 2024-01-01 | login      |
| 101     | 2024-01-02 | view       |
| 101     | 2024-01-03 | purchase   |

**Support:**

| user_id | ticket_date | resolved |
|---------|-------------|----------|
| 101     | 2024-02-01  | 1        |
| 101     | 2024-03-15  | 0        |

### Solution
```sql
-- Complete churn prediction feature set
WITH base AS (
  SELECT 
    user_id, 
    signup_date, 
    plan_type, 
    country,
    DATEDIFF(CURRENT_DATE, signup_date) AS account_age_days
  FROM users
  WHERE is_active = 1
),

order_features AS (
  SELECT 
    user_id,
    COUNT(*) AS total_orders,
    ROUND(SUM(amount), 2) AS ltv,
    ROUND(AVG(amount), 2) AS avg_order_value,
    DATEDIFF(CURRENT_DATE, MAX(order_date)) AS recency_days,
    DATEDIFF(MAX(order_date), MIN(order_date)) AS order_span_days,
    COUNT(DISTINCT category) AS category_diversity,
    
    -- Revenue last 30 days
    ROUND(SUM(CASE WHEN order_date >= CURRENT_DATE - INTERVAL 30 DAY 
             THEN amount ELSE 0 END), 2) AS revenue_last_30d,
    
    -- Revenue last 90 days
    ROUND(SUM(CASE WHEN order_date >= CURRENT_DATE - INTERVAL 90 DAY 
             THEN amount ELSE 0 END), 2) AS revenue_last_90d,
    
    -- Revenue trend (30d vs 90d)
    ROUND(
      SUM(CASE WHEN order_date >= CURRENT_DATE - INTERVAL 30 DAY THEN amount ELSE 0 END) /
      NULLIF(SUM(CASE WHEN order_date >= CURRENT_DATE - INTERVAL 90 DAY THEN amount ELSE 0 END), 0),
      2
    ) AS revenue_trend
    
  FROM orders 
  GROUP BY user_id
),

event_features AS (
  SELECT 
    user_id,
    COUNT(*) AS total_events,
    COUNT(DISTINCT event_type) AS event_diversity,
    DATEDIFF(CURRENT_DATE, MAX(event_date)) AS days_since_last_event,
    
    -- Events last 7 days
    SUM(CASE WHEN event_date >= CURRENT_DATE - INTERVAL 7 DAY 
        THEN 1 ELSE 0 END) AS events_last_7d,
    
    -- Events last 30 days
    SUM(CASE WHEN event_date >= CURRENT_DATE - INTERVAL 30 DAY 
        THEN 1 ELSE 0 END) AS events_last_30d,
    
    -- Engagement trend
    ROUND(
      SUM(CASE WHEN event_date >= CURRENT_DATE - INTERVAL 7 DAY THEN 1 ELSE 0 END) /
      NULLIF(SUM(CASE WHEN event_date >= CURRENT_DATE - INTERVAL 30 DAY THEN 1 ELSE 0 END), 0),
      2
    ) AS engagement_trend
  FROM events 
  GROUP BY user_id
),

support_features AS (
  SELECT 
    user_id,
    COUNT(*) AS support_tickets,
    ROUND(AVG(CASE WHEN resolved = 1 THEN 1.0 ELSE 0 END), 2) AS resolution_rate,
    
    -- Unresolved tickets
    SUM(CASE WHEN resolved = 0 THEN 1 ELSE 0 END) AS unresolved_tickets
  FROM support_tickets 
  GROUP BY user_id
),

-- RFM segmentation
rfm AS (
  SELECT 
    user_id,
    CASE 
      WHEN o.recency_days <= 7 AND o.total_orders >= 10 THEN 'High Value'
      WHEN o.recency_days <= 30 AND o.total_orders >= 5 THEN 'Medium Value'
      WHEN o.recency_days <= 30 AND o.total_orders < 5 THEN 'Low Value'
      WHEN o.
