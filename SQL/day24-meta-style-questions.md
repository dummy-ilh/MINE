# Day 24 — Meta-Style SQL Questions
**FAANG SQL 30-Day Prep**

---

## What Meta Tests

```
1. Social graph metrics (friends, mutual connections, PYMK)
2. Content performance (engagement rate, feed ranking, reach)
3. Viral growth (K-factor, invite funnels, 2nd-degree growth)
4. Ads funnel (impressions → clicks → installs → purchases)
5. Streak & retention patterns
6. Group / page health scoring
```

---

## 1. Friend Recommendations (PYMK)

```sql
WITH edges AS (
  SELECT user_a AS src, user_b AS dst FROM friendships
  UNION ALL
  SELECT user_b, user_a FROM friendships
),
mutual_friends AS (
  SELECT a.src AS user_x, b.src AS user_y, COUNT(*) AS mutual_count
  FROM edges a JOIN edges b ON a.dst = b.dst AND a.src < b.src
  GROUP BY a.src, b.src
),
existing AS (
  SELECT user_a AS u1, user_b AS u2 FROM friendships
  UNION ALL SELECT user_b, user_a FROM friendships
),
ranked AS (
  SELECT mf.user_x, mf.user_y, mf.mutual_count,
    RANK() OVER (PARTITION BY mf.user_x ORDER BY mf.mutual_count DESC) AS rk
  FROM mutual_friends mf
  WHERE NOT EXISTS (SELECT 1 FROM existing e WHERE e.u1=mf.user_x AND e.u2=mf.user_y)
)
SELECT user_x AS user_id, user_y AS recommended_user_id,
  mutual_count, rk AS recommendation_rank
FROM ranked WHERE rk <= 5
ORDER BY user_id, rk;
```

> 💡 Key steps: (1) normalize to bidirectional edges, (2) self-join on shared neighbor, (3) exclude existing friendships with NOT EXISTS.

---

## 2. K-Factor & Invite Funnel

```sql
WITH invite_stats AS (
  SELECT sender_id,
    COUNT(*) AS invites_sent,
    SUM(CASE WHEN accepted_at IS NOT NULL THEN 1 ELSE 0 END) AS accepted,
    ROUND(SUM(CASE WHEN accepted_at IS NOT NULL THEN 1 ELSE 0 END)*100.0/COUNT(*),2) AS accept_rate
  FROM invites GROUP BY sender_id
)
SELECT
  COUNT(DISTINCT sender_id) AS inviters,
  SUM(accepted)             AS total_new_users,
  ROUND(SUM(accepted)*1.0/NULLIF(COUNT(DISTINCT sender_id),0),4) AS k_factor
  -- K > 1 = viral growth
FROM invite_stats;
```

---

## 3. Feed Ranking Score

```sql
WITH signals AS (
  SELECT p.post_id, p.author_id, p.created_at,
    SUM(CASE WHEN i.interaction_type='like'    THEN 1 ELSE 0 END) AS likes,
    SUM(CASE WHEN i.interaction_type='comment' THEN 1 ELSE 0 END) AS comments,
    SUM(CASE WHEN i.interaction_type='share'   THEN 1 ELSE 0 END) AS shares,
    SUM(CASE WHEN i.interaction_type='report'  THEN 1 ELSE 0 END) AS reports,
    TIMESTAMPDIFF(HOUR, p.created_at, NOW())                       AS age_hours
  FROM posts p LEFT JOIN interactions i ON p.post_id = i.post_id
  GROUP BY p.post_id, p.author_id, p.created_at
)
SELECT post_id, author_id, likes, comments, shares, reports, age_hours,
  ROUND(
    (likes*1 + comments*3 + shares*5)
    * EXP(-0.1 * age_hours)                -- time decay
    * CASE WHEN reports > 0 THEN 0.2 ELSE 1 END  -- report penalty
  , 4) AS feed_score,
  RANK() OVER (ORDER BY feed_score DESC) AS feed_rank
FROM signals WHERE reports < 5
ORDER BY feed_score DESC;
```

> 💡 Time decay: `EXP(-0.1 × hours)` → score halves roughly every 7 hours.

---

## 4. 2-Hop Graph (Degree of Connection)

```sql
WITH edges AS (
  SELECT user_a AS src, user_b AS dst FROM friendships UNION ALL
  SELECT user_b, user_a FROM friendships
),
first_degree  AS (SELECT dst AS user, 1 AS degree FROM edges WHERE src = 1001),
second_degree AS (
  SELECT e.dst AS user, 2 AS degree FROM edges e
  JOIN first_degree fd ON e.src = fd.user
  WHERE e.dst != 1001 AND e.dst NOT IN (SELECT user FROM first_degree)
)
SELECT user, degree FROM first_degree
UNION ALL
SELECT user, MIN(degree) FROM second_degree GROUP BY user
ORDER BY degree, user;
```

---

## 5. Content Engagement Rate

```sql
WITH post_metrics AS (
  SELECT p.post_id, p.author_id,
    COUNT(DISTINCT pv.user_id) AS viewers,
    COUNT(DISTINCT CASE WHEN i.interaction_type IN ('like','comment','share')
          THEN i.user_id END)  AS engagers,
    COUNT(DISTINCT CASE WHEN i.interaction_type='share'
          THEN i.user_id END)  AS sharers
  FROM posts p
  LEFT JOIN post_views pv ON p.post_id = pv.post_id
  LEFT JOIN interactions i ON p.post_id = i.post_id
  GROUP BY p.post_id, p.author_id
)
SELECT *,
  ROUND(engagers*100.0/NULLIF(viewers,0),4) AS engagement_rate,
  ROUND(sharers*1.0/NULLIF(viewers,0),4)    AS virality_score
FROM post_metrics ORDER BY engagement_rate DESC;
```

---

## 6. Meta Ads Funnel

```sql
SELECT campaign_id, DATE_FORMAT(event_time,'%Y-%m') AS month,
  COUNT(DISTINCT CASE WHEN event_type='impression'  THEN user_id END) AS reached,
  COUNT(DISTINCT CASE WHEN event_type='click'       THEN user_id END) AS clicked,
  COUNT(DISTINCT CASE WHEN event_type='app_install' THEN user_id END) AS installed,
  COUNT(DISTINCT CASE WHEN event_type='purchase'    THEN user_id END) AS purchasers,
  SUM(CASE WHEN event_type='purchase' THEN revenue ELSE 0 END)        AS revenue,
  ROUND(COUNT(DISTINCT CASE WHEN event_type='click'
    THEN user_id END)*100.0/NULLIF(COUNT(DISTINCT CASE WHEN event_type='impression'
    THEN user_id END),0),4) AS ctr,
  ROUND(COUNT(DISTINCT CASE WHEN event_type='purchase'
    THEN user_id END)*100.0/NULLIF(COUNT(DISTINCT CASE WHEN event_type='impression'
    THEN user_id END),0),4) AS overall_cvr
FROM ad_events
GROUP BY campaign_id, DATE_FORMAT(event_time,'%Y-%m')
ORDER BY month, revenue DESC;
```

---

## 7. Daily Streak (Gaps & Islands)

```sql
WITH daily AS (SELECT DISTINCT user_id, event_date FROM user_events),
gaps AS (
  SELECT user_id, event_date,
    DATEDIFF(event_date,
      LAG(event_date) OVER (PARTITION BY user_id ORDER BY event_date)) AS day_gap
  FROM daily
),
streak_groups AS (
  SELECT user_id, event_date,
    SUM(CASE WHEN day_gap != 1 OR day_gap IS NULL THEN 1 ELSE 0 END)
      OVER (PARTITION BY user_id ORDER BY event_date) AS streak_id
  FROM gaps
)
SELECT user_id, MIN(event_date) AS streak_start,
  MAX(event_date) AS streak_end, COUNT(*) AS streak_days
FROM streak_groups GROUP BY user_id, streak_id
HAVING MAX(event_date) >= CURRENT_DATE - INTERVAL 1 DAY
ORDER BY streak_days DESC;
```

---

## Practice Questions

### Q1 — Medium ✅
Top 5 friend recommendations per user based on mutual friends.

```sql
WITH edges AS (
  SELECT user_a AS src, user_b AS dst FROM friendships UNION ALL
  SELECT user_b, user_a FROM friendships
),
mutual AS (
  SELECT a.src AS user_x, b.src AS user_y, COUNT(*) AS mutual_count
  FROM edges a JOIN edges b ON a.dst=b.dst AND a.src < b.src
  GROUP BY a.src, b.src
),
existing AS (
  SELECT user_a AS u1, user_b AS u2 FROM friendships UNION ALL SELECT user_b, user_a FROM friendships
),
ranked AS (
  SELECT user_x, user_y, mutual_count,
    RANK() OVER (PARTITION BY user_x ORDER BY mutual_count DESC) AS rk
  FROM mutual m
  WHERE NOT EXISTS (SELECT 1 FROM existing e WHERE e.u1=m.user_x AND e.u2=m.user_y)
)
SELECT user_x AS user_id, user_y AS recommended_user_id,
  mutual_count, rk AS recommendation_rank
FROM ranked WHERE rk <= 5
ORDER BY user_id, rk;
```

### Q2 — Hard ✅
Author engagement rate + best post + MoM trend flag.

```sql
WITH post_er AS (
  SELECT p.post_id, p.author_id,
    DATE_FORMAT(p.created_at,'%Y-%m-01') AS post_month,
    COUNT(DISTINCT pv.user_id) AS viewers,
    COUNT(DISTINCT CASE WHEN i.interaction_type IN ('like','comment','share')
          THEN i.user_id END) AS engagers,
    ROUND(COUNT(DISTINCT CASE WHEN i.interaction_type IN ('like','comment','share')
          THEN i.user_id END)*100.0/NULLIF(COUNT(DISTINCT pv.user_id),0),4) AS er
  FROM posts p
  LEFT JOIN post_views pv ON p.post_id = pv.post_id
  LEFT JOIN interactions i ON p.post_id = i.post_id
  GROUP BY p.post_id, p.author_id, DATE_FORMAT(p.created_at,'%Y-%m-01')
),
best_post AS (
  SELECT author_id, post_month, post_id AS best_post_id
  FROM (
    SELECT author_id, post_month, post_id,
      ROW_NUMBER() OVER (PARTITION BY author_id, post_month ORDER BY er DESC) AS rn
    FROM post_er
  ) t WHERE rn = 1
),
author_monthly AS (
  SELECT pe.author_id, pe.post_month,
    COUNT(*) AS total_posts,
    ROUND(AVG(er),4) AS avg_er,
    bp.best_post_id
  FROM post_er pe
  JOIN best_post bp ON pe.author_id=bp.author_id AND pe.post_month=bp.post_month
  GROUP BY pe.author_id, pe.post_month, bp.best_post_id
),
with_trend AS (
  SELECT *,
    LAG(avg_er) OVER (PARTITION BY author_id ORDER BY post_month) AS prev_er
  FROM author_monthly
)
SELECT author_id, post_month, total_posts, avg_er, prev_er, best_post_id,
  ROUND((avg_er-prev_er)*100.0/NULLIF(prev_er,0),2) AS er_mom_pct,
  CASE WHEN (avg_er-prev_er)*100.0/NULLIF(prev_er,0) < -20
       THEN '🔴 ENGAGEMENT DROP' ELSE '✅ STABLE' END AS trend_flag
FROM with_trend ORDER BY author_id, post_month;
```

### Q3 — Very Hard ✅
Weekly growth report: signups, invite vs organic, K-factor, 2nd-degree growth, new friendships.

```sql
WITH weekly_signups AS (
  SELECT u.user_id, u.country, u.signup_date,
    DATE_SUB(u.signup_date, INTERVAL DAYOFWEEK(u.signup_date)-1 DAY) AS signup_week,
    CASE WHEN i.new_user_id IS NOT NULL THEN 'invite' ELSE 'organic' END AS source,
    i.sender_id AS invited_by
  FROM users u LEFT JOIN invites i ON u.user_id = i.new_user_id
),
weekly_invites AS (
  SELECT DATE_SUB(DATE(sent_at), INTERVAL DAYOFWEEK(DATE(sent_at))-1 DAY) AS invite_week,
    COUNT(*) AS invites_sent,
    SUM(CASE WHEN accepted_at IS NOT NULL THEN 1 ELSE 0 END) AS accepted,
    COUNT(DISTINCT sender_id) AS inviters,
    ROUND(SUM(CASE WHEN accepted_at IS NOT NULL THEN 1 ELSE 0 END)*100.0/COUNT(*),2) AS accept_rate
  FROM invites GROUP BY invite_week
),
second_degree AS (
  SELECT ws.signup_week,
    COUNT(DISTINCT i.sender_id) AS new_users_who_invited,
    SUM(CASE WHEN i.accepted_at IS NOT NULL THEN 1 ELSE 0 END) AS second_gen_signups
  FROM weekly_signups ws
  JOIN invites i ON ws.user_id=i.sender_id
    AND i.sent_at BETWEEN ws.signup_date AND ws.signup_date + INTERVAL 7 DAY
  GROUP BY ws.signup_week
),
new_friendships AS (
  SELECT ws.signup_week, COUNT(*) AS new_user_friendships
  FROM weekly_signups ws
  JOIN friendships f ON (f.user_a=ws.user_id OR f.user_b=ws.user_id)
    AND f.created_at BETWEEN ws.signup_date AND ws.signup_date + INTERVAL 7 DAY
  GROUP BY ws.signup_week
),
base AS (
  SELECT signup_week, country,
    COUNT(DISTINCT user_id) AS total_signups,
    COUNT(DISTINCT CASE WHEN source='invite'  THEN user_id END) AS invite_signups,
    COUNT(DISTINCT CASE WHEN source='organic' THEN user_id END) AS organic_signups
  FROM weekly_signups GROUP BY signup_week, country
)
SELECT b.signup_week, b.country,
  b.total_signups, b.invite_signups, b.organic_signups,
  ROUND(b.invite_signups*100.0/NULLIF(b.total_signups,0),2) AS invite_pct,
  wi.accept_rate,
  ROUND(b.invite_signups*1.0/NULLIF(wi.inviters,0),4) AS k_factor,
  sd.new_users_who_invited, sd.second_gen_signups,
  ROUND(sd.second_gen_signups*1.0/NULLIF(b.invite_signups,0),4) AS second_gen_rate,
  nf.new_user_friendships
FROM base b
LEFT JOIN weekly_invites  wi ON b.signup_week = wi.invite_week
LEFT JOIN second_degree   sd ON b.signup_week = sd.signup_week
LEFT JOIN new_friendships nf ON b.signup_week = nf.signup_week
ORDER BY b.signup_week, b.country;
```

---

## Key Takeaways

- **PYMK** → normalize to bidirectional edges → self-join on shared neighbor → exclude existing friends
- **K-factor** → `accepted invites / unique inviters` — K > 1 = viral
- **Feed score** → weighted interactions × time decay (EXP) × report penalty
- **Engagement rate** → `unique engagers / unique viewers` (not interactions/impressions)
- **Streak** → gaps-and-islands: cumulative SUM of `day_gap != 1` flags
- **2-hop graph** → two CTEs: first_degree → join edges again for second_degree
- **Ads funnel** → COUNT DISTINCT per stage, never COUNT raw events
- **Second-gen growth** → join new users back to invites on sender_id within 7 days
- **Always LEFT JOIN** in funnel queries → preserve users who dropped off

---

*Day 24 complete — 6 days to go 🚀*
