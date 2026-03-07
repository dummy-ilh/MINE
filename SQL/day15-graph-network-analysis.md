# Day 15 — Graph & Network Analysis in SQL
**FAANG SQL 30-Day Prep**

---

## Concepts Covered

1. Representing Graphs in SQL
2. Basic Graph Queries (Degree)
3. Mutual Friends
4. Friend-of-Friend Recommendations
5. Shortest Path (BFS)
6. Connected Components
7. Influence & Centrality
8. Triangle Counting & Clustering Coefficient
9. FAANG Graph Patterns

---

## 1. Representing Graphs in SQL

```sql
-- Directed graph (follows, messages)
-- edges(src_user_id, dst_user_id, created_at)

-- Undirected graph (friendships)
-- friendships(user_id_1, user_id_2) — convention: user_id_1 < user_id_2

-- Normalize undirected to both directions
WITH all_edges AS (
  SELECT user_id_1 AS src, user_id_2 AS dst FROM friendships
  UNION ALL
  SELECT user_id_2, user_id_1 FROM friendships
)
```

---

## 2. Basic Graph Queries

```sql
-- Out-degree, in-degree, total degree
WITH out_deg AS (
  SELECT src_user_id AS user_id, COUNT(*) AS out_degree
  FROM edges GROUP BY src_user_id
),
in_deg AS (
  SELECT dst_user_id AS user_id, COUNT(*) AS in_degree
  FROM edges GROUP BY dst_user_id
)
SELECT
  COALESCE(o.user_id, i.user_id) AS user_id,
  COALESCE(out_degree, 0)        AS out_degree,
  COALESCE(in_degree, 0)         AS in_degree,
  COALESCE(out_degree, 0) +
  COALESCE(in_degree, 0)         AS total_degree
FROM out_deg o
FULL OUTER JOIN in_deg i ON o.user_id = i.user_id
ORDER BY total_degree DESC;
```

---

## 3. Mutual Friends

```sql
-- Mutual friends between user 101 and 202
WITH all_edges AS (
  SELECT user_id_1 AS src, user_id_2 AS dst FROM friendships
  UNION ALL
  SELECT user_id_2, user_id_1 FROM friendships
)
SELECT a.dst AS mutual_friend
FROM all_edges a
JOIN all_edges b
  ON  a.dst = b.dst
  AND a.src = 101
  AND b.src = 202;

-- Mutual friend count for ALL non-friend pairs
WITH all_edges AS (
  SELECT user_id_1 AS src, user_id_2 AS dst FROM friendships
  UNION ALL
  SELECT user_id_2, user_id_1 FROM friendships
)
SELECT a.src AS user_a, b.src AS user_b, COUNT(*) AS mutual_friends
FROM all_edges a
JOIN all_edges b ON a.dst = b.dst
WHERE a.src < b.src
  AND NOT EXISTS (
    SELECT 1 FROM friendships f
    WHERE (f.user_id_1 = a.src AND f.user_id_2 = b.src)
       OR (f.user_id_1 = b.src AND f.user_id_2 = a.src)
  )
GROUP BY a.src, b.src
ORDER BY mutual_friends DESC;
```

---

## 4. Friend-of-Friend Recommendations

```sql
WITH all_edges AS (
  SELECT user_id_1 AS src, user_id_2 AS dst FROM friendships
  UNION ALL
  SELECT user_id_2, user_id_1 FROM friendships
),
direct_friends AS (
  SELECT dst AS friend FROM all_edges WHERE src = 101
),
foaf AS (
  SELECT DISTINCT b.dst AS recommended_user
  FROM all_edges a
  JOIN all_edges b ON a.dst = b.src
  WHERE a.src = 101
    AND b.dst != 101
    AND b.dst NOT IN (SELECT friend FROM direct_friends)
)
SELECT f.recommended_user,
  COUNT(*) AS mutual_friend_count
FROM foaf f
JOIN all_edges a
  ON  a.dst = f.recommended_user
  AND a.src IN (SELECT friend FROM direct_friends)
GROUP BY f.recommended_user
ORDER BY mutual_friend_count DESC
LIMIT 10;
```

---

## 5. Shortest Path (BFS)

```sql
WITH RECURSIVE bfs AS (
  SELECT 1 AS src, user_id_2 AS dst, 1 AS hops,
    CAST(CONCAT('1->', user_id_2) AS CHAR(1000)) AS path
  FROM friendships WHERE user_id_1 = 1

  UNION ALL

  SELECT b.src, f.user_id_2, b.hops + 1,
    CONCAT(b.path, '->', f.user_id_2)
  FROM bfs b
  JOIN friendships f ON b.dst = f.user_id_1
  WHERE b.hops < 6
    AND FIND_IN_SET(f.user_id_2,
        REPLACE(b.path, '->', ',')) = 0
)
SELECT src, dst, hops, path
FROM bfs WHERE dst = 100
ORDER BY hops LIMIT 1;
```

---

## 6. Connected Components

```sql
WITH RECURSIVE components AS (
  SELECT user_id, user_id AS component_id FROM users

  UNION ALL

  SELECT f.user_id_2, LEAST(c.component_id, f.user_id_2)
  FROM components c
  JOIN friendships f ON c.user_id = f.user_id_1
  WHERE f.user_id_2 < c.component_id
)
SELECT component_id, COUNT(DISTINCT user_id) AS component_size
FROM components
GROUP BY component_id
ORDER BY component_size DESC;
```

---

## 7. Influence Score

```sql
WITH follower_counts AS (
  SELECT dst_user_id AS user_id, COUNT(*) AS followers
  FROM edges GROUP BY dst_user_id
),
influence AS (
  SELECT e.dst_user_id AS user_id,
    SUM(fc.followers) AS influence_score
  FROM edges e
  JOIN follower_counts fc ON e.src_user_id = fc.user_id
  GROUP BY e.dst_user_id
)
SELECT u.user_id, u.name,
  COALESCE(fc.followers, 0)     AS direct_followers,
  COALESCE(i.influence_score, 0) AS influence_score
FROM users u
LEFT JOIN follower_counts fc ON u.user_id = fc.user_id
LEFT JOIN influence i         ON u.user_id = i.user_id
ORDER BY influence_score DESC;
```

---

## 8. Triangle Counting

```sql
-- Total triangles in graph
WITH all_edges AS (
  SELECT user_id_1 AS a, user_id_2 AS b FROM friendships
  UNION ALL SELECT user_id_2, user_id_1 FROM friendships
)
SELECT COUNT(*) / 6 AS triangle_count
FROM all_edges ab
JOIN all_edges bc ON ab.b = bc.a AND ab.a != bc.b
JOIN all_edges ca ON bc.b = ca.a AND ca.b = ab.a;

-- Clustering coefficient per user
WITH all_edges AS (
  SELECT user_id_1 AS src, user_id_2 AS dst FROM friendships
  UNION ALL SELECT user_id_2, user_id_1 FROM friendships
),
degrees AS (
  SELECT src AS user_id, COUNT(*) AS degree
  FROM all_edges GROUP BY src
),
triangles AS (
  SELECT ab.src AS user_id, COUNT(*) AS tri_count
  FROM all_edges ab
  JOIN all_edges bc ON ab.dst = bc.src AND ab.src != bc.dst
  JOIN all_edges ca ON bc.dst = ca.src AND ca.dst = ab.src
  GROUP BY ab.src
)
SELECT d.user_id, d.degree,
  COALESCE(t.tri_count, 0) AS triangles,
  ROUND(COALESCE(t.tri_count, 0) * 1.0 /
    NULLIF(d.degree * (d.degree - 1), 0), 4) AS clustering_coeff
FROM degrees d
LEFT JOIN triangles t ON d.user_id = t.user_id
ORDER BY clustering_coeff DESC;
```

---

## 9. FAANG Graph Patterns

```sql
-- Viral content spread (Meta)
WITH RECURSIVE spread AS (
  SELECT poster_id AS user_id, post_id, 0 AS hop
  FROM posts WHERE post_id = 9999

  UNION ALL

  SELECT s2.sharer_id, s.post_id, s.hop + 1
  FROM spread s
  JOIN shares s2
    ON  s2.original_poster_id = s.user_id
    AND s2.post_id = s.post_id
  WHERE s.hop < 5
)
SELECT hop,
  COUNT(DISTINCT user_id)               AS users_reached,
  SUM(COUNT(DISTINCT user_id)) OVER (
    ORDER BY hop
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  )                                      AS cumulative_reach
FROM spread
GROUP BY hop ORDER BY hop;
```

```sql
-- 1st and 2nd degree connections (LinkedIn)
WITH all_edges AS (
  SELECT user_id_1 AS src, user_id_2 AS dst FROM connections
  UNION ALL SELECT user_id_2, user_id_1 FROM connections
),
first_degree AS (
  SELECT DISTINCT dst AS connection FROM all_edges WHERE src = 500
),
second_degree AS (
  SELECT DISTINCT b.dst AS connection
  FROM all_edges a JOIN all_edges b ON a.dst = b.src
  WHERE a.src = 500
    AND b.dst != 500
    AND b.dst NOT IN (SELECT connection FROM first_degree)
)
SELECT
  (SELECT COUNT(*) FROM first_degree)  AS first_degree_count,
  (SELECT COUNT(*) FROM second_degree) AS second_degree_count;
```

---

## Practice Questions

### Q1 — Easy ✅
Top 10 most connected users.

```sql
WITH all_edges AS (
  SELECT user_id_1 AS user_id, user_id_2 AS friend FROM friendships
  UNION ALL
  SELECT user_id_2, user_id_1 FROM friendships
)
SELECT user_id, COUNT(*) AS friend_count
FROM all_edges
GROUP BY user_id
ORDER BY friend_count DESC
LIMIT 10;
```

### Q2 — Medium ✅
Friend-of-friend recommendations for user 500.

```sql
WITH all_edges AS (
  SELECT user_id_1 AS src, user_id_2 AS dst FROM friendships
  UNION ALL SELECT user_id_2, user_id_1 FROM friendships
),
direct_friends AS (
  SELECT dst AS friend FROM all_edges WHERE src = 500
),
foaf AS (
  SELECT DISTINCT b.dst AS recommended_user_id
  FROM all_edges a
  JOIN all_edges b ON a.dst = b.src
  WHERE a.src = 500
    AND b.dst != 500
    AND b.dst NOT IN (SELECT friend FROM direct_friends)
)
SELECT f.recommended_user_id, COUNT(*) AS mutual_friend_count
FROM foaf f
JOIN all_edges a
  ON  a.dst = f.recommended_user_id
  AND a.src IN (SELECT friend FROM direct_friends)
GROUP BY f.recommended_user_id
ORDER BY mutual_friend_count DESC;
```

### Q3 — Hard ✅
Mutual follow pairs with shared follower count.

```sql
WITH mutual_follows AS (
  SELECT a.follower_id AS user_a, a.followee_id AS user_b
  FROM follows a
  JOIN follows b
    ON  a.follower_id = b.followee_id
    AND a.followee_id = b.follower_id
  WHERE a.follower_id < a.followee_id
),
mutual_followers AS (
  SELECT m.user_a, m.user_b,
    COUNT(DISTINCT f1.follower_id) AS shared_followers
  FROM mutual_follows m
  JOIN follows f1 ON f1.followee_id = m.user_a
  JOIN follows f2
    ON  f2.followee_id = m.user_b
    AND f2.follower_id = f1.follower_id
  GROUP BY m.user_a, m.user_b
)
SELECT ua.name AS user_a_name, ub.name AS user_b_name,
  mf.shared_followers
FROM mutual_followers mf
JOIN users ua ON mf.user_a = ua.user_id
JOIN users ub ON mf.user_b = ub.user_id
ORDER BY shared_followers DESC;
```

---

## Key Takeaways

- **Always normalize** undirected graphs with UNION ALL both directions
- **Mutual friends** → self-join all_edges on shared neighbor
- **FOAF** → 2-hop expansion, exclude self + existing friends
- **Shortest path** → recursive BFS + hop counter + cycle check
- **Connected components** → recursive CTE propagating minimum ID
- **Triangle counting** → 3-way self-join, divide by 6
- **Clustering coefficient** → triangles / (degree × (degree-1))
- **Viral spread** → recursive CTE from source, track hops

---

*Day 15 complete — 15 days to go 🚀*
