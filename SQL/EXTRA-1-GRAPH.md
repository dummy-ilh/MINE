# Graph SQL Masterclass
### FAANG 30-Day Prep — Complete Reference

---

## Table of Contents

1. [Graph Fundamentals in SQL](#1-graph-fundamentals-in-sql)
2. [Normalization — The Foundation CTE](#2-normalization--the-foundation-cte)
3. [Degree Counting](#3-degree-counting)
4. [Mutual Friends](#4-mutual-friends)
5. [Friend-of-Friend (FOAF) Recommendations](#5-friend-of-friend-foaf-recommendations)
6. [PYMK — People You May Know (Ranked)](#6-pymk--people-you-may-know-ranked)
7. [Shortest Path (BFS)](#7-shortest-path-bfs)
8. [Connected Components](#8-connected-components)
9. [Influence Score](#9-influence-score)
10. [Triangle Counting & Clustering Coefficient](#10-triangle-counting--clustering-coefficient)
11. [Viral Spread](#11-viral-spread)
12. [2-Hop Degree of Connection (LinkedIn)](#12-2-hop-degree-of-connection-linkedin)
13. [Feed Ranking Score](#13-feed-ranking-score)
14. [Daily Streak (Gaps & Islands)](#14-daily-streak-gaps--islands)
15. [Meta Ads Funnel](#15-meta-ads-funnel)
16. [K-Factor & Invite Funnel](#16-k-factor--invite-funnel)
17. [Weekly Growth Report (Very Hard)](#17-weekly-growth-report-very-hard)
18. [Key Takeaways Cheatsheet](#18-key-takeaways-cheatsheet)

---

## 1. Graph Fundamentals in SQL

### Core Idea

A graph is just **nodes + edges**. In SQL, edges live in a table. There are two flavors:

| Type | Example | Table Convention |
|------|---------|-----------------|
| **Directed** | Twitter follows, messages | `edges(src_user_id, dst_user_id)` |
| **Undirected** | Facebook friendships | `friendships(user_id_1, user_id_2)` where `user_id_1 < user_id_2` |

The undirected convention `user_id_1 < user_id_2` stores each edge **once** (no duplicates). But almost every query needs both directions, so you normalize first.

---

## 2. Normalization — The Foundation CTE

**Every graph query starts here. Memorize this pattern.**

### The Problem
`friendships(user_id_1, user_id_2)` stores (1, 5) but not (5, 1).  
When you want "all friends of user 5", a simple `WHERE user_id_1 = 5` misses half the rows.

### The Fix — `UNION ALL` both directions

```sql
WITH all_edges AS (
  SELECT user_id_1 AS src, user_id_2 AS dst FROM friendships
  UNION ALL
  SELECT user_id_2 AS src, user_id_1 AS dst FROM friendships
)
```

### Input → Output Trace

**Input `friendships`:**
```
user_id_1 | user_id_2
----------+----------
1         | 3
1         | 5
3         | 5
```

**`all_edges` output:**
```
src | dst
----+----
1   | 3      ← original row
1   | 5      ← original row
3   | 5      ← original row
3   | 1      ← flipped
5   | 1      ← flipped
5   | 3      ← flipped
```

Now for any user X: `WHERE src = X` gives ALL their friends.

> **Why `UNION ALL` not `UNION`?** `UNION` deduplicates, which is slower and unnecessary since (1→3) and (3→1) are never the same row.

---

## 3. Degree Counting

**"How many friends does each user have?"**

### Query

```sql
WITH all_edges AS (
  SELECT user_id_1 AS src, user_id_2 AS dst FROM friendships
  UNION ALL
  SELECT user_id_2, user_id_1 FROM friendships
)
SELECT user_id, COUNT(*) AS friend_count
FROM all_edges
GROUP BY user_id      -- wait, column is 'src' here, rename below
ORDER BY friend_count DESC
LIMIT 10;
```

Correct version:
```sql
WITH all_edges AS (
  SELECT user_id_1 AS src, user_id_2 AS dst FROM friendships
  UNION ALL
  SELECT user_id_2 AS src, user_id_1 AS dst FROM friendships
)
SELECT src AS user_id, COUNT(*) AS friend_count
FROM all_edges
GROUP BY src
ORDER BY friend_count DESC
LIMIT 10;
```

### CTE-Level I/O Trace

**`all_edges` (same as above — 6 rows):**
```
src | dst
1   | 3
1   | 5
3   | 5
3   | 1
5   | 1
5   | 3
```

**`GROUP BY src` + `COUNT(*)`:**
```
src | COUNT(*)
----+---------
1   | 2
3   | 2
5   | 2
```

**Final output (ORDER BY, LIMIT 10):**
```
user_id | friend_count
--------+-------------
1       | 2
3       | 2
5       | 2
```

### Directed Graph Variant

For a directed graph (followers), you want in-degree, out-degree, and total:

```sql
WITH out_deg AS (
  SELECT src_user_id AS user_id, COUNT(*) AS out_degree
  FROM edges GROUP BY src_user_id
),
in_deg AS (
  SELECT dst_user_id AS user_id, COUNT(*) AS in_degree
  FROM edges GROUP BY dst_user_id
)
SELECT
  COALESCE(o.user_id, i.user_id)       AS user_id,
  COALESCE(out_degree, 0)              AS out_degree,
  COALESCE(in_degree,  0)              AS in_degree,
  COALESCE(out_degree, 0) +
  COALESCE(in_degree,  0)              AS total_degree
FROM out_deg o
FULL OUTER JOIN in_deg i ON o.user_id = i.user_id
ORDER BY total_degree DESC;
```

**Why `FULL OUTER JOIN`?** A user might only appear as a source (no followers) or only as a destination (never followed anyone). FULL OUTER JOIN preserves both sides.

**I/O Trace:**

Input `edges`:
```
src | dst
1   | 2
1   | 3
2   | 3
```

`out_deg`:
```
user_id | out_degree
--------+-----------
1       | 2
2       | 1
```

`in_deg`:
```
user_id | in_degree
--------+----------
2       | 1
3       | 2
```

After FULL OUTER JOIN:
```
user_id | out_degree | in_degree | total_degree
--------+------------+-----------+-------------
1       | 2          | 0         | 2
2       | 1          | 1         | 2
3       | 0          | 2         | 2
```

---

## 4. Mutual Friends

**"What friends do user 101 and user 202 share?"**

### The Core Idea

A mutual friend of X and Y is a node Z where:
- Edge(X → Z) exists
- Edge(Y → Z) exists

In SQL: self-join `all_edges` matching on `dst`.

### Query — Mutual Friends Between Two Specific Users

```sql
WITH all_edges AS (
  SELECT user_id_1 AS src, user_id_2 AS dst FROM friendships
  UNION ALL
  SELECT user_id_2, user_id_1 FROM friendships
)
SELECT a.dst AS mutual_friend
FROM all_edges a
JOIN all_edges b
  ON  a.dst = b.dst      -- same destination = shared friend
  AND a.src = 101        -- friends of 101
  AND b.src = 202;       -- friends of 202
```

### CTE-Level I/O Trace

Input `friendships`:
```
user_id_1 | user_id_2
101       | 5
101       | 7
202       | 5
202       | 9
```

`all_edges`:
```
src | dst
101 | 5
101 | 7
202 | 5
202 | 9
5   | 101
7   | 101
5   | 202
9   | 202
```

Self-join `a` (src=101) × `b` (src=202) on `a.dst = b.dst`:
```
a.src | a.dst | b.src | b.dst  → selected: a.dst
101   | 5     | 202   | 5      → 5  ✅ mutual
101   | 7     | —     | —      → no match (202 not friends with 7)
```

**Output:**
```
mutual_friend
-------------
5
```

### Query — Mutual Friend Count for ALL Non-Friend Pairs

```sql
WITH all_edges AS (
  SELECT user_id_1 AS src, user_id_2 AS dst FROM friendships
  UNION ALL
  SELECT user_id_2, user_id_1 FROM friendships
)
SELECT a.src AS user_a, b.src AS user_b, COUNT(*) AS mutual_friends
FROM all_edges a
JOIN all_edges b ON a.dst = b.dst    -- shared neighbor
WHERE a.src < b.src                  -- avoid duplicates (1,2) and (2,1)
  AND NOT EXISTS (                   -- exclude pairs already friends
    SELECT 1 FROM friendships f
    WHERE (f.user_id_1 = a.src AND f.user_id_2 = b.src)
       OR (f.user_id_1 = b.src AND f.user_id_2 = a.src)
  )
GROUP BY a.src, b.src
ORDER BY mutual_friends DESC;
```

**Why `a.src < b.src`?**  
Without it, pair (1,2) appears twice: once as (1,2) and once as (2,1). The `<` keeps only the canonical form.

**Why `NOT EXISTS` instead of `NOT IN`?**  
`NOT IN` returns no rows if any value in the subquery is NULL. `NOT EXISTS` is NULL-safe.

---

## 5. Friend-of-Friend (FOAF) Recommendations

**"Recommend users exactly 2 hops from user 500 who are not already friends."**

### The Mental Model

```
500 → [direct friends] → [their friends = FOAFs]
```

We want FOAFs, minus user 500 themselves, minus existing direct friends.

### Query

```sql
WITH all_edges AS (
  SELECT user_id_1 AS src, user_id_2 AS dst FROM friendships
  UNION ALL
  SELECT user_id_2, user_id_1 FROM friendships
),
-- Step 1: Who are 500's direct friends?
direct_friends AS (
  SELECT dst AS friend
  FROM all_edges
  WHERE src = 500
),
-- Step 2: Who are friends of those friends?
foaf AS (
  SELECT DISTINCT b.dst AS recommended_user_id
  FROM all_edges a
  JOIN all_edges b ON a.dst = b.src    -- hop 1 → hop 2
  WHERE a.src = 500                    -- start from 500
    AND b.dst != 500                   -- exclude self
    AND b.dst NOT IN (SELECT friend FROM direct_friends)  -- exclude existing friends
)
-- Step 3: Count mutual friends for ranking
SELECT f.recommended_user_id, COUNT(*) AS mutual_friend_count
FROM foaf f
JOIN all_edges a
  ON  a.dst = f.recommended_user_id
  AND a.src IN (SELECT friend FROM direct_friends)
GROUP BY f.recommended_user_id
ORDER BY mutual_friend_count DESC;
```

### CTE-Level I/O Trace

Input `friendships`:
```
user_id_1 | user_id_2
500       | 1
500       | 2
1         | 3
2         | 3
2         | 4
```

`all_edges` (only subset shown):
```
src | dst
500 | 1
500 | 2
1   | 500
2   | 500
1   | 3
3   | 1
2   | 3
3   | 2
2   | 4
4   | 2
```

`direct_friends`:
```
friend
------
1
2
```

`foaf` — join `all_edges` (a: src=500) with `all_edges` (b) on `a.dst = b.src`:
```
a.src | a.dst | b.src | b.dst | filters
500   | 1     | 1     | 3     | 3 ≠ 500, 3 NOT IN {1,2} → KEEP
500   | 1     | 1     | 500   | 500 = 500 → EXCLUDE (self)
500   | 2     | 2     | 3     | 3 ≠ 500, 3 NOT IN {1,2} → KEEP (dup)
500   | 2     | 2     | 4     | 4 ≠ 500, 4 NOT IN {1,2} → KEEP
```

After DISTINCT:
```
recommended_user_id
-------------------
3
4
```

Mutual friend count — for user 3:
- Friends of 500 who also know 3: user 1 (1→3) and user 2 (2→3) → count = 2

For user 4:
- Friends of 500 who also know 4: user 2 (2→4) → count = 1

**Final output:**
```
recommended_user_id | mutual_friend_count
--------------------+--------------------
3                   | 2
4                   | 1
```

---

## 6. PYMK — People You May Know (Ranked)

**"For EVERY user, give top 5 friend recommendations ranked by mutual friend count."**

This generalizes FOAF to all users simultaneously.

### Query

```sql
WITH edges AS (
  SELECT user_a AS src, user_b AS dst FROM friendships
  UNION ALL
  SELECT user_b, user_a FROM friendships
),
-- Mutual friend count for all non-friend pairs
mutual_friends AS (
  SELECT a.src AS user_x, b.src AS user_y, COUNT(*) AS mutual_count
  FROM edges a
  JOIN edges b
    ON  a.dst = b.dst      -- shared neighbor
    AND a.src < b.src      -- canonical order, avoid duplicates
  GROUP BY a.src, b.src
),
-- All existing friendships (both directions for easy lookup)
existing AS (
  SELECT user_a AS u1, user_b AS u2 FROM friendships
  UNION ALL
  SELECT user_b, user_a FROM friendships
),
-- Rank recommendations per user
ranked AS (
  SELECT mf.user_x, mf.user_y, mf.mutual_count,
    RANK() OVER (
      PARTITION BY mf.user_x        -- rank within each user's candidates
      ORDER BY mf.mutual_count DESC -- higher mutual = better rank
    ) AS rk
  FROM mutual_friends mf
  WHERE NOT EXISTS (                -- exclude already-friends
    SELECT 1 FROM existing e
    WHERE e.u1 = mf.user_x AND e.u2 = mf.user_y
  )
)
SELECT user_x AS user_id,
       user_y AS recommended_user_id,
       mutual_count,
       rk AS recommendation_rank
FROM ranked
WHERE rk <= 5
ORDER BY user_id, rk;
```

### CTE-Level I/O Trace

Input `friendships`:
```
user_a | user_b
-------+-------
1      | 2
1      | 3
2      | 3
2      | 4
3      | 5
```

`edges` (bidirectional, 10 rows):
```
src | dst
1   | 2
1   | 3
2   | 3
2   | 4
3   | 5
2   | 1
3   | 1
3   | 2
4   | 2
5   | 3
```

`mutual_friends` — key join: `a.dst = b.dst AND a.src < b.src`  
Example: a=(src=1,dst=3) joins b=(src=2,dst=3) → user_x=1, user_y=2, but (1,2) are already friends.  
a=(src=1,dst=3) joins b=(src=5,dst=3) → user_x=1, user_y=5, mutual=1

```
user_x | user_y | mutual_count
-------+--------+-------------
1      | 4      | 1   (shared neighbor: 2)
1      | 5      | 1   (shared neighbor: 3)
2      | 5      | 1   (shared neighbor: 3)
3      | 4      | 1   (shared neighbor: 2)
```

`existing` → all directed pairs from friendships.

After `NOT EXISTS` filter (remove actual friends), same rows pass since (1,4), (1,5), (2,5), (3,4) are not friends.

`ranked` with `RANK() OVER (PARTITION BY user_x)`:
```
user_x | user_y | mutual_count | rk
-------+--------+--------------+----
1      | 4      | 1            | 1  (tied)
1      | 5      | 1            | 1  (tied — RANK gives same rank)
2      | 5      | 1            | 1
3      | 4      | 1            | 1
```

**Final output:**
```
user_id | recommended_user_id | mutual_count | recommendation_rank
--------+---------------------+--------------+--------------------
1       | 4                   | 1            | 1
1       | 5                   | 1            | 1
2       | 5                   | 1            | 1
3       | 4                   | 1            | 1
```

> **RANK vs ROW_NUMBER:** `RANK()` gives ties the same rank (1,1,3). `ROW_NUMBER()` forces unique ranks (1,2,3). Use `RANK()` for "top 5 by score" to be fair to tied candidates. Use `ROW_NUMBER()` when you need exactly N rows.

---

## 7. Shortest Path (BFS)

**"What's the shortest path between user 1 and user 100?"**

### The Mental Model — Recursive BFS

```
Level 0: {1}
Level 1: {friends of 1}
Level 2: {friends of friends of 1, not already seen}
...stop when target found
```

SQL implements this with a **recursive CTE**.

### Query

```sql
WITH RECURSIVE bfs AS (
  -- Anchor: start from user 1, expand to direct friends
  SELECT
    1 AS src,
    user_id_2 AS dst,
    1 AS hops,
    CAST(CONCAT('1->', user_id_2) AS CHAR(1000)) AS path
  FROM friendships
  WHERE user_id_1 = 1

  UNION ALL

  -- Recursive: expand one more hop
  SELECT
    b.src,
    f.user_id_2,
    b.hops + 1,
    CONCAT(b.path, '->', f.user_id_2)
  FROM bfs b
  JOIN friendships f ON b.dst = f.user_id_1
  WHERE b.hops < 6                                           -- max depth guard
    AND FIND_IN_SET(f.user_id_2, REPLACE(b.path, '->', ',')) = 0  -- cycle check
)
SELECT src, dst, hops, path
FROM bfs
WHERE dst = 100
ORDER BY hops
LIMIT 1;
```

### CTE-Level I/O Trace

Input `friendships`:
```
user_id_1 | user_id_2
1         | 2
1         | 3
2         | 4
3         | 100
```

**Anchor (hops=1):**
```
src | dst | hops | path
1   | 2   | 1    | 1->2
1   | 3   | 1    | 1->3
```

**Recursive iteration 1 (hops=2):**  
Take each row from anchor, join `friendships` on `b.dst = f.user_id_1`:
- Row (dst=2): joins → user_id_2=4 → path='1->2->4'
- Row (dst=3): joins → user_id_2=100 → path='1->3->100' ✅

```
src | dst | hops | path
1   | 4   | 2    | 1->2->4
1   | 100 | 2    | 1->3->100
```

**Filter `WHERE dst = 100`:**
```
src | dst | hops | path
1   | 100 | 2    | 1->3->100
```

**Output (ORDER BY hops LIMIT 1):**
```
src | dst | hops | path
1   | 100 | 2    | 1->3->100
```

### Key Concepts

**Cycle check — `FIND_IN_SET`:**  
Path '1->2->4' stored as string. To check if node 3 is already visited, convert '->' to ',' → '1,2,4', then `FIND_IN_SET(3, '1,2,4') = 0` → not visited. This prevents infinite loops.

**`hops < 6` guard:**  
Social networks follow the "six degrees of separation" rule. Capping at 6 prevents runaway recursion if the target is unreachable.

**Why `UNION ALL` not `UNION`?**  
Recursive CTEs require `UNION ALL`. The deduplication in `UNION` is incompatible with recursive semantics in most engines.

---

## 8. Connected Components

**"Which users are in the same friend network? How large is each component?"**

### The Idea — Minimum ID Propagation

Each user starts as their own component (ID = themselves). Recursively, a user inherits the MINIMUM component ID of their neighbors. Eventually, all users in the same connected group converge to the same minimum ID.

```
Users: 1, 2, 3, 4, 5
Friendships: (1,2), (2,3), (4,5)
Components: {1,2,3} → ID=1 and {4,5} → ID=4
```

### Query

```sql
WITH RECURSIVE components AS (
  -- Anchor: each user is their own component
  SELECT user_id, user_id AS component_id FROM users

  UNION ALL

  -- Recursive: pull minimum component_id from neighbors
  SELECT f.user_id_2, LEAST(c.component_id, f.user_id_2)
  FROM components c
  JOIN friendships f ON c.user_id = f.user_id_1
  WHERE f.user_id_2 < c.component_id   -- only update if we find a smaller ID
)
SELECT component_id,
       COUNT(DISTINCT user_id) AS component_size
FROM components
GROUP BY component_id
ORDER BY component_size DESC;
```

### CTE-Level I/O Trace

Input `users`: 1, 2, 3, 4, 5  
Input `friendships`: (1,2), (2,3), (4,5)

**Anchor:**
```
user_id | component_id
1       | 1
2       | 2
3       | 3
4       | 4
5       | 5
```

**Recursive iteration 1:**  
Join on `f.user_id_1 = c.user_id`, check `f.user_id_2 < c.component_id`:
- user 2 (comp=2) has neighbor 1 → LEAST(2,1)=1 → update user_id_2=... 

Actually the propagation goes: user 1 (comp=1) has neighbor 2 → LEAST(1,2)=1 → user 2 gets comp=1  
User 2 (comp=2) has neighbor 3 → LEAST(2,3)=2 → user 3 gets comp=2 (will converge next pass)  
User 4 (comp=4) has neighbor 5 → LEAST(4,5)=4 → user 5 gets comp=4

**After convergence:**
```
user_id | component_id
1       | 1
2       | 1
3       | 1
4       | 4
5       | 4
```

**Final output:**
```
component_id | component_size
-------------+---------------
1            | 3
4            | 2
```

---

## 9. Influence Score

**"Whose followers have the most followers? (Weighted reach)"**

A user is influential not just by follower count, but by having *followers who are themselves popular*.

### Query

```sql
-- Step 1: Direct follower counts
WITH follower_counts AS (
  SELECT dst_user_id AS user_id, COUNT(*) AS followers
  FROM edges
  GROUP BY dst_user_id
),
-- Step 2: Sum up followers-of-followers for each user
influence AS (
  SELECT e.dst_user_id AS user_id,
    SUM(fc.followers) AS influence_score
  FROM edges e
  JOIN follower_counts fc
    ON e.src_user_id = fc.user_id   -- follower's own follower count
  GROUP BY e.dst_user_id
)
SELECT
  u.user_id,
  u.name,
  COALESCE(fc.followers, 0)      AS direct_followers,
  COALESCE(i.influence_score, 0) AS influence_score
FROM users u
LEFT JOIN follower_counts fc ON u.user_id = fc.user_id
LEFT JOIN influence i         ON u.user_id = i.user_id
ORDER BY influence_score DESC;
```

### CTE-Level I/O Trace

Input `edges`:
```
src | dst
A   | C       (A follows C)
B   | C       (B follows C)
D   | A       (D follows A)
E   | A       (E follows A)
F   | B       (F follows B)
```

`follower_counts`:
```
user_id | followers
A       | 2   (D, E follow A)
B       | 1   (F follows B)
C       | 2   (A, B follow C)
```

`influence` — for user C:  
C is followed by A (who has 2 followers) and B (who has 1 follower).  
influence_score = 2 + 1 = 3

For user A:  
A is followed by D (0 followers) and E (0 followers).  
influence_score = 0

```
user_id | influence_score
C       | 3
A       | 0
```

**Final output:**
```
user_id | name | direct_followers | influence_score
--------+------+------------------+----------------
C       | ...  | 2                | 3
A       | ...  | 2                | 0
B       | ...  | 1                | 0
```

---

## 10. Triangle Counting & Clustering Coefficient

### Part A — Total Triangles

**"How many closed triangles exist in the graph?"**

A triangle is three nodes A-B-C where all three edges exist.

```sql
WITH all_edges AS (
  SELECT user_id_1 AS a, user_id_2 AS b FROM friendships
  UNION ALL SELECT user_id_2, user_id_1 FROM friendships
)
SELECT COUNT(*) / 6 AS triangle_count
FROM all_edges ab
JOIN all_edges bc ON ab.b = bc.a AND ab.a != bc.b   -- hop A→B→C
JOIN all_edges ca ON bc.b = ca.a AND ca.b = ab.a;   -- close C→A
```

**Why divide by 6?**  
Triangle A-B-C is counted 6 times in the directed representation:
- (A→B→C), (A→C→B), (B→A→C), (B→C→A), (C→A→B), (C→B→A)

### I/O Trace

`all_edges` for triangle {1,2,3}:
```
a | b
1 | 2
2 | 1
1 | 3
3 | 1
2 | 3
3 | 2
```

3-way join paths that complete a triangle:
```
ab(1→2) → bc(2→3) → ca(3→1) ✅
ab(1→3) → bc(3→2) → ca(2→1) ✅
... (4 more permutations)
```

COUNT(*) = 6 → 6/6 = **1 triangle**

---

### Part B — Clustering Coefficient

**"For user X with degree k and T triangles, what fraction of possible triangles are closed?"**

Formula: `CC = T / (k × (k-1))` where k=degree and T=triangles involving this node.

```sql
WITH all_edges AS (
  SELECT user_id_1 AS src, user_id_2 AS dst FROM friendships
  UNION ALL SELECT user_id_2, user_id_1 FROM friendships
),
-- Degree per user
degrees AS (
  SELECT src AS user_id, COUNT(*) AS degree
  FROM all_edges GROUP BY src
),
-- Triangles per user (as one vertex of the triangle)
triangles AS (
  SELECT ab.src AS user_id, COUNT(*) AS tri_count
  FROM all_edges ab
  JOIN all_edges bc ON ab.dst = bc.src AND ab.src != bc.dst   -- A→B→C
  JOIN all_edges ca ON bc.dst = ca.src AND ca.dst = ab.src    -- C→A
  GROUP BY ab.src
)
SELECT
  d.user_id,
  d.degree,
  COALESCE(t.tri_count, 0) AS triangles,
  ROUND(
    COALESCE(t.tri_count, 0) * 1.0
    / NULLIF(d.degree * (d.degree - 1), 0)   -- NULLIF prevents div/0 when degree<2
  , 4) AS clustering_coeff
FROM degrees d
LEFT JOIN triangles t ON d.user_id = t.user_id
ORDER BY clustering_coeff DESC;
```

### I/O Trace

For a graph where users {1,2,3,4} with friendships (1,2),(1,3),(1,4),(2,3):

User 1 has degree 3 (friends: 2,3,4).  
Triangles containing 1: only {1,2,3} (since 4 is not connected to 2 or 3).  
Possible triangles = 3×2 = 6.  
CC(1) = 1/6 ≈ 0.1667

User 2 has degree 2 (friends: 1,3).  
Triangles: {1,2,3} → tri_count = 1 (as counted from 2's perspective)  
Possible = 2×1 = 2.  
CC(2) = 1/2 = 0.5

```
user_id | degree | triangles | clustering_coeff
--------+--------+-----------+-----------------
2       | 2      | 1         | 0.5000
3       | 2      | 1         | 0.5000
1       | 3      | 1         | 0.1667
4       | 1      | 0         | NULL
```

---

## 11. Viral Spread

**"How does post 9999 spread through the network, hop by hop?"**

### Query

```sql
WITH RECURSIVE spread AS (
  -- Anchor: original poster
  SELECT poster_id AS user_id, post_id, 0 AS hop
  FROM posts WHERE post_id = 9999

  UNION ALL

  -- Recursive: who shared it next?
  SELECT s2.sharer_id, s.post_id, s.hop + 1
  FROM spread s
  JOIN shares s2
    ON  s2.original_poster_id = s.user_id
    AND s2.post_id = s.post_id
  WHERE s.hop < 5
)
SELECT
  hop,
  COUNT(DISTINCT user_id)  AS users_reached,
  SUM(COUNT(DISTINCT user_id)) OVER (
    ORDER BY hop
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  )                        AS cumulative_reach
FROM spread
GROUP BY hop
ORDER BY hop;
```

### CTE-Level I/O Trace

Input — `posts`: post_id=9999, poster_id=A  
Input — `shares`:
```
sharer_id | original_poster_id | post_id
B         | A                  | 9999
C         | A                  | 9999
D         | B                  | 9999
E         | C                  | 9999
```

**Anchor (hop=0):**
```
user_id | hop
A       | 0
```

**Iteration 1 (hop=1):** Who shared from A?
```
user_id | hop
B       | 1
C       | 1
```

**Iteration 2 (hop=2):** Who shared from B or C?
```
user_id | hop
D       | 2  (shared from B)
E       | 2  (shared from C)
```

**Grouped:**
```
hop | users_reached | cumulative_reach
0   | 1             | 1
1   | 2             | 3
2   | 2             | 5
```

---

## 12. 2-Hop Degree of Connection (LinkedIn)

**"Label each user as 1st or 2nd degree connection of user 1001."**

```sql
WITH edges AS (
  SELECT user_a AS src, user_b AS dst FROM connections UNION ALL
  SELECT user_b, user_a FROM connections
),
first_degree AS (
  SELECT dst AS user, 1 AS degree
  FROM edges
  WHERE src = 1001
),
second_degree AS (
  SELECT e.dst AS user, 2 AS degree
  FROM edges e
  JOIN first_degree fd ON e.src = fd.user    -- start from 1st-degree connections
  WHERE e.dst != 1001                         -- exclude self
    AND e.dst NOT IN (SELECT user FROM first_degree)  -- exclude already 1st-degree
)
SELECT user, degree FROM first_degree
UNION ALL
SELECT user, MIN(degree) FROM second_degree GROUP BY user
ORDER BY degree, user;
```

### CTE-Level I/O Trace

Connections: (1001,A), (1001,B), (A,C), (B,C), (B,D)

`first_degree`:
```
user | degree
A    | 1
B    | 1
```

`second_degree` — join edges where src IN {A,B}, excluding 1001 and {A,B}:
```
src | dst → user | degree
A   | C   → C   | 2
B   | C   → C   | 2  (dup, handled by MIN)
B   | D   → D   | 2
```

After `MIN(degree) GROUP BY user`:
```
user | degree
C    | 2
D    | 2
```

**Final output:**
```
user | degree
A    | 1
B    | 1
C    | 2
D    | 2
```

---

## 13. Feed Ranking Score

**"Score each post for feed ordering using weighted interactions, time decay, and report penalty."**

```sql
WITH signals AS (
  SELECT
    p.post_id,
    p.author_id,
    p.created_at,
    SUM(CASE WHEN i.interaction_type = 'like'    THEN 1 ELSE 0 END) AS likes,
    SUM(CASE WHEN i.interaction_type = 'comment' THEN 1 ELSE 0 END) AS comments,
    SUM(CASE WHEN i.interaction_type = 'share'   THEN 1 ELSE 0 END) AS shares,
    SUM(CASE WHEN i.interaction_type = 'report'  THEN 1 ELSE 0 END) AS reports,
    TIMESTAMPDIFF(HOUR, p.created_at, NOW())                         AS age_hours
  FROM posts p
  LEFT JOIN interactions i ON p.post_id = i.post_id
  GROUP BY p.post_id, p.author_id, p.created_at
)
SELECT
  post_id,
  author_id,
  likes, comments, shares, reports, age_hours,
  ROUND(
    (likes*1 + comments*3 + shares*5)    -- weighted interaction score
    * EXP(-0.1 * age_hours)              -- time decay (halves every ~7 hours)
    * CASE WHEN reports > 0 THEN 0.2 ELSE 1 END  -- report penalty
  , 4) AS feed_score,
  RANK() OVER (ORDER BY feed_score DESC) AS feed_rank
FROM signals
WHERE reports < 5                        -- hard filter: too many reports = hidden
ORDER BY feed_score DESC;
```

### I/O Trace

Input `posts`: post A (1 hour old), post B (10 hours old)  
Input `interactions`:
- A: 5 likes, 2 comments, 1 share
- B: 10 likes, 5 comments, 3 shares, 1 report

**`signals`:**
```
post | likes | comments | shares | reports | age_hours
A    | 5     | 2        | 1      | 0       | 1
B    | 10    | 5        | 3      | 1       | 10
```

**Score calculation:**

Post A: `(5×1 + 2×3 + 1×5) × EXP(-0.1×1) × 1`  
= `(5+6+5) × 0.9048 × 1`  
= `16 × 0.9048` = **14.48**

Post B: `(10×1 + 5×3 + 3×5) × EXP(-0.1×10) × 0.2`  
= `(10+15+15) × 0.3679 × 0.2`  
= `40 × 0.3679 × 0.2` = **2.94**

**Output:**
```
post | feed_score | feed_rank
A    | 14.4768    | 1
B    | 2.9432     | 2
```

Post B has more raw engagement but loses due to age decay and report penalty.

---

## 14. Daily Streak (Gaps & Islands)

**"What is each user's current streak of consecutive daily activity?"**

### The Pattern — Gaps & Islands

```
Events: Jan 1, Jan 2, Jan 3, Jan 5, Jan 6
Streaks: [Jan 1-3 = 3 days] [Jan 5-6 = 2 days]
```

**Key insight:** A new streak starts whenever `day_gap != 1`. Cumulative sum of those "new streak" flags creates a group ID.

```sql
WITH daily AS (
  SELECT DISTINCT user_id, event_date FROM user_events
),
-- Step 1: Compute gap between consecutive event days
gaps AS (
  SELECT user_id, event_date,
    DATEDIFF(event_date,
      LAG(event_date) OVER (PARTITION BY user_id ORDER BY event_date)
    ) AS day_gap
  FROM daily
),
-- Step 2: Flag start of new streak, then running sum = streak group ID
streak_groups AS (
  SELECT user_id, event_date,
    SUM(CASE WHEN day_gap != 1 OR day_gap IS NULL THEN 1 ELSE 0 END)
      OVER (PARTITION BY user_id ORDER BY event_date) AS streak_id
  FROM gaps
)
-- Step 3: Aggregate each group
SELECT user_id,
  MIN(event_date) AS streak_start,
  MAX(event_date) AS streak_end,
  COUNT(*)        AS streak_days
FROM streak_groups
GROUP BY user_id, streak_id
HAVING MAX(event_date) >= CURRENT_DATE - INTERVAL 1 DAY  -- only active streaks
ORDER BY streak_days DESC;
```

### CTE-Level I/O Trace

Input events for user 7:
```
event_date
Jan 1
Jan 2
Jan 3
Jan 5
Jan 6
```

`gaps`:
```
event_date | day_gap
Jan 1      | NULL    (first row, no LAG)
Jan 2      | 1
Jan 3      | 1
Jan 5      | 2       ← gap! new streak
Jan 6      | 1
```

`streak_groups` — running SUM of `day_gap != 1 OR NULL`:
```
event_date | day_gap | new_streak_flag | streak_id (running sum)
Jan 1      | NULL    | 1               | 1
Jan 2      | 1       | 0               | 1
Jan 3      | 1       | 0               | 1
Jan 5      | 2       | 1               | 2
Jan 6      | 1       | 0               | 2
```

`GROUP BY user_id, streak_id`:
```
streak_id | streak_start | streak_end | streak_days
1         | Jan 1        | Jan 3      | 3
2         | Jan 5        | Jan 6      | 2
```

---

## 15. Meta Ads Funnel

**"Track impression → click → install → purchase, with CTR and conversion rate."**

```sql
SELECT
  campaign_id,
  DATE_FORMAT(event_time, '%Y-%m') AS month,
  COUNT(DISTINCT CASE WHEN event_type='impression'  THEN user_id END) AS reached,
  COUNT(DISTINCT CASE WHEN event_type='click'       THEN user_id END) AS clicked,
  COUNT(DISTINCT CASE WHEN event_type='app_install' THEN user_id END) AS installed,
  COUNT(DISTINCT CASE WHEN event_type='purchase'    THEN user_id END) AS purchasers,
  SUM(CASE WHEN event_type='purchase' THEN revenue ELSE 0 END)        AS revenue,
  ROUND(
    COUNT(DISTINCT CASE WHEN event_type='click' THEN user_id END) * 100.0
    / NULLIF(COUNT(DISTINCT CASE WHEN event_type='impression' THEN user_id END), 0)
  , 4) AS ctr,
  ROUND(
    COUNT(DISTINCT CASE WHEN event_type='purchase' THEN user_id END) * 100.0
    / NULLIF(COUNT(DISTINCT CASE WHEN event_type='impression' THEN user_id END), 0)
  , 4) AS overall_cvr
FROM ad_events
GROUP BY campaign_id, DATE_FORMAT(event_time, '%Y-%m')
ORDER BY month, revenue DESC;
```

### I/O Trace

Input `ad_events` for campaign C1, month 2024-01:
```
user_id | event_type  | revenue
U1      | impression  | -
U2      | impression  | -
U3      | impression  | -
U1      | click       | -
U2      | click       | -
U1      | app_install | -
U1      | purchase    | 9.99
U2      | purchase    | 14.99
```

**Pivot:**
```
reached  | clicked | installed | purchasers | revenue | CTR    | CVR
3        | 2       | 1         | 2          | 24.98   | 66.67% | 66.67%
```

> **Why `COUNT(DISTINCT user_id)` not `COUNT(*)`?**  
> A user can have multiple impressions. We care about unique users at each funnel stage, not raw event count. Using `COUNT(*)` overstates reach and understates conversion rates.

---

## 16. K-Factor & Invite Funnel

**"Is our product growing virally? K-factor > 1 = viral growth."**

K-factor = average invites accepted per inviting user.

```sql
WITH invite_stats AS (
  SELECT
    sender_id,
    COUNT(*) AS invites_sent,
    SUM(CASE WHEN accepted_at IS NOT NULL THEN 1 ELSE 0 END) AS accepted,
    ROUND(
      SUM(CASE WHEN accepted_at IS NOT NULL THEN 1 ELSE 0 END) * 100.0
      / COUNT(*), 2
    ) AS accept_rate
  FROM invites
  GROUP BY sender_id
)
SELECT
  COUNT(DISTINCT sender_id)                                   AS inviters,
  SUM(accepted)                                               AS total_new_users,
  ROUND(SUM(accepted) * 1.0 / NULLIF(COUNT(DISTINCT sender_id), 0), 4) AS k_factor
FROM invite_stats;
```

### I/O Trace

Input `invites`:
```
sender_id | recipient | accepted_at
Alice     | Bob       | 2024-01-02
Alice     | Carol     | NULL
Alice     | Dave      | 2024-01-03
Bob       | Eve       | 2024-01-05
```

`invite_stats`:
```
sender_id | invites_sent | accepted | accept_rate
Alice     | 3            | 2        | 66.67
Bob       | 1            | 1        | 100.00
```

**Final:**
```
inviters | total_new_users | k_factor
2        | 3               | 1.5
```

K-factor = 3/2 = 1.5 → viral! (each inviter brings in 1.5 new users on average)

---

## 17. Weekly Growth Report (Very Hard)

**Full viral growth funnel: signups → invite vs organic → K-factor → 2nd-gen growth → friendships formed.**

```sql
WITH weekly_signups AS (
  SELECT
    u.user_id, u.country, u.signup_date,
    DATE_SUB(u.signup_date, INTERVAL DAYOFWEEK(u.signup_date)-1 DAY) AS signup_week,
    CASE WHEN i.new_user_id IS NOT NULL THEN 'invite' ELSE 'organic' END AS source,
    i.sender_id AS invited_by
  FROM users u
  LEFT JOIN invites i ON u.user_id = i.new_user_id
),
weekly_invites AS (
  SELECT
    DATE_SUB(DATE(sent_at), INTERVAL DAYOFWEEK(DATE(sent_at))-1 DAY) AS invite_week,
    COUNT(*) AS invites_sent,
    SUM(CASE WHEN accepted_at IS NOT NULL THEN 1 ELSE 0 END) AS accepted,
    COUNT(DISTINCT sender_id) AS inviters,
    ROUND(SUM(CASE WHEN accepted_at IS NOT NULL THEN 1 ELSE 0 END)*100.0/COUNT(*),2) AS accept_rate
  FROM invites
  GROUP BY invite_week
),
second_degree AS (
  SELECT ws.signup_week,
    COUNT(DISTINCT i.sender_id) AS new_users_who_invited,
    SUM(CASE WHEN i.accepted_at IS NOT NULL THEN 1 ELSE 0 END) AS second_gen_signups
  FROM weekly_signups ws
  JOIN invites i
    ON  ws.user_id = i.sender_id
    AND i.sent_at BETWEEN ws.signup_date AND ws.signup_date + INTERVAL 7 DAY
  GROUP BY ws.signup_week
),
new_friendships AS (
  SELECT ws.signup_week, COUNT(*) AS new_user_friendships
  FROM weekly_signups ws
  JOIN friendships f
    ON  (f.user_a = ws.user_id OR f.user_b = ws.user_id)
    AND f.created_at BETWEEN ws.signup_date AND ws.signup_date + INTERVAL 7 DAY
  GROUP BY ws.signup_week
),
base AS (
  SELECT signup_week, country,
    COUNT(DISTINCT user_id)                                              AS total_signups,
    COUNT(DISTINCT CASE WHEN source='invite'  THEN user_id END)         AS invite_signups,
    COUNT(DISTINCT CASE WHEN source='organic' THEN user_id END)         AS organic_signups
  FROM weekly_signups
  GROUP BY signup_week, country
)
SELECT
  b.signup_week, b.country,
  b.total_signups, b.invite_signups, b.organic_signups,
  ROUND(b.invite_signups * 100.0 / NULLIF(b.total_signups, 0), 2) AS invite_pct,
  wi.accept_rate,
  ROUND(b.invite_signups * 1.0 / NULLIF(wi.inviters, 0), 4)       AS k_factor,
  sd.new_users_who_invited,
  sd.second_gen_signups,
  ROUND(sd.second_gen_signups * 1.0 / NULLIF(b.invite_signups, 0), 4) AS second_gen_rate,
  nf.new_user_friendships
FROM base b
LEFT JOIN weekly_invites  wi ON b.signup_week = wi.invite_week
LEFT JOIN second_degree   sd ON b.signup_week = sd.signup_week
LEFT JOIN new_friendships nf ON b.signup_week = nf.signup_week
ORDER BY b.signup_week, b.country;
```

### CTE Map

```
weekly_signups  ──────────────────────────────────────► base (total/invite/organic per week)
     │                                                        │
     └──► second_degree (new users who invite within 7d)     │
     └──► new_friendships (friendships formed by new users)  │
                                                              │
weekly_invites ──────────────────────────────────────────────┤
(accept_rate, inviters, k_factor)                            │
                                                             ▼
                                                    FINAL JOIN → weekly growth report
```

### Key Design Decisions

| Decision | Why |
|----------|-----|
| `LEFT JOIN invites` in `weekly_signups` | Organic users have no invite record; LEFT JOIN keeps them with NULL |
| `DATE_SUB(... DAYOFWEEK)` | Sunday-anchored week bucketing |
| `BETWEEN signup_date AND signup_date + 7 DAY` | Capture viral behavior within first week |
| `LEFT JOIN` all metrics onto `base` | Some weeks may have no 2nd-gen signups; LEFT keeps those rows |

---

## 18. Key Takeaways Cheatsheet

### The Universal Starter

```sql
WITH all_edges AS (
  SELECT user_id_1 AS src, user_id_2 AS dst FROM friendships
  UNION ALL
  SELECT user_id_2 AS src, user_id_1 AS dst FROM friendships
)
-- Every undirected graph query starts here
```

### Pattern → Technique Map

| Problem | Core Technique | Key Clause |
|---------|---------------|------------|
| Friend count | Normalize + GROUP BY | `UNION ALL` both directions |
| Mutual friends | Self-join on shared neighbor | `JOIN all_edges b ON a.dst = b.dst` |
| FOAF | 2-hop expansion | `JOIN all_edges b ON a.dst = b.src` |
| PYMK (all users) | FOAF + `RANK() OVER` | `PARTITION BY user_x ORDER BY mutual_count DESC` |
| Shortest path | Recursive CTE + BFS | `WITH RECURSIVE` + cycle check |
| Connected components | Recursive min-ID propagation | `LEAST(component_id, neighbor_id)` |
| Viral spread | Recursive CTE from source | Track `hop` counter |
| Triangle count | 3-way self-join ÷ 6 | `JOIN ab, bc, ca` |
| Clustering coefficient | triangles / (k×(k-1)) | `NULLIF(degree*(degree-1), 0)` |
| Influence score | Sum follower-counts of followers | 2-level star join |
| Streak | Gaps & islands | Cumulative SUM of `day_gap != 1` |
| Feed score | Weighted sum × decay × penalty | `EXP(-0.1 × age_hours)` |
| Ads funnel | Conditional COUNT DISTINCT per stage | `CASE WHEN event_type=... THEN user_id END` |
| K-factor | accepted / unique inviters | `> 1` = viral growth |

### Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Missing friends in undirected graph | Always `UNION ALL` both directions |
| Duplicates in pair-wise queries | Add `a.src < b.src` constraint |
| Including existing friends in recommendations | `NOT EXISTS` subquery (not `NOT IN`, which fails on NULLs) |
| Infinite recursion in BFS | `hops < 6` guard + cycle check |
| Division by zero in rates | Wrap denominator with `NULLIF(..., 0)` |
| Counting events not users in funnels | Use `COUNT(DISTINCT user_id)` not `COUNT(*)` |
| Wrong triangle count | Divide by 6 (directed: each triangle counted 6 ways) |

---

*Graph SQL Masterclass — Day 15 & 24 Combined Reference*
