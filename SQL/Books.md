md
# 📘 Best SQL Books for FAANG Interviews (Only SQL)

Curated from **Reddit**, **FAANG interview experiences**, and **industry recommendations**.  
This list focuses **purely on SQL** — no databases theory fluff, no ORMs, no system design.

---

## ⭐ Tier 1: MUST-HAVE for FAANG SQL Interviews

These are the most frequently recommended and interview-relevant books.

### 1️⃣ SQL Practice Problems — *Sylvia Moestl Vasilik*
**Best single book for SQL interviews**

- Real interview-style problems
- Joins, subqueries, aggregations, window functions
- Increasing difficulty (basic → advanced)
- Clear, concise solutions

📌 *Reddit consensus:* closest to actual interview SQL questions

---

### 2️⃣ SQL Cookbook — *Anthony Molinaro*
**Problem-solving goldmine**

- “How do I write this query?” style
- Excellent for tricky joins & transformations
- Helps develop SQL intuition

📌 *Very popular among data engineers & analysts*

---

### 3️⃣ Grokking the SQL Interview — *Javin Paul*
**Explicitly interview-focused**

- Covers:
  - Joins
  - Window functions
  - CTEs
  - Ranking problems
- Short, direct, practical

📌 *Good final revision before interviews*

---

## 🧱 Tier 2: Strong Foundations (Highly Recommended)

These build the **mental model of SQL** interviewers expect.

### 4️⃣ Learning SQL — *Alan Beaulieu*
**Best fundamentals book**

- Clean explanations
- Covers ANSI SQL well
- Ideal first book if fundamentals are shaky

📌 *Frequently recommended on r/learnSQL*

---

### 5️⃣ SQL Queries for Mere Mortals — *John Viescas*
**Teaches how to THINK in SQL**

- Logical query construction
- Avoids common beginner mistakes
- Great for analytical SQL roles

📌 *Classic book, still very relevant*

---

### 6️⃣ The Art of SQL — *Stephane Faroult*
**Advanced SQL thinking**

- Query design philosophy
- Optimization mindset
- Read once fundamentals are solid

📌 *Less syntax, more reasoning*

---

## 🚀 Tier 3: Advanced / Bonus (For L5+ / Data Eng Roles)

Not required for all interviews, but extremely valuable.

### 7️⃣ SQL Antipatterns — *Bill Karwin*
**What NOT to do in SQL**

- Common design & query mistakes
- Helps avoid subtle interview traps

📌 *Great for debugging bad queries*

---

### 8️⃣ SQL Performance Explained — *Markus Winand*
**Query optimization & indexing**

- Index usage
- Query execution behavior
- Performance tradeoffs

📌 *Very useful for senior roles*

---

### 9️⃣ T-SQL Fundamentals — *Itzik Ben-Gan*
**Deep execution-level understanding**

- Logical query processing order
- Advanced query transformations
- T-SQL specific, but concepts transfer

📌 *Optional unless role is heavy SQL*

---

## 🎯 What FAANG SQL Interviews Actually Test

Expect heavy focus on:

- INNER / LEFT / SELF joins  
- GROUP BY + HAVING  
- Subqueries & correlated subqueries  
- Window functions:
  - `ROW_NUMBER`
  - `RANK`
  - `DENSE_RANK`
  - `LAG / LEAD`
- CTEs
- Query correctness > syntax tricks
- Performance intuition (at senior levels)

---

## 🧭 Recommended Reading Order (Interview-Optimized)

1. **Learning SQL** *(if fundamentals are weak)*  
2. **SQL Practice Problems**  
3. **SQL Cookbook**  
4. **Grokking the SQL Interview**  
5. **SQL Antipatterns** *(optional but powerful)*  

📌 If time is short → **SQL Practice Problems + Grokking SQL Interview**

---

## ✅ Final Verdict

**If you buy only ONE book:**  
➡️ **SQL Practice Problems – Sylvia Moestl Vasilik**

**If you buy TWO:**  
➡️ SQL Practice Problems + SQL Cookbook

**If you want FAANG-level mastery:**  
➡️ Add SQL Antipatterns

---
markdown
# 🚀 FAANG SQL Interview Prep Guide (2026)

## 📚 Top 5 SQL Books (Ranked by FAANG Success Rate)

| Rank | Book | Why FAANG Gold | Level | 
|------|------|----------------|-------|
| 🥇 | **T-SQL Fundamentals**<br>Itzik Ben-Gan | Window functions, gaps/islands, optimization | Advanced |
| 🥈 | **SQL Cookbook**<br>Anthony Molinaro | 1000+ query recipes | Intermediate |
| 🥉 | **SQL Practice Problems**<br>Sylvia Moestl Vasilik | 57 LeetCode-style exercises | Beginner-Int |
| 4️⃣ | **SQL for Data Scientists**<br>Renee Teate | Analysis-focused | Data Science |
| 5️⃣ | **Ace the Data Science Interview**<br>Kevin Huo/Nick Singh | 30 FAANG SQL questions |

## 🎯 Practice Platforms (FAANG-Tagged Questions)

| Platform | Problems | FAANG Focus | Free Tier |
|----------|----------|-------------|-----------|
| [LeetCode Database](https://leetcode.com/problemset/database/) | 500+ | **SQL 50** study plan ⭐ | ✅ 50 Hard |
| [StrataScratch](https://www.stratascratch.com/) | 800+ | Amazon/Google real questions | ✅ 100+ |
| [DataLemur](https://datalemur.com/) | 300+ | Interview simulator | ✅ All |
| [HackerRank SQL](https://www.hackerrank.com/domains/sql) | 200+ | Optimization | ✅ All |

## 📊 FAANG SQL Topics (Must Master)


Tier 1 (80% questions):
✅ Window Functions (RANK, LAG, ROW_NUMBER)
✅ Complex Multi-JOINs  
✅ GROUP BY + HAVING + CTEs
✅ Date/Time manipulation

Tier 2 (15%):
✅ Subquery optimization
✅ Pivot/Unpivot
✅ Running totals/cumulative sums

Tier 3 (5%):
✅ Schema design
✅ Index optimization


## 📅 30-Day Mastery Plan


Week 1: T-SQL Ch 1-5 + LeetCode Easy (50 problems)
Week 2: SQL Cookbook + StrataScratch FAANG (40 problems)
Week 3: SQL Practice Problems + DataLemur Hard (30 problems)  
Week 4: Mock interviews + Review ALL misses


## 🎖️ Success Stack (90% Coverage)

📖 T-SQL Fundamentals (theory )
💻 StrataScratch (FAANG questions)
⭐ LeetCode SQL 50 (must-do)
⏱️ DataLemur (timed practice)


**FAANG SQL = Window Functions + Complex JOINs + Practice**. Start now! 🚀


# 🚀 FAANG SQL Interview Prep Guide - Reddit's Top Picks (2026)

## 📚 The "Big Three" Books (Reddit Gold Standard)

| Book | Best For | Why Reddit Loves It |
|------|----------|-------------------|
| **🥇 SQL Cookbook**<br/>Anthony Molinaro (O'Reilly) | **FAANG Interview Prep** | Problem-solution format • Window functions • Pivoting • Medium/Hard LeetCode-style |
| **🥈 T-SQL Fundamentals**<br/>Itzik Ben-Gan | **Deep Understanding** | Logical Query Processing (FROM→WHERE→GROUP BY) • SQL engine execution order |
| **🥉 SQL Antipatterns**<br/>Bill Karwin | **Senior/Design Interviews** | Database design • Performance pitfalls • Scalability discussions |

## 🎯 Top Practice Platforms (Reddit Favorites)

| Platform | Why FAANG-Focused | Problems | Free Tier |
|----------|-------------------|----------|-----------|
| **DataLemur** | Ex-Facebook DS created • Real interview prompts | 300+ Hard | ✅ All |
| **StrataScratch** | "LeetCode for Data Science" • Airbnb/Netflix/Google | 800+ | ✅ 100+ |
| **LeetCode Database** | **SQL 50** study plan | 500+ | ✅ SQL 50 |
| **Mode Analytics** | Best Window Function tutorial | Tutorial | ✅ Free |

## 🔥 Core FAANG Topics (80% Coverage)

| Topic | Why Critical | Must-Know Functions |
|-------|-------------|--------------------|
| **Window Functions** | Most common "Hard" questions | `RANK()`, `LEAD()/LAG()`, `ROW_NUMBER()` |
| **CTEs** | Clean code > nested subqueries | `WITH clause` syntax |
| **Self-Joins** | First/last event problems | Same-table comparisons |
| **Aggregations** | Complex filtering | `HAVING vs WHERE`, `CASE WHEN` in `SUM()` |

## 💎 Reddit's "Final Boss" Tip
