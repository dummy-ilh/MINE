Absolutely! Here's your **complete and structured guide** to **Dynamic Programming (DP)** for beginners, formatted beautifully in **Markdown** so you can copy it, study it, or even share it.

---

# 🚀 Beginner's Guide to Dynamic Programming (DP)

> Welcome! If you're a **novice Python programmer** and have **never heard of DP**, this guide is made *just for you*. We’ll walk step-by-step through the basics of DP, using analogies, examples, and simple Python code.

---

## 📌 Table of Contents

1. [What is Dynamic Programming?](#what-is-dynamic-programming-dp---the-basics-and-introduction)
2. [Memoization vs Tabulation](#tabulation-vs-memoization-two-flavors-of-dp)
3. [Steps to Solve a DP Problem](#steps-to-solve-a-dp-problem)
4. [Example: Fibonacci with DP](#putting-it-all-together-a-simple-example-fibonacci)
5. [Resources for Practice](#where-to-go-next-with-geeksforgeeks)

---

## 🧠 What is Dynamic Programming (DP)? – The Basics and Introduction

Imagine you're solving a complex problem. Often, it can be **broken down into smaller subproblems**, and here’s the trick — **these subproblems repeat**.

### 🎯 The core idea:

> **Solve each subproblem only once and store the result.** If the same problem appears again, **just reuse the stored result.**

### 🍰 Analogy:

* **Without DP**: You're baking a cake and every time you need to mix eggs + flour, you start from scratch.
* **With DP**: You mix eggs + flour once, store the mix, and reuse it later.

### 🔍 Why "Dynamic"? Why "Programming"?

* **Dynamic**: The solution evolves as you solve smaller problems and build up.
* **Programming**: A structured method for solving problems systematically with code.

---

## 🔄 Tabulation vs Memoization: Two Flavors of DP

There are two main ways to implement DP:

### 🧠 1. Memoization (Top-Down)

* **What it does**: Start from the main problem, break it into subproblems.
* **Use recursion** and **store results** (memoize) in a dictionary or array.

#### 📌 Analogy:

> You're measuring mountains. If you've measured one before, just check your notes instead of re-measuring.

### 💾 2. Tabulation (Bottom-Up)

* **What it does**: Start with the smallest problem, solve it, and build up.
* **Use loops** to fill in a table.

#### 📌 Analogy:

> You’re building a tower, one level at a time, from the ground up.

### 🆚 Comparison Table:

| Feature           | Memoization (Top-Down)               | Tabulation (Bottom-Up)   |
| ----------------- | ------------------------------------ | ------------------------ |
| Approach          | Start big, break into smaller pieces | Start small, build up    |
| Method            | Recursion + caching                  | Iteration + table        |
| Storage           | Usually dict or array                | Table (array or matrix)  |
| Stack usage       | High (recursive calls)               | Low (no recursion)       |
| Lazy evaluation   | ✅ Only solves needed subproblems     | ❌ Solves all subproblems |
| Beginner friendly | ✅ Often easier to write              | ✅ Often easier to debug  |

---

## 🧩 Steps to Solve a DP Problem

Here’s the **DP Problem-Solving Recipe** — your go-to checklist every time.

---

### ✅ Step 1: Identify if DP is suitable

Check two key properties:

* **Overlapping Subproblems**: Do smaller problems repeat?
* **Optimal Substructure**: Can a solution be built from solutions to subproblems?

---

### 🧾 Step 2: Define the DP State

What does `dp[i]` or `dp[i][j]` represent?

Example:

* `dp[i] = i-th Fibonacci number`
* `dp[i][j] = number of paths to cell (i, j)`

---

### 🔁 Step 3: Form the Recurrence Relation

Find a way to express the big problem in terms of smaller ones.

Example:

* Fibonacci: `dp[i] = dp[i-1] + dp[i-2]`
* Paths in grid: `dp[i][j] = dp[i-1][j] + dp[i][j-1]`

---

### 🧱 Step 4: Define the Base Cases

The smallest problems that don’t depend on others.

Example:

* Fibonacci: `dp[0] = 0`, `dp[1] = 1`
* Paths: `dp[0][0] = 1`

---

### 🛠️ Step 5: Choose Memoization or Tabulation

* **Memoization**: Recursively solve and cache.
* **Tabulation**: Iteratively fill the DP table.

---

### 🎯 Step 6: Return the Final Answer

After building the DP table, return the value for the full problem.

---

## 🧪 Putting it all together: A Simple Example (Fibonacci)

### 🧮 Problem: Compute the nth Fibonacci number

```
F(0) = 0
F(1) = 1
F(n) = F(n-1) + F(n-2) for n > 1
```

---

### 🟩 Tabulation Approach (Bottom-Up)

```python
def fibonacci_tabulation(n):
    if n <= 1:
        return n

    dp = [0] * (n + 1)
    dp[0], dp[1] = 0, 1

    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]

    return dp[n]

print(fibonacci_tabulation(6))  # Output: 8
```

---

### 🟦 Memoization Approach (Top-Down)

```python
memo = {}

def fibonacci_memoization(n):
    if n in memo:
        return memo[n]
    
    if n <= 1:
        return n
    
    memo[n] = fibonacci_memoization(n - 1) + fibonacci_memoization(n - 2)
    return memo[n]

print(fibonacci_memoization(6))  # Output: 8
```

---

## 📚 Where to Go Next with GeeksforGeeks?

Now that you understand the foundation, it's time to practice. GeeksforGeeks is a great site for that.

### 🔗 Recommended Search Keywords:

* `Dynamic Programming Introduction site:geeksforgeeks.org`
* `Memoization vs Tabulation GeeksforGeeks`
* `Longest Common Subsequence GeeksforGeeks`
* `0/1 Knapsack Problem GeeksforGeeks`
* `Coin Change Problem GeeksforGeeks`

---

## 🎓 Final Tips for Beginners

* **Practice small problems first** (Fibonacci, Climbing Stairs).
* **Draw recursion trees** to spot repeated subproblems.
* **Write both recursive (memoized) and iterative (tabulated) versions.**
* **Don't memorize solutions — understand how to derive them.**

> 💡 *Dynamic Programming is less about code, and more about thinking!* Once you train your brain to see patterns, it becomes second nature.


Let's continue with the next problems in your **Dynamic Programming Guide**.

---

## 🧗‍♂️ 5. Climbing Stairs with 3 Moves

### 🔍 Problem

You can climb **1, 2, or 3 steps** at a time. Count the number of ways to reach the top of `n` stairs.

### 📌 Example:

```
n = 4 → [1+1+1+1, 1+2+1, 1+1+2, 2+1+1, 2+2, 1+3, 3+1] → 7 ways
```

### 🧠 Recursive (Brute Force)

```python
def climb(n):
    if n < 0:
        return 0
    if n == 0:
        return 1
    return climb(n-1) + climb(n-2) + climb(n-3)
```

* ❌ Time: O(3^n)
* ❌ Space: O(n)

---

### 💾 Memoization (Top-Down)

```python
def climb(n, memo={}):
    if n in memo:
        return memo[n]
    if n < 0:
        return 0
    if n == 0:
        return 1
    memo[n] = climb(n-1, memo) + climb(n-2, memo) + climb(n-3, memo)
    return memo[n]
```

* ✅ Time: O(n)
* ✅ Space: O(n)

---

### 📊 Tabulation (Bottom-Up)

```python
def climb(n):
    if n == 0:
        return 1
    if n == 1:
        return 1
    if n == 2:
        return 2
    dp = [0] * (n+1)
    dp[0], dp[1], dp[2] = 1, 1, 2
    for i in range(3, n+1):
        dp[i] = dp[i-1] + dp[i-2] + dp[i-3]
    return dp[n]
```

* ✅ Time: O(n)
* ✅ Space: O(n)

---

### ♻️ Optimized Space

```python
def climb(n):
    if n == 0:
        return 1
    if n == 1:
        return 1
    if n == 2:
        return 2
    a, b, c = 1, 1, 2
    for _ in range(3, n+1):
        a, b, c = b, c, a + b + c
    return c
```

* ✅ Time: O(n)
* ✅ Space: O(1)

---

## 🧮 6. Weighted Climbing Stairs

### 🔍 Problem

Each stair `i` has a cost `cost[i]`. You can take 1 or 2 steps. Find the **minimum cost to reach the top**.

### 📌 Example:

```
cost = [10, 15, 20]
Answer: 15 (step 1 → step 3)
```

### 🎯 Tabulation (Bottom-Up)

```python
def minCostClimbingStairs(cost):
    n = len(cost)
    dp = [0] * (n+1)
    for i in range(2, n+1):
        dp[i] = min(dp[i-1] + cost[i-1], dp[i-2] + cost[i-2])
    return dp[n]
```

* ✅ Time: O(n)
* ✅ Space: O(n)

---

### ♻️ Optimized Space

```python
def minCostClimbingStairs(cost):
    a = b = 0
    for i in range(2, len(cost)+1):
        a, b = b, min(b + cost[i-1], a + cost[i-2])
    return b
```

* ✅ Time: O(n)
* ✅ Space: O(1)

---

Would you like to continue with:

* ✅ Min Sum in a Triangle
* ✅ nth Catalan Number
* ✅ Count Unique BSTs
* ✅ Valid Parentheses Count
* ✅ Polygon Triangulation

Let me know the pace and style you prefer:
➡ one at a time with visuals,
➡ bulk updates with code blocks,
➡ or interactive GitHub-style Q\&A.
I've revised the document to:

1. **Clearly explain each problem** (starting with Fibonacci and Tribonacci).
2. Include **alternatives** like recursion, memoization, and tabulation.
3. Break down the **problem statement**, **approach**, and **complexity analysis** in a clean structure.

Would you like me to continue in this detailed format for the next problems (like Lucas numbers, climbing stairs, etc.)? Or adjust the depth/focus in any way?
You're absolutely right — and thank you for being patient. Let's **restart this properly**, exactly as you asked:

---

### 🔁 For Each Problem, You’ll Get:

| Section                        | What it Covers                   |
| ------------------------------ | -------------------------------- |
| **1. Problem Statement**       | Explained like you’re 5.         |
| **2. Brute-force / Recursion** | No DP yet. Show slowness.        |
| **3. Memoization**             | Top-down DP + Time/Space         |
| **4. Tabulation**              | Bottom-up DP                     |
| **5. Space Optimization**      | If possible                      |
| **6. Dry Run**                 | Small input hand-walk            |
| **7. Alternatives / Notes**    | Any tricks, math, other approach |

---

## ✅ PROBLEM 1: Fibonacci Numbers

### 1. 🧠 Problem

> A number sequence where each term is the sum of the previous two.
> F(0) = 0, F(1) = 1
> F(n) = F(n−1) + F(n−2) for n ≥ 2

---

### 2. 🔨 Brute-force (Recursion)

```python
def fib(n):
    if n == 0: return 0
    if n == 1: return 1
    return fib(n-1) + fib(n-2)
```

* ❌ Time: **O(2ⁿ)** – calls explode!
* ❌ Space: **O(n)** – recursion stack

---

### 3. 💡 Memoization (Top-down DP)

```python
def fib_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n == 0: return 0
    if n == 1: return 1
    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]
```

* ✅ Time: **O(n)**
* ✅ Space: **O(n)**

---

### 4. 📋 Tabulation (Bottom-up DP)

```python
def fib_tab(n):
    if n == 0: return 0
    dp = [0] * (n+1)
    dp[1] = 1
    for i in range(2, n+1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]
```

* ✅ Time: **O(n)**
* ✅ Space: **O(n)**

---

### 5. ♻️ Space Optimized

```python
def fib_optimized(n):
    if n == 0: return 0
    a, b = 0, 1
    for _ in range(2, n+1):
        a, b = b, a + b
    return b
```

* ✅ Time: **O(n)**
* ✅ Space: **O(1)**

---

### 6. 🧪 Dry Run (n = 5)

* 0 1 1 2 3 5 → returns **5**

---

### 7. 🛠️ Alternatives

* Use matrix exponentiation for O(log n)
* Binet’s formula (golden ratio) – not always accurate due to floats

---

## ✅ PROBLEM 2: Tribonacci Numbers

### 1. 🧠 Problem

> Like Fibonacci but with 3 terms:
> T(0)=0, T(1)=1, T(2)=1
> T(n) = T(n−1) + T(n−2) + T(n−3)

---

### 2. 🔨 Brute-force

```python
def trib(n):
    if n == 0: return 0
    if n <= 2: return 1
    return trib(n-1) + trib(n-2) + trib(n-3)
```

* ❌ Time: O(3ⁿ)

---

### 3. 💡 Memoization

```python
def trib_memo(n, memo={}):
    if n in memo: return memo[n]
    if n == 0: return 0
    if n <= 2: return 1
    memo[n] = trib_memo(n-1, memo) + trib_memo(n-2, memo) + trib_memo(n-3, memo)
    return memo[n]
```

* ✅ Time: O(n)

---

### 4. 📋 Tabulation

```python
def trib_tab(n):
    if n == 0: return 0
    if n <= 2: return 1
    dp = [0]*(n+1)
    dp[0], dp[1], dp[2] = 0, 1, 1
    for i in range(3, n+1):
        dp[i] = dp[i-1] + dp[i-2] + dp[i-3]
    return dp[n]
```

---

### 5. ♻️ Space Optimized

```python
def trib_opt(n):
    if n == 0: return 0
    if n <= 2: return 1
    a, b, c = 0, 1, 1
    for _ in range(3, n+1):
        a, b, c = b, c, a + b + c
    return c
```

---

### 6. 🧪 Dry Run (n=5)

* 0 1 1 2 4 7 → returns **7**

---

### 7. 🛠️ Alternatives

* Can build it as a general k-bonacci generator

---

Let me now continue with the next problems in the **same format**:

* Lucas Numbers
* Climbing stairs with 3 moves
* Weighted stairs
* Maximum segments

Would you like to continue with **Lucas Numbers** next?
Perfect! Let's continue from where we left off, using the **same structured, interview-optimized, beginner-friendly dynamic programming (DP) template**.

---

## ✅ PROBLEM 3: Lucas Numbers

### 1. 🧠 Problem

> Like Fibonacci, but different start:
>
> * L(0) = 2
> * L(1) = 1
> * L(n) = L(n−1) + L(n−2) for n ≥ 2

---

### 2. 🔨 Brute-force (Recursion)

```python
def lucas(n):
    if n == 0: return 2
    if n == 1: return 1
    return lucas(n-1) + lucas(n-2)
```

* ❌ Time: O(2ⁿ)
* ❌ Space: O(n)

---

### 3. 💡 Memoization (Top-down DP)

```python
def lucas_memo(n, memo={}):
    if n in memo: return memo[n]
    if n == 0: return 2
    if n == 1: return 1
    memo[n] = lucas_memo(n-1, memo) + lucas_memo(n-2, memo)
    return memo[n]
```

* ✅ Time: O(n)
* ✅ Space: O(n)

---

### 4. 📋 Tabulation

```python
def lucas_tab(n):
    if n == 0: return 2
    dp = [0] * (n+1)
    dp[0], dp[1] = 2, 1
    for i in range(2, n+1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]
```

---

### 5. ♻️ Space Optimized

```python
def lucas_opt(n):
    if n == 0: return 2
    if n == 1: return 1
    a, b = 2, 1
    for _ in range(2, n+1):
        a, b = b, a + b
    return b
```

---

### 6. 🧪 Dry Run (n = 5)

* 2 1 3 4 7 11 → returns **11**

---

### 7. 🛠️ Alternatives

* Just Fibonacci with different seeds: can reuse code.
* Binet-type formulas exist, but unnecessary for interviews.

---

## ✅ PROBLEM 4: Climbing Stairs with 3 Moves

### 1. 🧠 Problem

> You’re at step 0. You can jump 1, 2, or 3 steps.
> How many **distinct ways** to reach step **n**?

---

### 2. 🔨 Brute-force (Recursion)

```python
def climb(n):
    if n == 0: return 1  # 1 way: do nothing
    if n < 0: return 0
    return climb(n-1) + climb(n-2) + climb(n-3)
```

* ❌ Time: O(3ⁿ)

---

### 3. 💡 Memoization

```python
def climb_memo(n, memo={}):
    if n in memo: return memo[n]
    if n == 0: return 1
    if n < 0: return 0
    memo[n] = climb_memo(n-1, memo) + climb_memo(n-2, memo) + climb_memo(n-3, memo)
    return memo[n]
```

* ✅ Time: O(n)

---

### 4. 📋 Tabulation

```python
def climb_tab(n):
    dp = [0] * (n+1)
    dp[0] = 1
    for i in range(1, n+1):
        dp[i] += dp[i-1] if i-1 >= 0 else 0
        dp[i] += dp[i-2] if i-2 >= 0 else 0
        dp[i] += dp[i-3] if i-3 >= 0 else 0
    return dp[n]
```

---

### 5. ♻️ Space Optimized

```python
def climb_opt(n):
    if n == 0: return 1
    a, b, c = 1, 0, 0  # dp[0], dp[-1], dp[-2]
    for _ in range(1, n+1):
        a, b, c = a + b + c, a, b
    return a
```

---

### 6. 🧪 Dry Run (n=4)

* Ways: \[1,1,1,1], \[1,1,2], \[2,2], \[1,2,1], \[3,1], \[1,3], \[2,1,1]
  → **7 ways**

---

### 7. 🛠️ Alternatives

* Generalized to k steps.
* Can use matrix exponentiation or fast DP in rare cases.

---

## ✅ PROBLEM 5: Weighted Climbing Stairs

### 1. 🧠 Problem

> Each stair has a cost. You can jump 1 or 2 steps.
> Find **min cost** to reach top.

📦 Input: `cost = [10, 15, 20]`

---

### 2. 🔨 Brute-force

```python
def min_cost(i, cost):
    if i >= len(cost): return 0
    return cost[i] + min(min_cost(i+1, cost), min_cost(i+2, cost))
```

---

### 3. 💡 Memoization

```python
def min_cost_memo(i, cost, memo={}):
    if i in memo: return memo[i]
    if i >= len(cost): return 0
    memo[i] = cost[i] + min(min_cost_memo(i+1, cost, memo), min_cost_memo(i+2, cost, memo))
    return memo[i]
```

Start with: `min(min_cost_memo(0, cost), min_cost_memo(1, cost))`

---

### 4. 📋 Tabulation

```python
def min_cost_tab(cost):
    n = len(cost)
    dp = [0] * (n+1)
    dp[0], dp[1] = 0, 0
    for i in range(2, n+1):
        dp[i] = min(dp[i-1] + cost[i-1], dp[i-2] + cost[i-2])
    return dp[n]
```

---

### 5. ♻️ Space Optimized

```python
def min_cost_space(cost):
    a = b = 0
    for i in range(2, len(cost)+1):
        a, b = b, min(b + cost[i-1], a + cost[i-2])
    return b
```

---

### 6. 🧪 Dry Run (`cost = [10, 15, 20]`)

* Start at 0 or 1 → \[10 → 20], \[15] → min is 15

---

### 7. 🛠️ Alternatives

* Can add `dp[i] = cost[i] + min(dp[i-1], dp[i-2])` if starting from index 0 only.
* Can adapt this to k-jumps or path problems.

---

Do you want to continue with:

### ➤ Next Up:

* Maximum Segments (Rod Cutting)
* nth Catalan Number
* Count Unique BSTs

Let me know if you'd like **Maximum Segments** next!
Great! Let’s now continue our **Dynamic Programming Problems Guide** from where we left off. This time, **we'll strictly include alternatives**, such as recursion, memoization, tabulation, and optimized variations—plus observations, dry runs, and ideas when alternatives aren't needed.

---

## 🧗‍♂️ 5. Climbing Stairs with 3 Moves (Triple Step)

### 🔍 Problem:

You can climb **1, 2, or 3** steps. Count how many distinct ways to reach step `n`.

### 📌 Example:

```
n = 4
Ways = [1+1+1+1, 1+1+2, 1+2+1, 2+1+1, 2+2, 1+3, 3+1] = 7 ways
```

### 🔁 Recursion (Brute Force)

```python
def climb(n):
    if n < 0: return 0
    if n == 0: return 1
    return climb(n-1) + climb(n-2) + climb(n-3)
```

* ❗ Time: O(3^n) – exponential
* ❗ Space: O(n)

---

### 🧠 Memoization (Top-down DP)

```python
def climb(n, memo={}):
    if n in memo:
        return memo[n]
    if n < 0: return 0
    if n == 0: return 1
    memo[n] = climb(n-1, memo) + climb(n-2, memo) + climb(n-3, memo)
    return memo[n]
```

* ✅ Time: O(n)
* ✅ Space: O(n)

---

### 🧱 Tabulation (Bottom-up DP)

```python
def climb(n):
    if n == 0: return 1
    if n == 1: return 1
    if n == 2: return 2
    dp = [0]*(n+1)
    dp[0], dp[1], dp[2] = 1, 1, 2
    for i in range(3, n+1):
        dp[i] = dp[i-1] + dp[i-2] + dp[i-3]
    return dp[n]
```

---

### ♻️ Optimized Space (Three variables)

```python
def climb(n):
    if n == 0: return 1
    if n == 1: return 1
    if n == 2: return 2
    a, b, c = 1, 1, 2
    for _ in range(3, n+1):
        a, b, c = b, c, a + b + c
    return c
```

---

## 🪨 6. Weighted Climbing Stairs (Min Cost Climbing)

### 🔍 Problem:

You're given a list `cost[]`, where `cost[i]` is the cost to step on `i`. You can start at `cost[0]` or `cost[1]`, and can climb 1 or 2 steps at a time. Find **min total cost** to reach top.

### 📌 Example:

```
cost = [10, 15, 20] → output = 15
Because: min(10+20, 15) = 15
```

---

### 🧱 Tabulation (Bottom-up DP)

```python
def minCost(cost):
    n = len(cost)
    dp = [0]*n
    dp[0], dp[1] = cost[0], cost[1]
    for i in range(2, n):
        dp[i] = cost[i] + min(dp[i-1], dp[i-2])
    return min(dp[-1], dp[-2])
```

* ✅ Time: O(n)
* ✅ Space: O(n)

---

### ♻️ Optimized Space

```python
def minCost(cost):
    a, b = cost[0], cost[1]
    for i in range(2, len(cost)):
        a, b = b, cost[i] + min(a, b)
    return min(a, b)
```

* ✅ Time: O(n)
* ✅ Space: O(1)

---

✅ Alternatives:

* Tabulation is natural here.
* Recursion is not intuitive due to input structure.
* Greedy fails in corner cases.

---

Next up:

### 🪚 7. Maximum Segments (Cut Rope Problem)

Would you like me to continue with that?
Perfect! Let's continue with:

---

## ✂️ 7. Maximum Segments (Cut the Rope Problem)

### 🔍 **Problem Statement**

You are given a rope of length `n`. You can only cut the rope into lengths of `a`, `b`, or `c`. Find the **maximum number of segments** you can make from the rope.

### 📌 **Example:**

```
Input: n = 5, a = 2, b = 5, c = 1  
Output: 5  
Explanation: 5 cuts of length 1 → [1,1,1,1,1]
```

---

## 🧠 1. Recursion (Brute Force)

```python
def maxCuts(n, a, b, c):
    if n == 0:
        return 0
    if n < 0:
        return -1
    res = max(
        maxCuts(n - a, a, b, c),
        maxCuts(n - b, a, b, c),
        maxCuts(n - c, a, b, c)
    )
    if res == -1:
        return -1
    return res + 1
```

* ❌ Time: `O(3^n)`
* ❌ Space: Recursion stack = `O(n)`

---

## 🧠 2. Memoization (Top-down DP)

```python
def maxCuts(n, a, b, c, memo=None):
    if memo is None:
        memo = {}
    if n in memo:
        return memo[n]
    if n == 0:
        return 0
    if n < 0:
        return -1

    res = max(
        maxCuts(n - a, a, b, c, memo),
        maxCuts(n - b, a, b, c, memo),
        maxCuts(n - c, a, b, c, memo)
    )
    memo[n] = -1 if res == -1 else res + 1
    return memo[n]
```

* ✅ Time: `O(n)`
* ✅ Space: `O(n)` for memo

---

## 🧱 3. Tabulation (Bottom-up DP)

```python
def maxCuts(n, a, b, c):
    dp = [-1] * (n + 1)
    dp[0] = 0  # base case: 0 length needs 0 cuts

    for i in range(1, n + 1):
        if i - a >= 0 and dp[i - a] != -1:
            dp[i] = max(dp[i], dp[i - a] + 1)
        if i - b >= 0 and dp[i - b] != -1:
            dp[i] = max(dp[i], dp[i - b] + 1)
        if i - c >= 0 and dp[i - c] != -1:
            dp[i] = max(dp[i], dp[i - c] + 1)

    return dp[n]
```

* ✅ Time: `O(n)`
* ✅ Space: `O(n)`

---

### 🔁 Dry Run for `n = 5, a = 2, b = 5, c = 1`:

We want to find how many cuts we can make to reach 5:

* Start from 0 and build up using `+a`, `+b`, `+c`.
* At each i, store max segments possible to reach i.

`dp = [0,1,1,2,2,3]`
→ dp\[5] = 5 (using five 1-length cuts).

---

### 💡 Observations:

* Pure greedy (always take biggest cut) will fail:

  * E.g., n = 7, a = 5, b = 2, c = 2 → greedy takes 5+2 = 2 cuts
  * But 2+2+2+1 = 4 cuts → more, but invalid as 1 not allowed.

---

Next Problem:

## 🍃 8. Nth Catalan Number

Would you like to proceed?
