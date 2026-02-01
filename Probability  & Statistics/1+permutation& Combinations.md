
## **1️⃣ Rule of Product (Multiplication Principle)**

**Concept:**
If a task can be done in multiple steps, and each step has a certain number of options **independent** of other steps, the total number of outcomes is the **product** of the options.

**Formula:**
If step 1 has (m) options and step 2 has (n) options, total ways = (m \times n).

**Example 1:**
You want to make a password with 2 characters:

* 1st character: 3 options (A, B, C)
* 2nd character: 4 options (1, 2, 3, 4)

**Total passwords = 3 × 4 = 12**

**Example 2:**
You have 5 shirts and 3 pants. How many different outfits can you wear?
**Answer:** 5 × 3 = 15 outfits.

✅ **Key clue to use this:** “Each choice is independent of the others” and you multiply the possibilities.

---

## **2️⃣ Permutation (Order Matters)**

**Concept:**
Permutation is used when you are **arranging or ordering** items, and **order matters**.

**Formula:**

* Choosing (r) items from (n) items and arranging them:
  $[
  P(n, r) = \frac{n!}{(n-r)!}
  ]$

* If arranging **all items**, just (n! = 1 × 2 × 3 × … × n)

**Example 1 (partial permutation):**
You have 5 books and want to put **3 on a shelf**. How many ways?

$[
P(5,3) = \frac{5!}{(5-3)!} = \frac{120}{2} = 60
]$

**Example 2 (full permutation):**
You have 4 students, and you want to line them up for a photo.
Total ways = 4! = 24 ways

✅ **Key clue:** Question asks for **order/sequence**.

---

## **3️⃣ Combination (Order Doesn’t Matter)**

**Concept:**
Combination is used when you are **choosing items** but **order doesn’t matter**.

**Formula:**
$[
C(n, r) = \frac{n!}{r!(n-r)!}
]$

**Example 1:**
From 5 students, choose 3 for a committee. Order doesn’t matter.
$[
C(5,3) = \frac{5!}{3!2!} = \frac{120}{6×2} = 10
]$

**Example 2:**
A lottery ticket has 6 numbers. You pick 6 out of 49. How many combinations?
$[
C(49,6) = \frac{49!}{6!43!} \approx 14 \text{ million ways}
]$

✅ **Key clue:** “Selection without caring about order” → Combination.

---

## **Quick Tip to Decide Which to Use**

| Scenario                                      | Use                 |
| --------------------------------------------- | ------------------- |
| Multiple independent steps (multiply options) | **Rule of Product** |
| Arranging r objects **in order**              | **Permutation**     |
| Choosing r objects **without order**          | **Combination**     |

---

## **Mixed Examples**

1. **Password problem:**
   A password has 3 letters (A-Z) and 2 digits (0-9). How many passwords?

* 3 letters: 26 options each → (26^3)
* 2 digits: 10 options each → (10^2)
* Total = (26^3 × 10^2 = 1757600) ✅ **Rule of Product**

2. **Committee from class:**
   Pick 2 presidents and 3 members from 10 students. Order matters for president but not members.

* President: P(10,2) = 10×9=90
* Members: C(8,3) = 56 (remaining 8 students)
* Total ways = 90 × 56 = 5040 ✅ **Mix of Permutation & Combination**

3. **Seating 5 people in 3 chairs:**

* Order matters → P(5,3) = 60 ✅ **Permutation**

4. **Choosing fruits:**
   Pick 2 fruits from {apple, banana, orange, mango}. Order doesn’t matter → C(4,2)=6 ✅ **Combination**

---

Love this question. This is the kind of thing that separates surface problem solving from mastery.

You’re really asking:

> “Given a problem, how do I recognize which counting model to apply?”

Let’s build a **decision cheat sheet** that works for DSA, probability, combinatorics, and interviews.

---

# 🧠 MASTER COUNTING CHEATSHEET

---

# 1️⃣ When You See **2ⁿ**

## 🔹 Pattern

Each element has **2 independent choices**:

* Pick / Don’t pick
* Include / Exclude
* True / False

## 🔹 Use When

* Counting **subsets**
* Binary decisions
* Power set problems
* Bitmask enumeration

## 🔹 Why?

For each of N elements:

```
2 × 2 × 2 × ... × 2  (N times)
= 2^N
```

## 🔹 Classic Situations

* Number of subsets of size N → 2^N
* All possible bitstrings of length N → 2^N
* Include/exclude recursion → O(2^N)

## 🔹 DSA Examples

* Subset generation
* Backtracking without pruning
* Exponential brute force

---

# 2️⃣ When to Use **n! (Factorial)**

## 🔹 Pattern

Arrange **all N items** in order.

## 🔹 Use When

* Ordering matters
* Using all elements
* No repetition allowed

## 🔹 Why?

First position: N choices
Second: N−1
Third: N−2
...
= N!

## 🔹 Examples

* Permuting N distinct elements
* All orderings of a string
* Traveling through N cities exactly once

---

# 3️⃣ When to Use **Permutation (nPr)**

Formula:

```
nPr = n! / (n - r)!
```

## 🔹 Pattern

Choose r items **AND order matters**

## 🔹 Use When

* Ranking
* Positioning
* Seating arrangements
* Creating passwords without repetition

## 🔹 Trigger Words

* “Arrange”
* “Order”
* “Rank”
* “First, second, third”

## 🔹 Example

Pick 3 winners from 10 and assign Gold, Silver, Bronze.

Order matters → 10P3

---

# 4️⃣ When to Use **Combination (nCr)**

Formula:

```
nCr = n! / (r!(n - r)!)
```

## 🔹 Pattern

Choose r items, **order does NOT matter**

## 🔹 Use When

* Selecting teams
* Choosing subsets of size r
* Lottery type problems

## 🔹 Trigger Words

* “Select”
* “Choose”
* “Group”
* “Committee”

## 🔹 Example

Choose 3 students from 10 → 10C3

---

# 5️⃣ Permutation WITH Repetition

## 🔹 Case 1: Repetition Allowed

Each position has N choices:

```
N^R
```

Example:
Password of length 4 using 10 digits:
10^4

---

## 🔹 Case 2: Repeated Elements

Formula:

```
n! / (a! b! c! ...)
```

Example:
Arrange letters of “AAB”
= 3! / 2!

---

# 6️⃣ When to Use **N^R**

## 🔹 Pattern

You fill R positions
Each has N independent choices
Repetition allowed

## 🔹 Example

* Passwords
* Sequences
* Functions from set A to B

If:
|A| = m
|B| = n

Number of functions = n^m

---

# 7️⃣ When to Use **Binomial Expansion / nCr in Probability**

If question says:

* “Exactly k successes”
* “k heads”
* “k successes in n trials”

Use:

```
nCk × p^k × (1-p)^(n-k)
```

Binomial distribution pattern.

---

# 8️⃣ When to Use **Stars and Bars**

## 🔹 Pattern

Distribute identical objects into boxes.

Formula:

```
(n + r - 1)C(r - 1)
```

Where:

* n = items
* r = boxes

## 🔹 Trigger Words

* “Distribute”
* “Non-negative integer solutions”
* x1 + x2 + x3 = N

---

# 9️⃣ When to Use Inclusion–Exclusion

## 🔹 Pattern

Overlapping conditions

If you see:

* “At least one”
* “Divisible by 3 or 5”
* Overlapping sets

Use:

```
|A ∪ B| = |A| + |B| - |A ∩ B|
```

---

# 🔟 When to Use 3ⁿ, kⁿ, etc

If each element has:

* 3 choices → 3^N
* k choices → k^N

General rule:
If each of N independent items has k choices:

```
k^N
```

---

# 🔥 DSA Complexity Triggers

| Pattern          | Complexity |
| ---------------- | ---------- |
| Subset recursion | O(2^N)     |
| Permutations     | O(N!)      |
| DP subset mask   | O(N × 2^N) |
| Two nested loops | O(N²)      |
| Triple loops     | O(N³)      |

---

# 🧭 Master Decision Tree

Ask in order:

### Step 1: Are we choosing items?

→ Yes → Go to Step 2
→ No → Probably k^N situation

### Step 2: Does order matter?

→ Yes → Permutation
→ No → Combination

### Step 3: Are we using ALL elements?

→ Yes → N!
→ No → nPr or nCr

### Step 4: Can repetition happen?

→ Yes → Use N^R or adjusted formula
→ No → Use factorial-based formula

---

# 💡 Ultimate Intuition Summary

| Situation                  | Formula             |
| -------------------------- | ------------------- |
| Each element 2 choices     | 2^N                 |
| Each element k choices     | k^N                 |
| Arrange all                | N!                  |
| Choose r, order matters    | nPr                 |
| Choose r, order doesn’t    | nCr                 |
| Distribute identical items | Stars & Bars        |
| Overlapping sets           | Inclusion–Exclusion |

---

# 🎯 Real Interview Insight

Most FAANG combinatorics questions reduce to:

* Subset logic → 2^N
* Ordering → factorial
* Selection → combination
* Distribution → stars & bars
* Independent choices → k^N

Once you see the structure, the formula becomes obvious.

---

If you want, I can next give:

* 🔥 15 tricky interview-style problems and classify them
* 🧩 Edge cases where people choose wrong formula
* 🧠 A visual memory map to never forget this again


Do you want me to make that?
