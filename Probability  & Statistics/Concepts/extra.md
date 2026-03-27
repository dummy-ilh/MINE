🔷 Section 1: Counting (Warm-up → Medium)
Q1 (Warm-up)

You have 5 distinct books.
In how many ways can you arrange them on a shelf?

Q2 (Slight twist)

You have 5 distinct books, but 2 specific books must always stay together.
How many arrangements are possible?

Q3 (Classic FAANG-style)

A password consists of 4 digits (0–9).

How many passwords are possible if repetition is allowed?
How many if repetition is NOT allowed?
How many if the password must have at least one even digit?

# 🔷 Q1: Arranging 5 distinct books

Your answer: **(5!)** ✅

✔️ Correct
Because all books are distinct:
[
\text{Total arrangements} = 5! = 120
]

---

# 🔷 Q2: 2 books must stay together

Your answer: **(4!)** ❌ (incomplete)

### Let’s fix it properly:

Treat the 2 books as a **single block**.

So now you have:

* 1 block (the 2 books)
* 3 other books

👉 Total = 4 items → arrange in (4!) ways

BUT inside the block:

* The 2 books can be arranged in (2!) ways

### Final answer:

[
4! \times 2! = 24 \times 2 = 48
]

### 🔑 Interview insight:

You missed the **internal permutations** — very common mistake.

---

# 🔷 Q3: Passwords

## Part 1: Repetition allowed

Your answer: **(9^4)** ❌

Digits are **0–9 → total 10 digits**

[
10^4 = 10000
]

👉 Mistake: You excluded 0 unnecessarily

---

## Part 2: No repetition

Your answer: **(9 \times 8 \times 7 \times 6)** ❌

Correct logic:

* First digit: 10 choices
* Then: 9, 8, 7

[
10 \times 9 \times 8 \times 7 = 5040
]

👉 Again, you dropped 0 incorrectly

---

## Part 3: At least one even digit ❌ (not attempted properly)

This is where FAANG tests **thinking ability**.

---

## 💡 Best approach: Complement method

Instead of counting “at least one even”, count:

[
\text{Total} - \text{No even digits}
]

### Step 1: Total passwords

[
10^4
]

### Step 2: No even digits

Odd digits = {1,3,5,7,9} → 5 choices

[
5^4
]

### Final:

[
10^4 - 5^4 = 10000 - 625 = 9375
]

---

# 🔥 Key Takeaways (Important for interviews)

### 1. Always check:

* Are you **excluding valid options** (like 0)?
* Are there **hidden permutations inside groups**?

### 2. Use complement when:

* Condition says **“at least”, “at most”, “none”**


Great—this is a **core technique** that shows up everywhere in probability and counting. Let’s build it properly so you can *recognize when to use it instantly in interviews*.

---

# 🔷 The Idea: Complement Counting

When a problem says:

> “**at least one**”, “**at least k**”, “**not none**”

👉 Direct counting becomes messy because:

* Many overlapping cases
* Hard to ensure you don’t double count

So we flip the problem:

[
\text{At least one} = \text{Total} - \text{None}
]

---

# 🔷 Apply to Your Problem

### Problem:

Number of 4-digit passwords with **at least one even digit**

Digits = {0–9}

---

## Step 1: Total possibilities

Each position:

* 10 choices

[
\text{Total} = 10^4
]

---

## Step 2: Count the opposite (NO even digits)

What does “no even digits” mean?

👉 Only **odd digits allowed**

Odd digits = {1, 3, 5, 7, 9} → 5 choices

Each position:

* 5 choices

[
\text{No even digits} = 5^4
]

---

## Step 3: Subtract

[
\text{At least one even} = 10^4 - 5^4
]

[
= 10000 - 625 = 9375
]

---

# 🔥 Why This Works (Deep Intuition)

Think in sets:

Let:

* ( A ) = passwords with at least one even digit
* ( A^c ) = passwords with **no even digits**

Then:
[
|A| = |S| - |A^c|
]

👉 Because every password is either:

* Has ≥1 even digit
  **OR**
* Has 0 even digits

No overlap. No missing cases.

---

# 🔷 Why Direct Counting is Hard

If you try directly:

* Case 1: exactly 1 even
* Case 2: exactly 2 even
* Case 3: exactly 3 even
* Case 4: exactly 4 even

You’d have to compute:

[
\binom{4}{1} \cdot 5^1 \cdot 5^3 + \binom{4}{2} \cdot 5^2 \cdot 5^2 + \dots
]

👉 Messy, error-prone, and time-consuming in interviews.

---

# 🔷 Pattern Recognition (Very Important)

Use complement when you see:

| Phrase       | Strategy                      |
| ------------ | ----------------------------- |
| At least one | Total − None                  |
| At most k    | Total − (more than k)         |
| None         | Direct                        |
| Exactly k    | Direct (usually combinations) |

---

# 🔥 Interview-Level Insight

FAANG interviewers LOVE this trick because it tests:

* Mathematical maturity
* Ability to simplify problems
* Awareness of **combinatorial explosion**

---



