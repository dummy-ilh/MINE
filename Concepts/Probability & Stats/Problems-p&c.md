Great — here’s a large set of **numerical permutation & combination problems** with full, step-by-step solutions (exact numeric answers included). I arranged them from **easy → hard** and included different flavours you’ll see in FAANG-style interviews or Olympiad-style counting. Work through them, and tell me which types you want more of (e.g., circular, derangements, constrained, distributions, probability).

# 1. EASY / FOUNDATIONS

### Problem 1

How many distinct permutations of the letters in the word **APPLE**?

**Solution**
Letters: A, P, P, L, E → total 5 letters, P repeats twice.
Permutations = ( \dfrac{5!}{2!} ).
Compute: (5! = 120). (2! = 2). So (120/2 = 60).

**Answer:** **60**

---

### Problem 2

How many 3-digit numbers (100–999) can be formed using digits 1–9 with no repetition?

**Solution**
Number of choices: first digit: 1–9 except 0 → 9 choices. But we require 3-digit number and no repetition:

* First digit: 9 choices.
* Second digit: remaining 8.
* Third digit: remaining 7.
  Total = (9 \times 8 \times 7 = 504).

**Answer:** **504**

---

### Problem 3

How many 4-letter strings can you form (order matters) from the 26 English letters if repetition **is allowed**?

**Solution**
Each position: 26 choices. Total = (26^4).
Compute: (26^2 = 676). (676 \times 26 = 17576). (17576 \times 26 = 456976).

**Answer:** **456,976**

---

### Problem 4

From 10 people, how many committees of 4 can be chosen?

**Solution**
Choose 4 out of 10: ( \binom{10}{4} = \dfrac{10!}{4!6!} ).
Compute: (10 \times 9 \times 8 \times 7 / (4 \times 3 \times 2 \times 1) = 5040/24 = 210).

**Answer:** **210**

---

# 2. PERMUTATIONS (ORDER, restrictions, repeats)

### Problem 5

How many 5-digit numbers can be made using digits 0–9 without repetition if the number must be even?

**Solution**
Even digits: 0,2,4,6,8 (5 choices). But careful: leading digit cannot be 0.
Casework on last digit:

* If last digit ≠ 0 (choose from {2,4,6,8}) → 4 choices for last digit.

  * First digit: cannot be 0 and cannot be last digit. Available digits: from remaining 8 digits excluding chosen last. But 0 is allowed for first? No. So number of choices for first digit: 8 (digits 1–9 minus the chosen last → 8).
  * Remaining three middle places: choose and arrange from remaining 8 digits? Let's do sequentially:
    After choosing last (one of 4) and first (one of 8), remaining pool size = 10 − 2 = 8. For positions 2,3,4 there are (8 \times 7 \times 6).
    Total for this case = (4 \times 8 \times 8 \times 7 \times 6).

Compute: (8 \times 7 \times 6 = 336). (8 \times 336 = 2688). (4 \times 2688 = 10,752).

* If last digit = 0 → 1 choice for last.

  * First digit: must be 1–9 and ≠ 0 → 9 choices.
  * Remaining three middle digits: choose from remaining 8 digits (10 total minus first and last) → (8 \times 7 \times 6 = 336).
    Total = (1 \times 9 \times 336 = 3024).

Total even numbers = (10,752 + 3,024 = 13,776).

**Answer:** **13,776**

---

### Problem 6

How many distinct permutations of the letters in **MISSISSIPPI**?

**Solution**
Word letters count: M:1, I:4, S:4, P:2. Total letters = 11.
Permutations = (\dfrac{11!}{1!4!4!2!}).

Compute stepwise:

* (11! = 39,916,800).
* (4! = 24). So (4! \times 4! = 24 \times 24 = 576).
* (576 \times 2! = 576 \times 2 = 1,152).
  Now divide: (39,916,800 / 1,152).

Compute division: (1,152 \times 34,650 = 39,916,800). (You can check: (1,152 \times 34,650 = 1,152 \times 34,000 + 1,152 \times 650 = 39,168,000 + 748,800 = 39,916,800).)
So result = **34,650**.

**Answer:** **34,650**

---

### Problem 7

How many ways to seat 8 people around a circular table (rotations considered the same)?

**Solution**
Circular permutations of n distinct people: ((n-1)!).
So ((8-1)! = 7! = 5,040).

**Answer:** **5,040**

---

# 3. COMBINATIONS & DISTRIBUTIONS (stars & bars)

### Problem 8

In how many ways can you distribute 12 identical candies to 5 children, where each child may get zero or more?

**Solution**
Stars and bars: (\binom{12 + 5 -1}{5-1} = \binom{16}{4}).
Compute: (\binom{16}{4} = 1820).

(Compute: (16 \times 15 \times 14 \times 13 / 24 = 43680/24 = 1820).)

**Answer:** **1,820**

---

### Problem 9

Same as above, but each child receives **at least one** candy.

**Solution**
Give each child 1 candy → remaining candies = (12 - 5 = 7). Now distribute 7 identical to 5 children with zero allowed: (\binom{7+5-1}{5-1} = \binom{11}{4}).
Compute: (\binom{11}{4} = 330).

**Answer:** **330**

---

### Problem 10

How many solutions in nonnegative integers to (x_1 + x_2 + x_3 + x_4 = 20) with (0 \le x_i \le 10) for each (i)?

**Solution**
Count unrestricted nonnegative solutions: (\binom{20+4-1}{4-1} = \binom{23}{3} = 1771). Now subtract those where some (x_i \ge 11). Use inclusion–exclusion.

Let (A_i) = solutions with (x_i \ge 11). For (A_1): set (y_1 = x_1 - 11 \ge 0). Then (y_1 + x_2 + x_3 + x_4 = 9) → (\binom{9+4-1}{3} = \binom{12}{3} = 220). By symmetry, each single (A_i) count = 220. There are 4 such → subtract (4 \times 220 = 880).

For intersections (A_i \cap A_j): set two variables ≥11, subtract 11 twice: remaining sum = (20 - 22 = -2) → impossible. So intersections are zero.

Thus valid = (1771 - 880 = 891).

**Answer:** **891**

---

# 4. ADJACENCY & CONSTRAINTS

### Problem 11

In how many permutations of the letters A, B, C, D, E do **A and B appear next to each other** (as adjacent letters)?

**Solution**
Treat {A,B} as a block. Inside block A and B can be ordered two ways. So number = (2 \times) permutations of 4 items (the block + 3 others) = (2 \times 4! = 2 \times 24 = 48).

**Answer:** **48**

---

### Problem 12

How many permutations of digits 1–7 (all distinct) are there such that **no two even digits are adjacent**?

**Solution**
Even digits among 1–7: 2,4,6 → 3 evens. Odds: 1,3,5,7 → 4 odds.

Place the 4 odds first: they create 5 gaps (before/after and between) where evens may go. To ensure no adjacency between evens, place each even into a distinct gap. Number of ways:

* Choose 3 of the 5 gaps: (\binom{5}{3} = 10).
* Arrange 3 even digits among themselves: (3! = 6).
* Arrange 4 odd digits: (4! = 24).
  Total = (10 \times 6 \times 24 = 1,440).

Compute: (6 \times 24 = 144). (144 \times 10 = 1,440).

**Answer:** **1,440**

---

### Problem 13

How many 5-digit numbers can be formed from digits 1–9 (no repetition) such that **digits are in strictly increasing order**?

**Solution**
Every strictly increasing 5-digit sequence corresponds to a choice of 5 distinct digits from 1–9; once chosen there's only one order (ascending). So number = (\binom{9}{5}).

Compute (\binom{9}{5} = \binom{9}{4} = 126).

**Answer:** **126**

---

# 5. ADVANCED & CLASSIC COUNTS

### Problem 14 (Derangement)

How many permutations of 6 distinct letters have **no letter in its original position** (i.e., derangements (D_6))?

**Solution**
Derangements formula: (D_n = n! \sum_{k=0}^{n} \dfrac{(-1)^k}{k!}). For (n=6):

Compute:

* (6! = 720).
* Sum S = (1 - 1 + 1/2! - 1/3! + 1/4! - 1/5! + 1/6!).
  Compute term by term:

1. 1 = 1.
2. −1 = 0.
3. +1/2! = +1/2 = 0.5 → S = 0.5.
4. −1/6 = −0.166666... → S = 0.333333...
5. +1/24 = +0.041666... → S = 0.375
6. −1/120 = −0.008333333... → S = 0.366666666...
7. +1/720 ≈ +0.0013888889 → S ≈ 0.3680555556.

Now (D_6 = 720 \times S). Multiply: (720 \times 0.3680555556 ≈ 265). But derangement counts are integers; (D_6 = 265) (known exact).

Alternatively use recurrence: (D_n = (n-1)(D_{n-1} + D_{n-2})) with (D_0=1, D_1=0): compute (D_2=1), (D_3=2), (D_4=9), (D_5=44), (D_6=(6-1)(44+9)=5\times53=265).

**Answer:** **265**

---

### Problem 15

From digits 0–9, how many 4-digit numbers (leading digit ≠ 0) have **all digits distinct** and are **divisible by 5**?

**Solution**
Divisible by 5 → last digit must be 0 or 5.

Case 1: last digit = 0.

* First digit (thousands): cannot be 0 → choose from {1–9} excluding any used digits (last=0), so 9 choices.
* Remaining middle two digits: choose from remaining 8 digits (10 total − chosen thousand − last) and arrange → (8 \times 7 = 56).
  Total case1 = (9 \times 56 = 504).

Case 2: last digit = 5.

* First digit: cannot be 0 and cannot be 5. So choices: from {1–9} excluding 5 → 8 choices (digits 1–9 except 5).
* Middle two digits: choose from remaining 8 digits (10 − chosen first − last) → (8 \times 7 = 56).
  Total case2 = (8 \times 56 = 448).

Total = (504 + 448 = 952).

**Answer:** **952**

---

### Problem 16 (K-th permutation concept — numeric)

What is the **3rd lexicographic permutation** of digits ([1,2,3,4])?

**Solution**
List permutations in lexicographic order or use factorial number system.

Number of permutations = 4! = 24. Each leading digit block size = (3! = 6).

* 3rd permutation corresponds to index k = 3 (1-based).
  Find first digit: ( (k-1) / 3! = (3-1)/6 = 2/6 = 0) → index 0 among [1,2,3,4] → first digit = 1.
  Remaining k' = k - 0*6 = 3.
  Now for remaining 3 digits, block size = (2! = 2).
  Second digit index = (k'-1)/2! = (3-1)/2 = 2/2 = 1 → index 1 among [2,3,4] → second digit = 3.
  Update k' = k' - 1*2 = 3 - 2 = 1.
  Now remaining digits [2,4], block size = 1! =1.
  Third digit index = (1-1)/1 = 0 → digit = 2.
  Fourth digit = remaining = 4.

Permutation: **1 3 2 4** → as string: **1324**.

**Answer:** **1324**

---

# 6. PROBABILITY STYLE COUNTS (numerical)

### Problem 17

From a standard 52-card deck, if you draw 5 cards, how many 5-card hands contain **exactly 2 aces**?

**Solution**
Number of aces in deck = 4. Choose 2 aces: (\binom{4}{2} = 6).
Choose other 3 cards from non-aces (48 cards): (\binom{48}{3}).
Total = (6 \times \binom{48}{3}).

Compute (\binom{48}{3} = 48 \times 47 \times 46 / 6).
Compute numerator: (48 \times 47 = 2256). (2256 \times 46 = 103,776).
Divide by 6: (103,776 / 6 = 17,296).
So total = (6 \times 17,296 = 103,776).

**Answer:** **103,776**

---

### Problem 18

From numbers 1–20, how many ways to choose 3 so that their sum is divisible by 3?

**Solution (mod 3 classes)**
Numbers mod 3 can be 0,1,2.
Count numbers in each class among 1..20:

* Numbers divisible by 3: multiples of 3 up to 18 → 3,6,...,18 → count = ( \lfloor 20/3 \rfloor = 6).
* Remainder 1: numbers congruent 1 mod 3: 1,4,7,10,13,16,19 → 7 numbers.
* Remainder 2: 2,5,8,11,14,17,20 → 7 numbers.

We need triples whose remainders sum ≡ 0 mod 3. Possibilities:

1. (0,0,0): choose 3 from class0: (\binom{6}{3} = 20).
2. (1,1,1): (\binom{7}{3} = 35).
3. (2,2,2): (\binom{7}{3} = 35).
4. (0,1,2): choose one from each: (6 \times 7 \times 7 = 294).

Total = (20 + 35 + 35 + 294 = 384).

**Answer:** **384**

---

# 7. HARDER / COMBINATORIAL HYBRIDS

### Problem 19

How many different 8-character license plates are there of the form **3 letters followed by 5 digits**, where letters are A–Z (repetition allowed) and digits 0–9 (repetition allowed), **but the plate must contain at least one vowel among the letters** (vowels = A,E,I,O,U)?

**Solution**
Total plates without restriction: (26^3 \times 10^5).
We subtract plates where the 3 letters are all consonants.
Number of consonants = 21 (26 − 5).
Plates with 3 consonants: (21^3 \times 10^5).

So valid = ((26^3 - 21^3) \times 10^5).

Compute numeric values:

* (26^2 = 676). (26^3 = 676 \times 26 = 17,576).
* (21^2 = 441). (21^3 = 441 \times 21 = 9,261).
* (26^3 - 21^3 = 17,576 - 9,261 = 8,315).
* (10^5 = 100,000).
  Total plates = (8,315 \times 100,000 = 831,500,000).

**Answer:** **831,500,000**

---

### Problem 20 (Constrained arrangements)

In how many ways can 10 distinct books be placed on a shelf if 3 particular books must **not** be together (i.e., not all three consecutive)?

**Solution**
Total permutations without restriction: (10! = 3,628,800).

Count arrangements where the 3 particular books **are together** (as a block):
Treat the 3-book block as one item + the other 7 books → total 8 items → (8!) arrangements. The 3 books inside the block can be arranged in (3! = 6) ways. So number where they are together = (8! \times 3! = 40,320 \times 6 = 241,920). (Since (8! = 40,320).)

We want permutations where they are **not all three consecutive**. So subtract:
Total valid = (10! - 8! \times 3! = 3,628,800 - 241,920 = 3,386,880).

**Answer:** **3,386,880**

---

# Quick reference — formulas used

* Permutations of (n) distinct: (n!).
* Permutations with repeats: (\dfrac{n!}{n_1!n_2!\dots}).
* Combinations: (\binom{n}{k} = \dfrac{n!}{k!(n-k)!}).
* Stars and bars (nonnegative solutions to (x_1+\dots+x_r = n)): (\binom{n+r-1}{r-1}).
* Circular permutations of n distinct objects: ((n-1)!).
* Derangements: (D_n = n!\sum_{k=0}^n \dfrac{(-1)^k}{k!}) or recurrence (D_n=(n-1)(D_{n-1}+D_{n-2})).

---

Would you like:

* a PDF or a downloadable worksheet with these problems and space to write solutions?
* extra problems focused on **derangements**, **inclusion–exclusion**, **circular permutations**, or **combinatorics + DP**?
* step-by-step code (Python) that **generates answers** to verify your manual calculations?

Pick one and I’ll produce it next.


## 🧠 STRUCTURE

We'll organize this into **5 tiers**:

1. **Basic Counting (Core Math P&C)**
2. **Algorithmic Permutations (Backtracking, Recursion)**
3. **Combinatorial Generation (Subsets, Combinations, Bitmasking)**
4. **Advanced Pattern Problems (Unique permutations, constrained arrangements)**
5. **FAANG-Level Hybrid Problems (Combinatorics + DP / Probability / Graph)**

Each problem includes:

* **Problem description**
* **Approach**
* **Code (Python)**
* **Time complexity**
* **Key insights**

---

# 🧩 1. BASIC COUNTING (Core P&C Foundation)

---

### **Q1: How many ways can the letters of "APPLE" be arranged?**

**Explanation:**

* Total letters = 5
* But ‘P’ repeats twice.

[
\text{Total permutations} = \frac{5!}{2!} = 60
]

**Concept:** Division rule for identical elements.

---

### **Q2: How many 3-digit even numbers can be formed using digits 1–9 without repetition?**

**Explanation:**

* Last digit (must be even): 2, 4, 6, 8 → 4 choices
* First digit: any of remaining 8
* Middle digit: remaining 7

[
4 \times 8 \times 7 = 224
]

---

### **Q3: From 10 people, how many ways to choose 3 for a team?**

[
^nC_r = \frac{10!}{3!7!} = 120
]

---

### **Q4: How many 4-letter words can be made from letters A–Z if repetition is allowed?**

[
26^4 = 456,976
]

---

# 🔁 2. ALGORITHMIC PERMUTATIONS (Backtracking & Recursion)

---

### **Q5: Generate all permutations of a list [1, 2, 3]**

**Approach:** Use backtracking.

```python
def permute(nums):
    res = []
    def backtrack(path, used):
        if len(path) == len(nums):
            res.append(path[:])
            return
        for i in range(len(nums)):
            if used[i]:
                continue
            used[i] = True
            path.append(nums[i])
            backtrack(path, used)
            path.pop()
            used[i] = False
    backtrack([], [False]*len(nums))
    return res

print(permute([1,2,3]))
```

**Output:**

```
[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```

**Time complexity:** O(n!)
**Space:** O(n)

---

### **Q6: Generate all combinations of k numbers from 1–n**

```python
def combine(n, k):
    res = []
    def backtrack(start, path):
        if len(path) == k:
            res.append(path[:])
            return
        for i in range(start, n+1):
            path.append(i)
            backtrack(i+1, path)
            path.pop()
    backtrack(1, [])
    return res

print(combine(4,2))
```

**Output:**

```
[[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
```

**Time complexity:** O(C(n, k))

---

# 🧮 3. COMBINATORIAL GENERATION (Subsets / Bitmasking)

---

### **Q7: Generate all subsets of a set [1, 2, 3]**

#### (a) Recursive approach:

```python
def subsets(nums):
    res = []
    def dfs(i, path):
        if i == len(nums):
            res.append(path[:])
            return
        dfs(i+1, path)
        path.append(nums[i])
        dfs(i+1, path)
        path.pop()
    dfs(0, [])
    return res
```

**Output:**

```
[[], [3], [2], [2,3], [1], [1,3], [1,2], [1,2,3]]
```

#### (b) Bitmask approach:

Use binary from `0` to `2^n - 1`.

```python
def subsets_bitmask(nums):
    n = len(nums)
    res = []
    for mask in range(1 << n):
        subset = [nums[i] for i in range(n) if mask & (1 << i)]
        res.append(subset)
    return res
```

**Time complexity:** O(n * 2^n)

---

# 🧩 4. ADVANCED PATTERN PROBLEMS

---

### **Q8: Generate all unique permutations (with duplicates)**

Example: `[1,1,2]`

```python
def permuteUnique(nums):
    res = []
    nums.sort()
    def backtrack(path, used):
        if len(path) == len(nums):
            res.append(path[:])
            return
        for i in range(len(nums)):
            if used[i]:
                continue
            if i > 0 and nums[i] == nums[i-1] and not used[i-1]:
                continue
            used[i] = True
            path.append(nums[i])
            backtrack(path, used)
            path.pop()
            used[i] = False
    backtrack([], [False]*len(nums))
    return res

print(permuteUnique([1,1,2]))
```

**Output:**

```
[[1,1,2],[1,2,1],[2,1,1]]
```

**Insight:** Sorting + skipping duplicates avoids repetition.

---

### **Q9: Letter Case Permutation**

Given a string `s`, return all possible letter case permutations.

```python
def letterCasePermutation(s):
    res = []
    def backtrack(i, path):
        if i == len(s):
            res.append("".join(path))
            return
        if s[i].isalpha():
            path.append(s[i].lower())
            backtrack(i+1, path)
            path.pop()
            path.append(s[i].upper())
            backtrack(i+1, path)
            path.pop()
        else:
            path.append(s[i])
            backtrack(i+1, path)
            path.pop()
    backtrack(0, [])
    return res

print(letterCasePermutation("a1b2"))
```

**Output:**

```
['a1b2','a1B2','A1b2','A1B2']
```

---

# ⚙️ 5. FAANG-LEVEL COMBINATORIAL PROBLEMS

---

### **Q10: Combination Sum (LeetCode #39)**

Find all unique combinations of candidates that sum to target.

```python
def combinationSum(candidates, target):
    res = []
    def dfs(i, path, total):
        if total == target:
            res.append(path[:])
            return
        if total > target or i >= len(candidates):
            return
        path.append(candidates[i])
        dfs(i, path, total + candidates[i])  # reuse
        path.pop()
        dfs(i+1, path, total)                # skip
    dfs(0, [], 0)
    return res

print(combinationSum([2,3,6,7], 7))
```

**Output:**

```
[[2,2,3],[7]]
```

**Key DSA Concepts:** recursion + pruning.
**Complexity:** Exponential (O(2^n)), but efficient with pruning.

---

### **Q11: Generate Parentheses (Catalan Number Pattern)**

Generate all valid parentheses for `n` pairs.

```python
def generateParenthesis(n):
    res = []
    def backtrack(openN, closeN, s):
        if openN == closeN == n:
            res.append(s)
            return
        if openN < n:
            backtrack(openN+1, closeN, s+"(")
        if closeN < openN:
            backtrack(openN, closeN+1, s+")")
    backtrack(0,0,"")
    return res
```

**Output for n=3:**

```
["((()))","(()())","(())()","()(())","()()()"]
```

---

### **Q12: Subsets With Duplicates**

```python
def subsetsWithDup(nums):
    res = []
    nums.sort()
    def backtrack(start, path):
        res.append(path[:])
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i-1]:
                continue
            path.append(nums[i])
            backtrack(i+1, path)
            path.pop()
    backtrack(0, [])
    return res

print(subsetsWithDup([1,2,2]))
```

**Output:**

```
[[],[1],[1,2],[1,2,2],[2],[2,2]]
```

---

### **Q13: Combinations that form Palindromes**

Given a string, generate all unique permutations that form valid palindromes.

**Concept:** Count frequencies → check if ≤1 odd → generate half → mirror.

---

### **Q14: K-th Permutation Sequence (LeetCode #60)**

Return the k-th permutation of numbers `[1..n]` lexicographically.

Uses factorial numbering system.

---

### **Q15: Count Unique Paths (Combinatorics + DP)**

Grid m×n, start top-left, move only down/right.

[
\text{Paths} = \binom{m+n-2}{m-1}
]

or via DP.

```python
def uniquePaths(m, n):
    from math import comb
    return comb(m+n-2, m-1)
```

---

# 🌟 Summary Table

| Type         | Problem               | Concept             | Complexity  |
| ------------ | --------------------- | ------------------- | ----------- |
| Counting     | Arrangements of APPLE | Repetition          | O(1)        |
| Backtracking | Permutations of list  | Recursion           | O(n!)       |
| Backtracking | Combinations          | Recursion           | O(C(n,k))   |
| Bitmasking   | Subsets               | Bitmask             | O(n·2ⁿ)     |
| Unique       | Duplicates skip       | Sorting + backtrack | O(n!)       |
| FAANG        | Combination Sum       | DFS + pruning       | Exponential |
| FAANG        | Generate Parentheses  | Catalan             | Exponential |

---



Would you like me to generate that next?
