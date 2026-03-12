# Counting: Permutations, Combinations & the Rule of Product

> *"If you can't count, you can't do probability."*

---

## Table of Contents

1. [Why Counting = Probability](#1-why-counting--probability)
2. [Rule of Product](#2-rule-of-product)
3. [Permutations](#3-permutations)
4. [Combinations](#4-combinations)
5. [The Fundamental Relationship](#5-the-fundamental-relationship)
6. [Special Formulas](#6-special-formulas)
7. [Inclusion-Exclusion Principle](#7-inclusion-exclusion-principle)
8. [The Decision Framework](#8-the-decision-framework)
9. [Common Traps](#9-common-traps)
10. [Problem Sets](#10-problem-sets)
11. [Counting → Probability → DP](#11-counting--probability--dp)
12. [Master Cheat Sheet](#12-master-cheat-sheet)

---

## 1. Why Counting = Probability

When all outcomes are equally likely:

$$P(\text{event}) = \frac{\text{number of favorable outcomes}}{\text{total number of outcomes}}$$

This formula is deceptively simple. **The hard part is always the counting.** Two quantities to compute — both require counting techniques.

**Motivating example:** Flip a fair coin 10 times. What is the probability of exactly 3 heads?

| Quantity | How to count | Value |
|---|---|---|
| Total outcomes | Rule of product: $2^{10}$ | 1024 |
| Favorable outcomes | Combinations: $\binom{10}{3}$ | 120 |
| **Probability** | $120 / 1024$ | **≈ 0.117** |

Without counting, you have nothing.

---

## 2. Rule of Product

> **If there are $n$ ways to perform action 1 and $m$ ways to perform action 2, there are $n \cdot m$ ways to perform both in sequence.**

Also called the **multiplication principle**.

**Critical subtlety:** The rule holds even when the *specific* options for action 2 depend on the choice in action 1 — as long as the *number* of options remains the same regardless of that choice.

### Examples

**Shirts and pants:** 5 shirts × 3 pants = **15 outfits**

**Olympic medals:** 5 competitors, award gold / silver / bronze.
- 5 ways to pick gold → 4 remaining for silver → 3 for bronze
$$5 \times 4 \times 3 = 60 \text{ ways}$$

Who wins silver *depends on* who won gold, but the *count* of silver candidates is always 4. Rule of product applies.

**Password:** 3 letters then 2 digits (repetition allowed):
$$26^3 \times 10^2 = 1{,}757{,}600$$

### Case-Splitting (Rule of Product + Cases)

Sometimes the number of options in a later step depends on an earlier choice — then you must split into cases.

**Wardrobe problem:** *"I won't wear green and red together."*
- Shirts: 3 Black (B), 3 Red (R), 2 Green (G)
- Sweaters: 1B, 2R, 1G
- Pants: 2 Denim, 2 Black

| Shirt choice | Compatible sweaters | Pants | Subtotal |
|---|---|---|---|
| Red (3 shirts) | R, B (not G) → 3 sweaters | 4 pants | $3 \times 3 \times 4 = 36$ |
| Black (3 shirts) | R, B, G → 4 sweaters | 4 pants | $3 \times 4 \times 4 = 48$ |
| Green (2 shirts) | B, G (not R) → 2 sweaters | 4 pants | $2 \times 2 \times 4 = 16$ |
| **Total** | | | **100** |

---

## 3. Permutations

A **permutation** is an *ordered* arrangement. **Order matters.**

$${}_{n}P_{r} = \frac{n!}{(n-r)!} = n(n-1)\cdots(n-r+1)$$

### All elements

The set $\{a, b, c\}$ has $3! = 6$ permutations: $abc, acb, bac, bca, cab, cba$.

In general, $k$ distinct elements can be arranged in $k!$ ways.

### Choose $r$ from $n$

All permutations of 3 elements from $\{a, b, c, d\}$:

```
abc  acb  bac  bca  cab  cba
abd  adb  bad  bda  dab  dba
acd  adc  cad  cda  dac  dca
bcd  bdc  cbd  cdb  dbc  dcb
```

${}_{4}P_{3} = 4 \times 3 \times 2 = 24$ — no need to list when you understand the formula.

### Repeated elements

When some elements are identical, arrangements that differ only in the positions of identical elements are the *same* arrangement. Divide by the factorial of each repeat count.

$$\text{Arrangements of } n \text{ items with repeats} = \frac{n!}{a_1!\, a_2!\, \cdots\, a_k!}$$

| Word | Letters | Formula | Answer |
|---|---|---|---|
| APPLE | A,P,P,L,E — P repeats 2× | $5!/2!$ | 60 |
| BANANA | B,A,A,A,N,N — A×3, N×2 | $6!/(3!\cdot2!)$ | 60 |
| STATISTICS | S×3, T×3, A×1, I×2, C×1 | $10!/(3!\cdot3!\cdot2!)$ | 50,400 |

### Circular permutations

In a circle, rotations of the same arrangement are identical. Fix one person, arrange the rest:

$$\text{Circular permutations of } n = (n-1)!$$

5 people around a round table: $(5-1)! = 24$ ways.

---

## 4. Combinations

A **combination** is an *unordered* selection. **Order does not matter.**

$$\binom{n}{r} = {}_{n}C_{r} = \frac{n!}{r!(n-r)!}$$

Read as **"$n$ choose $r$."**

### Side-by-side comparison

All combinations of 3 from $\{a,b,c,d\}$:

| Permutations (all 6 orderings) | Combination |
|---|---|
| $abc,\; acb,\; bac,\; bca,\; cab,\; cba$ | $\{a,b,c\}$ |
| $abd,\; adb,\; bad,\; bda,\; dab,\; dba$ | $\{a,b,d\}$ |
| $acd,\; adc,\; cad,\; cda,\; dac,\; dca$ | $\{a,c,d\}$ |
| $bcd,\; bdc,\; cbd,\; cdb,\; dbc,\; dcb$ | $\{b,c,d\}$ |

24 permutations → 4 combinations. Each row collapses $3! = 6$ orderings into one subset.

### DNA sequences (worked example)

DNA nucleotides: {A, C, G, T}

| Question | Reasoning | Answer |
|---|---|---|
| Sequences of length 3, repeats OK | Each position: 4 choices, independently | $4^3 = 64$ |
| Sequences of length 3, no repeats | Order matters (ACG ≠ GAC) + no repetition | ${}_{4}P_{3} = 24$ |

---

## 5. The Fundamental Relationship

$$\boxed{{}_{n}P_{r} = \binom{n}{r} \times r!}$$

A permutation = first *choose* the $r$ elements (combinations), then *order* them ($r!$ ways).

**Worked example:** ${}_{4}P_{3} = \binom{4}{3} \times 3! = 4 \times 6 = 24$ ✓

This relationship is the key to *deriving* the combination formula from the permutation formula:

$$\binom{n}{r} = \frac{{}_{n}P_{r}}{r!} = \frac{n!}{r!(n-r)!}$$

---

## 6. Special Formulas

### Repetition allowed

| Situation | Formula | Example |
|---|---|---|
| Arrange $r$ items from $n$ with repetition | $n^r$ | 4-letter words from 3 letters: $3^4 = 81$ |
| Select $r$ items from $n$ with repetition (unordered) | $\binom{n+r-1}{r}$ | 3 identical balls, 2 distinct boxes: $\binom{4}{3} = 4$ |

The second formula is the **Stars and Bars** method. It counts non-negative integer solutions to:
$$x_1 + x_2 + \cdots + x_k = n$$
Answer: $\binom{n+k-1}{k-1}$

**With minimum constraints:** If each $x_i \geq m$, substitute $y_i = x_i - m$ to reduce to the standard form.

*Example:* Distribute 10 identical balls into 3 boxes, each box ≥ 2.
- Let $y_i = x_i - 2$: then $y_1 + y_2 + y_3 = 4$
- Answer: $\binom{4+3-1}{3-1} = \binom{6}{2} = 15$

### Subsets

Every element is either *in* or *out*: $2^n$ total subsets of a set of size $n$.

- Including empty set: $2^n$
- Excluding empty set (at least one element): $2^n - 1$
- Subsets of exactly size $k$: $\binom{n}{k}$

### Grid paths

Number of shortest paths from $(0,0)$ to $(m,n)$ moving only right or up:
$$\binom{m+n}{n}$$
You must make $m+n$ total moves; choose which $n$ of them are "up."

### Catalan numbers

$$C_n = \frac{1}{n+1}\binom{2n}{n}$$

Appears whenever you're counting balanced or non-crossing structures:

| Problem | Answer |
|---|---|
| Valid sequences of $n$ pairs of parentheses | $C_n$ |
| Number of distinct BSTs with $n$ keys | $C_n$ |
| Paths from $(0,0)$ to $(n,n)$ that stay below the diagonal | $C_n$ |
| Ways to triangulate a convex $(n+2)$-gon | $C_n$ |

---

## 7. Inclusion-Exclusion Principle

For two sets:
$$|A \cup B| = |A| + |B| - |A \cap B|$$

Translates directly to probability:
$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

**Band example:** 7 singers, 4 guitarists, 2 do both → band size = $7 + 4 - 2 = 9$.

### "At least one" — the complement trick

Computing $P(\text{at least one } \ldots)$ directly means summing many cases. The complement is usually one calculation:

$$P(\text{at least one}) = 1 - P(\text{none})$$

*Example:* P(at least one head in 10 flips):
$$1 - P(\text{no heads}) = 1 - \left(\frac{1}{2}\right)^{10} = 1 - \frac{1}{1024} \approx 0.999$$

*Example:* Committee of 4 from 6 men + 5 women, at least one woman:
$$\binom{11}{4} - \binom{6}{4} = 330 - 15 = 315$$

---

## 8. The Decision Framework

When you see a counting problem, run through this decision tree in order:

```
Is each position/slot filled independently with k options?
    └─ YES → k^N  (or product of options per position)

Are you selecting/arranging items from a pool?
    ├─ Does order matter?
    │      YES → Permutation: nPr
    │      NO  → Combination: nCr
    │
    ├─ Are objects identical (indistinguishable)?
    │      YES → Stars & Bars
    │
    └─ Are all items used?
           YES → N! (or N!/repeats for identical items)

Are there overlapping constraints ("at least", "at most", "divisible by")?
    └─ YES → Complement trick or Inclusion-Exclusion

Is there a "no adjacency" constraint?
    └─ YES → Often Fibonacci / DP recurrence

Does it involve balanced structures (parentheses, paths, trees)?
    └─ YES → Catalan number
```

---

## 9. Common Traps

### ❌ Using $\binom{n}{r}$ when order matters

> "Choose 3 winners: Gold, Silver, Bronze from 10 people"

Wrong: $\binom{10}{3} = 120$ — ignores ranking.
Correct: ${}_{10}P_{3} = 720$ — ranking is the whole point.

### ❌ Forgetting the circular permutation adjustment

6 people around a table:
Wrong: $6! = 720$. Correct: $(6-1)! = 120$.
Rotations are the same arrangement. Fix one person; only the *relative* positions matter.

### ❌ Identical vs. distinct objects in distribution

| Scenario | Method | Example ($n=10$, $k=3$) |
|---|---|---|
| Identical objects → distinct boxes | Stars & Bars | $\binom{12}{2} = 66$ |
| Distinct objects → distinct boxes | Each object chooses a box: $k^n$ | $3^{10} = 59{,}049$ |

Confusing these two is one of the most common errors in combinatorics.

### ❌ Forgetting repetition rules

Form a 4-digit code from 1–9:
- Repetition allowed: $9^4 = 6{,}561$
- No repetition: ${}_{9}P_{4} = 3{,}024$

Always ask: *can the same item be chosen more than once?*

### ❌ Leading-zero constraint

5-digit numbers from digits 0–9 (no repetition):
Wrong: ${}_{10}P_{5} = 30{,}240$ (treats leading zeros as valid).
Correct: First digit has 9 choices (1–9), remaining 4 positions from remaining 9 digits: $9 \times {}_{9}P_{4} = 27{,}216$.

### ❌ "At least" computed by summing cases instead of complement

"At least 2 women on a committee" — do not sum Case(2 women) + Case(3 women) + … unless forced. First ask: is $1 - P(\text{fewer than 2 women})$ simpler?

### ❌ Subsets vs. subsequences vs. subarrays

| Structure | Count |
|---|---|
| Subsets of array length $n$ | $2^n$ |
| Subsequences of array length $n$ | $2^n$ |
| Contiguous subarrays of length $n$ | $n(n+1)/2$ |

These are different objects. Know which one the problem is asking for.

---

## 10. Problem Sets

### Quick Reference (MIT 18.05 Class 1)

| Problem | Concept | Answer |
|---|---|---|
| DNA sequences, length 3, repeats OK | Rule of Product | $4^3 = 64$ |
| DNA sequences, length 3, no repeats | Permutation | ${}_{4}P_{3} = 24$ |
| Band members (singers ∪ guitarists) | Inclusion-Exclusion | $7+4-2=9$ |
| Olympic medals (5 competitors) | Permutation | $5 \times 4 \times 3 = 60$ |
| Wardrobe with color constraints | Rule of Product + cases | $36+48+16=100$ |
| Exactly 3 heads in 10 flips (count) | Combination | $\binom{10}{3}=120$ |
| Exactly 3 heads in 10 flips (prob.) | Combination + Rule of Product | $120/1024 \approx 0.117$ |

---

### Easy (E1–E15)

**E1.** How many ways to arrange 4 distinct books on a shelf?
$$4! = 24$$

**E2.** How many 2-digit numbers using 1,2,3,4,5 without repetition?
$${}_{5}P_{2} = 5 \times 4 = 20$$

**E3.** How many ways to select 2 students from 10?
$$\binom{10}{2} = 45$$

**E4.** How many arrangements of "APPLE"?
$$\frac{5!}{2!} = 60 \quad \text{(P repeats twice)}$$

**E5.** How many 3-letter words from A,B,C,D with repetition?
$$4^3 = 64$$

**E6.** From 5 men and 6 women, a team of 3 with exactly 1 woman.
$$\binom{6}{1}\binom{5}{2} = 6 \times 10 = 60$$

**E7.** How many 4-digit numbers from 1–6, no repeats?
$${}_{6}P_{4} = 6 \times 5 \times 4 \times 3 = 360$$

**E8.** How many ways can 3 students sit in 3 chairs?
$$3! = 6$$

**E9.** How many ways to choose 3 cards from 52?
$$\binom{52}{3} = 22{,}100$$

**E10.** How many ways to arrange 3 letters from A,B,C,D without repeats?
$${}_{4}P_{3} = 24$$

**E11.** How many ways to select president and VP from 5 members?
$${}_{5}P_{2} = 20$$

**E12.** How many arrangements of "MISS"?
$$\frac{4!}{2!} = 12 \quad \text{(S repeats twice)}$$

**E13.** 3-digit numbers divisible by 5, digits from 1–5, no repeats?
Last digit must be 5 (fixed). Remaining 2 positions from $\{1,2,3,4\}$:
$${}_{4}P_{2} = 12$$

**E14.** Select a team of 3 from 7 men.
$$\binom{7}{3} = 35$$

**E15.** 3-digit numbers with distinct digits from 0–5.
First digit ≠ 0: 5 choices. Next 2 positions from remaining 5 digits:
$$5 \times 5 \times 4 = 100$$

---

### Medium (M1–M15)

**M1.** Arrange A,B,C,D,E in a row with A always first.
A is fixed; arrange remaining 4:
$$4! = 24$$

**M2.** From 6 men and 4 women, committee of 3 with at least 1 woman.

| Case | Count |
|---|---|
| 1W, 2M | $\binom{4}{1}\binom{6}{2} = 60$ |
| 2W, 1M | $\binom{4}{2}\binom{6}{1} = 36$ |
| 3W | $\binom{4}{3} = 4$ |
| **Total** | **100** |

**M3.** Arrangements of "BANANA"? (B×1, A×3, N×2)
$$\frac{6!}{3!\cdot2!\cdot1!} = 60$$

**M4.** 4-digit even numbers from 1–6, no repeats.
Last digit ∈ {2,4,6}: 3 choices. Remaining 3 positions from other 5:
$$3 \times {}_{5}P_{3} = 3 \times 60 = 180$$

**M5.** 5 people around a round table.
$$(5-1)! = 24$$

**M6.** Distribute 3 identical balls into 2 distinct boxes.
$$\binom{3+2-1}{2-1} = \binom{4}{1} = 4$$
(Distributions: 3+0, 2+1, 1+2, 0+3)

**M7.** Team of 2 men and 2 women from 5 men and 6 women.
$$\binom{5}{2}\binom{6}{2} = 10 \times 15 = 150$$

**M8.** 4-digit numbers divisible by 5 using 1–5, no repeats.
Last digit = 5 (only even-of-5 option). Remaining 3 positions from {1,2,3,4}:
$${}_{4}P_{3} = 24$$

**M9.** 4 letters of ABCDE with A and B adjacent (in either order).
Treat AB as a unit → 4 objects, arrange in $4!$ ways, then AB can be BA:
$$4! \times 2 = 48$$

**M10.** 3-digit numbers from 1–5 with at least 2 odd digits (odd: 1,3,5; even: 2,4).

| Case | Count |
|---|---|
| Exactly 2 odd, 1 even | $\binom{3}{2}\binom{2}{1}\times 3! = 36$ |
| All 3 odd | $3! = 6$ |
| **Total** | **42** |

**M11.** 4-letter words from A,B,C,D,E with repetition.
$$5^4 = 625$$

**M12.** 4-digit numbers with distinct digits from 1–7.
$${}_{7}P_{4} = 840$$

**M13.** 6 students in a row; 2 specific students must NOT be adjacent.
$$\underbrace{6!}_{\text{total}} - \underbrace{5! \times 2}_{\text{adjacent}} = 720 - 240 = 480$$

**M14.** Select president, VP, and secretary from 10 students.
$${}_{10}P_{3} = 720$$

**M15.** From 5 men and 6 women, team of 4 with at least 2 women.

| Case | Count |
|---|---|
| 2W, 2M | $\binom{6}{2}\binom{5}{2} = 150$ |
| 3W, 1M | $\binom{6}{3}\binom{5}{1} = 100$ |
| 4W | $\binom{6}{4} = 15$ |
| **Total** | **265** |

---

### Hard (H1–H10)

**H1.** 7 people in a row, 3 specific people always together.
Treat the 3 as one unit → 5 units total:
$$\underbrace{5!}_{\text{arrange units}} \times \underbrace{3!}_{\text{internal order}} = 120 \times 6 = 720$$

**H2.** 8 people in a row, 2 specific people NOT adjacent.
$$8! - 7! \times 2 = 40{,}320 - 10{,}080 = 30{,}240$$

**H3.** 5 books to 3 distinct shelves, each shelf gets at least 1.

| Distribution | Count |
|---|---|
| 3,1,1: choose shelf for 3, arrange books | $3 \times {}_{5}P_{3} \times 2! = 360$ |
| 2,2,1: choose shelf for 1, then split 4 into 2+2 | $3 \times \binom{5}{1} \times \frac{\binom{4}{2}}{2!} \times 2! \times 2! = 180$ |
| **Total** | **540** |

**H4.** Arrange "SUCCESS". (S×3, C×2, U×1, E×1)
$$\frac{7!}{3!\cdot2!} = 420$$

**H5.** 4-digit numbers from 1–7 with at least 2 odd digits (odd: 1,3,5,7; even: 2,4,6).

| Case | Count |
|---|---|
| 2 odd, 2 even | $\binom{4}{2}\binom{3}{2} \times 4! = 6\times3\times24 = 432$ |
| 3 odd, 1 even | $\binom{4}{3}\binom{3}{1} \times 4! = 4\times3\times24 = 288$ |
| 4 odd | $1 \times 1 \times 4! = 24$ |
| **Total** | **744** |

**H6.** 6 students sit in a circle.
$$(6-1)! = 120$$

**H7.** Arrange "PROGRAM" with G and R always together.
Treat GR as one unit → 6 units:
$$6! \times 2 = 1{,}440 \quad \text{(GR or RG)}$$

**H8.** 4-men, 3-women in a row so no two men are adjacent.
Place women first in $3!$ ways. This creates 4 gaps (\_W\_W\_W\_). Place 4 men in those 4 gaps:
$$3! \times {}_{4}P_{4} = 6 \times 24 = 144$$

**H9.** How many 5-digit numbers have strictly increasing digits (from 0–9)?
Strictly increasing means choosing 5 digits determines the number uniquely (only one ordering). But 0 cannot be the leading digit:
- Total: $\binom{10}{5} = 252$
- Subtract those starting with 0: 0 is fixed as first; choose 4 more from 1–9 increasing: $\binom{9}{4} = 126$
$$252 - 126 = 126$$

**H10.** Arrange "STATISTICS". (S×3, T×3, I×2, A×1, C×1)
$$\frac{10!}{3!\cdot3!\cdot2!\cdot1!\cdot1!} = 50{,}400$$

---

### Brutal FAANG-Level

**B1. Binary strings of length $N$ with no two consecutive 1s.**

Cannot just say $2^N$. Define $f(N)$ = valid strings of length $N$:
- If last digit is 0: any valid string of length $N-1$ can precede it → $f(N-1)$ ways
- If last digit is 1: the previous must be 0 → $f(N-2)$ ways

$$f(N) = f(N-1) + f(N-2), \quad f(1)=2,\; f(2)=3$$

This is the **Fibonacci sequence**. The answer for length $N$ is $F_{N+2}$.

*Same structure:* counting subsets of $\{1,\ldots,N\}$ with no two consecutive elements.

---

**B2. Number of functions from set $A$ (size $m$) to set $B$ (size $n$).**

Each of the $m$ elements in $A$ independently maps to one of $n$ elements in $B$:
$$n^m$$

Special cases: $n=2$ gives $2^m$ (characteristic functions = subsets of $A$).

---

**B3. How many 5-digit numbers have strictly increasing digits (full analysis)?**

See H9 above — the key insight is that a strictly increasing sequence is entirely determined by which digits you choose. Choosing the set of digits gives exactly one valid number.

---

**B4. Number of paths from $(0,0)$ to $(N,N)$ that never go above the diagonal.**

Total unrestricted paths: $\binom{2N}{N}$

Valid (non-crossing) paths = **Catalan number**:
$$C_N = \frac{1}{N+1}\binom{2N}{N}$$

Appears in: balanced parentheses, BST counting, triangulating polygons, Dyck paths.

---

**B5. Expected number of fixed points in a random permutation of $N$ elements.**

Let $X_i = 1$ if element $i$ is in its original position. By linearity of expectation:
$$E\left[\sum_{i=1}^N X_i\right] = \sum_{i=1}^N P(X_i = 1) = \sum_{i=1}^N \frac{1}{N} = 1$$

The expected number of fixed points is **always 1**, regardless of $N$.

---

**B6. Seat $N$ couples in a row such that no couple sits together.**

Let $A_i$ = "couple $i$ sits together." Use inclusion-exclusion:

$$P\left(\bigcap_{i=1}^N A_i^c\right) = \sum_{k=0}^{N} (-1)^k \binom{N}{k} \frac{(2N-k)! \cdot 2^k}{(2N)!}$$

---

**B7. Color a cycle of $N$ nodes with 3 colors, adjacent nodes different.**

Using the chromatic polynomial for a cycle:
$$(3-1)^N + (-1)^N(3-1) = 2^N + (-1)^N \cdot 2$$

For $N=4$: $2^4 + 2 = 18$.

---

## 11. Counting → Probability → DP

### The Chain

$$\text{Counting} \xrightarrow{\div \text{ total outcomes}} \text{Probability} \xrightarrow{\text{recurrence}} \text{Dynamic Programming}$$

Many DP problems are **hidden counting problems**. Recognize the structure:

| DP Problem | What it's actually counting |
|---|---|
| Climbing stairs (1 or 2 steps) | Valid step sequences → Fibonacci |
| Unique paths in grid | Combinations $\binom{m+n}{n}$ |
| Decode ways | String partition count |
| Count subsets with sum $k$ | Subset combinations |
| Coin change (count ways) | Compositions with parts from coin set |
| Longest increasing subsequence | Constrained subsequence count |

### When combinatorics becomes DP

Use DP (instead of a closed-form formula) when:
- There are adjacency constraints (e.g., no consecutive 1s)
- There are running totals or cumulative conditions
- The structure is recursive with overlapping subproblems
- Problem size is $N \leq 20$ → likely $O(N \cdot 2^N)$ bitmask DP

### Bitmask DP

If a problem asks "count subsets" or "ways to select" with $N \leq 20$, each subset corresponds to a bitmask (an integer from $0$ to $2^N - 1$). Iterate over all $2^N$ states:
$$O(N \cdot 2^N)$$

---

## 12. Master Cheat Sheet

```
SITUATION                              FORMULA
─────────────────────────────────────────────────────────
Each of N positions has k options      k^N
Binary (include/exclude each element)  2^N
Subsets of size r                      nCr
Ordered selections of r from n         nPr  =  n!/(n-r)!
Arrange all n distinct items           n!
Arrange with repeated items            n! / (a! b! c! ...)
Circular arrangement of n              (n-1)!
Distribute n identical → k boxes       C(n+k-1, k-1)  [Stars & Bars]
Distribute n distinct → k boxes        k^n
Grid paths (m right, n up)             C(m+n, n)
Balanced/non-crossing structures       Catalan: C(2n,n)/(n+1)

CONSTRAINT STRATEGY
─────────────────────────────────────────────────────────
"At least one"        →  1 - P(none)    [complement]
"At most k"           →  sum cases 0..k
"None adjacent"       →  Fibonacci / DP
"Divisible by"        →  fix last digit(s), count rest
Leading digit ≠ 0     →  9 choices for first digit
Overlapping sets      →  Inclusion-Exclusion

DECISION TRIGGERS
─────────────────────────────────────────────────────────
Ranking / ordering     →  Permutation
Selection / committee  →  Combination
Fixed positions        →  Rule of Product + fix those positions
Identical objects      →  Stars & Bars
Circular seating       →  (n-1)!
Balanced structure     →  Catalan
No-adjacency           →  Fibonacci recurrence
```
# 18.05 Problem Set 1 — Solutions

**Spring 2022**

---

## Table of Contents

1. [Problem 1 — 6-Card Poker Hands](#problem-1--6-card-poker-hands)
2. [Problem 2 — Non-Transitive Dice](#problem-2--non-transitive-dice)
3. [Problem 3 — The Birthday Problem](#problem-3--the-birthday-problem)

---

## Problem 1 — 6-Card Poker Hands (20 pts.)

We draw a hand of 6 cards from a standard 52-card deck. All hands are equally likely.

**Total number of 6-card hands:**
$$\binom{52}{6} = 20{,}358{,}520$$

We compute the probability of two specific hand types.

---

### Part (a): Two Pair

**Definition:** Exactly two cards share one rank, exactly two others share a second rank, and the remaining two cards have two *different* ranks (neither matching each other or the pairs).

**Example:** $\{2\heartsuit, 2\spadesuit, 5\heartsuit, 5\clubsuit, Q\diamondsuit, K\diamondsuit\}$

**Why we must be careful:** The phrase "two pair" is precise. The remaining two cards must have *different* ranks (otherwise we'd have a full house or two more pairs).

#### Counting Step by Step

| Step | What we choose | Count |
|---|---|---|
| 1 | Choose 2 ranks for the pairs (from 13 ranks) | $\binom{13}{2}$ |
| 2 | Choose 2 suits for each pair rank | $\binom{4}{2}^2$ |
| 3 | Choose 2 *different* ranks for the remaining 2 cards (from the 11 leftover ranks) | $\binom{11}{2}$ |
| 4 | Choose 1 suit for each of those 2 cards | $\binom{4}{1}^2$ |

**Total two-pair hands:**
$$\binom{13}{2}\binom{4}{2}^2\binom{11}{2}\binom{4}{1}^2 = 78 \times 36 \times 55 \times 16 = 2{,}471{,}040$$

**Probability:**
$$P(\text{two pair}) = \frac{2{,}471{,}040}{20{,}358{,}520} \approx \boxed{0.1214}$$

---

### Part (b): Three of a Kind

**Definition:** Exactly three cards share one rank; the remaining three cards have three *distinct* other ranks (no two matching each other, and none matching the triple).

**Example:** $\{2\heartsuit, 2\spadesuit, 2\clubsuit, 5\clubsuit, 9\spadesuit, K\heartsuit\}$

#### Counting Step by Step

| Step | What we choose | Count |
|---|---|---|
| 1 | Choose the rank for the triple | $\binom{13}{1}$ |
| 2 | Choose 3 suits for that rank | $\binom{4}{3}$ |
| 3 | Choose 3 different ranks for the remaining 3 cards (from 12 leftover) | $\binom{12}{3}$ |
| 4 | Choose 1 suit for each of those 3 cards | $\binom{4}{1}^3$ |

**Total three-of-a-kind hands:**
$$\binom{13}{1}\binom{4}{3}\binom{12}{3}\binom{4}{1}^3 = 13 \times 4 \times 220 \times 64 = 732{,}160$$

**Probability:**
$$P(\text{three of a kind}) = \frac{732{,}160}{20{,}358{,}520} \approx \boxed{0.03596}$$

---

### Comparison

$$P(\text{two pair}) \approx 0.1214 \gg P(\text{three of a kind}) \approx 0.0360$$

Two pair is roughly **3.4 times more likely** than three of a kind. This is why three of a kind ranks higher in poker — rarity determines value.

**Intuition for the gap:** For two pair, you choose 2 ranks for pairs (from 13), then 2 more ranks for singletons. The singletons have a lot of flexibility ($\binom{11}{2} = 55$ choices). For three of a kind, you need 3 cards of the same rank — the suits constraint ($\binom{4}{3} = 4$) is tight. The 6-card setting makes two pair especially common since having 6 cards gives more opportunity for ranks to collide in pairs.

---

## Problem 2 — Non-Transitive Dice (20 pts.)

**Setup:** Three dice with non-standard faces:

| Die | Faces |
|---|---|
| **Blue** | 3, 3, 3, 3, 3, 6 |
| **Orange** | 1, 4, 4, 4, 4, 4 |
| **White** | 2, 2, 2, 5, 5, 5 |

---

### Part (a): Single Die Head-to-Head Comparisons

#### White vs. Orange

Build a 6×6 outcome table. White shows {2,2,2,5,5,5} and Orange shows {1,4,4,4,4,4}.

| | O=1 | O=4 | O=4 | O=4 | O=4 | O=4 |
|---|---|---|---|---|---|---|
| **W=2** | W wins | O wins | O wins | O wins | O wins | O wins |
| **W=2** | W wins | O wins | O wins | O wins | O wins | O wins |
| **W=2** | W wins | O wins | O wins | O wins | O wins | O wins |
| **W=5** | W wins | W wins | W wins | W wins | W wins | W wins |
| **W=5** | W wins | W wins | W wins | W wins | W wins | W wins |
| **W=5** | W wins | W wins | W wins | W wins | W wins | W wins |

White wins in: 3 (from W=2, O=1) + 15 (from W=5, all Orange) = **21 cells**

$$P(\text{White beats Orange}) = \frac{21}{36} = \frac{7}{12} \approx 0.583$$

#### Orange vs. Blue

| | B=3 | B=3 | B=3 | B=3 | B=3 | B=6 |
|---|---|---|---|---|---|---|
| **O=1** | B | B | B | B | B | B |
| **O=4** | O | O | O | O | O | B |
| **O=4** | O | O | O | O | O | B |
| **O=4** | O | O | O | O | O | B |
| **O=4** | O | O | O | O | O | B |
| **O=4** | O | O | O | O | O | B |

Orange wins: $5 \times 5 = 25$ cells (O=4 beats B=3).

$$P(\text{Orange beats Blue}) = \frac{25}{36} \approx 0.694$$

#### Blue vs. White

| | W=2 | W=2 | W=2 | W=5 | W=5 | W=5 |
|---|---|---|---|---|---|---|
| **B=3** | B | B | B | W | W | W |
| **B=3** | B | B | B | W | W | W |
| **B=3** | B | B | B | W | W | W |
| **B=3** | B | B | B | W | W | W |
| **B=3** | B | B | B | W | W | W |
| **B=6** | B | B | B | B | B | B |

Blue wins: $5 \times 3$ (B=3 beats W=2) + $1 \times 6$ (B=6 beats all) = $15 + 6 = 21$ cells.

$$P(\text{Blue beats White}) = \frac{21}{36} = \frac{7}{12} \approx 0.583$$

#### Summary: The Cycle

$$\text{Blue} \xrightarrow{7/12} \text{beats} \xrightarrow{} \text{White} \xrightarrow{7/12} \text{beats} \xrightarrow{} \text{Orange} \xrightarrow{25/36} \text{beats} \xrightarrow{} \text{Blue}$$

There is **no best die**. The relation "beats" is not transitive. This violates the assumption most people bring to comparisons — that if A > B and B > C, then A > C.

> **Why this is remarkable:** In everyday life, preferences and rankings are usually assumed transitive. Non-transitive dice show that pairwise "better than" does not guarantee a global ordering. This has real implications in voting theory (Condorcet's paradox) and game theory.

---

### Part (b): Two White vs. Two Blue (Sum Comparison)

Roll two White dice and two Blue dice; compare the sums.

**White sum:** Each die shows 2 (prob 1/2) or 5 (prob 1/2). Possible sums and their probabilities:

| White sum | Probability |
|---|---|
| 4 (2+2) | 1/4 |
| 7 (2+5 or 5+2) | 1/2 |
| 10 (5+5) | 1/4 |

**Blue sum:** Each die shows 3 (prob 5/6) or 6 (prob 1/6). Possible sums:

| Blue sum | Probability |
|---|---|
| 6 (3+3) | 25/36 |
| 9 (3+6 or 6+3) | 10/36 |
| 12 (6+6) | 1/36 |

**Computing $P(\text{White sum} > \text{Blue sum})$:** Check all 9 combinations:

| White\Blue | 6 (25/36) | 9 (10/36) | 12 (1/36) |
|---|---|---|---|
| **4** (1/4) | 4 < 6: B | 4 < 9: B | 4 < 12: B |
| **7** (1/2) | 7 > 6: W | 7 < 9: B | 7 < 12: B |
| **10** (1/4) | 10 > 6: W | 10 > 9: W | 10 < 12: B |

$P(\text{White wins}) = \frac{1}{2}\cdot\frac{25}{36} + \frac{1}{4}\cdot\frac{25}{36} + \frac{1}{4}\cdot\frac{10}{36}$
$= \frac{25}{72} + \frac{25}{144} + \frac{10}{144} = \frac{50}{144} + \frac{25}{144} + \frac{10}{144} = \frac{85}{144}$

$$\boxed{P(\text{White pair beats Blue pair}) = \frac{85}{144} \approx 0.590}$$

**The reversal:** A single Blue die beats a single White die with probability 7/12 ≈ 0.583. But two White dice out-sum two Blue dice with probability 85/144 ≈ 0.590.

> **Why this happens:** Aggregation changes the effective distributions. The Blue die's occasional 6 (which makes it beat White) becomes less impactful relative to White's high-end value of 10 when summing two dice. This is a concrete illustration of **Simpson's paradox**-like reversal through aggregation.

---

## Problem 3 — The Birthday Problem (55 pts.)

**Setup:**
- 365 equally likely birthdays
- $n$ people in a room
- Sample space: sequences $\omega = (b_1, b_2, \ldots, b_n)$ where each $b_i \in \{1, \ldots, 365\}$
- Total outcomes: $365^n$, each with probability $1/365^n$

---

### Part (a): Probability Function

The sample space is uniform:

$$P(\omega) = \frac{1}{365^n} \quad \text{for every } \omega$$

This is the **equally likely outcomes model**. It applies here because we assume each birthday is equally likely and birthdays are independent across people.

---

### Part (b): Defining the Events

Let your birthday be day $b$.

**Event A:** At least one of the $n$ people shares your birthday.
$$A = \{\omega : b_k = b \text{ for some } k \in \{1,\ldots,n\}\}$$

**Event B:** At least two people (among the $n$) share *any* birthday.
$$B = \{\omega : b_j = b_k \text{ for some } j \neq k\}$$

**Event C:** At least three people share *any* birthday.
$$C = \{\omega : b_j = b_k = b_\ell \text{ for some distinct } j, k, \ell\}$$

Note the key difference: Event A is about *your specific* birthday; Events B and C are about *any* shared birthday among the group.

---

### Part (c): Probability of Event A

**Complement is easier:** $A^c$ is the event that *none* of the $n$ people share your birthday. Each person independently avoids day $b$ with probability $364/365$:

$$P(A^c) = \left(\frac{364}{365}\right)^n$$

$$\boxed{P(A) = 1 - \left(\frac{364}{365}\right)^n}$$

**Solving for $P(A) > 0.5$:**

$$1 - \left(\frac{364}{365}\right)^n > 0.5 \implies \left(\frac{364}{365}\right)^n < 0.5$$

$$n > \frac{\ln 0.5}{\ln(364/365)} = \frac{-0.6931}{-0.002740} \approx 252.6$$

$$\boxed{n = 253 \text{ people needed}}$$

---

### Part (d): Why is 253 > 365/2?

**The short answer:** Event A is asking about *your specific* birthday. Only 1 out of 365 days qualifies. Each of the $n$ people has only a $1/365 \approx 0.27\%$ chance of matching you.

**The misconception:** One might think "there are 365 days, so 365/2 ≈ 183 people should suffice." But this reasoning confuses the *birthday problem* (Event B) with the *your birthday problem* (Event A).

For Event B (any shared birthday), only 23 people are needed for >50% probability. That's much smaller because there are $\binom{23}{2} = 253$ pairs to check — and any matching pair counts.

For Event A, there are only $n$ people to check (each against your fixed birthday), giving far fewer opportunities for a match. Hence you need many more people.

---

### Part (e): Simulation for Event B

Using 10,000 simulated trials, find the smallest $n$ such that $P(B) > 0.9$.

**Result from simulation:** $n = 41$

**On reliability:** With only 30 trials, the estimate of $P(B)$ fluctuates widely. Variance of the estimator $\hat{p}$ is $p(1-p)/30$. At $p \approx 0.9$, standard deviation $\approx \sqrt{0.09/30} \approx 0.055$. With 10,000 trials, standard deviation drops to $\approx 0.003$ — much more reliable.

---

### Part (f): Exact Formula for Event B

**Complement:** $B^c$ = all $n$ people have *distinct* birthdays.

Count outcomes where all birthdays are distinct: this is an ordered selection of $n$ distinct days from 365, i.e., ${}_{365}P_{n}$:

$$P(B^c) = \frac{365 \times 364 \times \cdots \times (365-n+1)}{365^n} = \frac{365!}{(365-n)!\cdot 365^n}$$

$$\boxed{P(B) = 1 - \frac{365!}{(365-n)!\cdot 365^n} = 1 - \prod_{k=0}^{n-1}\left(1 - \frac{k}{365}\right)}$$

**Verification of the famous threshold:** At $n = 23$:

$$P(B) = 1 - \frac{365 \times 364 \times \cdots \times 343}{365^{23}} \approx 0.5073 > 0.5 \checkmark$$

**Why does $P(B)$ grow so fast?** With $n$ people, there are $\binom{n}{2}$ pairs, each with probability $1/365$ of matching. For $n=23$, that's $\binom{23}{2} = 253$ pairs — the same number as the threshold for Event A. This is not a coincidence: the "birthday threshold" approximately equals the threshold from part (c).

---

### Part (g): Simulation for Event C

Find the smallest $n$ such that $P(C) > 0.5$, where $C$ is the event that some birthday is shared by at least 3 people.

**Result from simulation:** $n = 87$ or $88$ people.

**Why so much larger than for Event B?**

| Event | Threshold $n$ for 50% | What's being matched |
|---|---|---|
| B (≥ 2 share) | 23 | Any pair |
| C (≥ 3 share) | 87–88 | Any triple |

Going from a pair to a triple requires much larger groups because having a pair is common once $n$ is moderate, but having three people land on the same specific day is far rarer. The probability of a given day having $\geq 3$ occupants out of $n$ follows a binomial distribution.

---

### Summary Table

| Event | Description | Threshold ($>50\%$) |
|---|---|---|
| $A$ | Someone shares **your specific** birthday | $n = 253$ |
| $B$ | At least two people share **any** birthday | $n = 23$ |
| $C$ | At least three people share **any** birthday | $n \approx 87$ |

> **The key insight:** These three problems look superficially similar but are structurally very different. Event B has $\binom{n}{2}$ pairs contributing, each independently likely to match — which is why 23 suffices. Event A has only $n$ single-person checks — which is why 253 is needed. Event C requires a triple, which is dramatically rarer than a pair.

---

*These problems illustrate a core probabilistic skill: recognizing when the complement is easier to compute, and understanding how counting structures (pairs, triples, vs. single comparisons) govern the rate at which probabilities grow.*
