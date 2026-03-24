# 2 — Probability: Terminology, Rules & Examples

## Table of Contents

1. [The Cast List — What Every Probability Problem Is Made Of](#1-the-cast-list)
2. [Sample Spaces — Building Them Right](#2-sample-spaces)
3. [Events — Subsets That Matter](#3-events)
4. [The Probability Function — The Rules It Must Follow](#4-the-probability-function)
5. [Three Core Rules of Probability](#5-three-core-rules)
6. [Counting — The Rule of Product](#6-counting--the-rule-of-product)
7. [Permutations and Combinations](#7-permutations-and-combinations)
8. [Worked Problems (Full Solutions)](#8-worked-problems)
9. [Non-Transitive Dice — When Intuition Breaks](#9-non-transitive-dice)
10. [Lucky Lucy — Algebra Beats Intuition](#10-lucky-lucy)
11. [Quick Reference & Cheat Sheet](#11-quick-reference)

---

## 1. The Cast List

Every probability problem is a story with the same four characters. Identify them first, and the rest follows.

| Character | What it means | Symbol |
|---|---|---|
| **Experiment** | A repeatable procedure with well-defined outcomes | — |
| **Sample space** | The set of *all* possible outcomes | Ω or S |
| **Event** | Any subset of the sample space | A, B, E, … |
| **Probability function** | Assigns a number to each outcome | P(·) |

**Why this matters:** Most probability errors come from a sloppy sample space. Define it clearly first — everything else is just arithmetic on top of it.

---

## 2. Sample Spaces

### 2.1 Finite Sample Spaces

**Example 1 — One fair coin toss:**

- Experiment: toss a coin, report the result.
- Sample space: Ω = {H, T}
- Probability function: P(H) = 0.5, P(T) = 0.5

---

**Example 2 — Three fair coin tosses:**

- Experiment: toss three times, list all results in order.
- Sample space: Ω = {HHH, HHT, HTH, HTT, THH, THT, TTH, TTT}
- Probability function: each outcome has probability 1/8

| HHH | HHT | HTH | HTT | THH | THT | TTH | TTT |
|-----|-----|-----|-----|-----|-----|-----|-----|
| 1/8 | 1/8 | 1/8 | 1/8 | 1/8 | 1/8 | 1/8 | 1/8 |

Notice: there are 2³ = 8 outcomes. For n fair coin tosses, there are always 2ⁿ outcomes.

---

### 2.2 Infinite Discrete Sample Spaces

**Example 3 — Taxis passing 77 Mass Ave:**

- Experiment: count taxis during an 18.05 class.
- Sample space: Ω = {0, 1, 2, 3, …} — infinitely many outcomes, but all *listable*.
- Probability function: Poisson distribution.

$$P(k) = e^{-\lambda} \frac{\lambda^k}{k!}$$

where λ is the average number of taxis.

| k | 0 | 1 | 2 | 3 | … |
|---|---|---|---|---|---|
| P(k) | e^{-λ} | e^{-λ}·λ | e^{-λ}·λ²/2 | e^{-λ}·λ³/6 | … |

The sum of all probabilities = 1 (it equals the Taylor series for e^λ divided by e^λ).

---

**Example 4 — Flip until heads:**

- Experiment: flip a coin with P(H) = p. Count total flips until the first head.
- Sample space: Ω = {1, 2, 3, …}
- Probability function: P(n) = (1−p)^(n−1) · p

Why? To get the first head on flip n, you need (n−1) tails first, then one head. Each tail has probability (1−p), each head has probability p, and flips are independent.

| n | 1 | 2 | 3 | 4 | … |
|---|---|---|---|---|---|
| P(n) | p | (1−p)p | (1−p)²p | (1−p)³p | … |

This is the **Geometric distribution** and comes up constantly. Its expected value is E[n] = 1/p. For a fair coin (p = 0.5), you expect 2 flips on average.

> **Stopping problems.** This toy example is a clean version of a huge class of real problems called *stopping rule problems* — rules that tell you when to stop a process. Medical treatment protocols, A/B test stopping rules, quality control — all of these ask "when do we stop?" with some probabilistic criterion.

---

### 2.3 Continuous Sample Spaces

**Example 5 — Mass of a proton:**

- Sample space: Ω = [0, ∞) — every positive real number is a possible outcome.
- You *cannot* assign a probability to each individual outcome (there are uncountably many).
- Instead, you use a **probability density function** — covered later in the course.

---

### 2.4 Choosing Your Sample Space Wisely

There's often more than one valid sample space for the same experiment. The right choice depends on what question you're answering.

**Example: Two dice**

*Option 1:* Record the ordered pair (die 1, die 2).
- Sample space: {(1,1), (1,2), …, (6,6)} — 36 outcomes, all equally likely.

|       | **1** | **2** | **3** | **4** | **5** | **6** |
|-------|-------|-------|-------|-------|-------|-------|
| **1** | 1/36  | 1/36  | 1/36  | 1/36  | 1/36  | 1/36  |
| **2** | 1/36  | 1/36  | 1/36  | 1/36  | 1/36  | 1/36  |
| **3** | 1/36  | 1/36  | 1/36  | 1/36  | 1/36  | 1/36  |
| **4** | 1/36  | 1/36  | 1/36  | 1/36  | 1/36  | 1/36  |
| **5** | 1/36  | 1/36  | 1/36  | 1/36  | 1/36  | 1/36  |
| **6** | 1/36  | 1/36  | 1/36  | 1/36  | 1/36  | 1/36  |

*Option 2:* Record only the sum.
- Sample space: {2, 3, 4, …, 12} — 11 outcomes, **not** equally likely.

| Sum | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|-----|---|---|---|---|---|---|---|---|----|----|----|
| P   | 1/36 | 2/36 | 3/36 | 4/36 | 5/36 | 6/36 | 5/36 | 4/36 | 3/36 | 2/36 | 1/36 |

**The key insight:** Option 1 is richer — you can always recover the sum from the pair, but not vice versa. When in doubt, use the more detailed sample space.

> **Habit to build:** Before doing any calculation, write down: (1) the experiment, (2) the sample space, (3) the probability function. This catches most mistakes before they happen.

---

## 3. Events

An **event** is any subset of the sample space. It corresponds to the ordinary meaning: "the event that exactly 2 heads appeared" is the *set of all outcomes that contain exactly 2 heads*.

**Example — Three coin tosses:**

The event "exactly 2 heads" = {THH, HTH, HHT}

Notice: the same event can be described in multiple ways:
- "exactly 2 heads" = "exactly 1 tail"

Both descriptions carve out the same subset. They are the *same event*.

### Event Notation Reference

| Notation | Plain English | Example (Ω = coin toss 3x) |
|---|---|---|
| A ∪ B | A **or** B (at least one) | "2 heads or 2 tails" |
| A ∩ B | A **and** B (both) | "2 heads and first flip is H" |
| Aᶜ | **not** A (complement) | "not exactly 2 heads" |
| A ⊆ B | A **implies** B | "exactly 2 heads" ⊆ "at least 1 head" |
| A ∩ B = ∅ | A and B are **disjoint** (mutually exclusive) | "exactly 2 heads" and "exactly 2 tails" |

### "A implies B" = A ⊆ B

If every outcome in A is also in B, then whenever A happens, B must have happened too. That's what implication means — and it's exactly the subset relationship.

**Example:** "Exactly 2 heads" ⊆ "At least 2 heads":
- {THH, HTH, HHT} ⊂ {THH, HTH, HHT, HHH} ✓

### Disjoint vs. Independent — The Most Common Confusion

These sound related. They are not. They are almost opposites.

| | What it means | Formula |
|---|---|---|
| **Disjoint (mutually exclusive)** | A and B cannot both happen | P(A ∩ B) = 0 |
| **Independent** | Knowing A happened tells you nothing about B | P(A ∩ B) = P(A)·P(B) |

If A and B are disjoint and both have positive probability, they are *dependent* — knowing A occurred tells you B definitely did not. That's the definition of dependence.

The only way disjoint events can be independent is if at least one of them has probability zero.

---

## 4. The Probability Function

### The Two Rules It Must Satisfy

For a discrete sample space S = {ω₁, ω₂, …, ωₙ}, a probability function P must satisfy:

**Rule 1:** 0 ≤ P(ω) ≤ 1 for every outcome ω

**Rule 2:** The probabilities of all outcomes sum to 1:
$$\sum_{j=1}^{n} P(\omega_j) = 1$$

### Probability of an Event

The probability of event E is the sum of the probabilities of all outcomes inside E:

$$P(E) = \sum_{\omega \in E} P(\omega)$$

For equally likely outcomes (probability 1/|Ω| each):

$$P(E) = \frac{|E|}{|\Omega|} = \frac{\text{number of outcomes in } E}{\text{total number of outcomes}}$$

This is the formula you use for fair coins, fair dice, well-shuffled decks.

---

## 5. Three Core Rules of Probability

These three rules are all you need to solve most probability problems.

---

### Rule 1 — Complement Rule

$$P(A^c) = 1 - P(A)$$

**When to use it:** Any time you see "at least one," "not all," or "doesn't happen." These are almost always easier to compute via their complement.

**Example (from class):**
- 10 servers, each fails independently with P = 0.01.
- P(at least one fails) = 1 − P(none fail) = 1 − (0.99)¹⁰ ≈ 0.096

Computing "at least one fails" directly requires summing over all possible failure combinations. Computing "none fail" is one multiplication. Always take the complement path when it's available.

---

### Rule 2 — Addition Rule for Disjoint Events

If A and B are **disjoint** (cannot both occur):

$$P(A \cup B) = P(A) + P(B)$$

This extends to any number of disjoint events:

$$P(A_1 \cup A_2 \cup \cdots \cup A_n) = P(A_1) + P(A_2) + \cdots + P(A_n)$$

**Example:**
- P(X is a multiple of 2) = 0.6
- P(X is odd and less than 10) = 0.25
- These are disjoint (even vs odd numbers can't overlap)
- P(either happens) = 0.6 + 0.25 = 0.85

---

### Rule 3 — Inclusion-Exclusion Principle

If A and B **can** overlap:

$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

**Why subtract?** When you add P(A) + P(B), the overlap A ∩ B gets counted twice. Subtracting it once fixes the double-count.

**Three-event version:**
$$P(A \cup B \cup C) = P(A) + P(B) + P(C) - P(A \cap B) - P(A \cap C) - P(B \cap C) + P(A \cap B \cap C)$$

The alternating pattern (add singles, subtract pairs, add triple) continues for more sets.

---

### The Range Problem — Class Example

**Setup:** A class has 50 students: 20 male (M), 25 brown-eyed (B). What is the range of P(M ∪ B)?

Using inclusion-exclusion:
$$P(M \cup B) = P(M) + P(B) - P(M \cap B) = 0.4 + 0.5 - P(M \cap B) = 0.9 - P(M \cap B)$$

Now, what are the limits on P(M ∩ B)?

- **Maximum overlap:** all 20 males are brown-eyed → P(M ∩ B) = 0.4 → P(M ∪ B) = 0.5
- **Minimum overlap:** no male is brown-eyed → P(M ∩ B) = 0 → P(M ∪ B) = 0.9

**Answer: 0.5 ≤ P(M ∪ B) ≤ 0.9**

> **The thinking trick here:** You didn't need to know the actual overlap. You found the bounds by asking "what's the most/least overlap that's physically possible?" This kind of bounding argument appears everywhere in probability.

---

## 6. Counting — The Rule of Product

Before you can compute P(E) = |E|/|Ω|, you need to be able to count |E| and |Ω| correctly.

### The Rule of Product

If a procedure has k steps, and:
- Step 1 can be done in n₁ ways
- Step 2 can be done in n₂ ways (regardless of how Step 1 went)
- …
- Step k can be done in nₖ ways

Then the total number of ways to complete the procedure is:

$$n_1 \times n_2 \times \cdots \times n_k$$

**The key condition:** steps must be *independent* — the number of choices at each step can't depend on previous steps. If it does, you need to be more careful.

---

### Counting Carefully — A D20 Example

**Problem:** Roll a 20-sided die 9 times. What is the probability all 9 rolls are distinct?

**Sample space:** all sequences of 9 numbers, each between 1 and 20.
- Total outcomes: 20 × 20 × … × 20 (9 times) = 20⁹

**Event A:** all 9 rolls are distinct.
- First roll: 20 choices
- Second roll: 19 choices (can't repeat the first)
- Third roll: 18 choices
- …
- Ninth roll: 12 choices

|A| = 20 × 19 × 18 × 17 × 16 × 15 × 14 × 13 × 12 = ₂₀P₉

$$P(A) = \frac{20 \times 19 \times 18 \times 17 \times 16 \times 15 \times 14 \times 13 \times 12}{20^9} \approx 0.119$$

So there's only about a 12% chance all 9 rolls are distinct. With 9 rolls on a 20-sided die, repeats are actually the norm.

---

## 7. Permutations and Combinations

### Permutations — Order Matters, No Repeats

The number of ways to arrange r items chosen from n distinct items, where **order matters** and **no repetition** is allowed:

$$_nP_r = \frac{n!}{(n-r)!} = n \times (n-1) \times \cdots \times (n-r+1)$$

**Example:** ₅P₃ = 5 × 4 × 3 = 60 (choosing 3 items from 5, order matters)

---

### Combinations — Order Doesn't Matter, No Repeats

The number of ways to choose r items from n distinct items, where **order doesn't matter**:

$$\binom{n}{r} = \frac{n!}{r!(n-r)!}$$

The relationship: C(n,r) = P(n,r) / r! — you divide out the r! orderings you're no longer distinguishing.

**Example:** C(5,3) = 60/6 = 10

---

### When to Use Which

| Situation | Formula | Example |
|---|---|---|
| Order matters, no repeats | _nP_r | Arranging 3 books on a shelf from 10 |
| Order doesn't matter, no repeats | C(n,r) | Choosing a committee of 3 from 10 |
| Order matters, repeats allowed | nʳ | 4-digit PIN code |
| Order doesn't matter, repeats allowed | C(n+r−1, r) | Stars & bars |

---

### Multinomial Coefficients — Repeated Elements

If you have n total items, with n₁ identical items of type 1, n₂ of type 2, …, nₖ of type k (where n₁ + n₂ + … + nₖ = n):

$$\text{Number of distinct arrangements} = \frac{n!}{n_1! \cdot n_2! \cdots n_k!}$$

**Example — MISSISSIPPI:**
- Letters: M×1, I×4, S×4, P×2. Total = 11.
- Arrangements = 11! / (1! · 4! · 4! · 2!) = 39,916,800 / 1,152 = **34,650**

---

## 8. Worked Problems

### Problem 1 — Poker Hands (One Pair)

**Setup:** A standard 52-card deck. A poker hand is 5 cards. A *one-pair* hand has exactly two cards of one rank, and three cards of three other distinct ranks.

**Example:** {2♡, 2♠, 5♡, 8♣, K♢} — pair of 2s, with 5, 8, K as kickers.

---

#### Part (a): How many one-pair hands exist?

We build a systematic algorithm that produces every possible hand exactly once. The trick is to break it into small, clean steps.

**Combinations approach** (treating a hand as an unordered set of 5 cards):

| Action | What you're choosing | Count |
|---|---|---|
| 1. Choose the rank for the pair | 1 rank from 13 | C(13,1) = 13 |
| 2. Choose 2 suits for the pair | 2 suits from 4 | C(4,2) = 6 |
| 3. Choose 3 ranks for the kickers | 3 ranks from remaining 12 | C(12,3) = 220 |
| 4. Choose 1 suit for each kicker | 1 suit from 4, three times | 4³ = 64 |

By the rule of product:

$$13 \times 6 \times 220 \times 64 = 1{,}098{,}240$$

**Why are the kicker ranks chosen from 12, not 13?** The pair's rank is already taken — if a kicker matched the pair's rank, we'd have three of a kind, not one pair.

**Why choose 3 kicker ranks together (not one at a time)?** If you chose them sequentially — "first kicker from 12 ranks, second from 11, third from 10" — you'd get ordered selections and need to divide by 3! at the end to avoid counting the same hand multiple times. Choosing with C(12,3) handles this cleanly.

---

#### Permutations approach (keeping track of deal order):

| Action | Count |
|---|---|
| Choose which 2 positions hold the pair | C(5,2) = 10 |
| First card of the pair | 52 |
| Second card of the pair (same rank, different card) | 3 |
| First open slot (can't match the pair's rank) | 48 |
| Second open slot (can't match pair or previous card) | 44 |
| Third open slot | 40 |

$$10 \times 52 \times 3 \times 48 \times 44 \times 40 = 131{,}788{,}800$$

This counts ordered hands (where deal order matters).

---

#### Part (b): Probability of a one-pair hand

Total 5-card hands (order doesn't matter): C(52,5) = 2,598,960

$$P(\text{one pair}) = \frac{1{,}098{,}240}{2{,}598{,}960} \approx 0.4226$$

**Sanity check with permutations:**
Total ordered 5-card hands = ₅₂P₅ = 311,875,200

$$P(\text{one pair}) = \frac{131{,}788{,}800}{311{,}875{,}200} \approx 0.4226 \checkmark$$

Both approaches give the same answer. They must — the probability is a physical fact about the deck, not about how you count.

> About 42% of all 5-card poker hands are one-pair hands. This makes them the most common hand type that pays out.

---

### Problem 2 — Inclusion-Exclusion on a Class

**Setup:** A class has 50 students: 20 male (M), 25 brown-eyed (B).

$$P(M) = 20/50 = 0.4, \quad P(B) = 25/50 = 0.5$$

**Question:** What is the range of possible values for P(M ∪ B)?

By inclusion-exclusion:

$$P(M \cup B) = 0.4 + 0.5 - P(M \cap B) = 0.9 - P(M \cap B)$$

We need the range of P(M ∩ B):

- **Minimum:** if M ⊆ B (all males have brown eyes), then P(M ∩ B) = P(M) = 0.4, so P(M ∪ B) = 0.5
- **Maximum:** if M and B are disjoint, P(M ∩ B) = 0, so P(M ∪ B) = 0.9

**Answer: 0.5 ≤ P(M ∪ B) ≤ 0.9**

---

### Problem 3 — D20 (All Distinct Rolls)

**Setup:** Roll a D20 nine times. What is the probability all 9 results are distinct?

**Sample space:** ordered sequences of 9 numbers from {1, …, 20}.

$$|\Omega| = 20^9$$

**Event A:** all 9 numbers are distinct.

$$|A| = 20 \times 19 \times 18 \times 17 \times 16 \times 15 \times 14 \times 13 \times 12 = {_{20}}P_9$$

$$P(A) = \frac{{_{20}}P_9}{20^9} = \frac{20 \times 19 \times \cdots \times 12}{20^9} \approx 0.119$$

Only a ~12% chance. **This is the birthday problem in disguise** — with 9 "birthdays" drawn from a pool of 20, getting no repeats is surprisingly unlikely.

> **The birthday problem intuition:** With n people and a pool of N possibilities, the probability of no repeats drops quickly as n grows. With n = 23 people and N = 365 days, P(at least one shared birthday) ≈ 50.7% — which feels impossible until you count the C(23,2) = 253 pairs.

---

## 9. Non-Transitive Dice

This is one of the most counterintuitive results in elementary probability. Read it carefully.

### The Setup

Jon has three dice with unusual face values:

| Die | Values | Probabilities |
|---|---|---|
| **Blue** | 3 (×5), 6 (×1) | P(3) = 5/6, P(6) = 1/6 |
| **White** | 2 (×3), 5 (×3) | P(2) = 3/6, P(5) = 3/6 |
| **Orange** | 1 (×1), 4 (×5) | P(1) = 1/6, P(4) = 5/6 |

Two players each pick a die and roll once. Highest number wins. **Which die would you pick?**

---

### Blue vs. White

Build the product sample space — all (Blue, White) pairs and their probabilities:

|  | White = 2 (prob 3/6) | White = 5 (prob 3/6) |
|---|---|---|
| **Blue = 3** (prob 5/6) | 15/36 → **Blue wins** | 15/36 → White wins |
| **Blue = 6** (prob 1/6) | 3/36 → **Blue wins** | 3/36 → **Blue wins** |

P(Blue beats White) = (15 + 3 + 3)/36 = **21/36 = 7/12 ≈ 58.3%**

Blue beats White.

---

### White vs. Orange

|  | Orange = 1 (prob 1/6) | Orange = 4 (prob 5/6) |
|---|---|---|
| **White = 2** (prob 3/6) | 3/36 → **White wins** | 15/36 → Orange wins |
| **White = 5** (prob 3/6) | 3/36 → **White wins** | 15/36 → **White wins** |

P(White beats Orange) = (3 + 3 + 15)/36 = **21/36 = 7/12 ≈ 58.3%**

White beats Orange.

---

### Orange vs. Blue

|  | Blue = 3 (prob 5/6) | Blue = 6 (prob 1/6) |
|---|---|---|
| **Orange = 1** (prob 1/6) | 5/36 → Blue wins | 1/36 → Blue wins |
| **Orange = 4** (prob 5/6) | 25/36 → **Orange wins** | 5/36 → Blue wins |

P(Orange beats Blue) = 25/36 ≈ 69.4%

Orange beats Blue.

---

### The Paradox

$$\text{Blue beats White} \quad (7/12)$$
$$\text{White beats Orange} \quad (7/12)$$
$$\text{Orange beats Blue} \quad (25/36)$$

**There is no best die.** The "beats" relationship cycles: Blue → White → Orange → Blue.

This is called **non-transitivity**. In everyday life, "better than" is transitive: if A is taller than B and B is taller than C, then A is taller than C. But probabilistic "beats" is not. The pairwise comparison depends on the specific distribution of face values, and different matchups activate different regions of the joint probability table.

**Practical consequence:** If your opponent picks first, you can always pick a die that beats theirs with probability 7/12. The second mover has the advantage — completely unlike games where all options are equivalent.

> This is why questions like "Die A beats Die B 60% of the time, Die B beats Die C 60% of the time — what's P(A beats C)?" **cannot be answered** without more information. The correct answer is "it depends on the specific distributions." Don't assume transitivity.

---

## 10. Lucky Lucy

**Setup:** Lucy has a *biased* coin with P(H) = p, where p ≠ 1/2.

- Event A: both tosses the same {HH, TT}
- Event B: both tosses different {HT, TH}

**Question:** Which is more likely?

---

### Setting Up the Probability Table

Let q = 1 − p (so p + q = 1).

| Outcome | HH | TT | HT | TH |
|---|---|---|---|---|
| Probability | p² | q² | pq | pq |

$$P(A) = p^2 + q^2$$
$$P(B) = 2pq$$

---

### The Comparison

Subtract:

$$P(A) - P(B) = p^2 + q^2 - 2pq = (p - q)^2$$

Since the coin is *unfair*, p ≠ q, so (p − q)² > 0.

**Therefore P(A) > P(B) always** — for any unfair coin, both tosses being the same is strictly more likely than them being different.

**Intuition:** If the coin strongly favors heads (say p = 0.9), then HH has probability 0.81, swamping the difference outcomes. The "same" outcomes dominate. The algebraic identity just confirms this holds for *any* p ≠ 0.5.

**When p = 0.5:** (p − q)² = 0, so P(A) = P(B) = 0.5. A fair coin makes "same" and "different" equally likely. Makes sense — all four outcomes {HH, HT, TH, TT} have equal probability 1/4.

---

## 11. Quick Reference

### Sample Space Types

| Type | Description | Example |
|---|---|---|
| Finite | Finitely many outcomes | Fair coin: {H, T} |
| Infinite discrete | Listable but infinite | Natural numbers {0,1,2,…} |
| Continuous | Uncountably infinite | Measuring a length |

---

### Three Rules of Probability

| Rule | Formula | When to use |
|---|---|---|
| Complement | P(Aᶜ) = 1 − P(A) | "at least one," "not all" |
| Addition (disjoint) | P(A ∪ B) = P(A) + P(B) | Events can't both happen |
| Inclusion-exclusion | P(A ∪ B) = P(A) + P(B) − P(A ∩ B) | Events can overlap |

---

### Counting Formulas

| What you want | Formula | Notes |
|---|---|---|
| Ordered, no repeats | n!/(n−r)! | Permutation |
| Unordered, no repeats | n!/(r!(n−r)!) | Combination |
| Ordered, with repeats | nʳ | PINs, sequences |
| Arrangements of n with repeats | n!/(n₁!·n₂!·…) | MISSISSIPPI-type |
| Distribute n items into k bins (each ≥1) | C(n−1, k−1) | Stars & bars with minimum |

---

### Canonical Probability Values to Remember

| Problem | Answer |
|---|---|
| P(one pair in 5-card poker hand) | ≈ 0.4226 |
| P(all 9 D20 rolls distinct) | ≈ 0.119 |
| P(at least 1 of 10 servers fails) | ≈ 0.0956 |
| P(at least 2 share a birthday, 23 people) | ≈ 0.5073 |
| E[flips until first head, fair coin] | 2 |
| E[fixed points in random permutation] | 1 (always!) |
| P(Blue beats White), Jon's dice | 7/12 |

---

### Mindset Notes

- **Complement first.** Any "at least one" or "not all" problem is almost certainly easier via 1 − P(complement).
- **Define the sample space explicitly.** Most errors come from an implicit, sloppy sample space.
- **Probability is not transitive.** A beating B and B beating C tells you nothing about A vs C without knowing the distributions.
- **Disjoint ≠ independent.** Disjoint events with positive probability are actually *more* dependent than independent events.
- **P(A) + P(B) − P(A ∩ B) = P(A ∪ B).** This is just bookkeeping — don't double-count the overlap.
- **The right counting method depends on whether order matters.** Always ask this explicitly before picking a formula.

---
