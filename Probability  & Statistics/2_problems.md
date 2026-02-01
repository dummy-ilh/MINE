Here is your content rewritten cleanly and clearly in **well-structured Markdown**.

---

# 18.05 – Class 2 Problems

**Spring 2022**

---

# Concept Questions

---

## Concept Question 1 — What’s the Event?

**Experiment:** Toss a coin 3 times.

Which of the following equals the event *“exactly two heads”*?

* **A** = {THH, HTH, HHT, HHH}
* **B** = {THH, HTH, HHT}
* **C** = {HTH, THH}

**Answer:** (2) **B**

The event “exactly two heads” contains **all and only** the outcomes with exactly two H’s:

$[
{THH, HTH, HHT}
]$

An event corresponds to a unique subset of the sample space.

---

## Concept Question 2 — Describe the Event

**Experiment:** Toss a coin 3 times.

Which description matches the event:

$[
{THH, HTH, HHT}
]$

Options:

1. Exactly one head
2. Exactly one tail
3. At most one tail
4. None of the above

**Answer:** (2) **Exactly one tail**

Notice:

* “Exactly two heads”
* “Exactly one tail”

describe the **same event**.

Different wording can represent the same subset of outcomes.

---

## Concept Question 3 — Are They Disjoint?

**Experiment:** Toss a coin 3 times.

Are the events:

* “Exactly 2 heads”
* “Exactly 2 tails”

disjoint?

**Answer:** True.

$[
{THH, HTH, HHT}
\cap
{TTH, THT, HTT}
===============

\emptyset
]$

No outcome can have both exactly two heads and exactly two tails simultaneously.

---

## Concept Question 4 — Does A Imply B?

Are the statements:

> “A implies B”

and

$[
A \subseteq B
]$

equivalent?

**Answer:** True.

If event A happens, then B must also happen.
This is exactly the definition of subset.

Example:

* A = “exactly 2 heads”
* B = “at least 2 heads”

$[
{THH, HTH, HHT}
\subset
{THH, HTH, HHT, HHH}
]$

---

# Board Problems

---

# Problem 1 — Poker Hands (One Pair)

Deck:

* 13 ranks: 2, 3, …, 10, J, Q, K, A
* 4 suits

A **one-pair hand**:

* Two cards of one rank
* Three other cards of distinct different ranks

Example:

{2♥, 2♠, 5♥, 8♣, K♦}

---

## (a) Counting One-Pair Hands

### Combination Approach

Treat hand as an unordered set.

### Step-by-step construction

1. Choose rank of the pair:

$[
\binom{13}{1}
]$

2. Choose 2 suits from that rank:

$[
\binom{4}{2}
]$

3. Choose 3 different ranks from remaining 12:

$[
\binom{12}{3}
]$

4. Choose 1 suit from each:

$[
4^3
]$

### Total one-pair hands

$[
\binom{13}{1}
\binom{4}{2}
\binom{12}{3}
4^3
===

1,098,240
]$

---

## (b) Probability of One Pair

Total 5-card hands:

$[
\binom{52}{5} = 2,598,960
]$

Probability:

$[
P(\text{one pair})
==================

\frac{1,098,240}{2,598,960}
\approx 0.42257
]$

---

### Permutation Approach (Order Matters)

Count ordered deals.

1. Choose 2 positions for the pair:

$[
\binom{5}{2}
]$

2. First card of pair: 52 choices
3. Second matching card: 3 choices
4. First different card: 48 choices
5. Next different card: 44 choices
6. Final different card: 40 choices

Total ordered one-pair hands:

$[
\binom{5}{2}
\cdot 52 \cdot 3 \cdot 48 \cdot 44 \cdot 40
===========================================

131,788,800
]$

Total ordered 5-card deals:

$[
52P5
====

# 52 \cdot 51 \cdot 50 \cdot 49 \cdot 48

311,875,200
]$

Probability:

$[
\frac{131,788,800}{311,875,200}
===============================

0.42257
]$

Both approaches agree.

---

# Problem 2 — Inclusion–Exclusion

Class of 50 students:

* 20 male (M)
* 25 brown-eyed (B)

We want:

$[
P(M \cup B)
]$

---

### Minimum

Occurs when all males are brown-eyed.

Then:

$[
|M \cup B| = 25
]$

$[
P = 25/50 = 0.5
]$

---

### Maximum

Occurs when no male is brown-eyed.

$[
|M \cup B| = 20 + 25 = 45
]$

$[
P = 45/50 = 0.9
]$

---

### Final Answer

$[
0.5 \le P(M \cup B) \le 0.9
]$

Using inclusion–exclusion:

$[
P(M \cup B)
===========

# P(M) + P(B) - P(M \cap B)

0.9 - P(M \cap B)
]$

---

# Problem 3 — D20 (9 Rolls)

Roll a 20-sided die 9 times.

---

## Sample Space

All sequences of 9 numbers from {1,…,20}.

$[
|S| = 20^9
]$

All outcomes equally likely.

---

## Event A

All 9 rolls are distinct.

Number of favorable outcomes:

$[
20P9
====

20 \cdot 19 \cdot 18 \cdots 12
]$

---

## Probability

$[
P(A)
====

\frac{20 \cdot 19 \cdot \dots \cdot 12}{20^9}
\approx 0.119
]$

---

# Problem 4 — Jon’s Dice

Dice:

* Blue: 3 (five times), 6 (once)
* White: 2 (three times), 5 (three times)
* Orange: 1 (once), 4 (five times)

---

## Pairwise Comparisons

$[
P(\text{Blue beats White}) = \frac{7}{12}
]$

$[
P(\text{White beats Orange}) = \frac{7}{12}
]$

$[
P(\text{Orange beats Blue}) = \frac{25}{36}
]$

---

### Conclusion

Blue > White
White > Orange
Orange > Blue

No best die.

The relation “better than” is **not transitive**.

---

# Problem 5 — Lucky Lucy

Lucy has a biased coin.

Let:

$[
P(H) = p, \quad P(T) = q = 1 - p
]$

Two tosses.

---

## Outcomes and Probabilities

| Outcome | Probability |
| ------- | ----------- |
| HH      | ( p^2 )     |
| TT      | ( q^2 )     |
| HT      | ( pq )      |
| TH      | ( pq )      |

---

## Define Events

A = same result = {HH, TT}

$[
P(A) = p^2 + q^2
]$

B = different results = {HT, TH}

$[
P(B) = 2pq
]$

---

## Compare A and B

Since the coin is unfair:

$[
p \ne q
]$

Consider:

$[
(p - q)^2 > 0
]$

Expanding:

$[
p^2 - 2pq + q^2 > 0
]$

$[
p^2 + q^2 > 2pq
]$

Therefore:

$[
P(A) > P(B)
]$

---

### Final Insight

For a biased coin:

The probability of getting **the same result** is greater than getting different results.

Lucy is more likely to get matching tosses.

---

If you'd like, I can next compress all of Class 2 into a one-page formula sheet for rapid revision.
