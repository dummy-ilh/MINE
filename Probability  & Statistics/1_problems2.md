Here is your content rewritten cleanly in **well-structured Markdown**, with proper notation and formatting.

---

# 18.05 – Problem Set 1 Solutions

**Spring 2022**

---

# Problem 1 (20 pts.) — 6 Card Draw

We consider 6-card poker hands and compute probabilities of two types:

* **Two Pair**
* **Three of a Kind**

All hands are assumed equally likely.

Total number of 6-card hands:

$[
\binom{52}{6}
]$

---

## Two Pair

**Definition:**
Two cards have one rank, two cards have another rank, and the remaining two cards have two different ranks.

Example:
{2♥, 2♠, 5♥, 5♣, Q♦, K♦}

### Counting Method

We build the hand step-by-step.

1. Choose 2 ranks for the pairs:
   $[
   \binom{13}{2}
   ]$

2. Choose 2 cards from each selected rank:
   $[
   \binom{4}{2}\binom{4}{2}
   ]$

3. Choose 2 different ranks (from remaining 11) for the last two cards:
   $[
   \binom{11}{2}
   ]$

4. Choose 1 card from each of those ranks:
   $[
   \binom{4}{1}\binom{4}{1}
   ]$

### Total Two-Pair Hands

$[
\binom{13}{2}
\binom{4}{2}^2
\binom{11}{2}
\binom{4}{1}^2
==============

2,471,040
]$

### Probability

$[
P(\text{two pair}) =
\frac{2,471,040}{\binom{52}{6}}
\approx 0.1214
]$

---

## Three of a Kind

**Definition:**
Three cards share one rank; the remaining three cards have three distinct other ranks.

Example:
{2♥, 2♠, 2♣, 5♣, 9♠, K♥}

### Counting Method

1. Choose rank of the triple:
   $[
   \binom{13}{1}
   ]$

2. Choose 3 cards from that rank:
   $[
   \binom{4}{3}
   ]$

3. Choose 3 different ranks from remaining 12:
   $[
   \binom{12}{3}
   ]$

4. Choose 1 card from each:
   $[
   \binom{4}{1}^3
   ]$

### Total Three-of-a-Kind Hands

$[
\binom{13}{1}
\binom{4}{3}
\binom{12}{3}
\binom{4}{1}^3
==============

732,160
]$

### Probability

$[
P(\text{three of a kind}) =
\frac{732,160}{\binom{52}{6}}
\approx 0.03596
]$

---

### Conclusion

$[
P(\text{two pair}) \approx 0.1214
]$

$[
P(\text{three of a kind}) \approx 0.03596
]$

**Two pair is much more likely.**

---

# Problem 2 (20 pts.) — Non-Transitive Dice

Dice:

* **Blue:** 3, 3, 3, 3, 3, 6
* **Orange:** 1, 4, 4, 4, 4, 4
* **White:** 2, 2, 2, 5, 5, 5

---

## (a) Single Die Comparisons

### Results

$[
P(\text{White beats Orange}) = \frac{7}{12}
]$

$[
P(\text{Orange beats Blue}) = \frac{25}{36}
]$

$[
P(\text{Blue beats White}) = \frac{7}{12}
]$

### Interpretation

There is no linear ranking.

Instead:

Blue → beats → White
White → beats → Orange
Orange → beats → Blue

They form a **cycle**, hence "non-transitive".

---

## (b) Two Dice vs Two Dice

We roll:

* Two White dice
* Two Blue dice

We compare sums.

### Result

$[
P(\text{White sum > Blue sum}) =
\frac{85}{144}
]$

Even though a single Blue tends to beat a single White, two Whites outperform two Blues overall.

This shows how aggregation can reverse advantage.

---

# Problem 3 (55 pts.) — Birthday Problem

Assume:

* 365 equally likely birthdays
* Group size = ( n )
* Sample space consists of sequences:

$[
\omega = (b_1, b_2, \dots, b_n)
]$

Total outcomes:

$[
365^n
]$

Each outcome has probability:

$[
P(\omega) = \frac{1}{365^n}
]$

---

## (a) Probability Function

Uniform over all sequences:

$[
P(\omega) = \frac{1}{365^n}
]$

---

## (b) Events

Let your birthday be day ( b ).

### Event A

Someone shares your birthday.

$[
\exists k \in {1,\dots,n} \text{ such that } b_k = b
]$

---

### Event B

At least two people share a birthday.

$[
\exists j \neq k \text{ such that } b_j = b_k
]$

---

### Event C

At least three people share a birthday.

$[
\exists j,k,l \text{ distinct such that } b_j = b_k = b_l
]$

---

## (c) Probability of A

Easier to compute complement.

$[
P(A^c) =
\left(\frac{364}{365}\right)^n
]$

$[
P(A) =
1 -
\left(\frac{364}{365}\right)^n
]$

Solving ( P(A) > 0.5 ):

$[
n \approx 253
]$

So **253 people** are needed for >50% chance someone shares your birthday.

---

## (d) Why is 253 > 365/2?

Because people in the group will likely share birthdays with each other, reducing the number of distinct birthdays represented.

---

## (e) Simulation for Event B

Using 10,000 trials:

Smallest ( n ) such that:

$[
P(B) > 0.9
]$

Answer:

$[
n = 41
]$

With only 30 trials, estimates vary widely (high sampling variability).

---

## (f) Exact Formula for B

Compute complement (all birthdays distinct).

$[
P(B^c) =
\frac{365 \cdot 364 \cdots (365-n+1)}{365^n}
============================================

\frac{365!}{(365-n)! , 365^n}
]$

$[
P(B) =
1 -
\frac{365!}{(365-n)! , 365^n}
]$

---

## (g) Simulation for Event C

Using 10,000 trials.

Smallest ( n ) such that:

$[
P(C) > 0.5
]$

Answer:

$[
n = 87 \text{ or } 88
]$

With 88 people, it is clearly above 0.5.

---

# Final Summary

| Event                        | Threshold    |
| ---------------------------- | ------------ |
| Someone shares your birthday | 253 people   |
| Some two share birthday      | 41 people    |
| Some three share birthday    | 87–88 people |

---

If you'd like, I can also convert this into a LaTeX-ready version or add intuition diagrams for each counting argument.
