# MIT 18.05 — Introduction to Probability and Statistics
## Class 3: Conditional Probability, Independence, and Bayes' Theorem
### Complete Study Notes | Spring 2022

> **Source:** MIT OpenCourseWare 18.05, Jeremy Orloff and Jonathan Bloom  
> **Coverage:** Prep notes + In-class problems with full solutions and expanded explanations

---

## Table of Contents

1. [Learning Goals](#1-learning-goals)
2. [Conditional Probability](#2-conditional-probability)
3. [The Multiplication Rule](#3-the-multiplication-rule)
4. [Law of Total Probability](#4-law-of-total-probability)
5. [Using Trees to Organize Computations](#5-using-trees-to-organize-computations)
6. [Independence](#6-independence)
7. [Bayes' Theorem](#7-bayes-theorem)
8. [The Base Rate Fallacy](#8-the-base-rate-fallacy)
9. [In-Class Concept Questions — Full Solutions](#9-in-class-concept-questions--full-solutions)
10. [In-Class Board Problems — Full Solutions](#10-in-class-board-problems--full-solutions)
11. [Common Mistakes Reference](#11-common-mistakes-reference)
12. [Quick Summary & Formula Sheet](#12-quick-summary--formula-sheet)

---

## 1. Learning Goals

By the end of this class you should be able to:

1. **Define** conditional probability and independence of events precisely.
2. **Compute** conditional probability directly from the definition.
3. **Apply** the multiplication rule to find the probability of an intersection.
4. **Use** the law of total probability to decompose complex probability calculations.
5. **Check** whether two events are independent.
6. **Apply** Bayes' formula to invert conditional probabilities.
7. **Organize** conditional probability computations using probability trees and tables.
8. **Understand** the base rate fallacy and why it matters in data science and medicine.

---

## 2. Conditional Probability

### 2.1 Concept Overview

Conditional probability answers the question:

> *"How does the probability of an event change when we have extra information?"*

In everyday reasoning, new information restricts what we consider possible. Conditional probability formalizes this by **shrinking the sample space** to only those outcomes consistent with the new information.

---

### 2.2 Intuition

Imagine flipping a fair coin 3 times. You know the sample space is:

$$\Omega = \{HHH, HHT, HTH, HTT, THH, THT, TTH, TTT\}$$

**Without extra information:** $P(\text{3 heads}) = 1/8$ (1 favourable out of 8 equally likely outcomes).

**With extra information** — suppose someone tells you the first toss was Heads. The set of possible outcomes shrinks to:

$$\Omega' = \{HHH, HHT, HTH, HTT\}$$

Now there is only 1 outcome with 3 heads out of 4 remaining: $P(\text{3 heads} \mid \text{first is H}) = 1/4$.

> **Key Intuition:** Conditioning on $B$ means we *restrict our universe* to the event $B$ and ask what fraction of $B$ is covered by $A$.

---

### 2.3 Formal Definition

> **Definition (Conditional Probability):** For events $A$ and $B$ with $P(B) \neq 0$, the **conditional probability of $A$ given $B$** is:
>
> $$\boxed{P(A \mid B) = \frac{P(A \cap B)}{P(B)}}$$

**Reading:** "$P(A \mid B)$" is read as "the probability of $A$ given $B$" or "the probability of $A$ conditioned on $B$."

**Geometric interpretation:** Think of $P(A)$ as the proportion of the entire sample space area taken up by event $A$. Then $P(A \mid B)$ is the proportion of the area of $B$ that is covered by $A$ — i.e., $P(A \cap B) / P(B)$.

---

### 2.4 Key Formulas

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) \neq 0$$

$$P(B \mid A) = \frac{P(A \cap B)}{P(A)}, \quad P(A) \neq 0$$

> **Warning:** $P(A \mid B) \neq P(B \mid A)$ in general. These are two fundamentally different quantities. Confusing them is one of the most common and consequential errors in probability and statistics.

---

### 2.5 Worked Examples

---

#### Example 1 — Coin Tossing (Basic Conditional Probability)

**Problem:** Toss a fair coin 3 times.  
(a) What is $P(\text{3 heads})$?  
(b) Given that the first toss is heads, what is $P(\text{3 heads} \mid \text{first toss is heads})$?

**Step 1: Identify the sample space.**

$$\Omega = \{HHH, HHT, HTH, HTT, THH, THT, TTH, TTT\}, \quad |\Omega| = 8$$

All outcomes are equally likely since the coin is fair.

**Step 2: Define events.**

$$A = \{\text{3 heads}\} = \{HHH\}, \quad |A| = 1$$
$$B = \{\text{first toss is heads}\} = \{HHH, HHT, HTH, HTT\}, \quad |B| = 4$$

**Step 3: Answer part (a).**

$$P(A) = \frac{|A|}{|\Omega|} = \frac{1}{8}$$

**Step 4: Answer part (b) — find $P(A \mid B)$.**

First find $A \cap B$: outcomes that are in both $A$ and $B$.

$$A \cap B = \{HHH\}, \quad |A \cap B| = 1$$

Apply the definition:

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)} = \frac{1/8}{4/8} = \frac{1/8}{1/2} = \frac{1}{4}$$

**Final Answer:** $P(\text{3 heads} \mid \text{first is heads}) = 1/4$

**Interpretation:** Knowing the first toss was heads doubled our probability of getting 3 heads (from $1/8$ to $1/4$). This makes intuitive sense — we already have one head guaranteed, so now we just need 2 more heads from 2 remaining tosses.

---

#### Example 2 — Drawing Cards (Conditional Probability by Counting)

**Problem:** Draw two cards from a standard deck (52 cards, without replacement).  
Define: $S_1$ = "first card is a spade", $S_2$ = "second card is a spade".  
Find $P(S_2 \mid S_1)$.

**Step 1: Find $P(S_1)$.**

There are 52 equally probable ways to draw the first card. 13 of them are spades.

$$P(S_1) = \frac{13}{52} = \frac{1}{4}$$

**Step 2: Find $P(S_1 \cap S_2)$.**

Count ordered pairs (first card, second card) where both are spades:
- Ways to pick a spade first: 13
- Ways to pick a spade second (given one spade already removed): 12
- Total ordered pairs of any two cards: $52 \times 51$

$$P(S_1 \cap S_2) = \frac{13 \times 12}{52 \times 51} = \frac{156}{2652} = \frac{3}{51}$$

**Step 3: Apply the conditional probability formula.**

$$P(S_2 \mid S_1) = \frac{P(S_1 \cap S_2)}{P(S_1)} = \frac{3/51}{1/4} = \frac{3}{51} \times 4 = \frac{12}{51}$$

**Sanity check (direct reasoning):** If the first card was a spade, 51 cards remain, of which 12 are spades. So $P(S_2 \mid S_1) = 12/51$. ✓

**Final Answer:** $P(S_2 \mid S_1) = \dfrac{12}{51} \approx 0.235$

**Aside — Why is $P(S_2) = 1/4$?**  
Even though the first card affects the second, if we don't condition on the first card, by symmetry every card in the deck is equally likely to be the second card drawn. So $P(S_2) = 13/52 = 1/4$. This is a subtle but important point: unconditionally, the position of the draw doesn't matter.

**Think:** What is $P(S_2 \mid S_1^c)$? (When the first card is NOT a spade.)  
*Answer:* If $S_1^c$ occurred, 51 cards remain and 13 of them are spades, so $P(S_2 \mid S_1^c) = 13/51$.

---

## 3. The Multiplication Rule

### 3.1 Concept Overview

The **multiplication rule** is simply a rearrangement of the conditional probability definition. It lets us compute the probability of a joint event $A \cap B$ by breaking it into a sequence of steps.

---

### 3.2 Formal Definition

> **Multiplication Rule:**
>
> $$\boxed{P(A \cap B) = P(A \mid B) \cdot P(B)}$$
>
> Equivalently:
>
> $$P(A \cap B) = P(B \mid A) \cdot P(A)$$

**Why it matters:** When computing $P(A \cap B)$ directly is hard, we can instead compute $P(B)$ (marginal probability of the first event) and $P(A \mid B)$ (conditional probability of the second given the first). This is exactly how probability trees work.

---

### 3.3 Extension to Three Events

$$P(A \cap B \cap C) = P(A) \cdot P(B \mid A) \cdot P(C \mid A \cap B)$$

This extends naturally to any number of events and is the mathematical foundation for reading probabilities along tree branches.

---

### 3.4 Worked Example

#### Example 3 — Verifying the Multiplication Rule

**Problem:** From the card example above, verify that $P(S_2 \mid S_1) \cdot P(S_1) = P(S_1 \cap S_2)$.

From Example 2: $P(S_2 \mid S_1) = 12/51$, $P(S_1) = 1/4$, $P(S_1 \cap S_2) = 3/51$.

$$P(S_2 \mid S_1) \cdot P(S_1) = \frac{12}{51} \cdot \frac{1}{4} = \frac{12}{204} = \frac{3}{51} = P(S_1 \cap S_2) \checkmark$$

---

## 4. Law of Total Probability

### 4.1 Concept Overview

The law of total probability lets us compute $P(A)$ when the sample space can be naturally divided into mutually exclusive, exhaustive cases. It is the engine that powers probability trees.

---

### 4.2 Intuition

Suppose you want to find the probability of event $A$, but $A$ can happen in several different "scenarios" $B_1, B_2, B_3$ (which together cover all possibilities and don't overlap). Then:

$$P(A) = \sum_i P(A \mid B_i) \cdot P(B_i)$$

This is a **weighted average** of conditional probabilities, where each weight is the probability of being in that scenario.

> **Key Intuition:** If you don't know which "branch" you're on, sum up the contributions from all branches weighted by how likely each branch is.

---

### 4.3 Formal Definition

> **Law of Total Probability:** Let $B_1, B_2, \ldots, B_n$ be a **partition** of the sample space $\Omega$ (i.e., they are mutually disjoint and $B_1 \cup B_2 \cup \cdots \cup B_n = \Omega$). Then for any event $A$:
>
> $$\boxed{P(A) = \sum_{i=1}^{n} P(A \mid B_i) \cdot P(B_i)}$$

**For two events ($B$ and $B^c$):**

$$P(A) = P(A \mid B) \cdot P(B) + P(A \mid B^c) \cdot P(B^c)$$

**For three events:**

$$P(A) = P(A \mid B_1) P(B_1) + P(A \mid B_2) P(B_2) + P(A \mid B_3) P(B_3)$$

---

### 4.4 Worked Examples

---

#### Example 4 — Red Balls from Urn (Simple Version)

**Problem:** An urn contains **5 red** and **2 green** balls. Two balls are drawn one after the other (without replacement). What is the probability the second ball is red?

**Step 1: Set up the partition.**

Let $R_1$ = "first ball is red", $G_1$ = "first ball is green". These partition $\Omega$ since either the first is red or green.

$$P(R_1) = \frac{5}{7}, \qquad P(G_1) = \frac{2}{7}$$

**Step 2: Find conditional probabilities for the second draw.**

- *If first was red* (5 red remain out of 6 total): $P(R_2 \mid R_1) = 4/6$
- *If first was green* (5 red remain out of 6 total): $P(R_2 \mid G_1) = 5/6$

**Step 3: Apply the law of total probability.**

$$P(R_2) = P(R_2 \mid R_1) \cdot P(R_1) + P(R_2 \mid G_1) \cdot P(G_1)$$

$$= \frac{4}{6} \cdot \frac{5}{7} + \frac{5}{6} \cdot \frac{2}{7} = \frac{20}{42} + \frac{10}{42} = \frac{30}{42} = \frac{5}{7}$$

**Final Answer:** $P(R_2) = 5/7$

**Interpretation:** Interestingly, $P(R_2) = P(R_1) = 5/7$. By symmetry, without any conditioning, each position in the draw sequence has the same probability for any given colour. The second draw is just as likely to be red as the first.

---

#### Example 5 — Urn with Ball Replacement Rule (More Complex)

**Problem:** Urn has **5 red** and **2 green** balls. A ball is drawn. If it's **green**, a red ball is added. If it's **red**, a green ball is added. (The original ball is removed.) Then a second ball is drawn. What is $P(R_2)$?

**Step 1: Identify the two scenarios.**

$$P(R_1) = \frac{5}{7}, \qquad P(G_1) = \frac{2}{7}$$

**Step 2: Find the urn composition after each scenario, then compute conditional probabilities.**

- *Scenario 1 — Red drawn first:* Remove the red, add a green. Urn now has 4 red + 3 green = 7 balls.
  $$P(R_2 \mid R_1) = \frac{4}{7}$$

- *Scenario 2 — Green drawn first:* Remove the green, add a red. Urn now has 6 red + 1 green = 7 balls.
  $$P(R_2 \mid G_1) = \frac{6}{7}$$

**Step 3: Apply the law of total probability.**

$$P(R_2) = P(R_2 \mid R_1) \cdot P(R_1) + P(R_2 \mid G_1) \cdot P(G_1)$$

$$= \frac{4}{7} \cdot \frac{5}{7} + \frac{6}{7} \cdot \frac{2}{7} = \frac{20}{49} + \frac{12}{49} = \frac{32}{49}$$

**Final Answer:** $P(R_2) = 32/49 \approx 0.653$

**Why this differs from Example 4:** The replacement rule makes red balls more likely to persist (a green first draw increases red count), which inflates $P(R_2)$ above $5/7 \approx 0.714$... actually $32/49 \approx 0.653 < 5/7$. The rule swaps colours, so drawing red initially decreases red balls and drawing green increases them — a counteracting effect that brings $P(R_2)$ close to but slightly below $5/7$.

---

## 5. Using Trees to Organize Computations

### 5.1 Concept Overview

A **probability tree** is a diagram that organizes sequential random processes. Each branch represents an outcome, and the probability is written on the branch. Trees make the law of total probability visual and systematic.

---

### 5.2 How to Read a Probability Tree

- **Root node** (level 0): Starting point (no information yet).
- **Level 1 nodes**: Outcomes of the first event. Branch probabilities are **marginal** (unconditional) probabilities.
- **Level 2 nodes**: Outcomes of the second event given the first. Branch probabilities are **conditional** probabilities.
- **Leaf node probability** = product of all branch probabilities along the path (by the multiplication rule).
- **Total probability of an outcome** = sum of leaf probabilities corresponding to that outcome.

---

### 5.3 Tree for Example 5 (Urn with Replacement Rule)

```
                    ●  (root)
                 /      \
             5/7           2/7
             /               \
           R₁                 G₁
          /  \               /   \
        4/7  3/7           6/7   1/7
        /      \           /       \
      (R₂)    (G₂)      (R₂)     (G₂)
    P=20/49  P=15/49  P=12/49   P=2/49
```

**Leaf probabilities (using multiplication rule):**
- $P(R_1 \cap R_2) = \frac{5}{7} \times \frac{4}{7} = \frac{20}{49}$
- $P(R_1 \cap G_2) = \frac{5}{7} \times \frac{3}{7} = \frac{15}{49}$
- $P(G_1 \cap R_2) = \frac{2}{7} \times \frac{6}{7} = \frac{12}{49}$
- $P(G_1 \cap G_2) = \frac{2}{7} \times \frac{1}{7} = \frac{2}{49}$

**Verify:** $20/49 + 15/49 + 12/49 + 2/49 = 49/49 = 1$ ✓

**Law of Total Probability from the tree:** Add all leaf probabilities leading to $R_2$:

$$P(R_2) = \frac{20}{49} + \frac{12}{49} = \frac{32}{49}$$

---

### 5.4 Shorthand vs. Precise Trees

In the shorthand tree, a leaf node labeled "$R_2$" under a path through $G_1$ actually represents the joint event $G_1 \cap R_2$. The shorthand is compact; the precise version labels everything explicitly. Always remember the shorthand convention when reading trees.

---

### 5.5 Three-Level Trees

For events with three stages (e.g., $A_i$, $B_j$, $C_k$), the tree extends to three levels. Branch probabilities at level 3 are conditional on both the level 1 and level 2 outcomes:

- Level 1 branch from root to $A_i$: $P(A_i)$
- Level 2 branch from $A_i$ to $B_j$: $P(B_j \mid A_i)$
- Level 3 branch from $A_i \cap B_j$ to $C_k$: $P(C_k \mid A_i \cap B_j)$

Leaf probability $= P(A_i) \cdot P(B_j \mid A_i) \cdot P(C_k \mid A_i \cap B_j) = P(A_i \cap B_j \cap C_k)$

---

## 6. Independence

### 6.1 Concept Overview

Two events are **independent** if knowing that one occurred gives you **no information** about whether the other occurred. The probability of one event is unchanged by knowledge of the other.

---

### 6.2 Intuition

> **Key Intuition:** If I tell you event $B$ happened, and you shrug and say "so what, that doesn't change my prediction for $A$" — then $A$ and $B$ are independent.

Examples of where independence can fail:
- Tossing a coin dipped in honey: the second toss outcome may be affected by the stickiness caused by the first.
- Survey respondents who all heard the same rumor from one source are not independent sources of evidence.

---

### 6.3 Formal Definition

> **Informal:** $A$ is independent of $B$ if $P(A \mid B) = P(A)$.
>
> **Formal Definition (Independence):** Events $A$ and $B$ are **independent** if:
>
> $$\boxed{P(A \cap B) = P(A) \cdot P(B)}$$

**Why the formal definition is preferred:**
- It's **symmetric**: $A$ independent of $B$ $\Leftrightarrow$ $B$ independent of $A$.
- It works even when $P(B) = 0$ (where the conditional probability is undefined).
- It directly generalises to multiple events.

**Equivalences (when probabilities are nonzero):**

| Condition | Equivalent independence check |
|---|---|
| $P(B) \neq 0$ | $A$ and $B$ independent $\Leftrightarrow$ $P(A \mid B) = P(A)$ |
| $P(A) \neq 0$ | $A$ and $B$ independent $\Leftrightarrow$ $P(B \mid A) = P(B)$ |
| Always | $A$ and $B$ independent $\Leftrightarrow$ $P(A \cap B) = P(A)P(B)$ |

---

### 6.4 Worked Examples

---

#### Example 6 — Two Coin Tosses

**Problem:** Toss a fair coin twice. Let $H_1$ = "heads on first toss", $H_2$ = "heads on second toss". Are they independent?

**Step 1: Find probabilities.**

$$P(H_1) = \frac{1}{2}, \qquad P(H_2) = \frac{1}{2}$$

$$P(H_1 \cap H_2) = P(\text{HH}) = \frac{1}{4}$$

**Step 2: Check the independence condition.**

$$P(H_1) \cdot P(H_2) = \frac{1}{2} \cdot \frac{1}{2} = \frac{1}{4} = P(H_1 \cap H_2) \checkmark$$

**Conclusion:** $H_1$ and $H_2$ are **independent**.

---

#### Example 7 — Are Two Heads and a Specific First Toss Independent?

**Problem:** Toss a fair coin 3 times. Let $H_1$ = "heads on first toss" and $A$ = "exactly two heads total". Are $H_1$ and $A$ independent?

**Step 1: List the sample space and events.**

$$\Omega = \{HHH, HHT, HTH, HTT, THH, THT, TTH, TTT\}$$

$$H_1 = \{HHH, HHT, HTH, HTT\}$$
$$A = \{HHT, HTH, THH\}, \quad P(A) = 3/8$$

**Step 2: Find $P(A \mid H_1)$.**

$A \cap H_1 = \{HHT, HTH\}$ (outcomes in both events), so $|A \cap H_1| = 2$.

$$P(A \mid H_1) = \frac{|A \cap H_1|}{|H_1|} = \frac{2}{4} = \frac{1}{2}$$

**Step 3: Compare.**

$$P(A \mid H_1) = \frac{1}{2} \neq \frac{3}{8} = P(A)$$

**Conclusion:** $H_1$ and $A$ are **not independent**. Knowing the first toss was heads increases the probability of getting exactly 2 heads total (from 3/8 to 1/2).

---

#### Example 8 — Cards: Ace, Hearts, Red

**Problem:** Draw one card from a standard deck. Define:  
$A$ = "the card is an ace", $H$ = "the card is a heart", $R$ = "the card is red".

**Part (a): Are $A$ and $H$ independent?**

$$P(A) = \frac{4}{52} = \frac{1}{13}, \qquad P(A \mid H) = \frac{1}{13} \quad \text{(1 ace of hearts out of 13 hearts)}$$

Since $P(A) = P(A \mid H)$, events $A$ and $H$ are **independent**.

**Part (b): Are $A$ and $R$ independent?**

$$P(A \mid R) = \frac{2}{26} = \frac{1}{13} = P(A)$$

(2 red aces out of 26 red cards). Since they are equal, $A$ and $R$ are **independent**.

**Part (c): Are $H$ and $R$ independent?**

$$P(H) = \frac{13}{52} = \frac{1}{4}, \qquad P(H \mid R) = \frac{13}{26} = \frac{1}{2}$$

Since $P(H) \neq P(H \mid R)$, $H$ and $R$ are **not independent**.

**Intuition:** If you know the card is red, it can only be hearts or diamonds — so the probability it's hearts jumps from 1/4 to 1/2. Colour and suit are not independent because suit *determines* colour.

---

### 6.5 A Paradox of Independence

An event $A$ with $P(A) = 0$ is independent of itself, since:

$$P(A \cap A) = P(A) = 0 = 0 \cdot 0 = P(A) \cdot P(A)$$

This seems paradoxical because knowing $A$ occurred certainly tells you $A$ occurred! The resolution: since $P(A) = 0$, the statement "$A$ occurred" is effectively impossible and thus vacuous — we never actually get information from it.

**Think:** For what other value of $P(A)$ is $A$ independent of itself?  
*Answer:* $P(A) = 1$. If $P(A) = 1$, then $P(A \cap A) = P(A) = 1 = 1 \times 1$.

---

## 7. Bayes' Theorem

### 7.1 Concept Overview

Bayes' theorem is one of the most important results in all of probability and statistics. It tells you how to **invert conditional probabilities**: if you know $P(A \mid B)$ but want $P(B \mid A)$, Bayes' rule gives you the answer.

In machine learning and statistics, this is the heart of **Bayesian inference**: updating beliefs about a hypothesis given observed data.

---

### 7.2 Intuition

> **Key Intuition:** We observe effect $A$ and want to reason backward to cause $B$. Bayes' theorem converts forward-direction conditional probabilities (probability of effect given cause) into backward-direction conditional probabilities (probability of cause given effect).

---

### 7.3 Formal Statement

> **Bayes' Theorem (Bayes' Rule / Bayes' Formula):** For events $A$ and $B$ with $P(A) \neq 0$:
>
> $$\boxed{P(B \mid A) = \frac{P(A \mid B) \cdot P(B)}{P(A)}}$$

---

### 7.4 Derivation

The proof is elegant and uses only the definition of conditional probability.

**Step 1:** From the definition of conditional probability:

$$P(B \mid A) \cdot P(A) = P(A \cap B)$$

**Step 2:** Also from the definition of conditional probability (in the other direction):

$$P(A \mid B) \cdot P(B) = P(A \cap B)$$

**Step 3:** Since $A \cap B = B \cap A$ (intersection is symmetric):

$$P(B \mid A) \cdot P(A) = P(A \mid B) \cdot P(B)$$

**Step 4:** Divide both sides by $P(A)$:

$$P(B \mid A) = \frac{P(A \mid B) \cdot P(B)}{P(A)} \qquad \square$$

---

### 7.5 Bayes' Theorem with Total Probability

In practice, $P(A)$ in the denominator is rarely given directly. We compute it using the law of total probability. When $B$ and $B^c$ partition the sample space:

$$P(B \mid A) = \frac{P(A \mid B) \cdot P(B)}{P(A \mid B) \cdot P(B) + P(A \mid B^c) \cdot P(B^c)}$$

**More generally** (with partition $B_1, \ldots, B_n$):

$$P(B_i \mid A) = \frac{P(A \mid B_i) \cdot P(B_i)}{\displaystyle\sum_{j=1}^{n} P(A \mid B_j) \cdot P(B_j)}$$

---

### 7.6 Worked Examples

---

#### Example 9 — Coin Tosses: Inverting Conditional Probability

**Problem:** Toss a coin 5 times. Let $H_1$ = "first toss is heads" and $H_A$ = "all 5 tosses are heads".

First, directly: $P(H_1 \mid H_A) = 1$ (if all 5 are heads, certainly the first is heads).

Now verify using Bayes' theorem.

**Step 1: Identify all terms.**

$$P(H_A \mid H_1) = \frac{1}{2^4} = \frac{1}{16} \quad \text{(need 4 more heads after first)}$$

$$P(H_1) = \frac{1}{2}, \qquad P(H_A) = \frac{1}{2^5} = \frac{1}{32}$$

**Step 2: Apply Bayes' theorem.**

$$P(H_1 \mid H_A) = \frac{P(H_A \mid H_1) \cdot P(H_1)}{P(H_A)} = \frac{(1/16) \cdot (1/2)}{1/32} = \frac{1/32}{1/32} = 1 \checkmark$$

**Interpretation:** Bayes' rule confirmed the direct reasoning. This simple example illustrates the mechanics without the complication of the total probability denominator.

---

## 8. The Base Rate Fallacy

### 8.1 Concept Overview

The base rate fallacy is one of the most important and practically consequential insights from probability. It explains why highly accurate tests can have surprisingly low predictive power when the condition being tested is rare.

> **The trap:** People instinctively use the accuracy of the test ($P(\text{positive} \mid \text{disease})$) as a proxy for what they actually want ($P(\text{disease} \mid \text{positive})$). These can be very different numbers.

---

### 8.2 The Core Insight

$$\underbrace{P(\text{test accurate})}_{\approx 95\%} \neq \underbrace{P(\text{positive test is correct})}_{\text{can be very low!}}$$

When a disease is rare, there are vastly more healthy people than sick people. Even a small false positive rate will generate **many false positives** (because it's applied to a huge population of healthy people), drowning out the true positives.

---

### 8.3 Worked Example — Disease Screening (Example 11)

**Problem:** Consider a routine screening test for a disease.
- **Base rate** (disease prevalence): $P(D^+) = 0.005$ (0.5%)
- **False positive rate:** $P(T^+ \mid D^-) = 0.05$ (5%)
- **False negative rate:** $P(T^- \mid D^+) = 0.10$ (10%)

You test positive. What is the probability you actually have the disease, $P(D^+ \mid T^+)$?

---

**Method 1: Using Bayes' Theorem**

**Step 1: Set up all known probabilities.**

| Quantity | Value |
|---|---|
| $P(D^+)$ | 0.005 |
| $P(D^-)$ | 0.995 |
| $P(T^+ \mid D^+)$ = true positive rate | $1 - 0.10 = 0.90$ |
| $P(T^- \mid D^+)$ = false negative rate | 0.10 |
| $P(T^+ \mid D^-)$ = false positive rate | 0.05 |
| $P(T^- \mid D^-)$ = true negative rate | $1 - 0.05 = 0.95$ |

**Step 2: Compute $P(T^+)$ using the law of total probability.**

$$P(T^+) = P(T^+ \mid D^+) \cdot P(D^+) + P(T^+ \mid D^-) \cdot P(D^-)$$

$$= (0.90)(0.005) + (0.05)(0.995)$$

$$= 0.0045 + 0.04975 = 0.05425$$

**Step 3: Apply Bayes' theorem.**

$$P(D^+ \mid T^+) = \frac{P(T^+ \mid D^+) \cdot P(D^+)}{P(T^+)} = \frac{0.90 \times 0.005}{0.05425} = \frac{0.0045}{0.05425} \approx 0.083$$

**Final Answer:** $P(D^+ \mid T^+) \approx 8.3\%$

> **Shocking insight:** Even though the test is 90% accurate for sick people and 95% accurate for healthy people, a positive test result only means you have an 8.3% chance of being sick!

---

**Method 2: Probability Tree**

```
                        ●
                    /       \
               0.995          0.005
              /                   \
            D⁻                    D⁺
           /  \                  /   \
        0.05  0.95            0.90   0.10
        /       \             /         \
      T⁺        T⁻          T⁺          T⁻
  P=0.04975  P=0.94525  P=0.0045    P=0.0005
```

Leaf probabilities for $T^+$ are 0.04975 (from healthy) and 0.0045 (from sick).

$$P(D^+ \mid T^+) = \frac{0.0045}{0.04975 + 0.0045} = \frac{0.0045}{0.05425} \approx 8.3\%$$

---

**Method 3: Table (Most Intuitive)**

Scale to **10,000 people**:

| | $D^+$ (Sick) | $D^-$ (Healthy) | Total |
|---|---|---|---|
| $T^+$ (Positive) | 45 | 498 | **543** |
| $T^-$ (Negative) | 5 | 9,452 | 9,457 |
| **Total** | 50 | 9,950 | 10,000 |

**Construction:**
- Total sick: $10000 \times 0.005 = 50$
- Total healthy: $10000 \times 0.995 = 9950$
- True positives: $50 \times 0.90 = 45$
- False negatives: $50 \times 0.10 = 5$
- False positives: $9950 \times 0.05 = 498$ (rounded from 497.5)
- True negatives: $9950 \times 0.95 = 9452$ (rounded)

$$P(D^+ \mid T^+) = \frac{45}{543} \approx 8.3\%$$

**Why is this the most intuitive?** You can see directly: out of 543 positive tests, only 45 come from actually sick people. The other 498 are false alarms from healthy people.

---

### 8.4 Summary Comparison

| What the test claims | What Bayes' theorem says |
|---|---|
| 95% of tests are accurate | Only ~8% of positive tests identify sick people |
| P(positive \| sick) = 90% | P(sick \| positive) ≈ 8.3% |

This table captures the essence of the base rate fallacy.

---

### 8.5 Why This Matters in Real Life

- **Medical testing:** Low-prevalence disease screening will generate many false positives even with accurate tests. Always ask your doctor about the base rate and the predictive value of a positive test.
- **Spam filtering:** A rare pattern triggering a spam filter might not actually be spam.
- **Security screening:** Rare terrorist profiles mean that even accurate screening generates many false positives.
- **Machine learning:** Class imbalance in training data leads to similar issues. A model that says "everyone is healthy" achieves 99.5% accuracy when 0.5% of people are sick.

---

## 9. In-Class Concept Questions — Full Solutions

### Concept Question 1 — Coin Toss and Conditional Probability

**Problem:** Toss a coin 4 times. Let $A$ = "at least three heads" and $B$ = "first toss is tails".

**(1) What is $P(A \mid B)$?**  
Options: (a) 1/16, (b) 1/8, (c) 1/4, (d) 1/5

**(2) What is $P(B \mid A)$?**  
Options: (a) 1/16, (b) 1/8, (c) 1/4, (d) 1/5

**Solution:**

**Step 1: List all outcomes with at least 3 heads (event $A$).**

At least 3 heads means exactly 3 or exactly 4 heads:
- Exactly 4 heads: HHHH (1 outcome)
- Exactly 3 heads: HHHT, HHTH, HTHH, THHH (4 outcomes)

$$|A| = 5$$

**Step 2: List all outcomes where first toss is tails (event $B$).**

First toss is T, remaining 3 tosses are arbitrary: $2^3 = 8$ outcomes.

$$|B| = 8$$

**Step 3: Find $A \cap B$ — at least 3 heads AND first toss is tails.**

If the first toss is tails, we need at least 3 heads from the remaining 3 tosses. The only way to get at least 3 heads total is if the last 3 are all heads: **THHH** (1 outcome).

$$|A \cap B| = 1$$

**Step 4: Compute $P(A \mid B)$.**

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)} = \frac{|A \cap B|/16}{|B|/16} = \frac{|A \cap B|}{|B|} = \frac{1}{8}$$

**Answer to (1): (b) 1/8**

**Step 5: Compute $P(B \mid A)$.**

$$P(B \mid A) = \frac{P(A \cap B)}{P(A)} = \frac{|A \cap B|}{|A|} = \frac{1}{5}$$

**Answer to (2): (d) 1/5**

**Interpretation:**
- $P(A \mid B) = 1/8$: Given the first toss was tails, the chance of at least 3 heads is small — only THHH works.
- $P(B \mid A) = 1/5$: Given we got at least 3 heads, only 1 of the 5 qualifying outcomes (THHH) has the first toss as tails.

Notice: $P(A \mid B) = 1/8 \neq 1/5 = P(B \mid A)$ — a clear illustration that conditional probability is not symmetric.

---

### Concept Questions 2–5 — Probability Trees

These questions use the following three-level probability tree:

```
                        x
                    /       \
                  A₁          A₂
                y              ...
             /     \
           B₁       B₂
                   z
                 /    \
               C₁      C₂
               ●
```

The circled node is $C_1$ under the path $A_1 \to B_2$.

---

**Concept Question 2:** The probability $x$ represents:  
(a) $P(A_1)$ (b) $P(A_1 \mid B_2)$ (c) $P(B_2 \mid A_1)$ (d) $P(C_1 \mid B_2 \cap A_1)$

**Answer: (a) $P(A_1)$**

**Reasoning:** $x$ is the branch probability from the **root node** to $A_1$. Root-to-level-1 branches always carry unconditional (marginal) probabilities.

---

**Concept Question 3:** The probability $y$ represents:  
(a) $P(B_2)$ (b) $P(A_1 \mid B_2)$ (c) $P(B_2 \mid A_1)$ (d) $P(C_1 \mid B_2 \cap A_1)$

**Answer: (c) $P(B_2 \mid A_1)$**

**Reasoning:** $y$ is the branch from $A_1$ to $B_2$ at level 2. Level 2 branches carry probabilities conditioned on the level 1 outcome — you're already at $A_1$ and branching to $B_2$.

---

**Concept Question 4:** The probability $z$ represents:  
(a) $P(C_1)$ (b) $P(B_2 \mid C_1)$ (c) $P(C_1 \mid B_2)$ (d) $P(C_1 \mid B_2 \cap A_1)$

**Answer: (d) $P(C_1 \mid B_2 \cap A_1)$**

**Reasoning:** $z$ is the branch at level 3 from the node reached via $A_1 \to B_2$ to $C_1$. The condition includes all ancestors on the path: $A_1$ and $B_2$.

---

**Concept Question 5:** The circled node represents the event:  
(a) $C_1$ (b) $B_2 \cap C_1$ (c) $A_1 \cap B_2 \cap C_1$ (d) $C_1 \mid B_2 \cap A_1$

**Answer: (c) $A_1 \cap B_2 \cap C_1$**

**Reasoning:** A leaf node at the end of a path represents the **joint event** of all outcomes along the path. The path goes through $A_1$, then $B_2$, then $C_1$, so the leaf represents $A_1 \cap B_2 \cap C_1$.

> **Key rule:** A node in a tree represents the joint event of its entire ancestral path, not just the label at that node.

---

## 10. In-Class Board Problems — Full Solutions

### Problem 1 — The Monty Hall Problem

**Setup:**
- 3 doors: 1 hides a car, 2 hide goats.
- Contestant chooses a door.
- Monty (who knows where the car is) opens a **different** door revealing a goat.
- Contestant may switch to the remaining unopened door or stay.

**Question:** What is the best strategy?  
(a) Switch, (b) Don't switch, (c) It doesn't matter.

**Answer: (a) Switch. Switching gives probability 2/3 of winning the car.**

---

**Full Solution with Probability Tree:**

The key is to organise the process into a sequence:
1. Contestant chooses a door (which is either a car door or a goat door).
2. Monty reveals a goat.
3. Contestant switches.

**After Monty reveals a goat, the switching strategy is:**
- If the contestant initially chose the **car** ($C_1$, probability 1/3): switching leads to a goat. **Loss.**
- If the contestant initially chose a **goat** ($G_1$, probability 2/3): the other goat is revealed, switching leads to the car. **Win.**

```
                    ●
                1/3   2/3
               /           \
             C₁              G₁
        (chose car)      (chose goat)
            |  0              |  1
            |                 |
           G₂               C₂
        (switch→goat)   (switch→car)
```

**Probability of winning when switching:**

$$P_{\text{switch}}(C_2) = P_{\text{switch}}(C_2 \mid C_1) \cdot P(C_1) + P_{\text{switch}}(C_2 \mid G_1) \cdot P(G_1)$$

$$= 0 \cdot \frac{1}{3} + 1 \cdot \frac{2}{3} = \frac{2}{3}$$

**Conclusion:** Always switch. Switching wins the car with probability 2/3; staying wins with probability 1/3.

---

**Why this surprises people:**

The intuition that "two doors remain, so it's 50-50" is wrong because Monty's action is not random — he always reveals a goat, and his choice carries information about where the car is.

More specifically: when you first pick, you have a 1/3 chance of being right. That doesn't change. The probability 2/3 that the car is behind one of the other two doors gets "concentrated" onto the single remaining door after Monty opens one.

---

### Problem 2 — Independence of Dice Events

**Problem:** Roll two dice. Consider:
- $A$ = "first die is 3"
- $B$ = "sum is 6"
- $C$ = "sum is 7"

Is $A$ independent of: (a) $B$ and $C$, (b) $B$ alone, (c) $C$ alone, (d) neither?

**Answer: (c) $C$ alone.**

---

**Full Solution:**

**Step 1: Find $P(A)$.**

The first die is 3 with probability $1/6$.

$$P(A) = \frac{1}{6}$$

**Step 2: Check $A$ and $B$ (sum is 6).**

The ways to roll a sum of 6 are: (1,5), (2,4), (3,3), (4,2), (5,1) → 5 outcomes.

$$P(B) = \frac{5}{36}$$

$A \cap B$ = "first die is 3 AND sum is 6" → only (3,3).

$$P(A \cap B) = \frac{1}{36}$$

Check independence: $P(A) \cdot P(B) = \frac{1}{6} \cdot \frac{5}{36} = \frac{5}{216} \neq \frac{1}{36} = \frac{6}{216}$

Or equivalently: $P(A \mid B) = \frac{1/36}{5/36} = \frac{1}{5} \neq \frac{1}{6} = P(A)$.

**$A$ and $B$ are NOT independent.** Knowing $B$ occurred means the first die can only be 1, 2, 3, 4, or 5 (since a 6 can't sum to 6 with a positive die), changing the probability of $A$.

**Step 3: Check $A$ and $C$ (sum is 7).**

The ways to roll a sum of 7 are: (1,6), (2,5), (3,4), (4,3), (5,2), (6,1) → 6 outcomes.

$$P(C) = \frac{6}{36} = \frac{1}{6}$$

$A \cap C$ = "first die is 3 AND sum is 7" → only (3,4).

$$P(A \cap C) = \frac{1}{36}$$

Check: $P(A) \cdot P(C) = \frac{1}{6} \cdot \frac{1}{6} = \frac{1}{36} = P(A \cap C)$ ✓

Or: $P(A \mid C) = \frac{1/36}{6/36} = \frac{1}{6} = P(A)$ ✓

**$A$ and $C$ ARE independent.**

**Why?** For any value of the first die (1–6), there is exactly one value of the second die that makes the sum 7. So knowing the sum is 7 tells us *nothing* about the first die — all values remain equally likely at 1/6. The sum of 7 is "democratic" across all first-die values.

**Why does knowing $B$ (sum = 6) change things?** Because rolling a 6 on the first die makes sum = 6 impossible (you'd need 0 on the second die). So $B$ excludes the value 6 from the first die, making the remaining 5 values each have probability 1/5 instead of 1/6.

---

### Problem 3 — Evil Squirrels (Base Rate Fallacy in Action)

**Problem:** MIT has 1,000,000 squirrels, 100 of which are "pure evil." An "Evil Squirrel Alarm" has:
- 99% true positive rate: $P(\text{alarm} \mid \text{evil}) = 0.99$
- 1% false positive rate: $P(\text{alarm} \mid \text{nice}) = 0.01$

**(a)** If a squirrel sets off the alarm, what is $P(\text{evil} \mid \text{alarm})$?  
**(b)** Should MIT employ the system?

---

**Solution:**

**Step 1: Identify the base rates.**

$$P(\text{evil}) = \frac{100}{1{,}000{,}000} = 0.0001, \qquad P(\text{nice}) = 0.9999$$

**Step 2: Apply Bayes' theorem.**

$$P(\text{evil} \mid \text{alarm}) = \frac{P(\text{alarm} \mid \text{evil}) \cdot P(\text{evil})}{P(\text{alarm} \mid \text{evil}) \cdot P(\text{evil}) + P(\text{alarm} \mid \text{nice}) \cdot P(\text{nice})}$$

$$= \frac{(0.99)(0.0001)}{(0.99)(0.0001) + (0.01)(0.9999)}$$

$$= \frac{0.000099}{0.000099 + 0.009999} = \frac{0.000099}{0.010098} \approx 0.0098 \approx 1\%$$

**Final Answer (a):** $P(\text{evil} \mid \text{alarm}) \approx 1\%$

---

**Table Method (most transparent):**

| | Evil | Nice | Total |
|---|---|---|---|
| Alarm goes off | 99 | 9,999 | **10,098** |
| Alarm silent | 1 | 989,901 | 989,902 |
| **Total** | 100 | 999,900 | 1,000,000 |

- Evil + alarm: $100 \times 0.99 = 99$
- Nice + alarm: $999{,}900 \times 0.01 = 9{,}999$

$$P(\text{evil} \mid \text{alarm}) = \frac{99}{10{,}098} \approx 0.0098 \approx 1\%$$

---

**Answer (b): No. MIT should not employ the system.**

The alarm would be virtually useless. For every 1 truly evil squirrel correctly flagged, approximately **99 innocent squirrels** would also trigger the alarm. The false positive rate is 99 times the true positive rate — the alarm is more trouble than it's worth.

**Key lesson (comparison):**

| Metric | Value |
|---|---|
| Accuracy of alarm on a random squirrel | $\approx 99\%$ |
| Probability that an alarm indicates evil | $\approx 1\%$ |

These look completely contradictory. The resolution is the base rate: evil squirrels are so rare (1 in 10,000) that even a 1% false positive rate generates a flood of false alarms.

---

### Problem 4 — The Dice Game (Bayes Applied)

**Problem:**
1. The Randomizer secretly holds either a 6-sided die or an 8-sided die in each fist.
2. The Roller picks a fist (each with probability 1/2) and rolls the die in secret.
3. The result is reported to the table.

**Question:** Given the reported number, what is the probability the 6-sided die was chosen?

---

**Solution:**

Let $D_6$ = "6-sided die chosen", $D_8$ = "8-sided die chosen".

$$P(D_6) = P(D_8) = \frac{1}{2}$$

**Case 1: Roll is a number $k \in \{1, 2, 3, 4, 5, 6\}$**

$$P(\text{roll } k \mid D_6) = \frac{1}{6}, \qquad P(\text{roll } k \mid D_8) = \frac{1}{8}$$

Apply Bayes' theorem:

$$P(D_6 \mid \text{roll } k) = \frac{P(\text{roll } k \mid D_6) \cdot P(D_6)}{P(\text{roll } k)}$$

The denominator (law of total probability):

$$P(\text{roll } k) = \frac{1}{6} \cdot \frac{1}{2} + \frac{1}{8} \cdot \frac{1}{2} = \frac{1}{12} + \frac{1}{16} = \frac{4}{48} + \frac{3}{48} = \frac{7}{48}$$

Therefore:

$$P(D_6 \mid \text{roll } k) = \frac{(1/6)(1/2)}{7/48} = \frac{1/12}{7/48} = \frac{1}{12} \cdot \frac{48}{7} = \frac{4}{7}$$

**Case 2: Roll is 7 or 8**

The 6-sided die cannot produce 7 or 8:

$$P(\text{roll 7} \mid D_6) = 0 \implies P(D_6 \mid \text{roll 7}) = 0$$

**Summary:**

| Reported Roll | $P(D_6 \mid \text{roll})$ | $P(D_8 \mid \text{roll})$ |
|---|---|---|
| 1 through 6 | $4/7 \approx 57.1\%$ | $3/7 \approx 42.9\%$ |
| 7 or 8 | $0\%$ | $100\%$ |

**Interpretation:** If a low number (1–6) is rolled, the 6-sided die is more likely to have been chosen (4/7 vs. 3/7), because the 6-sided die produces each number with probability 1/6, which is higher than the 8-sided die's 1/8 for the same outcomes. If 7 or 8 is rolled, only the 8-sided die could have produced it, so we're certain it's the 8-sided die.

**Worked numerical detail for roll = 4:**

$$P(D_6 \mid \text{roll 4}) = \frac{(1/6)(1/2)}{(1/6)(1/2) + (1/8)(1/2)} = \frac{1/12}{1/12 + 1/16} = \frac{1/12}{7/48} = \frac{4}{7}$$

The same calculation applies for any roll of 1–6 — the specific number doesn't matter, only whether it's reachable by both dice.

---

### Bonus — In-Class Example: Urn Game

**Problem:** Game with 5 orange and 2 blue balls in an urn. A random ball is selected and **replaced by a ball of the other color** (as in Example 5 from the prep notes). Then a second ball is drawn.

1. What is the probability the second ball is orange?
2. What is $P(\text{first was orange} \mid \text{second is orange})$?

**Solution:**

Let $O_1, B_1, O_2, B_2$ denote the obvious events.

**Part 1:** $P(O_2)$ using law of total probability.

$$P(O_1) = 5/7, \quad P(B_1) = 2/7$$

After drawing orange: 4 orange + 3 blue remain (6 orange removed, 1 blue added):
$$P(O_2 \mid O_1) = 4/7$$

After drawing blue: 6 orange + 1 blue remain:
$$P(O_2 \mid B_1) = 6/7$$

$$P(O_2) = \frac{5}{7} \cdot \frac{4}{7} + \frac{2}{7} \cdot \frac{6}{7} = \frac{20}{49} + \frac{12}{49} = \frac{32}{49}$$

**Part 2:** $P(O_1 \mid O_2)$ using Bayes' theorem.

$$P(O_1 \mid O_2) = \frac{P(O_2 \mid O_1) \cdot P(O_1)}{P(O_2)} = \frac{(4/7)(5/7)}{32/49} = \frac{20/49}{32/49} = \frac{20}{32} = \frac{5}{8}$$

**Answers:** $P(O_2) = 32/49 \approx 65.3\%$; $P(O_1 \mid O_2) = 5/8 = 62.5\%$.

---

## 11. Common Mistakes Reference

| Mistake | Why it's wrong | Correct approach |
|---|---|---|
| Confusing $P(A \mid B)$ with $P(B \mid A)$ | These are completely different quantities (base rate fallacy) | Always identify clearly which event is being conditioned on |
| Using $P(A \mid B) = P(B \mid A)$ | Only equal in special cases where $P(A) = P(B)$ | Use Bayes' theorem to convert between them |
| Thinking "independent" means "mutually exclusive" | Independent events can both occur; mutually exclusive events cannot | $P(A \cap B) = P(A)P(B)$ for independence; $P(A \cap B) = 0$ for mutual exclusivity |
| Forgetting the base rate | Makes the denominator of Bayes' theorem wrong | Always include $P(B)$ or compute $P(A)$ via total probability |
| Multiplying non-independent probabilities directly | $P(A \cap B) \neq P(A) \cdot P(B)$ when events are dependent | Use the multiplication rule: $P(A \cap B) = P(A \mid B) P(B)$ |
| Misreading tree branches as unconditional probabilities | Level 2+ branches are conditional on parent node | Level 1: unconditional; Level 2+: conditional on the path |
| Assuming a leaf node represents just one event label | Leaf represents the joint event of the entire path | The leaf after path $A_1 \to B_2 \to C_1$ represents $A_1 \cap B_2 \cap C_1$ |
| Thinking "accurate test" means "positive = disease" | A test can be very accurate yet most positives are false | Account for prevalence (base rate) in computing predictive value |

---

## 12. Quick Summary & Formula Sheet

### Core Definitions

| Concept | Formula |
|---|---|
| Conditional probability | $P(A \mid B) = \dfrac{P(A \cap B)}{P(B)}$ |
| Multiplication rule | $P(A \cap B) = P(A \mid B) \cdot P(B)$ |
| Law of total probability | $P(A) = \sum_i P(A \mid B_i) P(B_i)$ |
| Independence (formal) | $P(A \cap B) = P(A) \cdot P(B)$ |
| Independence (conditional form) | $P(A \mid B) = P(A)$ |
| Bayes' theorem | $P(B \mid A) = \dfrac{P(A \mid B) \cdot P(B)}{P(A)}$ |
| Bayes' + total probability | $P(B \mid A) = \dfrac{P(A \mid B) P(B)}{P(A \mid B) P(B) + P(A \mid B^c) P(B^c)}$ |

### Key Insights

- **Conditional probability restricts the sample space** to the conditioning event.
- **The multiplication rule** lets you compute $P(A \cap B)$ as a sequential product of probabilities.
- **Probability trees** make the law of total probability visual: branch probabilities multiply (multiplication rule), and leaves over the same outcome sum (total probability).
- **Independence** means one event gives no information about the other. Test it with $P(A \cap B) \stackrel{?}{=} P(A) P(B)$.
- **Bayes' theorem inverts** conditional probabilities: it converts $P(\text{effect} \mid \text{cause})$ into $P(\text{cause} \mid \text{effect})$.
- **The base rate fallacy**: $P(\text{positive test is correct}) \neq P(\text{test is accurate})$ — they can be very different when the condition is rare.

### Reading Probability Trees

| Tree element | What it represents |
|---|---|
| Root node | Start; no information |
| Level 1 branch probability | $P(\text{outcome}_i)$ — unconditional |
| Level 2 branch probability from node $X$ | $P(\text{outcome}_j \mid X)$ — conditional on parent |
| Level 3 branch from $X \to Y$ | $P(\text{outcome}_k \mid X \cap Y)$ — conditional on full path |
| Leaf node probability | Product of all branch probabilities on path = joint probability |
| $P(\text{event})$ from tree | Sum of all leaf probabilities corresponding to that event |

---

*End of MIT 18.05 Class 3 Study Notes*  
*Source: MIT OpenCourseWare, 18.05 Introduction to Probability and Statistics, Spring 2022*  
*Jeremy Orloff and Jonathan Bloom — https://ocw.mit.edu*
