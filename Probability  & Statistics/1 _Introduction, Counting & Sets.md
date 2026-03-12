## 1 : Introduction, Counting & Sets
> *Based on MIT OpenCourseWare materials by Jeremy Orloff and Jonathan Bloom, Spring 2022*

---

## Table of Contents

1. [What Are Probability and Statistics?](#1-what-are-probability-and-statistics)
2. [Frequentist vs. Bayesian Interpretations](#2-frequentist-vs-bayesian-interpretations)
3. [Applications, Toy Models, and Simulation](#3-applications-toy-models-and-simulation)
4. [Sets and Notation](#4-sets-and-notation)
   - [Definitions](#41-definitions)
   - [Venn Diagrams](#42-venn-diagrams)
   - [DeMorgan's Laws](#43-demorgans-laws)
   - [Products of Sets](#44-products-of-sets)
5. [Counting Techniques](#5-counting-techniques)
   - [The Core Principle](#51-the-core-principle)
   - [Inclusion-Exclusion Principle](#52-inclusion-exclusion-principle)
   - [Rule of Product](#53-rule-of-product)
6. [Permutations and Combinations](#6-permutations-and-combinations)
   - [Permutations](#61-permutations)
   - [Combinations](#62-combinations)
   - [Formulas Summary](#63-formulas-summary)
7. [Practice Problems & Solutions](#7-practice-problems--solutions)

---

## 1. What Are Probability and Statistics?

Probability and statistics are **deeply connected** — all statistical statements are, at bottom, statements about probability. Yet the two subjects can feel surprisingly different in practice.

| | **Probability** | **Statistics** |
|---|---|---|
| **What you know** | The random process | The outcome (data) |
| **What you find** | The probability of an outcome | The nature of the random process |
| **Feel** | Logically self-contained | Messy — as much art as science |
| **Example** | Fair coin, 100 tosses: P(≥60 heads) = ? | You got 60 heads: is the coin fair? |

### Probability Example
> You have a **fair coin** (equal probability of heads or tails). You toss it 100 times.  
> **What is the probability of 60 or more heads?**  
>
> There is exactly **one correct answer: ≈ 0.028444**. We will learn how to compute this.

The key point: the random process is **fully known** (P(heads) = 0.5). The goal is to find the probability of a particular outcome arising from that process.

### Statistics Example
> You have a coin of **unknown provenance**. To investigate whether it is fair, you toss it 100 times and observe **60 heads**. Your job as a statistician is to **draw an inference** from this data.

The key difference: the **outcome is known** (60 heads), and the goal is to **illuminate the unknown random process** (what is the true probability of heads for this coin?).

Notice how the two examples are *mirror images* of each other. Different statisticians might even draw **different conclusions** from the same data — which is why statistics involves judgment, not just calculation.

---

## 2. Frequentist vs. Bayesian Interpretations

There are two major schools of thought in statistics. Their approaches are rooted in **fundamentally different interpretations of what probability means**.

### Frequentist View
> **Probability measures the long-run frequency of outcomes** in a repeated experiment.

- Saying a fair coin has a 50% probability of heads means: if you toss it **many times**, about half the tosses will land heads.
- Probability is a property of the *physical world*, not of your mind.
- Has long been dominant in biology, medicine, public health, and the social sciences.

### Bayesian View
> **Probability is an abstract concept measuring a state of knowledge or degree of belief** in a proposition.

- In practice, a Bayesian does not assign a single value for "the probability this coin is fair." Instead, they consider **a range of values, each with its own probability of being correct**.
- Especially useful when **incorporating new data** into an existing model (e.g., training a speech recognition system).
- Has enjoyed a major resurgence in the era of powerful computers and big data.

### The Big Picture
Today, statisticians are creating powerful tools by using **both approaches in complementary ways**. In 18.05, we study and compare them directly. You should understand both, not just one.

---

## 3. Applications, Toy Models, and Simulation

### Why Does Any of This Matter?

Probability and statistics appear across virtually every quantitative field:

- **Medicine**: Testing one treatment against another (or a placebo)
- **Genetics**: Measures of genetic linkage
- **Physics**: The search for elementary particles
- **Computer Science**: Machine learning for vision and speech
- **Finance**: Gambling probabilities, economic forecasting
- **Climate Science**: Climate modeling
- **Epidemiology**: Tracking the spread of disease
- **Marketing**: Targeting and conversion optimization
- **Web**: How Google ranks search results

### Why Coins and Dice?

You may wonder why so much time in probability is spent on toys like coins and dice. The answer is elegant:

> **By understanding toy models thoroughly, you develop a feel for the simple essence hiding inside many complex real-world problems.**

The modest coin is a realistic model for **any situation with two possible outcomes**: success or failure of a treatment, an airplane engine, a bet, or even passing a class.

### Simulation

Sometimes a problem is so complicated that the best approach is **computer simulation** — using software to run virtual experiments thousands or millions of times to *estimate* probabilities. In 18.05, we use **R** for simulation, computation, and visualization.

---

## 4. Sets and Notation

Since we compute probabilities by counting elements of sets, we need a solid foundation in set theory.

### 4.1 Definitions

A **set** $S$ is a collection of elements. Here is the notation you need to know:

| Symbol | Meaning | Example |
|---|---|---|
| $x \in S$ | Element $x$ is in set $S$ | $3 \in \{1,2,3\}$ |
| $A \subset S$ | $A$ is a **subset** of $S$ (all elements of $A$ are in $S$) | $\{1,2\} \subset \{1,2,3\}$ |
| $A^c$ or $S - A$ | **Complement** of $A$ in $S$: elements of $S$ not in $A$ | |
| $A \cup B$ | **Union**: elements in $A$ *or* $B$ (or both) | |
| $A \cap B$ | **Intersection**: elements in *both* $A$ and $B$ | |
| $\emptyset$ | **Empty set**: the set with no elements | |
| $A \cap B = \emptyset$ | $A$ and $B$ are **disjoint** (no common elements) | |
| $A - B$ | **Difference**: elements in $A$ but *not* in $B$ | |
| $\|S\|$ or $\S$ | **Cardinality**: number of elements in $S$ | $|\{a,b,c\}| = 3$ |

### Worked Example — Animals

Start with the set of 10 animals:

$$S = \{\text{Antelope, Bee, Cat, Dog, Elephant, Frog, Gnat, Hyena, Iguana, Jaguar}\}$$

Define two subsets:

$$M = \text{mammals} = \{\text{Antelope, Cat, Dog, Elephant, Hyena, Jaguar}\}$$

$$W = \text{wild animals} = \{\text{Antelope, Bee, Elephant, Frog, Gnat, Hyena, Iguana, Jaguar}\}$$

Now applying set operations:

- **Intersection** $M \cap W$ — wild *and* mammal:
  $$M \cap W = \{\text{Antelope, Elephant, Hyena, Jaguar}\}$$

- **Union** $M \cup W$ — mammal *or* wild (or both):
  $$M \cup W = \{\text{Antelope, Bee, Cat, Dog, Elephant, Frog, Gnat, Hyena, Iguana, Jaguar}\}$$
  (Everything — because every animal is either a mammal or lives in the wild.)

- **Complement** $M^c$ — not a mammal:
  $$M^c = \{\text{Bee, Frog, Gnat, Iguana}\}$$

- **Difference** $M - W$ — mammal but *not* wild (i.e., domesticated mammals):
  $$M - W = \{\text{Cat, Dog}\}$$

> **Note:** There are often many different ways to express the same set.  
> For example: $M^c = S - M$, and $M - W = M \cap W^c$.

---

### 4.2 Venn Diagrams

Venn diagrams provide an invaluable visual tool for understanding set operations. In all figures, $S$ is the large rectangle, $L$ is the left circle, and $R$ is the right circle.

```
     S
  ┌─────────────────────┐
  │    ╭───╮   ╭───╮    │
  │   │  L │ ∩ │ R │   │
  │   │    │   │    │   │
  │    ╰───╯   ╰───╯    │
  └─────────────────────┘
```

| Expression | What's Shaded |
|---|---|
| $L \cup R$ | Everything inside either circle |
| $L \cap R$ | Only the overlapping (middle) region |
| $L^c$ | Everything in $S$ *outside* the left circle |
| $L - R$ | The left circle *excluding* the overlap |

---

### 4.3 DeMorgan's Laws

The relationship between union, intersection, and complement is captured by **DeMorgan's Laws**:

$$\boxed{(A \cup B)^c = A^c \cap B^c}$$

$$\boxed{(A \cap B)^c = A^c \cup B^c}$$

**In plain English:**
- First law: "Everything that is *not in A or B*" = "everything that is *not in A* AND *not in B*."
- Second law: "Everything that is *not in both A and B*" = "everything that is *not in A* OR *not in B*."

#### Proof via Example

Let $A = \{1,2,3\}$, $B = \{3,4\}$, $S = \{1,2,3,4,5\}$.

**Verify Law 1:** $(A \cup B)^c = A^c \cap B^c$

- Left side: $A \cup B = \{1,2,3,4\}$, so $(A \cup B)^c = \{5\}$
- Right side: $A^c = \{4,5\}$, $B^c = \{1,2,5\}$, so $A^c \cap B^c = \{5\}$ ✓

**Verify Law 2:** $(A \cap B)^c = A^c \cup B^c$

- Left side: $A \cap B = \{3\}$, so $(A \cap B)^c = \{1,2,4,5\}$
- Right side: $A^c = \{4,5\}$, $B^c = \{1,2,5\}$, so $A^c \cup B^c = \{1,2,4,5\}$ ✓

---

### 4.4 Products of Sets

The **product** of sets $S$ and $T$ is the set of all *ordered pairs*:

$$S \times T = \{(s, t) \mid s \in S, t \in T\}$$

#### Example

$$\{1,2,3\} \times \{1,2,3,4\}$$

| $\times$ | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| **1** | (1,1) | (1,2) | (1,3) | (1,4) |
| **2** | (2,1) | (2,2) | (2,3) | (2,4) |
| **3** | (3,1) | (3,2) | (3,3) | (3,4) |

This gives $3 \times 4 = 12$ ordered pairs total. This is the set-theoretic foundation of the **Rule of Product** (see below).

> **Key fact:** If $A \subset S$ and $B \subset T$, then $A \times B \subset S \times T$.

---

## 5. Counting Techniques

### 5.1 The Core Principle

> **Principle:** Suppose there are $n$ equally probable outcomes for an experiment. If $k$ of those outcomes are "desirable," then:
> $$P(\text{desirable outcome}) = \frac{k}{n}$$

This is the bridge between **counting** and **probability**. Before we can compute any probabilities, we need to count things well.

**Motivating Example:**
You flip a fair coin 3 times. What is the probability of exactly one head?

List all 8 equally probable outcomes:

$$\{TTT,\ TTH,\ THT,\ THH,\ HTT,\ HTH,\ HHT,\ HHH\}$$

Outcomes with exactly one head: $\{TTH,\ THT,\ HTT\}$ — there are 3.

$$P(\text{exactly 1 head}) = \frac{3}{8}$$

> **Think:** Would listing all outcomes be practical with 10 flips? There would be $2^{10} = 1024$ outcomes. Listing is clearly infeasible — we need better techniques.

---

### 5.2 Inclusion-Exclusion Principle

$$\boxed{|A \cup B| = |A| + |B| - |A \cap B|}$$

**Why?** When you add $|A| + |B|$, the elements in the overlap $A \cap B$ get counted **twice** — so you subtract them once to correct.

#### Worked Example — The Band

> A band consists of singers and guitar players. 7 people sing, 4 play guitar, and 2 do both. How many people are in the band?

Let $S$ = singers, $G$ = guitar players. The whole band is $S \cup G$.

$$|S \cup G| = |S| + |G| - |S \cap G| = 7 + 4 - 2 = \boxed{9}$$

> **Common mistake:** Simply adding 7 + 4 = 11 double-counts the 2 people who do both. Inclusion-exclusion fixes this.

---

### 5.3 Rule of Product

> **Rule of Product:** If there are $n$ ways to perform action 1, and then $m$ ways to perform action 2, then there are $n \cdot m$ ways to perform action 1 *followed by* action 2.

Also called the **multiplication rule**.

> **Critical subtlety:** The rule holds even if the *specific* options for action 2 depend on what was chosen in action 1 — as long as the **number** of options for action 2 is the same regardless of what was chosen in action 1.

#### Example — Shirts and Pants

3 shirts × 4 pants = **12 outfits** total.

#### Example — Olympic Medals

> There are 5 competitors in the 100m final. How many ways can gold, silver, and bronze be awarded?

- 5 ways to pick gold
- Once gold is awarded, 4 remaining athletes can take silver
- Once gold and silver are awarded, 3 remaining can take bronze

$$5 \times 4 \times 3 = \boxed{60 \text{ ways}}$$

Note: who wins silver *depends* on who won gold, but the *number* of silver candidates is always 4.

#### Example — Outfits with Color Constraints

> *"I won't wear green and red together. I think black or denim goes with anything."*
>
> Wardrobe: Shirts: 3 Black (B), 3 Red (R), 2 Green (G); Sweaters: 1 Black, 2 Red, 1 Green; Pants: 2 Denim, 2 Black.

Here, the number of compatible sweaters **depends** on which shirt you pick, so we cannot directly multiply shirt × sweater × pants. We must **split by cases** using a multiplication tree:

```
                     ┌──3──R──┬──3 sweaters──┬──4 pants
Shirts               │        │ (R, B)        │
                     ├──3──B──┼──4 sweaters──┼──4 pants
                     │        │ (R, B, G)     │
                     └──2──G──┴──2 sweaters──┴──4 pants
                                (B, G)
```

Multiplying down each path:

$$\underbrace{(3 \times 3 \times 4)}_{\text{Red shirt}} + \underbrace{(3 \times 4 \times 4)}_{\text{Black shirt}} + \underbrace{(2 \times 2 \times 4)}_{\text{Green shirt}} = 36 + 48 + 16 = \boxed{100}$$

---

## 6. Permutations and Combinations

### 6.1 Permutations

A **permutation** is an *ordered* arrangement of elements. **Order matters.**

The set $\{a, b, c\}$ has $3! = 6$ permutations:
$$abc,\ acb,\ bac,\ bca,\ cab,\ cba$$

**Why $3!$?** By the rule of product: 3 choices for position 1, then 2 choices for position 2, then 1 choice for position 3: $3 \times 2 \times 1 = 6$.

In general, the number of permutations of a set of $k$ elements is:
$$k! = k \cdot (k-1) \cdots 3 \cdot 2 \cdot 1$$

#### Permutations of $k$ things from $n$

> **Example:** List all permutations of 3 elements out of $\{a, b, c, d\}$.

$$\begin{array}{cccccc}
abc & acb & bac & bca & cab & cba \\
abd & adb & bad & bda & dab & dba \\
acd & adc & cad & cda & dac & dca \\
bcd & bdc & cbd & cdb & dbc & dcb \\
\end{array}$$

There are **24 permutations** (4 × 3 × 2 = 24, by the rule of product — no need to list them all).

---

### 6.2 Combinations

A **combination** is an *unordered* collection (subset) of elements. **Order does not matter.**

> **Example:** List all combinations of 3 elements out of $\{a, b, c, d\}$.

$$\{a,b,c\} \quad \{a,b,d\} \quad \{a,c,d\} \quad \{b,c,d\}$$

Only **4 combinations**, compared to **24 permutations**. The factor of $6 = 3!$ comes from the fact that every combination of 3 elements can be arranged in $3!$ different orders.

#### Permutations vs. Combinations — Side by Side

| Permutations (order matters) | Combination (order doesn't) |
|---|---|
| $abc,\ acb,\ bac,\ bca,\ cab,\ cba$ | $\{a,b,c\}$ |
| $abd,\ adb,\ bad,\ bda,\ dab,\ dba$ | $\{a,b,d\}$ |
| $acd,\ adc,\ cad,\ cda,\ dac,\ dca$ | $\{a,c,d\}$ |
| $bcd,\ bdc,\ cbd,\ cdb,\ dbc,\ dcb$ | $\{b,c,d\}$ |

Each row on the left shows all $3! = 6$ orderings of the corresponding set on the right.

---

### 6.3 Formulas Summary

$$\boxed{{}_{n}P_{k} = \frac{n!}{(n-k)!} = n(n-1)\cdots(n-k+1)}$$

$$\boxed{{}_{n}C_{k} = \binom{n}{k} = \frac{n!}{k!\,(n-k)!} = \frac{{}_{n}P_{k}}{k!}}$$

- ${}_{n}P_{k}$ = number of **ordered** lists of $k$ distinct elements from a set of size $n$
- ${}_{n}C_{k} = \binom{n}{k}$ = number of **subsets** of size $k$ from a set of size $n$
- Read $\binom{n}{k}$ as **"$n$ choose $k$"**

**The relationship:** ${}_{n}C_{k} = \frac{{}_{n}P_{k}}{k!}$ because each subset of $k$ elements can be arranged in $k!$ ways.

#### Quick Reference Examples

| Problem | Type | Formula | Answer |
|---|---|---|---|
| Choose 2 from 4 (unordered) | Combination | $\binom{4}{2} = \frac{4!}{2!\,2!}$ | $6$ |
| List 2 from 4 (ordered) | Permutation | ${}_{4}P_{2} = \frac{4!}{2!}$ | $12$ |
| Choose 3 from 10 (unordered) | Combination | $\binom{10}{3} = \frac{10 \cdot 9 \cdot 8}{3!}$ | $120$ |

---

## 7. Practice Problems & Solutions

### Problem 1 — DNA Sequences
> DNA is made of sequences of nucleotides: A, C, G, T.
> 1. How many DNA sequences of length 3 are there?
> 2. How many DNA sequences of length 3 are there **with no repeats**?

**Solution:**
1. Each position can be any of 4 nucleotides, independently: $4 \times 4 \times 4 = \mathbf{64}$
2. No repeats — each position must be different from all previous: $4 \times 3 \times 2 = \mathbf{24}$

---

### Problem 2 — The Band (Inclusion-Exclusion)
> A band consists of singers and guitar players: 7 people sing, 4 play guitar, 2 do both. How many people are in the band?

**Solution:**
$$|S \cup G| = 7 + 4 - 2 = \boxed{9}$$

---

### Problem 3 — Olympic Medals (Rule of Product)
> There are 5 competitors in the Olympics 100m final. How many ways can gold, silver, and bronze be awarded?

**Solution:**
$$5 \times 4 \times 3 = \boxed{60}$$

There are 5 ways to pick the gold medalist, then 4 remaining for silver, then 3 remaining for bronze.

---

### Problem 4 — Coin Flips (Combinations + Probability)
> (a) Count the number of ways to get exactly 3 heads in 10 flips of a coin.  
> (b) For a fair coin, what is the probability of exactly 3 heads in 10 flips?

**Solution:**

**(a)** We need to choose exactly which 3 out of 10 flips will be heads. Order of the positions doesn't matter (we just want to *which* positions are heads), so this is a combination:

$$\binom{10}{3} = \frac{10!}{3!\,7!} = \frac{10 \times 9 \times 8}{3 \times 2 \times 1} = \mathbf{120}$$

**(b)** By the rule of product, there are $2^{10} = 1024$ total possible outcomes from 10 coin flips. For a fair coin, each outcome is equally probable. So:

$$P(\text{exactly 3 heads}) = \frac{\binom{10}{3}}{2^{10}} = \frac{120}{1024} \approx \boxed{0.117}$$

---

### Problem 5 — Poker One-Pair (Intuition Test)
> A deck of 52 cards has 13 ranks and 4 suits. A **one-pair hand** consists of exactly two cards of one rank and three cards of three *different* other ranks (e.g., {2♡, 2♠, 5♡, 8♣, K♢}).  
> What is the probability of a one-pair hand?

**Intuition check answer: greater than 40%.**

This is surprisingly high! We'll be able to compute it exactly once we've built more tools, but the formula is:

$$P(\text{one pair}) = \frac{\underbrace{13}_{\text{rank for pair}} \times \underbrace{\binom{4}{2}}_{\text{suits}} \times \underbrace{\binom{12}{3}}_{\text{3 other ranks}} \times \underbrace{4^3}_{\text{suits for kickers}}}{\underbrace{\binom{52}{5}}_{\text{total 5-card hands}}} = \frac{1{,}098{,}240}{2{,}598{,}960} \approx 42.3\%$$

---

## Appendix: Key Formulas at a Glance

| Formula | Name | When to Use |
|---|---|---|
| $P(A) = k/n$ | Basic probability | $n$ equally likely outcomes, $k$ are favorable |
| $\|A \cup B\| = \|A\| + \|B\| - \|A \cap B\|$ | Inclusion-exclusion | Counting union of two sets |
| $n \times m$ | Rule of product | Sequential independent-count choices |
| $k!$ | Factorial | Permutations of all $k$ elements |
| ${}_{n}P_{k} = \frac{n!}{(n-k)!}$ | Permutations | Ordered selection of $k$ from $n$ |
| $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ | Combinations | Unordered selection of $k$ from $n$ |
| $(A \cup B)^c = A^c \cap B^c$ | DeMorgan's Law 1 | Simplifying set expressions |
| $(A \cap B)^c = A^c \cup B^c$ | DeMorgan's Law 2 | Simplifying set expressions |

---

## A Note on Two Schools of Thought

As you go through 18.05, keep this framing in mind:

| | Frequentist | Bayesian |
|---|---|---|
| **Probability is...** | Long-run frequency | Degree of belief / state of knowledge |
| **Parameters are...** | Fixed (unknown) constants | Random variables with their own distributions |
| **Inference uses...** | Only the observed data | Prior belief + observed data |
| **Output is...** | Point estimate, confidence interval, p-value | Posterior distribution |
| **Dominant in...** | Medicine, social sciences (historically) | Machine learning, big data, AI |

Neither school is universally "correct" — each has strengths. Understanding both is what makes 18.05 a genuine education in statistical thinking.

---

