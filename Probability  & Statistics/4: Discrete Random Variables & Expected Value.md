## 4: Discrete Random Variables & Expected Value


---

## Table of Contents

1. [Learning Goals](#1-learning-goals)
2. [Random Variables — Core Concepts](#2-random-variables--core-concepts)
3. [Probability Mass Function (PMF)](#3-probability-mass-function-pmf)
4. [Cumulative Distribution Function (CDF)](#4-cumulative-distribution-function-cdf)
5. [Named Discrete Distributions](#5-named-discrete-distributions)
6. [Arithmetic with Random Variables](#6-arithmetic-with-random-variables)
7. [Expected Value (Mean)](#7-expected-value-mean)
8. [Linearity of Expectation](#8-linearity-of-expectation)
9. [Expected Value of Functions of a Random Variable](#9-expected-value-of-functions-of-a-random-variable)
10. [In-Class Concept Questions — Full Solutions](#10-in-class-concept-questions--full-solutions)
11. [In-Class Board Problems — Full Solutions](#11-in-class-board-problems--full-solutions)
12. [Cognitive Biases: Gambler's Fallacy & Loss Aversion](#12-cognitive-biases-gamblers-fallacy--loss-aversion)
13. [Named Distribution Reference Table](#13-named-distribution-reference-table)
14. [Common Mistakes Reference](#14-common-mistakes-reference)
15. [Quick Summary & Formula Sheet](#15-quick-summary--formula-sheet)

---

## 1. Learning Goals

By the end of this class you should be able to:

1. **Define** a discrete random variable precisely as a function from a sample space to the real numbers.
2. **Know** the Bernoulli, Binomial, Geometric, and Uniform distributions and what real-world phenomena they model.
3. **Describe** a distribution using a probability mass function (pmf) table or formula, and a cumulative distribution function (cdf).
4. **Construct** new random variables from old ones using arithmetic operations.
5. **Compute** the expected value (mean) of any discrete random variable from its pmf.
6. **Know** the expected values of Bernoulli, Binomial, and Geometric random variables by heart.
7. **Apply** linearity of expectation to simplify complex calculations.
8. **Understand** why $E[h(X)] \neq h(E[X])$ in general.

---

## 2. Random Variables — Core Concepts

### 2.1 Concept Overview

A **random variable** is a bridge between the abstract world of sample spaces (outcomes of experiments) and the real number line. It assigns a numerical value to each outcome so we can do arithmetic with probability.

Think of it as a **payoff function**: each outcome $\omega$ in the sample space $\Omega$ maps to a number $X(\omega)$ — typically a measurement, score, count, or dollar amount.

> **Key insight for ML/AI:** Every loss function you minimise, every prediction your model makes, every accuracy score — these are all random variables. Understanding their distributions is the foundation of statistical learning theory.

---

### 2.2 Intuition

Imagine rolling two dice. The sample space is all 36 pairs $(i, j)$. Now suppose you care about *how much money you win*. The payoff (say, $500 if sum = 7, else $-100$) is a function that converts each outcome into a dollar amount. That function is a random variable.

The word "random" refers to the fact that which outcome $\omega$ occurs is random. The word "variable" reflects that we treat $X$ algebraically — we add random variables, square them, compare them, just like ordinary variables.

---

### 2.3 Formal Definition

> **Definition (Discrete Random Variable):** Let $\Omega$ be a sample space. A **discrete random variable** is a function
> $$X : \Omega \to \mathbb{R}$$
> that takes a discrete (finite or countably infinite) set of values.

**Important:** $X$ is a *function*, not a number. It becomes a specific number only after the experiment produces an outcome $\omega$.

---

### 2.4 Events Defined by Random Variables

For any value $a$, the notation $X = a$ denotes the **event** — the subset of $\Omega$ — consisting of all outcomes that map to $a$:

$$\{X = a\} = \{\omega \in \Omega : X(\omega) = a\}$$

Similarly, inequalities define events:

$$\{X \leq a\} = \{\omega \in \Omega : X(\omega) \leq a\}$$

This lets us compute probabilities like $P(X = a)$ and $P(X \leq a)$.

---

### 2.5 Worked Examples — Random Variables as Payoff Functions

---

#### Example 1 — Dice Game with Payoff Function

**Problem:** Roll two dice. Record the outcome as $(i, j)$. You win $500 if the sum is 7 and lose $100 otherwise. Define $X$ to be the payoff.

**Step 1: Write the formal definition of $X$.**

$$X(i, j) = \begin{cases} 500 & \text{if } i + j = 7 \\ -100 & \text{if } i + j \neq 7 \end{cases}$$

**Step 2: Identify the sample space and probabilities.**

$$\Omega = \{(i,j) \mid i, j \in \{1,2,3,4,5,6\}\}, \quad |\Omega| = 36, \quad P(i,j) = \frac{1}{36}$$

**Step 3: Find $P(X = 500)$.**

The event $\{X = 500\} = \{(1,6),(2,5),(3,4),(4,3),(5,2),(6,1)\}$, which has 6 outcomes.

$$P(X = 500) = \frac{6}{36} = \frac{1}{6}$$

**Step 4: Find $P(X = -100)$.**

$$P(X = -100) = 1 - \frac{1}{6} = \frac{5}{6}$$

**Interpretation:** The game has two possible financial outcomes — winning $500 (probability 1/6) or losing $100 (probability 5/6). We'll compute whether this game is worth playing once we learn expected value.

---

#### Example 2 — Another Payoff Function

**Problem:** Same two-dice setup, but now $Y(i,j) = ij - 10$.

If you roll $(6, 2)$: $Y = 12 - 10 = 2$, so you win $2.  
If you roll $(2, 3)$: $Y = 6 - 10 = -4$, so you lose $4.

**Which game ($X$ or $Y$) is the better bet?** We'll answer this using expected value in Section 7.

---

#### Example 3 — Maximum of Two Dice

**Problem:** Let $M(i, j) = \max(i, j)$, the maximum value showing on two dice.

**Step 1: Find the pmf of $M$.**

$M$ takes values $1, 2, 3, 4, 5, 6$. How many pairs have maximum exactly $k$?

- Maximum is exactly $k$ when: at least one die shows $k$, and no die shows more than $k$.
- Count of pairs $(i,j)$ where $\max(i,j) \leq k$: $k^2$ pairs.
- Count of pairs where $\max(i,j) \leq k-1$: $(k-1)^2$ pairs.
- So count of pairs where $\max(i,j) = k$: $k^2 - (k-1)^2 = 2k - 1$.

| $a$ | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| $p(a) = P(M=a)$ | 1/36 | 3/36 | 5/36 | 7/36 | 9/36 | 11/36 |

**Pattern:** $p(k) = (2k-1)/36$ — an arithmetic sequence of odd numbers. This makes intuitive sense: more ways to achieve a higher maximum.

**Verification:** $1+3+5+7+9+11 = 36$. ✓

---

## 3. Probability Mass Function (PMF)

### 3.1 Concept Overview

The **probability mass function** (pmf) is the complete probability specification for a discrete random variable. It tells you the probability of every possible value.

> **Intuition:** The pmf is like a "probability histogram" — it distributes the total probability mass of 1 across the possible values of $X$.

---

### 3.2 Formal Definition

> **Definition (PMF):** The **probability mass function** of a discrete random variable $X$ is the function:
> $$p(a) = P(X = a)$$
> We may also write $p_X(a)$ to make explicit which random variable we mean.

**Required properties:**

1. $0 \leq p(a) \leq 1$ for all $a$
2. $p(a) = 0$ for any value $a$ that $X$ never takes
3. $\sum_{\text{all } a} p(a) = 1$ (total probability sums to 1)

---

### 3.3 Key Formulas

$$\boxed{p(a) = P(X = a)}$$

For any event defined by $X$:
$$P(a < X \leq b) = \sum_{a < x \leq b} p(x)$$

---

### 3.4 Worked Example — PMF from Sample Space

#### Example 4 — PMF of Maximum (continued)

**Problem:** Compute the full pmf table for $M = \max(i,j)$ on two dice, and find $p(8)$.

| Value $a$ | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| pmf $p(a)$ | 1/36 | 3/36 | 5/36 | 7/36 | 9/36 | 11/36 |

$p(8) = 0$ — since two standard dice cannot produce a maximum of 8.

**Also:** What is the pmf of the sum $Z(i,j) = i + j$?

The sum ranges from 2 to 12. This should look familiar — it's a triangle-shaped distribution peaked at 7:

| Value $a$ | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pmf $p(a)$ | 1/36 | 2/36 | 3/36 | 4/36 | 5/36 | 6/36 | 5/36 | 4/36 | 3/36 | 2/36 | 1/36 |

---

## 4. Cumulative Distribution Function (CDF)

### 4.1 Concept Overview

The **cumulative distribution function** (cdf) answers a different but equally important question: "What is the probability that $X$ takes a value at most $a$?" It accumulates probability as $a$ increases from $-\infty$ to $+\infty$.

> **Intuition:** Imagine pouring probability "mass" from left to right along the number line. The cdf at any point $a$ is the total mass that has accumulated up to and including that point.

---

### 4.2 Formal Definition

> **Definition (CDF):** The **cumulative distribution function** of a random variable $X$ is:
> $$F(a) = P(X \leq a)$$
> Note carefully: the definition uses **$\leq$** (less than or equal), not $<$ (strictly less than). This distinction is critical for discrete random variables.

---

### 4.3 Key Formulas

$$\boxed{F(a) = P(X \leq a) = \sum_{x \leq a} p(x)}$$

**Recovering the pmf from the cdf:**

$$p(a) = F(a) - F(a^-) = F(a) - \lim_{x \to a^-} F(x)$$

For discrete random variables with values $x_1 < x_2 < \cdots$:

$$p(x_k) = F(x_k) - F(x_{k-1})$$

**Probability of an interval:**

$$P(a < X \leq b) = F(b) - F(a)$$

---

### 4.4 Properties of the CDF

Every valid cdf satisfies these three properties:

1. **Non-decreasing:** If $a \leq b$, then $F(a) \leq F(b)$.  
   *Why:* Adding more probability mass can only keep $F$ flat or increase it, never decrease it.

2. **Bounded:** $0 \leq F(a) \leq 1$ for all $a$.  
   *Why:* $F(a)$ is a probability.

3. **Limits:**
$$\lim_{a \to +\infty} F(a) = 1, \qquad \lim_{a \to -\infty} F(a) = 0$$
   *Why:* For very large $a$, $X \leq a$ is certain. For very negative $a$, $X \leq a$ is impossible.

> **Warning:** For discrete CDFs, the graph is a **step function** — it jumps at each value of $X$ and is flat (constant) between values. It is **right-continuous** (closed dots on the right of each horizontal segment).

---

### 4.5 Worked Examples

---

#### Example 5 — CDF of Maximum of Two Dice

Building the cdf of $M = \max(i,j)$ by accumulating the pmf:

| Value $a$ | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| pmf $p(a)$ | 1/36 | 3/36 | 5/36 | 7/36 | 9/36 | 11/36 |
| cdf $F(a)$ | 1/36 | 4/36 | 9/36 | 16/36 | 25/36 | 36/36 |

**Verification of the pattern:** $F(k) = P(M \leq k) = P(\text{both dice} \leq k) = (k/6)^2 = k^2/36$. ✓ (This is the "square" pattern: $1, 4, 9, 16, 25, 36$.)

**Key CDF calculations:**
- $F(8) = 1$ (the max of two standard dice is always $\leq 8$)
- $F(-2) = 0$ (the max can never be $\leq -2$)
- $F(2.5) = F(2) = 4/36$ (the cdf is constant between integer values)
- $F(\pi) = F(3) = 9/36$ (since $3 < \pi < 4$, and $M$ takes only integer values)

---

#### Example 6 — PMF and CDF of Number of Heads in 3 Coin Flips

**Setup:** $X$ = number of heads in 3 fair coin flips.

| Value $a$ | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| pmf $p(a)$ | 1/8 | 3/8 | 3/8 | 1/8 |
| cdf $F(a)$ | 1/8 | 4/8 | 7/8 | 1 |

**How to read the CDF graph:** The cdf is a staircase. It starts at 0 for $a < 0$, jumps to 1/8 at $a=0$, jumps to 4/8 at $a=1$, jumps to 7/8 at $a=2$, and reaches 1 at $a=3$. Between jumps it is flat.

**Computing $P(1 \leq X \leq 2)$:**
$$P(1 \leq X \leq 2) = F(2) - F(0) = \frac{7}{8} - \frac{1}{8} = \frac{6}{8} = \frac{3}{4}$$

Or directly: $p(1) + p(2) = 3/8 + 3/8 = 6/8$. ✓

---

## 5. Named Discrete Distributions

### 5.1 Bernoulli Distribution

#### Concept Overview

The **Bernoulli distribution** is the simplest possible distribution — it models a single trial with only two outcomes: **success** (1) or **failure** (0). It is the atomic building block from which many other distributions are constructed.

> **Examples in ML:** A single classification prediction (correct/incorrect), a single A/B test user conversion (converted/not), a single bit in a transmission (0/1).

---

#### Formal Definition

> **$X \sim \text{Bernoulli}(p)$** (also written $\text{Ber}(p)$) if:
> - $X$ takes only the values 0 and 1.
> - $P(X = 1) = p$ (success probability)
> - $P(X = 0) = 1 - p$ (failure probability)

**PMF table:**

| Value $a$ | 0 | 1 |
|---|---|---|
| pmf $p(a)$ | $1-p$ | $p$ |
| cdf $F(a)$ | $1-p$ | 1 |

**Key formulas:**

$$\boxed{E[X] = p} \qquad \text{(memorize this!)}$$

$$\text{Var}(X) = p(1-p)$$

#### Intuition

Think of $X$ as the outcome of flipping a biased coin: heads (1) with probability $p$, tails (0) with probability $1-p$. The indicator function $\mathbf{1}[\text{event } A]$ — which equals 1 if $A$ occurs and 0 otherwise — is a Bernoulli random variable with $p = P(A)$.

> **Power of indicator variables:** Almost any counting problem can be decomposed into a sum of Bernoulli indicators. This is the key technique behind the Musical Chairs problem (Section 11).

---

### 5.2 Binomial Distribution

#### Concept Overview

The **Binomial distribution** counts the number of successes in $n$ independent, identical Bernoulli trials. It is the most widely used discrete distribution in statistics.

> **Examples:** Number of heads in $n$ coin flips; number of defective items in a batch of $n$; number of customers who click on an ad out of $n$ shown.

---

#### Formal Definition

> **$X \sim \text{Binomial}(n, p)$** (also written $\text{Bin}(n, p)$) if $X$ is the number of successes in $n$ independent $\text{Bernoulli}(p)$ trials.

**PMF formula:** For $k = 0, 1, 2, \ldots, n$:

$$\boxed{p(k) = P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}}$$

where the **binomial coefficient** is:

$$\binom{n}{k} = \frac{n!}{k!(n-k)!}$$

**Full PMF table for $X \sim \text{Binomial}(n, p)$:**

| Value $a$ | 0 | 1 | 2 | $\cdots$ | $k$ | $\cdots$ | $n$ |
|---|---|---|---|---|---|---|---|
| pmf $p(a)$ | $(1-p)^n$ | $\binom{n}{1}p(1-p)^{n-1}$ | $\binom{n}{2}p^2(1-p)^{n-2}$ | $\cdots$ | $\binom{n}{k}p^k(1-p)^{n-k}$ | $\cdots$ | $p^n$ |

**Key formulas:**

$$\boxed{E[X] = np}$$

$$\text{Var}(X) = np(1-p)$$

---

#### Why the Binomial Formula Works

The derivation of the binomial pmf has two beautiful components:

**Step 1: Probability of one specific sequence with $k$ successes.**  
Any specific sequence with exactly $k$ successes and $n-k$ failures (e.g., SSFFSF...) has probability:

$$p^k \cdot (1-p)^{n-k}$$

because each success has probability $p$, each failure has probability $1-p$, and the trials are independent (so we multiply).

**Step 2: Count the number of such sequences.**  
We need to choose which $k$ of the $n$ trials are successes. The number of ways to do this is $\binom{n}{k}$.

**Step 3: Total probability = count × single-sequence probability:**

$$P(X = k) = \binom{n}{k} \cdot p^k (1-p)^{n-k}$$

---

#### Worked Example — Binomial(5, p)

**Problem:** $X \sim \text{Binomial}(5, p)$. Write the full PMF table.

The binomial coefficients for $n = 5$ are:
$$\binom{5}{0}=1, \quad \binom{5}{1}=5, \quad \binom{5}{2}=10, \quad \binom{5}{3}=10, \quad \binom{5}{4}=5, \quad \binom{5}{5}=1$$

| Value $a$ | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| pmf $p(a)$ | $(1-p)^5$ | $5p(1-p)^4$ | $10p^2(1-p)^3$ | $10p^3(1-p)^2$ | $5p^4(1-p)$ | $p^5$ |

---

#### Worked Example — P(3 or more heads in 5 fair coin flips)

**Problem:** $X \sim \text{Binomial}(5, 1/2)$. Find $P(X \geq 3)$.

**Step 1:** Apply the formula with $p = 1/2$.

$$P(X \geq 3) = P(X=3) + P(X=4) + P(X=5)$$

$$= 10\left(\frac{1}{2}\right)^3\left(\frac{1}{2}\right)^2 + 5\left(\frac{1}{2}\right)^4\left(\frac{1}{2}\right)^1 + \left(\frac{1}{2}\right)^5$$

$$= \frac{10}{32} + \frac{5}{32} + \frac{1}{32} = \frac{16}{32} = \frac{1}{2}$$

**Why is 1/2 not surprising?** By symmetry, $P(X \geq 3) = P(X \leq 2)$ when $p = 1/2$ and $n = 5$ (since $3 + 2 = 5$ and the distribution is symmetric around 2.5). So both must equal 1/2.

---

#### Relationship: Binomial is a Sum of Bernoullis

This is one of the most important structural facts in probability:

> **Key fact:** If $X_1, X_2, \ldots, X_n$ are independent $\text{Bernoulli}(p)$ random variables, then:
> $$X = X_1 + X_2 + \cdots + X_n \sim \text{Binomial}(n, p)$$

Each $X_i = 1$ if the $i$-th trial is a success, $X_i = 0$ otherwise. The sum counts the total number of successes. This decomposition is why linearity of expectation gives $E[X] = np$ so elegantly (see Section 8).

---

#### Shape of the Binomial Distribution

- $\text{Binomial}(10, 0.5)$: **Symmetric**, bell-shaped, peaked at 5.
- $\text{Binomial}(10, 0.1)$: **Right-skewed**, peaked at 1, fast decay toward 10.
- $\text{Binomial}(20, 0.1)$: **Right-skewed**, more spread out, peaked around 2.

As $n \to \infty$ with $p$ fixed, the Binomial distribution converges to a Normal distribution (Central Limit Theorem — a later topic).

---

### 5.3 Geometric Distribution

#### Concept Overview

The **Geometric distribution** models the number of "tails" before the first "heads" in a sequence of independent coin flips. It answers the question: "How long do I have to wait before my first success?"

> **Examples in ML/engineering:** Number of attempts before a neural network converges; number of packets dropped before a successful transmission; number of customers who decline before one accepts.

---

#### Formal Definition

> **$X \sim \text{Geometric}(p)$** (also written $\text{geo}(p)$) if $X$ is the number of failures before the first success in a sequence of independent $\text{Bernoulli}(p)$ trials.

**PMF formula:** For $k = 0, 1, 2, 3, \ldots$:

$$\boxed{p(k) = P(X = k) = (1-p)^k \cdot p}$$

**PMF table:**

| Value $a$ | 0 | 1 | 2 | 3 | $\cdots$ | $k$ | $\cdots$ |
|---|---|---|---|---|---|---|---|
| pmf $p(a)$ | $p$ | $(1-p)p$ | $(1-p)^2 p$ | $(1-p)^3 p$ | $\cdots$ | $(1-p)^k p$ | $\cdots$ |

**Key formulas:**

$$\boxed{E[X] = \frac{1-p}{p}}$$

$$\text{Var}(X) = \frac{1-p}{p^2}$$

---

#### Why the Geometric PMF Makes Sense

To get $X = k$ (exactly $k$ failures before the first success), the sequence must look like:

$$\underbrace{F, F, \ldots, F}_{k \text{ failures}}, S$$

The probability of this specific sequence is:

$$\underbrace{(1-p) \cdot (1-p) \cdots (1-p)}_{k \text{ terms}} \cdot p = (1-p)^k \cdot p$$

There is exactly **one** such sequence (the failures must all come first), so no counting coefficient is needed. This contrasts with the binomial, where failures and successes can be in any order.

---

#### Verification: Probabilities Sum to 1

$$\sum_{k=0}^{\infty} (1-p)^k p = p \sum_{k=0}^{\infty} (1-p)^k = p \cdot \frac{1}{1-(1-p)} = p \cdot \frac{1}{p} = 1 \checkmark$$

This uses the geometric series formula: $\sum_{k=0}^{\infty} x^k = \frac{1}{1-x}$ for $|x| < 1$.

---

#### Worked Example — Island Family Planning

**Problem:** Families on an island have children until their first daughter. Assume $P(\text{girl}) = 0.5$ independently. What is the probability a family has exactly $k$ boys?

**Step 1: Set up the model.**

Boys = "failures" (tails), girls = "successes" (heads). Number of boys before first girl: $X \sim \text{Geometric}(0.5)$.

**Step 2: Compute the PMF.**

$$P(X = k) = \left(\frac{1}{2}\right)^k \cdot \frac{1}{2} = \left(\frac{1}{2}\right)^{k+1}$$

**Verification:** $P(\text{BBBG}) = (1/2)^3 \cdot (1/2) = (1/2)^4 = 1/16$. Directly: 3 boys then 1 girl. ✓

**Think — What is the ratio of boys to girls on the island?**

$E[X] = (1-0.5)/0.5 = 1$. Each family has on average 1 boy and exactly 1 girl. So the expected ratio of boys to girls is **1:1**, even though families stop at the first girl! The island's population has equal sex ratios regardless of this stopping rule. This is a beautiful and counterintuitive result.

---

#### Alternative Definition (Common Confusion)

Some textbooks define the Geometric distribution as the number of **trials** (not failures) until the first success. In that case $X$ takes values $1, 2, 3, \ldots$ and the pmf is $p(k) = (1-p)^{k-1}p$.

This is just our Geometric random variable **plus 1**. Always check which convention a resource uses.

---

#### Expected Value of the Geometric Distribution — Derivation

Let $X \sim \text{geo}(p)$. We want $E[X] = \sum_{k=0}^{\infty} k(1-p)^k p$.

**The trick uses differentiation of the geometric series:**

**Step 1:** Geometric series formula:
$$\sum_{k=0}^{\infty} x^k = \frac{1}{1-x} \quad \text{for } |x| < 1$$

**Step 2:** Differentiate both sides with respect to $x$:
$$\sum_{k=0}^{\infty} k x^{k-1} = \frac{1}{(1-x)^2}$$

**Step 3:** Multiply both sides by $x$:
$$\sum_{k=0}^{\infty} k x^k = \frac{x}{(1-x)^2}$$

**Step 4:** Substitute $x = 1-p$:
$$\sum_{k=0}^{\infty} k(1-p)^k = \frac{1-p}{p^2}$$

**Step 5:** Multiply both sides by $p$:
$$E[X] = \sum_{k=0}^{\infty} k(1-p)^k p = \frac{1-p}{p}$$

---

#### Worked Example — Michael Jordan's Free Throws

**Problem:** Michael Jordan made 80% of his free throws. What is the expected number of free throws he makes before his first miss?

**Step 1: Identify the model.**

"Success" = makes a free throw (probability $1 - p$ in neutral language).  
"Failure" = misses (probability $p$).

We want: number of makes (tails) before first miss (heads).  
Using the neutral Geometric model: $p_{\text{miss}} = 0.2$, $p_{\text{make}} = 0.8$.

**Step 2: Apply the formula.**

$$E[X] = \frac{1 - p_{\text{miss}}}{p_{\text{miss}}} = \frac{0.8}{0.2} = 4$$

**Answer:** On average, Jordan makes **4 free throws** before his first miss.

---

### 5.4 Uniform Distribution

#### Formal Definition

> **$X \sim \text{Uniform}(N)$** if $X$ takes values $1, 2, 3, \ldots, N$, each with probability $1/N$.

This models any situation where all outcomes are equally likely. We've already used this distribution many times:

| Application | $N$ |
|---|---|
| Fair coin | 2 |
| Fair die | 6 |
| Day of year (birthdays) | 365 |
| 5-card poker hand | $\binom{52}{5} = 2{,}598{,}960$ |

**Key formulas:**

$$E[X] = \frac{N+1}{2}, \qquad \text{Var}(X) = \frac{N^2 - 1}{12}$$

---

## 6. Arithmetic with Random Variables

### 6.1 Concept Overview

Random variables can be combined algebraically: added, subtracted, multiplied, or transformed by functions. The result is a new random variable. This is fundamental to building complex probability models from simple components.

---

### 6.2 The Indicator Trick (Counting via Bernoulli Variables)

> **Key idea:** Any sequence of 0s and 1s sums to the count of 1s.

If $X_1, X_2, \ldots, X_n$ each take only values 0 or 1, then:
$$X_1 + X_2 + \cdots + X_n = \text{(number of } X_i \text{ equal to 1)}$$

This is trivial to verify but enormously powerful: **any counting problem can be solved by expressing the count as a sum of Bernoulli indicator variables**.

---

### 6.3 Worked Example — Sum of Independent Random Variables

#### Example 7 — Joint Table for $X + Y$

**Problem:** $X$ and $Y$ are independent with the following pmfs:

| $x$ | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| $p_X(x)$ | 1/10 | 2/10 | 3/10 | 4/10 |

| $y$ | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| $p_Y(y)$ | 1/15 | 2/15 | 3/15 | 4/15 | 5/15 |

**Verification:** $\sum p_X = 1/10+2/10+3/10+4/10 = 10/10 = 1$ ✓ and $\sum p_Y = 1/15+2/15+3/15+4/15+5/15 = 15/15 = 1$ ✓

**Step 1: Build the joint probability table.**

Since $X$ and $Y$ are independent, $P(X = x, Y = y) = p_X(x) \cdot p_Y(y)$.

| $X \backslash Y$ | $y=1$ | $y=2$ | $y=3$ | $y=4$ | $y=5$ | $p_X(x)$ |
|---|---|---|---|---|---|---|
| $x=1$ | 1/150 | 2/150 | 3/150 | 4/150 | 5/150 | 1/10 |
| $x=2$ | 2/150 | 4/150 | 6/150 | 8/150 | 10/150 | 2/10 |
| $x=3$ | 3/150 | 6/150 | 9/150 | 12/150 | 15/150 | 3/10 |
| $x=4$ | 4/150 | 8/150 | 12/150 | 16/150 | 20/150 | 4/10 |
| $p_Y(y)$ | 1/15 | 2/15 | 3/15 | 4/15 | 5/15 | |

**Step 2: Sum the anti-diagonals** (all cells where $X + Y = s$).

For each value $s$ of $X + Y$, sum probabilities along the anti-diagonal where $x + y = s$:

| $X+Y$ value | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|
| pmf | 1/150 | 4/150 | 10/150 | 20/150 | 30/150 | 34/150 | 31/150 | 20/150 |

**Verification:** $1+4+10+20+30+34+31+20 = 150$. Divided by 150: sum is 1. ✓

**Interpretation:** The distribution of the sum is more concentrated around the middle values (5, 6, 7), similar to how the sum of two dice is most likely to be 7. The heavier weighting of larger values in both $X$ and $Y$ shifts the peak toward 6-7.

---

## 7. Expected Value (Mean)

### 7.1 Concept Overview

The **expected value** (also called the **mean** or **average**) of a random variable is a single number that summarises the "centre" or "typical value" of the distribution. It is a probability-weighted average of all possible values.

> **Key insight for ML:** Expected value is everywhere in machine learning — expected loss, expected reward (reinforcement learning), expected accuracy. Minimising expected loss is the objective of almost every learning algorithm.

---

### 7.2 Intuition — Why This Formula?

Suppose you play a game 1000 times. If value $x_j$ occurs with probability $p(x_j)$, then it occurs about $1000 \cdot p(x_j)$ times.

The total sum of all outcomes is approximately:
$$\sum_j x_j \cdot [1000 \cdot p(x_j)]$$

Dividing by 1000 to get the average per game:
$$\text{average} \approx \sum_j x_j \cdot p(x_j)$$

This is exactly the definition of expected value. As the number of games $\to \infty$, this approximation becomes exact (the **Law of Large Numbers**).

---

### 7.3 Formal Definition

> **Definition (Expected Value):** If $X$ is a discrete random variable with values $x_1, x_2, \ldots$ and pmf $p$, the **expected value** is:
>
> $$\boxed{E[X] = \sum_j p(x_j) \cdot x_j}$$
>
> Other notations: $\mu_X$, $\mu$, $\langle X \rangle$.

**Important notes:**

1. $E[X]$ need not be a possible value of $X$ (e.g., expected value of a die roll is 3.5, which isn't a face value).
2. $E[X]$ is a **weighted average** where the weights are probabilities.
3. $E[X]$ is a **summary statistic** — it tells you the "location" or "centre" of the distribution.
4. For equally likely values $x_1, \ldots, x_n$: $E[X] = \frac{x_1 + x_2 + \cdots + x_n}{n}$ (ordinary average).

---

### 7.4 The Centre of Mass Interpretation

Place a "mass" of $p(x_j)$ at position $x_j$ on the number line for each $j$. The expected value $E[X]$ is the position of the **centre of mass** — the balance point of this distribution of masses.

**Physics analogy:** Two masses $m_1 = 500$ at position $x_1 = 3$ and $m_2 = 100$ at position $x_2 = 6$:

$$\bar{x} = \frac{m_1 x_1 + m_2 x_2}{m_1 + m_2} = \frac{500 \cdot 3 + 100 \cdot 6}{600} = \frac{2100}{600} = 3.5$$

**Connection to expected value:** With $p(3) = 5/6$ and $p(6) = 1/6$, we get $E[X] = 3 \cdot \frac{5}{6} + 6 \cdot \frac{1}{6} = 3.5$.

The "probability mass function" is so named precisely because of this centre-of-mass interpretation.

---

### 7.5 Worked Examples

---

#### Example 8 — Computing Expected Value from a Table

**Problem:** $X$ has pmf:

| Value $a$ | 1 | 3 | 5 |
|---|---|---|---|
| pmf $p(a)$ | 1/6 | 1/6 | 2/3 |

Find $E[X]$.

**Step 1:** Verify total probability: $1/6 + 1/6 + 2/3 = 1/6 + 1/6 + 4/6 = 6/6 = 1$. ✓

**Step 2:** Apply the definition:

$$E[X] = 1 \cdot \frac{1}{6} + 3 \cdot \frac{1}{6} + 5 \cdot \frac{2}{3} = \frac{1}{6} + \frac{3}{6} + \frac{10}{6} = \frac{14}{6} \cdot \frac{6}{6}$$

Wait — let me redo: $5 \cdot (2/3) = 10/3 = 20/6$. So:

$$E[X] = \frac{1}{6} + \frac{3}{6} + \frac{20}{6} = \frac{24}{6} = 4$$

**Final Answer:** $E[X] = 4$

**Sanity check:** 4 is between the smallest value (1) and largest value (5), and closer to 5 since 5 has the most probability mass. ✓

---

#### Example 9 — Expected Value of a Bernoulli Random Variable

**Problem:** Let $X \sim \text{Bernoulli}(p)$. Find $E[X]$.

**Solution:**

$$E[X] = 0 \cdot (1-p) + 1 \cdot p = p$$

**This is one of the most important results in probability. Memorise it:**

$$\boxed{E[\text{Bernoulli}(p)] = p}$$

**Interpretation:** On average, a Bernoulli trial produces the value $p$. Since the trial equals either 0 or 1, this means $p$ is the long-run fraction of 1s (successes).

---

#### Example 10 — Average Roll of a Loaded Die

**Problem:** A die has five 3s and one 6. What is the expected average over many rolls?

| Value | 3 | 6 |
|---|---|---|
| Expected frequency per 6 rolls | 5 | 1 |
| Probability | 5/6 | 1/6 |

$$E[X] = 3 \cdot \frac{5}{6} + 6 \cdot \frac{1}{6} = \frac{15}{6} + \frac{6}{6} = \frac{21}{6} = 3.5$$

**Interpretation:** Despite the die being heavily loaded toward 3, the average is 3.5 — the same as a fair die! The single 6 contributes enough to balance out.

---

#### Example 11 — Expected Winnings in a Casino Game

**Problem:** Roll two standard dice. Win $1000 if sum = 2, lose $100 otherwise. Expected winnings per game?

**Step 1:** Find probabilities.

$P(\text{sum} = 2) = P(1,1) = 1/36$ and $P(\text{sum} \neq 2) = 35/36$.

**Step 2:** Apply the definition.

$$E[\text{winnings}] = 1000 \cdot \frac{1}{36} + (-100) \cdot \frac{35}{36} = \frac{1000 - 3500}{36} = \frac{-2500}{36} \approx -\$69.44$$

**Interpretation:** You **lose** about $69.44 per game on average. Would you play this game? Once might be fun, but repeatedly it's financially ruinous. This is a classic casino structure — small chance of big win, large chance of moderate loss, and the house always wins in expectation.

---

#### Example 12 — Expected Value of the Geometric Distribution

**Problem:** Flip a fair coin until the first heads. What is the expected number of tails you flip?

**Model:** $X \sim \text{Geometric}(1/2)$.

$$E[X] = \frac{1 - 1/2}{1/2} = \frac{1/2}{1/2} = 1$$

**Interpretation:** You expect to flip just **1 tail** before the first heads. This might seem small — but intuitively: half the time you get heads immediately (0 tails), a quarter of the time you get exactly 1 tail, etc. The average really is 1.

---

#### Example 13 — Expected Value of $X^2$ for a Die Roll

**Problem:** Roll one fair die. Let $Y = X^2$. Find $E[Y]$.

| $X$ | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| $Y = X^2$ | 1 | 4 | 9 | 16 | 25 | 36 |
| $P$ | 1/6 | 1/6 | 1/6 | 1/6 | 1/6 | 1/6 |

$$E[Y] = E[X^2] = \frac{1+4+9+16+25+36}{6} = \frac{91}{6} \approx 15.167$$

**Important:** $E[X] = 3.5$ but $E[X^2] = 15.167 \neq (3.5)^2 = 12.25$.  
This demonstrates that $E[h(X)] \neq h(E[X])$ in general!

---

#### Example 14 — Sum of Two Dice (Using Linearity)

**Problem:** Roll two dice. Let $X$ = sum. Find $E[X]$.

**Method 1 (Direct, hard):** Use the pmf of the sum from Section 3.4.

**Method 2 (Linearity, easy):**

$$X = X_1 + X_2 \implies E[X] = E[X_1] + E[X_2] = 3.5 + 3.5 = 7$$

The expected value of a single fair die is $E[X_i] = (1+2+3+4+5+6)/6 = 21/6 = 3.5$.

---

#### Example 15 — Non-Existent Expected Value (St. Petersburg Paradox)

**Problem:** $X$ has pmf $p(2^k) = 1/2^k$ for $k = 1, 2, 3, \ldots$. Find $E[X]$.

| Value $x$ | 2 | 4 | 8 | $2^k$ | $\cdots$ |
|---|---|---|---|---|---|
| pmf $p(x)$ | 1/2 | 1/4 | 1/8 | $1/2^k$ | $\cdots$ |

$$E[X] = \sum_{k=1}^{\infty} 2^k \cdot \frac{1}{2^k} = \sum_{k=1}^{\infty} 1 = \infty$$

**The expected value does not exist** (is infinite)! This is the **St. Petersburg Paradox**: a game where you toss a coin until heads and win $2^k$ dollars if heads first appears on toss $k$ has infinite expected value, yet no rational person would pay more than a small amount to play it. This reveals the limitation of expected value as a decision criterion, motivating utility theory and variance considerations.

---

## 8. Linearity of Expectation

### 8.1 Concept Overview

Linearity of expectation is arguably the **most powerful tool** in probability calculations. It states that expected values add linearly — regardless of whether the random variables are independent.

> **Key insight:** Unlike probabilities (which only multiply for independent events), expected values **always** add, even for dependent variables. This is what makes the Musical Chairs calculation (Problem 3) so elegant.

---

### 8.2 Formal Statement

> **Theorem (Linearity of Expectation):**
>
> **Property 1:** For any random variables $X$ and $Y$ defined on the same sample space:
> $$\boxed{E[X + Y] = E[X] + E[Y]}$$
>
> **Property 2:** For constants $a$ and $b$:
> $$\boxed{E[aX + b] = aE[X] + b}$$
>
> **Important:** Property 1 holds regardless of whether $X$ and $Y$ are independent.

---

### 8.3 Proof of Property 1

Let the sample space be $\Omega = \{\omega_1, \omega_2, \ldots, \omega_n\}$ with $X(\omega_i) = x_i$ and $Y(\omega_i) = y_i$.

$X + Y$ is the random variable that equals $x_i + y_i$ when outcome $\omega_i$ occurs.

$$E[X + Y] = \sum_{i=1}^n (x_i + y_i) P(\omega_i) = \sum_{i=1}^n x_i P(\omega_i) + \sum_{i=1}^n y_i P(\omega_i) = E[X] + E[Y] \quad \square$$

**Key subtlety:** Note that we add the values of $X$ and $Y$ *for the same outcome $\omega_i$*. This is what it means to add random variables, and it doesn't require independence.

---

### 8.4 Proof of Property 2

$$E[aX + b] = \sum_i p(x_i)(ax_i + b) = a\sum_i p(x_i)x_i + b\sum_i p(x_i) = aE[X] + b \cdot 1 = aE[X] + b \quad \square$$

The last step uses the fact that probabilities sum to 1.

---

### 8.5 Expected Value of the Binomial (via Linearity)

**Problem:** $X \sim \text{Binomial}(n, p)$. Find $E[X]$.

**The hard way:** Compute $\sum_{k=0}^n k \binom{n}{k} p^k (1-p)^{n-k}$ directly. (Tedious algebra.)

**The elegant way using linearity:**

Write $X = X_1 + X_2 + \cdots + X_n$ where each $X_j \sim \text{Bernoulli}(p)$.

$$E[X] = E\left[\sum_{j=1}^n X_j\right] = \sum_{j=1}^n E[X_j] = \sum_{j=1}^n p = np$$

$$\boxed{E[\text{Binomial}(n,p)] = np}$$

This derivation is a template: **decompose a complex random variable into a sum of simple Bernoulli indicators, then apply linearity**.

---

## 9. Expected Value of Functions of a Random Variable

### 9.1 The Change of Variables Formula

If $Y = h(X)$ where $h$ is any function, then $Y$ is a new random variable. We can compute its expected value using the **change of variables formula** (also called LOTUS — Law of the Unconscious Statistician):

$$\boxed{E[h(X)] = \sum_j h(x_j) \cdot p(x_j)}$$

We don't need to first find the pmf of $Y = h(X)$ separately — we can directly use the pmf of $X$.

---

### 9.2 Critical Warning: Jensen's Inequality

$$\boxed{E[h(X)] \neq h(E[X]) \text{ in general!}}$$

This is one of the most important distinctions in probability. It fails whenever $h$ is nonlinear.

**The only exceptions:** When $h(x) = ax + b$ (linear/affine), then $E[h(X)] = h(E[X])$ — this is exactly Property 2 of linearity.

---

### 9.3 Worked Example — Payoff Function $Y = X^2 - 6X + 1$

**Problem:** Roll two dice. Let $X$ = sum and $Y = X^2 - 6X + 1$. Is this a good bet?

**Step 1: Build the joint table.**

| $X$ | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| $Y = X^2-6X+1$ | $-7$ | $-8$ | $-7$ | $-4$ | $1$ | $8$ | $17$ | $28$ | $41$ | $56$ | $73$ |
| $P(X)$ | 1/36 | 2/36 | 3/36 | 4/36 | 5/36 | 6/36 | 5/36 | 4/36 | 3/36 | 2/36 | 1/36 |

**Step 2:** Apply the change-of-variables formula.

$$E[Y] = \sum_{j=2}^{12} (j^2 - 6j + 1) \cdot P(X = j)$$

Computing each term and summing gives $E[Y] \approx 13.833$.

**Verification via linearity:** Since $E[X] = 7$, we might naively compute $(7)^2 - 6(7) + 1 = 49 - 42 + 1 = 8$. But $E[Y] = 13.833 \neq 8$. This confirms $E[h(X)] \neq h(E[X])$.

**Answer:** Since $E[Y] \approx 13.83 > 0$, this appears to be a **positive expected-value bet** worth taking.

---

## 10. In-Class Concept Questions — Full Solutions

### Concept Question 1 — Reading a CDF

**Problem:** $X$ has the CDF given by this table:

| Value of $X$ | 1 | 3 | 5 | 7 |
|---|---|---|---|---|
| CDF $F(a)$ | 0.5 | 0.75 | 0.9 | 1 |

**(Part 1) What is $P(X \leq 3)$?**  
Options: (a) 0.15, (b) 0.25, (c) 0.5, (d) 0.75

**Answer: (d) 0.75**

The CDF is defined as $F(a) = P(X \leq a)$, so:
$$P(X \leq 3) = F(3) = 0.75$$

No calculation needed — just read the table directly.

---

**(Part 2) What is $P(X = 3)$?**  
Options: (a) 0.15, (b) 0.25, (c) 0.5, (d) 0.75

**Answer: (b) 0.25**

Use the relationship: $p(a) = F(a) - F(\text{previous value})$.

The previous value is 1, so:
$$P(X = 3) = F(3) - F(1) = 0.75 - 0.50 = 0.25$$

**Interpretation:** The CDF jumped from 0.5 to 0.75 at $a = 3$, meaning $X = 3$ contributes exactly 0.25 of probability mass.

**Why not just read $F(3) = 0.75$ directly?** $F(3) = P(X \leq 3)$, which includes the probability of $X = 1$ (which is 0.5) plus the probability of $X = 3$ (which is 0.25). Reading the CDF value directly gives the cumulative probability, not the probability at exactly that point.

---

## 11. In-Class Board Problems — Full Solutions

### Problem 1 — Computing Expectation

**Problem:** $X$ has pmf:

| $X$ | 1 | 2 | 3 |
|---|---|---|---|
| pmf | 1/4 | 1/2 | 1/4 |

Find $E[X]$ and $E[1/X]$.

**Solution — $E[X]$:**

$$E[X] = 1 \cdot \frac{1}{4} + 2 \cdot \frac{1}{2} + 3 \cdot \frac{1}{4} = \frac{1}{4} + 1 + \frac{3}{4} = 2$$

**Solution — $E[1/X]$:**

Using the change-of-variables formula with $h(x) = 1/x$:

$$E\left[\frac{1}{X}\right] = \frac{1}{1} \cdot \frac{1}{4} + \frac{1}{2} \cdot \frac{1}{2} + \frac{1}{3} \cdot \frac{1}{4} = \frac{1}{4} + \frac{1}{4} + \frac{1}{12} = \frac{3}{12} + \frac{3}{12} + \frac{1}{12} = \frac{7}{12}$$

**Key observation:** $E[1/X] = 7/12 \neq 1/E[X] = 1/2$. This is another demonstration that $E[h(X)] \neq h(E[X])$ for nonlinear $h$.

---

### Problem 2 — Interpreting Expectation / Framing Bias

**Problem (a):** Would you accept a gamble offering a 10% chance to win $95 and a 90% chance of losing $5?

**Problem (b):** Would you pay $5 to participate in a lottery offering a 10% chance to win $100 and a 90% chance of winning nothing?

---

**Solution:**

**Part (a):**

| Value | 95 | -5 |
|---|---|---|
| PMF | 0.1 | 0.9 |

$$E[\text{gain}] = 95 \cdot 0.1 + (-5) \cdot 0.9 = 9.5 - 4.5 = \$5$$

**Part (b):**

If you pay $5 and win $100, your net gain is $95. If you pay $5 and win nothing, your net gain is $-5$.

| Value | 95 | -5 |
|---|---|---|
| PMF | 0.1 | 0.9 |

$$E[\text{gain}] = 95 \cdot 0.1 + (-5) \cdot 0.9 = \$5$$

**The two situations are mathematically identical.** Both have expected gain of $5. If you play enough times, you average winning $5 per game.

---

**Discussion — Framing Bias and Loss Aversion:**

In a famous study (Tversky and Kahneman), 132 undergraduates were given these two questions (in different orders). 55 gave *different answers* to the two identical situations. Of these 55, **42 rejected (a) but accepted (b)**.

Why? **Loss aversion**: we are psychologically far more unwilling to risk a *loss* than to pay an equivalent *cost* upfront. Paying $5 to play a lottery feels like a controllable cost. Risking losing $5 in a gamble feels like a painful loss — even though the dollar amounts are identical.

**Real-world implications:**
- **Insurance:** People pay premiums larger than their expected claims. Insurance companies are profitable precisely because of this asymmetry.
- **Example:** Pay $1/year to protect against a 1-in-1000 chance of losing $100. Expected value of change in wealth: without insurance $= -10$ cents/year; with insurance $= -\$1$/year. Insurance is mathematically worse, but people prefer the certainty of a known $1 loss over the possibility of a $100 loss.

---

### Problem 3 — Musical Chairs (Linearity of Expectation)

**Problem:** $n$ people run around and sit in $n$ chairs uniformly at random. What is the expected number of people who end up in their original seat?

---

**Solution:**

**Step 1: Define indicator variables.**

For each person $i = 1, 2, \ldots, n$, define:

$$X_i = \begin{cases} 1 & \text{if person } i \text{ returns to their original seat} \\ 0 & \text{otherwise} \end{cases}$$

Each $X_i \sim \text{Bernoulli}(1/n)$ since person $i$ is equally likely to sit in any of the $n$ chairs, so the probability of returning to their own is $1/n$.

Therefore: $E[X_i] = 1/n$.

**Step 2: Express the total as a sum.**

$$X = X_1 + X_2 + \cdots + X_n = \sum_{i=1}^n X_i$$

**Step 3: Apply linearity of expectation.**

$$E[X] = \sum_{i=1}^n E[X_i] = \sum_{i=1}^n \frac{1}{n} = n \cdot \frac{1}{n} = 1$$

**Final Answer:** $E[X] = 1$ regardless of $n$.

---

**Why this is surprising and beautiful:**

- For $n = 2$: Either both people return to their seats or both switch. $P(X=0) = 1/2$, $P(X=2) = 1/2$. Notice $X$ *never* equals its expected value of 1!
- For $n = 1000$: $E[X] = 1$ — on average, exactly 1 person out of 1000 sits back in their original chair.
- The $X_i$ are **not independent** (if $n-1$ people are in their correct seats, the $n$-th must be too). Yet linearity of expectation works regardless.

**Derangements bonus fact:** A permutation where nobody returns to their seat is called a **derangement**. The number of derangements of $n$ objects is approximately $n!/e$, so:

$$P(\text{everyone in a different seat}) \approx \frac{n!/e}{n!} = \frac{1}{e} \approx 36.8\%$$

Remarkably, this probability converges to $1/e$ as $n \to \infty$ and is approximately $36.8\%$ for all large $n$.

---

### Problem 4 — Bernoulli and Binomial

**Part (a):** $X \sim \text{Bernoulli}(p)$. Find $E[X]$.

| $X$ | 0 | 1 |
|---|---|---|
| pmf | $1-p$ | $p$ |

$$E[X] = (1-p) \cdot 0 + p \cdot 1 = p \qquad \checkmark$$

**This is important! $E[\text{Bernoulli}(p)] = p$. Memorise it.**

---

**Part (b):** $Y = X_1 + X_2 + \cdots + X_{12}$ where each $X_i \sim \text{Bernoulli}(0.25)$, independent. Find $E[Y]$.

$$E[Y] = E[X_1] + E[X_2] + \cdots + E[X_{12}] = 12 \cdot 0.25 = 3$$

**Note:** $Y \sim \text{Binomial}(12, 0.25)$ since it is the sum of 12 independent Bernoulli(0.25) trials. This confirms $E[\text{Binomial}(n,p)] = np = 12 \times 0.25 = 3$.

---

### Problem 5 — Counting Successes Before the Second Failure

**Problem:** Let $X$ = number of successes before the second failure in a sequence of independent $\text{Bernoulli}(p)$ trials. Find the pmf of $X$.

---

**Solution:**

$X$ takes values $0, 1, 2, \ldots$. We claim:

$$\boxed{p(n) = P(X = n) = (n+1) \cdot p^n \cdot (1-p)^2}$$

**Derivation for $n = 3$:** We want exactly 3 successes before the 2nd failure. The sequence must:
- Have exactly 3 successes (S) and 2 failures (F)
- End with a failure (F) on the 5th trial

So the sequence looks like: __ __ __ __ F with 3 S's and 1 F in the first 4 positions.

The number of ways to arrange 3 S's and 1 F in 4 positions = $\binom{4}{1} = 4$ (choosing which position gets the F):

$$\{FSSSF, SFSSF, SSFSF, SSSSF\}$$

Wait — correcting: FSSSF, SFSSF, SSFSF, SSSFF. Each has 3 S's and 2 F's, probability $p^3(1-p)^2$.

$$P(X = 3) = 4 \cdot p^3(1-p)^2 = (3+1)p^3(1-p)^2 \checkmark$$

**General argument for $n$:** We want exactly $n$ successes then the 2nd failure. The last trial must be a failure. In the first $n+1$ trials, there must be exactly $n$ successes and 1 failure (the first failure). Choose which of the $n+1$ positions gets the failure: $\binom{n+1}{1} = n+1$ ways. Each such sequence has probability $p^n(1-p)^2$.

$$p(n) = (n+1) p^n (1-p)^2$$

**This is the Negative Binomial distribution** with parameters $r=2$ (number of failures) and success probability $p$.

---

## 12. Cognitive Biases: Gambler's Fallacy & Loss Aversion

### 12.1 Gambler's Fallacy

**Claim (false):** "If red came up 10 times in a row on roulette, the next spin is more likely to be black."

**Truth:** Each spin is **independent**. $P(\text{red}) = 18/38$ regardless of previous outcomes. The roulette wheel has no memory.

**Historical example:** On August 18, 1913, at Monte Carlo casino, black came up 26 times in a row at roulette. Players began frantically betting on red after the 15th consecutive black, believing a correction was "due." They doubled and tripled their stakes. The casino won millions of francs.

**Why the fallacy is seductive:** The sequence BBBBBBBBBBBBBBBBBBBBBBBBBB *does* have low probability (approximately $(18/38)^{26} \approx 10^{-8}$). But once 25 blacks have occurred, the probability of the next spin being black is still exactly $18/38$. The low probability of the *entire* run doesn't affect individual independent outcomes.

---

### 12.2 Hot Hand Fallacy

**Claim (debated):** NBA players get "hot" — a player who made 5 shots in a row is more likely to make the next one.

**Research finding:** Gilovich, Vallone & Tversky (1985) analysed NBA game data and concluded the "hot hand" is a cognitive illusion — a player's probability of making a shot is **not** higher after consecutive makes.

**Why this reverses the Gambler's Fallacy:** The Gambler's Fallacy assumes outcomes will *correct* after a streak. The Hot Hand fallacy assumes outcomes will *continue* after a streak. Both arise from misunderstanding randomness, but in opposite directions.

**Caveat:** Some later statisticians have argued Gilovich et al.'s analysis had methodological flaws, and some evidence for a small hot-hand effect has been published. The question remains somewhat open.

**Lesson for data science:** Humans are extraordinarily bad at intuiting independence. We see patterns in random data. This is why careful statistical testing exists — to overcome our cognitive biases.

---

## 13. Named Distribution Reference Table

| Distribution | Notation | Takes values | PMF $p(k)$ | Mean $E[X]$ | Variance |
|---|---|---|---|---|---|
| Bernoulli | $\text{Ber}(p)$ | 0, 1 | $p^k(1-p)^{1-k}$ | $p$ | $p(1-p)$ |
| Binomial | $\text{Bin}(n,p)$ | $0, 1, \ldots, n$ | $\binom{n}{k}p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ |
| Geometric | $\text{Geo}(p)$ | $0, 1, 2, \ldots$ | $(1-p)^k p$ | $(1-p)/p$ | $(1-p)/p^2$ |
| Uniform | $\text{Unif}(N)$ | $1, 2, \ldots, N$ | $1/N$ | $(N+1)/2$ | $(N^2-1)/12$ |

---

## 14. Common Mistakes Reference

| Mistake | Why it's wrong | Correct approach |
|---|---|---|
| $E[h(X)] = h(E[X])$ | Only holds when $h$ is linear (or affine) | Use the change-of-variables formula $\sum h(x_j)p(x_j)$ directly |
| Reading $p(a)$ directly from the CDF | $F(a) = P(X \leq a)$, not $P(X = a)$ | Use $p(a) = F(a) - F(\text{previous value})$ |
| Thinking linearity requires independence | $E[X+Y] = E[X] + E[Y]$ always holds | Linearity of expectation is unconditional |
| Confusing Geometric($p$) conventions | Some define it as # failures before 1st success, others as # trials until 1st success | Always clarify: does $X$ start at 0 or 1? |
| "The mean must be a possible value of $X$" | A fair die has mean 3.5, which is not a face | The mean is a weighted average, not necessarily achievable |
| Gambler's fallacy | Past outcomes of independent trials affect future ones | Each trial is independent; no "correction" occurs |
| Applying Binomial when trials aren't independent | Binomial requires independence between trials | Check independence; use hypergeometric for sampling without replacement |
| Confusing pmf and cdf | pmf is probability at a point; cdf is probability up to a point | $p(a) = P(X=a)$; $F(a) = P(X \leq a) = \sum_{x \leq a} p(x)$ |

---

## 15. Quick Summary & Formula Sheet

### Core Definitions

| Concept | Definition |
|---|---|
| Random variable $X$ | Function $X: \Omega \to \mathbb{R}$ mapping outcomes to numbers |
| PMF | $p(a) = P(X = a)$ |
| CDF | $F(a) = P(X \leq a) = \sum_{x \leq a} p(x)$ |
| Expected value | $E[X] = \sum_j x_j \cdot p(x_j)$ |
| Change-of-variables | $E[h(X)] = \sum_j h(x_j) \cdot p(x_j)$ |

### Key Formulas

$$E[X + Y] = E[X] + E[Y] \quad \text{(always, no independence needed)}$$

$$E[aX + b] = aE[X] + b$$

$$E[h(X)] \neq h(E[X]) \quad \text{in general}$$

### Named Distribution Expected Values (Must Know)

$$E[\text{Bernoulli}(p)] = p$$

$$E[\text{Binomial}(n,p)] = np$$

$$E[\text{Geometric}(p)] = \frac{1-p}{p}$$

$$E[\text{Uniform}(N)] = \frac{N+1}{2}$$

### Key Insights

- **Bernoulli indicators** are the atoms of counting: any count = sum of indicators.
- **Linearity of expectation** works even for *dependent* random variables — this is what makes it so powerful.
- **$E[X]$ is the centre of mass** of the probability distribution.
- **Binomial = sum of Bernoullis** — this is why $E[\text{Bin}(n,p)] = np$ follows immediately.
- **The Geometric distribution** has memoryless property: given you haven't succeeded yet, the distribution of remaining waiting time is identical.
- **CDFs are non-decreasing step functions** for discrete distributions — always right-continuous.
- **Human intuition about randomness is systematically biased** — the Gambler's Fallacy and Hot Hand Fallacy both arise from pattern-seeking in random sequences.

---

Here's everything you need — crisp definitions first, then every FAANG question pattern from this chapter.---

## FAANG Interview Questions — Chapter 4

---

### 🔴 Tier 1 — Near Certain to Appear

---

**Q1. What is a random variable? How is it different from a regular variable?**

"A random variable is a function $X: \Omega \to \mathbb{R}$ — it maps each outcome in the sample space to a real number. A regular variable is a fixed (possibly unknown) number. A random variable is random because the experiment's outcome is random, and algebraic because we can add, square, and take expectations of it. The key insight: it is a *function*, not a number — it only becomes a number after the experiment runs."

---

**Q2. What is the difference between a PMF and a CDF?**

"The PMF $p(a) = P(X = a)$ gives probability *at a single point*. The CDF $F(a) = P(X \leq a)$ gives cumulative probability *up to and including* $a$. You recover the PMF from the CDF by differencing: $p(a) = F(a) - F(a^-)$. The CDF is always non-decreasing, lives in $[0,1]$, and for discrete variables is a right-continuous staircase. The $\leq$ in the CDF definition (not $<$) is critical for discrete distributions."

---

**Q3. What is $E[X]$, and what does it represent?**

"The expected value is the probability-weighted average of all values $X$ can take: $E[X] = \sum_j x_j \cdot p(x_j)$. Intuitively it is the centre of mass of the distribution — the balance point if you place masses $p(x_j)$ at positions $x_j$ on a number line. It is the long-run average you'd observe over many independent repetitions (Law of Large Numbers). It need not be a value $X$ can actually take — a fair die has $E[X] = 3.5$."

---

**Q4. State and explain linearity of expectation. Does it require independence?**

"$E[X + Y] = E[X] + E[Y]$, always — no independence required. And $E[aX + b] = aE[X] + b$. The proof is one line: $E[X+Y] = \sum_i (x_i + y_i)P(\omega_i) = \sum_i x_i P(\omega_i) + \sum_i y_i P(\omega_i) = E[X] + E[Y]$. The power is that it works even for *dependent* variables, enabling elegant solutions like the Musical Chairs problem: $n$ people reshuffling, expected number back in their seat is always exactly 1, regardless of $n$."

---

**Q5. Is $E[h(X)] = h(E[X])$?**

"No — only when $h$ is affine (linear). In general $E[h(X)] = \sum_j h(x_j) p(x_j) \neq h(E[X])$. Classic example: $E[X^2] \neq (E[X])^2$. The gap between them is exactly the variance: $\text{Var}(X) = E[X^2] - (E[X])^2 \geq 0$. For convex $h$, Jensen's inequality says $E[h(X)] \geq h(E[X])$. This matters enormously in ML — for example, minimising expected squared loss is not the same as squaring the expected error."

---

**Q6. What are the Bernoulli, Binomial, and Geometric distributions? When do you use each?**

| | Bernoulli($p$) | Binomial($n,p$) | Geometric($p$) |
|---|---|---|---|
| Models | One trial | $n$ trials, count successes | Failures before 1st success |
| Support | $\{0,1\}$ | $\{0,1,\ldots,n\}$ | $\{0,1,2,\ldots\}$ |
| PMF | $p^k(1-p)^{1-k}$ | $\binom{n}{k}p^k(1-p)^{n-k}$ | $(1-p)^k p$ |
| Mean | $p$ | $np$ | $(1-p)/p$ |

"Use Bernoulli for a single binary outcome (click/no-click). Use Binomial for a fixed number of trials (how many of 1000 users convert). Use Geometric for waiting time (how many attempts until first purchase)."

---

### 🟠 Tier 2 — High Frequency

---

**Q7. Why is $E[\text{Binomial}(n,p)] = np$? Prove it without summing the binomial series.**

"Write $X = X_1 + X_2 + \cdots + X_n$ where each $X_i \sim \text{Bernoulli}(p)$ is the indicator of success on trial $i$. By linearity: $E[X] = \sum_{i=1}^n E[X_i] = \sum_{i=1}^n p = np$. The binomial is a sum of Bernoullis — this decomposition is the entire insight, and linearity does the rest. No messy algebra needed."

---

**Q8. You flip a biased coin with $P(\text{H}) = 0.3$ until the first head. What is the expected number of flips?**

"Number of flips = number of tails before first head, plus 1. Tails before first head follows $\text{Geometric}(0.3)$, so $E[\text{tails}] = (1-0.3)/0.3 = 0.7/0.3 \approx 2.33$. Expected total flips = $2.33 + 1 = 10/3 \approx 3.33$. Alternatively if you define Geometric as the number of trials until first success, $E[X] = 1/p = 1/0.3 \approx 3.33$ directly. Always clarify which convention you're using."

---

**Q9. A CDF table gives $F(1)=0.5$, $F(3)=0.75$, $F(5)=0.9$, $F(7)=1$. Find $P(X=3)$ and $P(1 < X \leq 5)$.**

"$P(X=3) = F(3) - F(1) = 0.75 - 0.50 = 0.25$. For the interval: $P(1 < X \leq 5) = F(5) - F(1) = 0.90 - 0.50 = 0.40$. The strict inequality on the left means we subtract $F(1)$, which correctly excludes $X = 1$."

---

**Q10. Musical chairs / indicator variable trick.** $n$ items shuffled randomly — expected number back in original position?

"Define indicator $X_i = 1$ if item $i$ returns to its position. Each $P(X_i = 1) = 1/n$ by symmetry, so $E[X_i] = 1/n$. Total $X = \sum_{i=1}^n X_i$. By linearity: $E[X] = n \cdot (1/n) = 1$, regardless of $n$. Note: the $X_i$ are not independent — but linearity doesn't need independence. This technique appears constantly in algorithm analysis (e.g., expected comparisons in quicksort)."

---

**Q11. What is the Gambler's Fallacy and why is it wrong?**

"The fallacy: after many consecutive outcomes of one type, the other type is 'due.' It is wrong because independent trials have no memory — $P(\text{heads})$ is always $p$ regardless of history. The roulette wheel doesn't know what it previously showed. The 26 consecutive blacks at Monte Carlo in 1913 caused players to lose millions betting on red, when the true probability of red never changed. In data science: a sequence of model errors does not make the next prediction more likely to be correct if errors are independent."

---

### 🟡 Tier 3 — Conceptual / Applied

---

**Q12. Two games: (A) 10% chance of winning $95, 90% chance of losing $5. (B) Pay $5 upfront, 10% chance of winning $100. Which would you take?**

"Both have identical expected value: $E[\text{gain}] = 0.1(95) + 0.9(-5) = \$5$. Mathematically equivalent. In practice, Kahneman and Tversky showed that most people reject (A) but accept (B) — loss aversion makes us treat potential losses as psychologically worse than equivalent costs. For a data scientist: this is why users respond differently to 'free trial then charge' vs. 'pay upfront then refund' even when the expected cost is identical."

---

**Q13. You build a model that makes independent errors with probability $p$ on each of $n$ predictions. What is the expected number of errors?**

"Each prediction $X_i \sim \text{Bernoulli}(p)$. Total errors $X = \sum X_i \sim \text{Binomial}(n,p)$. By linearity: $E[X] = np$. For example, $n=1000$ predictions with $p=0.05$ error rate gives $E[\text{errors}] = 50$. The variance is $np(1-p) = 47.5$, so the standard deviation is $\approx 6.9$ errors."

---

**Q14. Why does $E[1/X] \neq 1/E[X]$? Give a concrete example.**

"Because $h(x) = 1/x$ is nonlinear. Direct counterexample: $X$ takes values 1, 2, 3 each with probability 1/4, 1/2, 1/4. $E[X] = 2$, so $1/E[X] = 0.5$. But $E[1/X] = (1)(1/4) + (1/2)(1/2) + (1/3)(1/4) = 1/4 + 1/4 + 1/12 = 7/12 \approx 0.583 \neq 0.5$. Jensen's inequality: $1/x$ is convex, so $E[1/X] \geq 1/E[X]$."

---

**Q15. (Amazon / quant) Derive the expected value of the Geometric distribution.**

"$E[X] = \sum_{k=0}^\infty k(1-p)^k p$. Use the identity: differentiate $\sum_{k=0}^\infty x^k = 1/(1-x)$ to get $\sum_{k=0}^\infty kx^{k-1} = 1/(1-x)^2$. Multiply by $x$: $\sum_{k=0}^\infty kx^k = x/(1-x)^2$. Substitute $x = 1-p$: $\sum k(1-p)^k = (1-p)/p^2$. Multiply by $p$: $E[X] = (1-p)/p$." This derivation — differentiating a power series — is a standard technique at quant/research roles.

---

## The 3 Things Every FAANG Interviewer Tests from Chapter 4

1. **Linearity of expectation with dependent variables** — the Musical Chairs / indicator trick. If you can't decompose a complex count into Bernoulli indicators and apply linearity, you will miss half the problems in this domain.

2. **$E[h(X)] \neq h(E[X])$** — they will hand you a nonlinear function and ask you to compute the expected value, watching whether you blindly substitute $E[X]$ into $h$.

3. **PMF vs CDF reading** — given a CDF table, find $P(X = a)$ and $P(a < X \leq b)$. The $\leq$ boundary is deliberately chosen to trap you.
