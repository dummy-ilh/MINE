
##  7 : Joint Distributions · Independence · Covariance · Correlation

---

## Table of Contents

1. [Introduction to Joint Distributions](#1-introduction-to-joint-distributions)
2. [Joint PMF — Discrete Case](#2-joint-pmf--discrete-case)
3. [Joint PDF — Continuous Case](#3-joint-pdf--continuous-case)
4. [Joint CDF](#4-joint-cdf)
5. [Marginal Distributions](#5-marginal-distributions)
6. [Independence of Random Variables](#6-independence-of-random-variables)
7. [Covariance](#7-covariance)
8. [Correlation](#8-correlation)
9. [Bivariate Normal Distribution](#9-bivariate-normal-distribution)
10. [Real-Life Correlations and Causation](#10-real-life-correlations-and-causation)
11. [Hospital Problem — Extended Application](#11-hospital-problem--extended-application)
12. [Quick Reference Summary](#12-quick-reference-summary)

---

## 1. Introduction to Joint Distributions

### 1.1 Concept Overview

In science and data science, we are almost always interested in **multiple random variables simultaneously**. A single variable tells you one thing; a joint distribution tells you how two (or more) variables relate to each other.

Real examples where joint distributions matter:

| Pair of Variables | Expected Relationship |
|---|---|
| Height and weight of giraffes | Positive: taller giraffes tend to be heavier |
| IQ and birth weight of children | Mild positive correlation |
| Frequency of exercise and heart disease rate | Negative: more exercise, less disease |
| Air pollution level and respiratory illness rate | Positive: more pollution, more illness |
| Number of Facebook friends and age | Non-linear: peaks in 20s–30s |

The **joint distribution** of $(X, Y)$ allows us to:
- Compute probabilities involving **both** variables simultaneously
- Understand the **relationship** between variables
- Measure **independence** or **dependence** precisely
- Compute **covariance** and **correlation**

> **Intuition:** Think of a single random variable as a 1D histogram. A joint distribution is a 2D histogram over a plane — it assigns probability to every pair of values $(x, y)$, not just to single values.

---

## 2. Joint PMF — Discrete Case

### 2.1 Formal Definition

Suppose $X$ and $Y$ are **discrete** random variables taking values $\{x_1, \ldots, x_n\}$ and $\{y_1, \ldots, y_m\}$ respectively.

**Definition — Joint PMF:**

The **joint probability mass function** (joint pmf) is:

$$p(x_i, y_j) = P(X = x_i \text{ and } Y = y_j)$$

**Validity requirements:**

1. $0 \leq p(x_i, y_j) \leq 1$ for all $i, j$
2. $\displaystyle\sum_{i=1}^{n} \sum_{j=1}^{m} p(x_i, y_j) = 1$ (total probability = 1)

### 2.2 Organization: The Joint Probability Table

The joint pmf is organized as a table where:
- **Rows** = values of $X$
- **Columns** = values of $Y$
- **Cell** $(i,j)$ = $P(X = x_i, Y = y_j)$
- **Row margins** = marginal pmf of $X$
- **Column margins** = marginal pmf of $Y$

---

### 2.3 Worked Examples — Discrete Joint Distributions

---

#### Example 1 — Two Dice: Independent Variables

**Problem:** Roll two fair dice. Let $X$ = value on the first die, $Y$ = value on the second die.

| $X \backslash Y$ | 1 | 2 | 3 | 4 | 5 | 6 | $p(x_i)$ |
|---|---|---|---|---|---|---|---|
| 1 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 1/6 |
| 2 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 1/6 |
| 3 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 1/6 |
| 4 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 1/6 |
| 5 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 1/6 |
| 6 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 1/6 |
| $p(y_j)$ | 1/6 | 1/6 | 1/6 | 1/6 | 1/6 | 1/6 | 1 |

**Key observation:** Every cell = $\frac{1}{36} = \frac{1}{6} \cdot \frac{1}{6}$ = (row margin) × (column margin). This is the product rule for **independence**.

---

#### Example 2 — Two Dice: Dependent Variables

**Problem:** Roll two fair dice. Let $X$ = value on first die, $T$ = **sum** on both dice.

| $X \backslash T$ | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | $p(x_i)$ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 0 | 0 | 0 | 0 | 0 | 1/6 |
| 2 | 0 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 0 | 0 | 0 | 0 | 1/6 |
| 3 | 0 | 0 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 0 | 0 | 0 | 1/6 |
| 4 | 0 | 0 | 0 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 0 | 0 | 1/6 |
| 5 | 0 | 0 | 0 | 0 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 0 | 1/6 |
| 6 | 0 | 0 | 0 | 0 | 0 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 1/6 |
| $p(t_j)$ | 1/36 | 2/36 | 3/36 | 4/36 | 5/36 | 6/36 | 5/36 | 4/36 | 3/36 | 2/36 | 1/36 | 1 |

**Key observation:** Many cells have probability 0, but the marginals are all non-zero. Since $0 \neq \frac{1}{6} \cdot \frac{1}{36}$, the product rule fails → **$X$ and $T$ are not independent**.

**Why does dependence occur?** If you know $X = 1$, then $T$ can only be 2–7, never 8–12. Knowing $X$ completely restricts the possible values of $T$. This is the essence of dependence.

---

#### Example 3 — Events from Joint Table: $B = \{Y - X \geq 2\}$

**Problem:** Using the two-dice table (Example 1), describe and find the probability of event $B = \{Y - X \geq 2\}$.

**Step 1:** List all $(X, Y)$ pairs where $Y - X \geq 2$:

$$B = \{(1,3),(1,4),(1,5),(1,6),(2,4),(2,5),(2,6),(3,5),(3,6),(4,6)\}$$

**Step 2:** Count the outcomes: 10 pairs.

**Step 3:** Each outcome has probability $1/36$.

$$P(B) = \frac{10}{36} = \frac{5}{18} \approx 0.278$$

**Interpretation:** About 27.8% of the time, the second die shows a value at least 2 more than the first.

---

### 2.4 Concept Question — Independence I

**Question:** Roll two dice: $X$ = value on first, $Y$ = value on second. Are $X$ and $Y$ independent?

**Answer: Yes.**

**Reasoning:** Every cell probability $= 1/36 = (1/6)(1/6)$ = product of marginal probabilities. The product rule holds everywhere, so $X$ and $Y$ are independent.

> **Intuition:** The outcome of one die physically cannot affect the other. This physical independence is mirrored in the mathematical structure: no cell "knows" what row or column it's in.

---

### 2.5 Concept Question — Independence II

**Question:** Roll two dice: $X$ = value on first, $T$ = sum. Are $X$ and $T$ independent?

**Answer: No.**

**Reasoning:** Cells with probability 0 cannot be the product of two positive marginals. For example, $P(X=1, T=8) = 0$ but $P(X=1) \cdot P(T=8) = \frac{1}{6} \cdot \frac{5}{36} \neq 0$.

---

## 3. Joint PDF — Continuous Case

### 3.1 Concept Overview

For continuous random variables, the joint pmf becomes a **joint probability density function (joint pdf)**. Instead of a table of probabilities, we have a surface over the $(x,y)$ plane.

### 3.2 Formal Definition

If $X$ takes values in $[a,b]$ and $Y$ takes values in $[c,d]$, the pair $(X,Y)$ takes values in the rectangle $[a,b] \times [c,d]$.

**Definition — Joint PDF:**

The **joint probability density function** $f(x,y)$ satisfies:

$$P\bigl((X,Y) \in \text{tiny rectangle at } (x,y)\bigr) = f(x,y)\, dx\, dy$$

**Validity requirements:**

1. $f(x,y) \geq 0$ for all $(x,y)$
2. $\displaystyle\int_c^d \int_a^b f(x,y)\, dx\, dy = 1$

> **Important:** Just like a 1D PDF, the joint PDF value $f(x,y)$ can be **greater than 1**. It is a density, not a probability. Probability comes from integrating over a region.

### 3.3 Computing Probabilities from a Joint PDF

$$P\bigl((X,Y) \in \text{region } R\bigr) = \iint_R f(x,y)\, dx\, dy$$

For rectangular regions $[x_1, x_2] \times [y_1, y_2]$:

$$P(x_1 \leq X \leq x_2,\; y_1 \leq Y \leq y_2) = \int_{y_1}^{y_2} \int_{x_1}^{x_2} f(x,y)\, dx\, dy$$

---

### 3.4 Worked Examples — Continuous Joint Distributions

---

#### Example 4 — Uniform Joint Density: $P(X > Y)$

**Problem:** $X$ and $Y$ both take values in $[0,1]$ with uniform density $f(x,y) = 1$. Visualize and find $P(X > Y)$.

**Step 1:** $(X,Y)$ takes values in the unit square $[0,1]^2$.

**Step 2:** The event $\{X > Y\}$ is the lower-right triangle below the diagonal line $y = x$.

```
y
1 |-----/
  |    /
  |   / (X > Y region)
  |  /
  | /
  |/___________
0             1  x
```

**Step 3:** Since $f(x,y) = 1$ (constant), probability = density × area.

$$P(X > Y) = 1 \times \text{area of lower triangle} = \frac{1}{2}$$

**Interpretation:** By symmetry, $X$ and $Y$ are interchangeable, so there's a 50% chance either one is larger.

---

#### Example 5 — Non-Uniform Joint Density: Validation and Probability

**Problem:** $X$ and $Y$ on $[0,1]^2$ with $f(x,y) = 4xy$. Show this is a valid pdf, then find $P(A)$ where $A = \{X < 0.5 \text{ and } Y > 0.5\}$.

**Step 1: Verify validity — non-negativity.**

$4xy \geq 0$ for $x,y \in [0,1]$. ✓

**Step 2: Verify total probability = 1.**

$$\int_0^1 \int_0^1 4xy\, dx\, dy = \int_0^1 \left[2x^2 y\right]_0^1 dy = \int_0^1 2y\, dy = \left[y^2\right]_0^1 = 1 \checkmark$$

**Step 3: Identify event $A$.**

$A$ is the **upper-left quadrant**: $x \in [0, 0.5]$, $y \in [0.5, 1]$.

**Step 4: Compute $P(A)$.**

$$P(A) = \int_0^{0.5} \int_{0.5}^{1} 4xy\, dy\, dx$$

Inner integral:
$$\int_{0.5}^{1} 4xy\, dy = \left[2xy^2\right]_{0.5}^{1} = 2x(1 - 0.25) = \frac{3x}{2} \cdot 2 = 3x \cdot \frac{1}{2}$$

Wait — let me compute carefully:
$$\int_{0.5}^{1} 4xy\, dy = 4x \cdot \left[\frac{y^2}{2}\right]_{0.5}^{1} = 4x \cdot \frac{1 - 0.25}{2} = 4x \cdot \frac{0.75}{2} = \frac{3x}{2}$$

Outer integral:
$$P(A) = \int_0^{0.5} \frac{3x}{2}\, dx = \frac{3}{2} \cdot \left[\frac{x^2}{2}\right]_0^{0.5} = \frac{3}{2} \cdot \frac{0.25}{2} = \frac{3}{16}$$

**Final Answer:** $P(A) = \dfrac{3}{16} \approx 0.1875$

**Interpretation:** With $f(x,y) = 4xy$, probability concentrates toward the upper-right corner (where both $x$ and $y$ are large). The upper-left quadrant (small $x$, large $y$) therefore has relatively low probability.

---

#### Example — Problem 1: Joint PDF $f(x,y) = x + y$ on $[0,1]^2$

**Full problem from MIT 18.05 board questions:**

$(X,Y)$ takes values in $[0,1] \times [0,1]$ with joint pdf $f(x,y) = x + y$.

---

**Part (a): Show $f(x,y)$ is a valid pdf.**

**Non-negativity:** For $x,y \in [0,1]$: $x + y \geq 0$. ✓

**Total probability:**
$$\int_0^1 \int_0^1 (x+y)\, dx\, dy = \int_0^1 \left[\frac{x^2}{2} + xy\right]_0^1 dy = \int_0^1 \left(\frac{1}{2} + y\right) dy = \left[\frac{y}{2} + \frac{y^2}{2}\right]_0^1 = \frac{1}{2} + \frac{1}{2} = 1 \checkmark$$

---

**Part (b): Find $P(A)$ where $A = \{X > 0.3 \text{ and } Y > 0.5\}$.**

The event $A$ is the rectangle $[0.3, 1] \times [0.5, 1]$ in the unit square.

$$P(A) = \int_{0.5}^{1} \int_{0.3}^{1} (x+y)\, dx\, dy$$

**Inner integral** (integrate over $x$ from 0.3 to 1):
$$\int_{0.3}^{1} (x+y)\, dx = \left[\frac{x^2}{2} + xy\right]_{0.3}^{1} = \left(\frac{1}{2} + y\right) - \left(\frac{0.09}{2} + 0.3y\right) = \frac{1 - 0.09}{2} + 0.7y = 0.455 + 0.7y$$

**Outer integral** (integrate over $y$ from 0.5 to 1):
$$\int_{0.5}^{1} (0.455 + 0.7y)\, dy = \left[0.455y + 0.35y^2\right]_{0.5}^{1}$$
$$= (0.455 + 0.35) - (0.2275 + 0.0875) = 0.805 - 0.315 = 0.49$$

**Final Answer:** $P(A) = 0.49$

---

**Part (c): Find the joint CDF $F(x,y)$.**

$$F(x,y) = \int_0^y \int_0^x (u+v)\, du\, dv$$

**Inner integral:**
$$\int_0^x (u+v)\, du = \left[\frac{u^2}{2} + vu\right]_0^x = \frac{x^2}{2} + vx$$

**Outer integral:**
$$F(x,y) = \int_0^y \left(\frac{x^2}{2} + vx\right) dv = \left[\frac{x^2 v}{2} + \frac{v^2 x}{2}\right]_0^y = \frac{x^2 y}{2} + \frac{xy^2}{2}$$

$$\boxed{F(x,y) = \frac{x^2 y + xy^2}{2}}$$

---

**Part (d): Marginal CDF $F_X(x)$ and $P(X < 0.5)$.**

The marginal CDF of $X$ is obtained by setting $y = 1$ (the top of the $y$-range):

$$F_X(x) = F(x, 1) = \frac{x^2 \cdot 1 + x \cdot 1^2}{2} = \frac{x^2 + x}{2}$$

$$P(X < 0.5) = F_X(0.5) = \frac{0.25 + 0.5}{2} = \frac{0.75}{2} = \frac{3}{8}$$

---

**Part (e): Marginal PDF $f_X(x)$ and verify $P(X < 0.5)$.**

**Method 1 — Differentiate the marginal CDF:**

$$f_X(x) = F_X'(x) = \frac{d}{dx}\left(\frac{x^2 + x}{2}\right) = \frac{2x + 1}{2} = x + \frac{1}{2}$$

**Method 2 — Integrate out $Y$ from the joint PDF:**

$$f_X(x) = \int_0^1 (x+y)\, dy = \left[xy + \frac{y^2}{2}\right]_0^1 = x + \frac{1}{2}$$

Both methods agree: $f_X(x) = x + \dfrac{1}{2}$.

**Verify $P(X < 0.5)$:**

$$P(X < 0.5) = \int_0^{0.5} \left(x + \frac{1}{2}\right) dx = \left[\frac{x^2}{2} + \frac{x}{2}\right]_0^{0.5} = \frac{0.25}{2} + \frac{0.5}{2} = 0.125 + 0.25 = \frac{3}{8} \checkmark$$

---

**Part (f): Discrete CDF $F(3.5, 4)$ from the two-dice table.**

$F(3.5, 4) = P(X \leq 3.5, Y \leq 4)$

From the joint pmf table, this includes rows $X \in \{1, 2, 3\}$ (since $3.5$ rounds down to $3$ in integers) and columns $Y \in \{1, 2, 3, 4\}$:

Shaded cells: rows 1, 2, 3 × columns 1, 2, 3, 4 = $3 \times 4 = 12$ cells, each with probability $1/36$.

$$F(3.5, 4) = \frac{12}{36} = \frac{1}{3}$$

---

## 4. Joint CDF

### 4.1 Formal Definition

$$F(x,y) = P(X \leq x,\; Y \leq y)$$

**Continuous case:**
$$F(x,y) = \int_c^y \int_a^x f(u,v)\, du\, dv$$

**Recovering the PDF from the CDF:**
$$f(x,y) = \frac{\partial^2 F}{\partial x \partial y}(x,y)$$

**Discrete case:**
$$F(x,y) = \sum_{x_i \leq x} \sum_{y_j \leq y} p(x_i, y_j)$$

### 4.2 Properties of the Joint CDF

1. **Non-decreasing:** If $x$ or $y$ increases, $F(x,y)$ stays constant or increases.
2. **Lower-left boundary:** $F(x,y) = 0$ at the lower-left of the joint range.
3. **Upper-right boundary:** $F(x,y) = 1$ at the upper-right of the joint range.

---

#### Example 6 — Joint CDF for $f(x,y) = 4xy$

**Problem:** Find $F(x,y)$ for the joint pdf in Example 5.

$$F(x,y) = \int_0^y \int_0^x 4uv\, du\, dv$$

**Inner integral:**
$$\int_0^x 4uv\, du = 4v \cdot \frac{x^2}{2} = 2x^2 v$$

**Outer integral:**
$$F(x,y) = \int_0^y 2x^2 v\, dv = 2x^2 \cdot \frac{y^2}{2} = x^2 y^2$$

$$\boxed{F(x,y) = x^2 y^2}$$

**Verification:** $F(1,1) = 1^2 \cdot 1^2 = 1$ ✓ (total probability = 1 at upper-right corner)

---

## 5. Marginal Distributions

### 5.1 Concept Overview

When $X$ and $Y$ have a joint distribution, the **marginal distribution** of $X$ alone (or $Y$ alone) is what you get by "ignoring" the other variable. The term "marginal" comes from the fact that the values appear in the **margins** of the joint table.

> **Intuition:** Imagine projecting the 2D probability surface down onto the $x$-axis. The marginal PDF of $X$ is the "shadow" of the joint distribution along the $x$-axis direction.

### 5.2 Marginal PMF (Discrete)

$$p_X(x_i) = \sum_j p(x_i, y_j) \quad \text{(sum over all } y \text{)}$$

$$p_Y(y_j) = \sum_i p(x_i, y_j) \quad \text{(sum over all } x \text{)}$$

### 5.3 Marginal PDF (Continuous)

$$f_X(x) = \int_c^d f(x,y)\, dy \quad \text{(integrate out } y \text{)}$$

$$f_Y(y) = \int_a^b f(x,y)\, dx \quad \text{(integrate out } x \text{)}$$

### 5.4 Marginal CDF

If $(X,Y)$ takes values on $[a,b] \times [c,d]$:

$$F_X(x) = F(x, d) \quad \text{(set } y \text{ to the top of its range)}$$

$$F_Y(y) = F(b, y) \quad \text{(set } x \text{ to the top of its range)}$$

---

### 5.5 Worked Examples — Marginals

---

#### Example 8 — Marginal PMF from Two-Dice Sum Table

**Problem:** From Example 2, compute marginal pmf for $X$ and $T$.

For $P(X = 3)$: sum row 3: $0 + 0 + 1/36 + 1/36 + 1/36 + 1/36 + 1/36 + 1/36 + 0 + 0 + 0 = 6/36 = 1/6$ ✓

For $P(T = 5)$: sum column $T = 5$: $1/36 + 1/36 + 1/36 + 1/36 + 0 + 0 = 4/36$ ✓

---

#### Example 9 — Marginal PDFs from $f(x,y) = \frac{8}{3}x^3 y$

**Problem:** $(X,Y)$ takes values on $[0,1] \times [1,2]$ with joint pdf $f(x,y) = \frac{8}{3}x^3 y$. Find $f_X(x)$ and $f_Y(y)$.

**Finding $f_X(x)$ — integrate out $y$:**

$$f_X(x) = \int_1^2 \frac{8}{3}x^3 y\, dy = \frac{8}{3}x^3 \left[\frac{y^2}{2}\right]_1^2 = \frac{8}{3}x^3 \cdot \frac{4-1}{2} = \frac{8}{3}x^3 \cdot \frac{3}{2} = 4x^3$$

**Finding $f_Y(y)$ — integrate out $x$:**

$$f_Y(y) = \int_0^1 \frac{8}{3}x^3 y\, dx = \frac{8}{3}y \left[\frac{x^4}{4}\right]_0^1 = \frac{8}{3}y \cdot \frac{1}{4} = \frac{2y}{3}$$

**Verification:** $\int_0^1 4x^3\, dx = [x^4]_0^1 = 1$ ✓ and $\int_1^2 \frac{2y}{3}\, dy = \frac{2}{3} \cdot \frac{4-1}{2} = 1$ ✓

---

#### Example 10 — Marginal PDF and Probability

**Problem:** $(X,Y)$ on $[0,1]^2$ with $f(x,y) = \frac{3}{2}(x^2 + y^2)$. Find $f_X(x)$ and $P(X < 0.5)$.

**Find $f_X(x)$:**

$$f_X(x) = \int_0^1 \frac{3}{2}(x^2 + y^2)\, dy = \frac{3}{2}\left[x^2 y + \frac{y^3}{3}\right]_0^1 = \frac{3}{2}\left(x^2 + \frac{1}{3}\right) = \frac{3}{2}x^2 + \frac{1}{2}$$

**Find $P(X < 0.5)$:**

$$P(X < 0.5) = \int_0^{0.5} \left(\frac{3}{2}x^2 + \frac{1}{2}\right) dx = \left[\frac{x^3}{2} + \frac{x}{2}\right]_0^{0.5} = \frac{0.125}{2} + \frac{0.5}{2} = 0.0625 + 0.25 = \frac{5}{16}$$

---

#### Example 11 — Marginal CDF

**Problem:** The joint CDF from Example — Problem 1 is $F(x,y) = \frac{1}{2}(x^3 y + xy^3)$. Find $F_X(x)$ and compute $P(X < 0.5)$.

$$F_X(x) = F(x, 1) = \frac{1}{2}(x^3 + x)$$

$$P(X < 0.5) = F_X(0.5) = \frac{1}{2}\left(\frac{1}{8} + \frac{1}{2}\right) = \frac{1}{2} \cdot \frac{5}{8} = \frac{5}{16} \checkmark$$

---

## 6. Independence of Random Variables

### 6.1 Concept Overview

We have an intuitive sense of independence — two variables are independent if knowing one tells you nothing about the other. The formal definition makes this precise.

**Definition:**

Jointly-distributed random variables $X$ and $Y$ are **independent** if and only if:

$$F(x,y) = F_X(x) \cdot F_Y(y) \quad \text{for all } (x,y)$$

**Equivalent conditions:**

For **discrete** variables:
$$p(x_i, y_j) = p_X(x_i) \cdot p_Y(y_j) \quad \text{for all } i, j$$

For **continuous** variables:
$$f(x,y) = f_X(x) \cdot f_Y(y) \quad \text{for all } (x,y)$$

> **Quick Test:** For a continuous joint pdf, independence holds if and only if you can **factor** $f(x,y)$ as a product of a function of $x$ alone times a function of $y$ alone — even without computing the marginals explicitly.

---

### 6.2 How to Test for Independence

**Step 1 — Try to factor:** Can you write $f(x,y) = g(x) \cdot h(y)$ for some functions $g$ and $h$?

**Step 2 — Check the support:** If the joint range is **not** a rectangle, the variables are automatically dependent.

**Step 3 — Verify one cell:** For discrete: find any cell where $p(x_i, y_j) \neq p_X(x_i) p_Y(y_j)$.

---

### 6.3 Concept Question — Independence III

**Which of the following joint pdfs have independent variables? (Range is a rectangle.)**

**(i) $f(x,y) = 4x^2 y^3$**

**Answer: Independent.**

This can be factored as:
$$f(x,y) = 4x^2 y^3 = (ax^2)(by^3) \quad \text{where } ab = 4$$

The marginal densities are $f_X(x) \propto x^2$ and $f_Y(y) \propto y^3$. The joint is the product of the marginals. ✓

**(ii) $f(x,y) = \frac{1}{2}(x^3 y + xy^3)$**

**Answer: Not independent.**

There is **no way** to factor this as a product $g(x) \cdot h(y)$. The cross-term structure $x^3 y + xy^3 = xy(x^2 + y^2)$ cannot be separated into a product of a function of $x$ alone times a function of $y$ alone.

**(iii) $f(x,y) = 6e^{-3x-2y}$**

**Answer: Independent.**

$$f(x,y) = 6e^{-3x-2y} = 6e^{-3x} \cdot e^{-2y} = (ae^{-3x})(be^{-2y}) \quad \text{where } ab = 6$$

The exponential of a sum factors into a product of exponentials. ✓

**Summary:**

| PDF | Independent? | Reason |
|---|---|---|
| $4x^2 y^3$ | ✓ Yes | Factors as $g(x) \cdot h(y)$ |
| $\frac{1}{2}(x^3 y + xy^3)$ | ✗ No | Cannot factor |
| $6e^{-3x-2y}$ | ✓ Yes | $e^{A+B} = e^A \cdot e^B$ |

---

### 6.4 The Factorization Shortcut

> **Key Fact:** For a joint pdf over a **rectangular** range, $X$ and $Y$ are independent if and only if $f(x,y)$ can be written as $g(x) \cdot h(y)$ for any functions $g$ and $h$ (not necessarily normalized). This is the fastest independence check.

**Example:**

- $f(x,y) = x^2 \sin(y)$ → factors → independent ✓
- $f(x,y) = x^2 + y^2$ → does not factor (a sum, not a product) → not independent ✗
- $f(x,y) = e^{-(x^2 + y^2)} = e^{-x^2} \cdot e^{-y^2}$ → factors → independent ✓

---

### 6.5 Common Mistakes

| Mistake | Correction |
|---|---|
| Assuming zero covariance means independence | Zero covariance implies only no **linear** relationship; dependence can still exist |
| Testing independence at only one cell | Must verify product rule holds at **all** cells |
| Forgetting to check rectangular support | Non-rectangular support automatically implies dependence |
| Confusing marginals with conditionals | Marginals average over the other variable; conditionals fix the other variable |

---

## 7. Covariance

### 7.1 Concept Overview

Covariance measures **how much two random variables vary together**. If when $X$ is large, $Y$ also tends to be large, the covariance is positive. If when $X$ is large, $Y$ tends to be small, the covariance is negative. If the two are unrelated, the covariance is zero.

**Real examples:**

| Variable Pair | Covariance Direction |
|---|---|
| Height and weight | Positive |
| Temperature and sweater sales | Negative |
| Coin flip 1 and coin flip 2 | Zero (independent) |
| Overlapping coin flip sums | Positive (shared flips) |

### 7.2 Formal Definition

$$\text{Cov}(X,Y) = E\bigl[(X - \mu_X)(Y - \mu_Y)\bigr]$$

where $\mu_X = E[X]$ and $\mu_Y = E[Y]$.

**Computation formulas:**

**Discrete:**
$$\text{Cov}(X,Y) = \sum_i \sum_j p(x_i, y_j)(x_i - \mu_X)(y_j - \mu_Y)$$

**Continuous:**
$$\text{Cov}(X,Y) = \int \int (x - \mu_X)(y - \mu_Y) f(x,y)\, dx\, dy$$

**Shortcut (analogous to $\text{Var}(X) = E[X^2] - \mu^2$):**

$$\boxed{\text{Cov}(X,Y) = E[XY] - \mu_X \mu_Y}$$

---

### 7.3 Properties of Covariance

| # | Property | Formula |
|---|---|---|
| 1 | Scaling | $\text{Cov}(aX+b, cY+d) = ac\,\text{Cov}(X,Y)$ |
| 2 | Linearity (bilinearity) | $\text{Cov}(X_1 + X_2, Y) = \text{Cov}(X_1, Y) + \text{Cov}(X_2, Y)$ |
| 3 | Self-covariance = Variance | $\text{Cov}(X,X) = \text{Var}(X)$ |
| 4 | Shortcut formula | $\text{Cov}(X,Y) = E[XY] - \mu_X \mu_Y$ |
| 5 | Variance of sum | $\text{Var}(X+Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X,Y)$ |
| 6 | Independence → zero covariance | If $X \perp Y$ then $\text{Cov}(X,Y) = 0$ |

> **Warning:** Property 6 does NOT reverse. Zero covariance does NOT imply independence (see Example below). Covariance only captures **linear** relationships.

---

### 7.4 Proof of Key Properties

**Proof of Property 4:** $\text{Cov}(X,Y) = E[XY] - \mu_X \mu_Y$

Starting from the definition and expanding:

$$\text{Cov}(X,Y) = E[(X-\mu_X)(Y-\mu_Y)]$$
$$= E[XY - \mu_X Y - \mu_Y X + \mu_X \mu_Y]$$
$$= E[XY] - \mu_X E[Y] - \mu_Y E[X] + \mu_X \mu_Y$$
$$= E[XY] - \mu_X \mu_Y - \mu_Y \mu_X + \mu_X \mu_Y$$
$$= E[XY] - \mu_X \mu_Y \quad \square$$

**Proof of Property 5:** $\text{Var}(X+Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X,Y)$

$$\text{Var}(X+Y) = \text{Cov}(X+Y, X+Y)$$
$$= \text{Cov}(X,X) + \text{Cov}(X,Y) + \text{Cov}(Y,X) + \text{Cov}(Y,Y)$$
$$= \text{Var}(X) + 2\text{Cov}(X,Y) + \text{Var}(Y) \quad \square$$

**Proof of Property 6:** If $X \perp Y$ then $\text{Cov}(X,Y) = 0$

When independent, $f(x,y) = f_X(x) f_Y(y)$, so:

$$\text{Cov}(X,Y) = \int\!\int (x-\mu_X)(y-\mu_Y) f_X(x) f_Y(y)\, dx\, dy$$
$$= \left[\int (x-\mu_X) f_X(x)\, dx\right] \cdot \left[\int (y-\mu_Y) f_Y(y)\, dy\right]$$
$$= E[X - \mu_X] \cdot E[Y - \mu_Y] = 0 \cdot 0 = 0 \quad \square$$

---

### 7.5 Worked Examples — Covariance

---

#### Example — Covariance via Joint Table (3 Coin Flips)

**Problem (Example 1 from prep notes):** Flip a fair coin 3 times. Let $X$ = heads in first 2 flips, $Y$ = heads in last 2 flips (overlapping middle flip). Compute $\text{Cov}(X,Y)$.

**Step 1:** All 8 outcomes: HHH, HHT, HTH, HTT, THH, THT, TTH, TTT.

Build the joint table:

| $X \backslash Y$ | 0 | 1 | 2 | $p(x_i)$ |
|---|---|---|---|---|
| 0 | 1/8 | 1/8 | 0 | 1/4 |
| 1 | 1/8 | 2/8 | 1/8 | 1/2 |
| 2 | 0 | 1/8 | 1/8 | 1/4 |
| $p(y_j)$ | 1/4 | 1/2 | 1/4 | 1 |

**Step 2:** Compute means from marginals.

$$E[X] = 0 \cdot \frac{1}{4} + 1 \cdot \frac{1}{2} + 2 \cdot \frac{1}{4} = 1 = E[Y]$$

**Step 3: Method 1 — Direct definition.**

$\text{Cov}(X,Y) = \sum_{i,j} p(x_i, y_j)(x_i - 1)(y_j - 1)$

Only terms with $x_i \neq 1$ AND $y_j \neq 1$ AND $p(x_i, y_j) \neq 0$ are nonzero:

- $(x=0, y=0)$: $\frac{1}{8}(0-1)(0-1) = \frac{1}{8}$
- $(x=2, y=2)$: $\frac{1}{8}(2-1)(2-1) = \frac{1}{8}$

$$\text{Cov}(X,Y) = \frac{1}{8} + \frac{1}{8} = \frac{1}{4}$$

**Step 4: Method 2 — Shortcut via $E[XY]$.**

$$E[XY] = 1 \cdot \frac{2}{8} + 2 \cdot \frac{1}{8} + 4 \cdot \frac{1}{8} = \frac{2+2+4}{8} = \frac{5}{4}$$

Wait — let me compute carefully from the table (only nonzero cells):
- $(x=1, y=1)$: $1 \cdot 1 \cdot \frac{2}{8} = \frac{2}{8}$
- $(x=1, y=2)$: $1 \cdot 2 \cdot \frac{1}{8} = \frac{2}{8}$
- $(x=2, y=1)$: $2 \cdot 1 \cdot \frac{1}{8} = \frac{2}{8}$
- $(x=2, y=2)$: $2 \cdot 2 \cdot \frac{1}{8} = \frac{4}{8}$

$$E[XY] = \frac{2+2+2+4}{8} = \frac{10}{8} = \frac{5}{4}$$

$$\text{Cov}(X,Y) = E[XY] - \mu_X \mu_Y = \frac{5}{4} - 1 \cdot 1 = \frac{1}{4} \checkmark$$

**Step 5: Method 3 — Linearity of covariance (most elegant).**

Let $X_i$ = result of $i$th flip ($X_i \sim \text{Bernoulli}(0.5)$, $\text{Var}(X_i) = 1/4$).

$X = X_1 + X_2$ and $Y = X_2 + X_3$.

$$\text{Cov}(X,Y) = \text{Cov}(X_1+X_2,\; X_2+X_3)$$
$$= \text{Cov}(X_1,X_2) + \text{Cov}(X_1,X_3) + \text{Cov}(X_2,X_2) + \text{Cov}(X_2,X_3)$$

Since different flips are independent: $\text{Cov}(X_i, X_j) = 0$ for $i \neq j$.

Only the shared flip $X_2$ contributes:
$$\text{Cov}(X,Y) = \text{Cov}(X_2, X_2) = \text{Var}(X_2) = \frac{1}{4}$$

**Final Answer:** $\text{Cov}(X,Y) = \dfrac{1}{4}$

**Interpretation:** The positive covariance makes sense: both $X$ and $Y$ include flip 2. If flip 2 lands heads, it contributes to both $X$ and $Y$. So large $X$ slightly predicts large $Y$.

---

#### Example — Zero Covariance ≠ Independence

**Problem (Example 2):** Let $X$ take values $\{-2,-1,0,1,2\}$ each with probability $1/5$. Let $Y = X^2$. Show $\text{Cov}(X,Y) = 0$ but $X$ and $Y$ are not independent.

**Joint table:**

| $Y \backslash X$ | -2 | -1 | 0 | 1 | 2 | $p(y_j)$ |
|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 1/5 | 0 | 0 | 1/5 |
| 1 | 0 | 1/5 | 0 | 1/5 | 0 | 2/5 |
| 4 | 1/5 | 0 | 0 | 0 | 1/5 | 2/5 |
| $p(x_i)$ | 1/5 | 1/5 | 1/5 | 1/5 | 1/5 | 1 |

**Step 1: Compute means.**

$$E[X] = \frac{1}{5}(-2-1+0+1+2) = 0$$
$$E[Y] = 0 \cdot \frac{1}{5} + 1 \cdot \frac{2}{5} + 4 \cdot \frac{2}{5} = \frac{10}{5} = 2$$

**Step 2: Compute $E[XY] = E[X \cdot X^2] = E[X^3]$.**

$$E[X^3] = \frac{1}{5}[(-8) + (-1) + 0 + 1 + 8] = \frac{0}{5} = 0$$

**Step 3: Compute covariance.**

$$\text{Cov}(X,Y) = E[XY] - \mu_X \mu_Y = 0 - 0 \cdot 2 = 0$$

**Step 4: Show they are NOT independent.**

$P(X = -2, Y = 0) = 0$ but $P(X = -2) \cdot P(Y = 0) = \frac{1}{5} \cdot \frac{1}{5} = \frac{1}{25} \neq 0$.

Product rule fails → NOT independent.

**Key insight:** $\text{Cov}(X, X^2) = 0$ even though $X$ and $X^2$ are **perfectly determined** by each other. The symmetric distribution cancels out all linear correlation. Covariance only captures linear relationships — it completely misses the quadratic relationship here.

---

#### Problem 2 — Covariance with 11 Coin Flips

**Problem:** Flip a fair coin 11 times. Let $X$ = heads in first 6 flips, $Y$ = heads in last 6 flips. Compute $\text{Cov}(X,Y)$ and $\text{Cor}(X,Y)$.

**Setup:**

Let $X_i$ = result of flip $i$, $X_i \sim \text{Bernoulli}(0.5)$, $\text{Var}(X_i) = 1/4$.

$$X = X_1 + X_2 + \cdots + X_6 \quad Y = X_6 + X_7 + \cdots + X_{11}$$

Note: flip 6 appears in BOTH $X$ and $Y$.

**Covariance by linearity:**

$$\text{Cov}(X,Y) = \text{Cov}\!\left(\sum_{i=1}^{6} X_i,\; \sum_{j=6}^{11} X_j\right) = \sum_{i=1}^{6}\sum_{j=6}^{11} \text{Cov}(X_i, X_j)$$

By independence, $\text{Cov}(X_i, X_j) = 0$ for $i \neq j$. The only nonzero term is when $i = j = 6$:

$$\text{Cov}(X,Y) = \text{Cov}(X_6, X_6) = \text{Var}(X_6) = \frac{1}{4}$$

**Correlation:**

$X$ = sum of 6 independent Bernoulli(0.5), so $\text{Var}(X) = 6/4$, giving $\sigma_X = \sqrt{3}/2$.
Similarly $\sigma_Y = \sqrt{3}/2$.

$$\text{Cor}(X,Y) = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y} = \frac{1/4}{(\sqrt{3}/2)(\sqrt{3}/2)} = \frac{1/4}{3/4} = \frac{1}{6}$$

**Interpretation:** The correlation is $1/6 \approx 0.167$, a mild positive correlation. This makes sense: out of 6 flips in each group, only 1 flip (flip 6) is shared. That single shared flip creates a small positive correlation.

---

#### Problem 3 — General $2n+1$ Coin Flips

**Problem:** Toss a fair coin $2n+1$ times. $X$ = heads in first $n+1$ tosses, $Y$ = heads in last $n+1$ tosses. Find $\text{Cov}(X,Y)$ and $\text{Cor}(X,Y)$.

**Setup:**

$$X = \sum_{i=1}^{n+1} X_i \qquad Y = \sum_{i=n+1}^{2n+1} X_i$$

The **only shared flip** is flip $n+1$.

**Covariance:**

By the same argument as above, only $\text{Cov}(X_{n+1}, X_{n+1}) = \text{Var}(X_{n+1}) = 1/4$ is nonzero.

$$\text{Cov}(X,Y) = \frac{1}{4}$$

This is **independent of $n$** — the covariance is always $1/4$.

**Variance of each sum:**

$X$ and $Y$ are each the sum of $n+1$ independent Bernoulli(0.5):

$$\text{Var}(X) = \text{Var}(Y) = \frac{n+1}{4} \quad \Rightarrow \quad \sigma_X = \sigma_Y = \sqrt{\frac{n+1}{4}} = \frac{\sqrt{n+1}}{2}$$

**Correlation:**

$$\text{Cor}(X,Y) = \frac{1/4}{\frac{\sqrt{n+1}}{2} \cdot \frac{\sqrt{n+1}}{2}} = \frac{1/4}{(n+1)/4} = \frac{1}{n+1}$$

**Key insight:** As $n$ increases, the correlation **decreases** like $1/(n+1)$. This makes intuitive sense: as each group contains more and more flips, the one shared flip becomes a smaller and smaller fraction of each sum, so its influence diminishes.

| $n$ | Group size $n+1$ | $\text{Cor}(X,Y)$ |
|---|---|---|
| 1 | 2 | 1/2 = 0.5 |
| 2 | 3 | 1/3 ≈ 0.333 |
| 5 | 6 | 1/6 ≈ 0.167 |
| 9 | 10 | 1/10 = 0.1 |
| 99 | 100 | 1/100 = 0.01 |

---

#### Example — Continuous Covariance (Example 3)

**Problem:** $f(x,y) = 2x^3 + 2y^3$ on $[0,1]^2$.

**(i) Verify validity:**

$$\int_0^1 \int_0^1 (2x^3 + 2y^3)\, dx\, dy$$

Inner integral: $\int_0^1 (2x^3 + 2y^3)\, dx = \left[\frac{x^4}{2} + 2xy^3\right]_0^1 = \frac{1}{2} + 2y^3$

Outer integral: $\int_0^1 \left(\frac{1}{2} + 2y^3\right) dy = \left[\frac{y}{2} + \frac{y^4}{2}\right]_0^1 = \frac{1}{2} + \frac{1}{2} = 1$ ✓

**(ii) Compute $\mu_X$ and $\mu_Y$:**

By symmetry ($f$ treats $x$ and $y$ identically), $\mu_X = \mu_Y$.

$$\mu_X = \int_0^1 \int_0^1 x(2x^3 + 2y^3)\, dx\, dy = \int_0^1 \int_0^1 (2x^4 + 2xy^3)\, dx\, dy = \frac{13}{20}$$

**(iii) Compute $\text{Cov}(X,Y)$:**

Using the shortcut $\text{Cov}(X,Y) = E[XY] - \mu_X \mu_Y$:

$$E[XY] = \int_0^1 \int_0^1 xy(2x^3 + 2y^3)\, dx\, dy = \int_0^1 \int_0^1 (2x^4 y + 2xy^4)\, dx\, dy$$

$$= 2\int_0^1 \left[\frac{x^5}{5}\right]_0^1 y\, dy + 2\int_0^1 x \left[\frac{y^5}{5}\right]_0^1 dx = 2 \cdot \frac{1}{5} \cdot \frac{1}{2} + 2 \cdot \frac{1}{2} \cdot \frac{1}{5} = \frac{1}{5} + \frac{1}{5} = \frac{2}{5} = 0.4$$

$$\text{Cov}(X,Y) = 0.4 - \left(\frac{13}{20}\right)^2 = 0.4 - 0.4225 = -0.0225 = -\frac{9}{400}$$

**Interpretation:** The negative covariance means that high values of $X$ tend to go with lower values of $Y$. Looking at $f(x,y) = 2x^3 + 2y^3$: the density is high when **either** $x$ or $y$ is large, but not necessarily when both are. This creates a slight tendency for large $X$ to pair with smaller $Y$.

---

#### Extra Problem 2a — Covariance via Joint Table (3 Flips)

**Problem:** Flip a coin 3 times. $X$ = heads in first 2 flips, $Y$ = heads in last 2 flips (same as Example 1 above). Compute $\text{Cor}(X,Y)$.

From Example 1: $\text{Cov}(X,Y) = 1/4$.

$X$ = sum of 2 independent Bernoulli(0.5): $\text{Var}(X) = 2 \cdot \frac{1}{4} = \frac{1}{2}$, so $\sigma_X = \frac{1}{\sqrt{2}}$.

Similarly $\sigma_Y = \frac{1}{\sqrt{2}}$.

$$\text{Cor}(X,Y) = \frac{1/4}{(1/\sqrt{2})(1/\sqrt{2})} = \frac{1/4}{1/2} = \frac{1}{2}$$

---

#### Extra Problem 2b — Covariance for 5 Flips, 3+3 Overlap

**Problem:** Flip a coin 5 times. $X$ = heads in first 3 flips ($X_1 + X_2 + X_3$), $Y$ = heads in last 3 flips ($X_3 + X_4 + X_5$). Only flip 3 is shared.

$$\text{Cov}(X,Y) = \text{Var}(X_3) = \frac{1}{4}$$

$\text{Var}(X) = 3 \cdot \frac{1}{4} = \frac{3}{4}$, so $\sigma_X = \frac{\sqrt{3}}{2}$. Similarly $\sigma_Y = \frac{\sqrt{3}}{2}$.

$$\text{Cor}(X,Y) = \frac{1/4}{({\sqrt{3}}/{2})({\sqrt{3}}/{2})} = \frac{1/4}{3/4} = \frac{1}{3}$$

---

## 8. Correlation

### 8.1 Concept Overview

Covariance has **units** (it's in units of $X$ times units of $Y$). This makes it hard to compare covariances across different variables or scales. **Correlation** removes the units by normalizing by the standard deviations.

### 8.2 Formal Definition

$$\text{Cor}(X,Y) = \rho = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}$$

### 8.3 Properties of Correlation

1. **$\rho$ is dimensionless** — it's a pure number, comparable across any variables.

2. **Bounded:** $-1 \leq \rho \leq 1$

3. **Extreme values:**
   - $\rho = +1$ if and only if $Y = aX + b$ for some $a > 0$ (perfect positive linear relationship)
   - $\rho = -1$ if and only if $Y = aX + b$ for some $a < 0$ (perfect negative linear relationship)

4. **Linear measure only:** $\rho$ measures the **linear** relationship between $X$ and $Y$. It completely misses nonlinear relationships (as shown in the $X, X^2$ example).

5. **Positive correlation:** When $X$ is large, $Y$ tends to be large. Data tends to lie in 1st and 3rd quadrants relative to the mean.

6. **Negative correlation:** When $X$ is large, $Y$ tends to be small. Data tends to lie in 2nd and 4th quadrants relative to the mean.

### 8.4 Proof that $-1 \leq \rho \leq 1$

For any random variables $X$ and $Y$ with standard deviations $\sigma_X, \sigma_Y$:

$$0 \leq \text{Var}\!\left(\frac{X}{\sigma_X} - \frac{Y}{\sigma_Y}\right) = \text{Var}\!\left(\frac{X}{\sigma_X}\right) + \text{Var}\!\left(\frac{Y}{\sigma_Y}\right) - 2\text{Cov}\!\left(\frac{X}{\sigma_X}, \frac{Y}{\sigma_Y}\right) = 1 + 1 - 2\rho = 2 - 2\rho$$

This implies $\rho \leq 1$.

Similarly $0 \leq \text{Var}(X/\sigma_X + Y/\sigma_Y) = 2 + 2\rho$ implies $\rho \geq -1$.

If $\rho = 1$ then $\text{Var}(X/\sigma_X - Y/\sigma_Y) = 0$, which means $X/\sigma_X - Y/\sigma_Y = c$ (constant), i.e., $Y = \frac{\sigma_Y}{\sigma_X} X - c\sigma_Y$, a perfect positive linear relationship. $\square$

---

### 8.5 Correlation Visual Guide

| $\rho$ value | Visual pattern |
|---|---|
| $\rho = 0.00$ | Circular cloud, no pattern |
| $\rho = 0.30$ | Slightly elongated ellipse, tilted up-right |
| $\rho = 0.70$ | Clearly elongated ellipse |
| $\rho = 1.00$ | Perfect line with positive slope |
| $\rho = -0.50$ | Ellipse tilted down-right |
| $\rho = -0.90$ | Tightly clustered near a line with negative slope |

**Key visual test:** Draw vertical and horizontal lines at the means of $X$ and $Y$. These create 4 quadrants. Positive correlation → data clusters in Q1 (upper-right) and Q3 (lower-left). Negative correlation → data clusters in Q2 (upper-left) and Q4 (lower-right).

---

### 8.6 Correlation from Overlapping Uniform Sums

In the simulations from MIT 18.05, $X_1, \ldots, X_{20}$ are i.i.d. $U(0,1)$. $X$ and $Y$ are both sums of $k$ of the $X_i$ with some overlap. The theoretical correlation is:

$$\rho = \frac{\text{overlap}}{\text{total terms in each sum}}$$

| Notation | $k$ (terms each) | Overlap | Theoretical $\rho$ | Simulated $\rho$ |
|---|---|---|---|---|
| $(1, 0)$ | 1 | 0 | 0.00 | -0.07 |
| $(2, 1)$ | 2 | 1 | 0.50 | 0.48 |
| $(5, 1)$ | 5 | 1 | 0.20 | 0.21 |
| $(5, 3)$ | 5 | 3 | 0.60 | 0.63 |
| $(10, 5)$ | 10 | 5 | 0.50 | 0.53 |
| $(10, 8)$ | 10 | 8 | 0.80 | 0.81 |

> **Formula:** For sums of $k$ i.i.d. terms with $m$ shared:
> $$\rho = \frac{m \cdot \text{Var}(X_i)}{k \cdot \text{Var}(X_i)} = \frac{m}{k}$$

This can be derived using the linearity of covariance:
$$\text{Cov}(X,Y) = m \cdot \text{Var}(X_i), \qquad \text{Var}(X) = \text{Var}(Y) = k \cdot \text{Var}(X_i)$$
$$\rho = \frac{m \cdot \text{Var}(X_i)}{k \cdot \text{Var}(X_i)} = \frac{m}{k}$$

---

### 8.7 Common Mistakes with Covariance and Correlation

| Mistake | Correction |
|---|---|
| Concluding independence from zero covariance | Zero covariance ≠ independence (see $X, X^2$ example) |
| Thinking $|\rho| = 1$ means exact equality | It means a perfect **linear** relationship, not $X = Y$ |
| Confusing $\text{Cov}(X,Y)$ scale with $\rho$ | Covariance depends on units; correlation is always in $[-1,1]$ |
| Applying $\text{Var}(X+Y) = \text{Var}(X) + \text{Var}(Y)$ when not independent | Only valid when $\text{Cov}(X,Y) = 0$ |
| Inferring causation from correlation | Correlation only measures statistical association, not causation |

---

## 9. Bivariate Normal Distribution

### 9.1 Definition

The **bivariate normal distribution** is the joint distribution of two normally distributed variables with a specified correlation $\rho$.

$$f(x,y) = \frac{1}{2\pi \sigma_X \sigma_Y \sqrt{1-\rho^2}} \exp\!\left\{-\frac{1}{2(1-\rho^2)}\left[\frac{(x-\mu_X)^2}{\sigma_X^2} + \frac{(y-\mu_Y)^2}{\sigma_Y^2} - \frac{2\rho(x-\mu_X)(y-\mu_Y)}{\sigma_X \sigma_Y}\right]\right\}$$

### 9.2 Key Properties

- **Marginal distributions:** Each marginal is normal. $X \sim N(\mu_X, \sigma_X^2)$ and $Y \sim N(\mu_Y, \sigma_Y^2)$.
- **Correlation parameter:** The parameter $\rho$ in the formula is exactly the correlation between $X$ and $Y$.
- **Independence:** For the bivariate normal, $\rho = 0$ is both necessary AND sufficient for independence. (This is a special property of the bivariate normal — in general, zero covariance does not imply independence!)

### 9.3 Why the Bivariate Normal Matters for ML

The bivariate normal is the foundation for:
- **Linear regression** (joint normal model)
- **Gaussian processes** (extension to infinite dimensions)
- **Principal Component Analysis** (finding principal axes of a bivariate/multivariate normal)
- **Kalman filters** (state estimation in robotics, finance)
- **Gaussian copulas** (modeling dependence in finance)

---

## 10. Real-Life Correlations and Causation

### 10.1 Correlation Does Not Imply Causation

From the MIT 18.05 notes, several examples illustrate this critical principle:

---

**Example A: Ice Cream and Drownings**

*Ice cream consumption is correlated with pool drownings.*

**Not** because ice cream causes drowning. Both are caused by a **common confounder**: hot summer weather drives both ice cream sales and pool attendance.

```
Hot Weather → More Ice Cream Consumption
Hot Weather → More Swimming → More Drownings
```

This creates correlation between ice cream and drownings with **no causal link** between them.

---

**Example B: Students and Early Death (1685 Paradox)**

*In a 1685 study, "student" was the profession with the lowest average age at death.*

**Not** because being a student causes early death. Students are young by definition. The study looked at the ages of people who had already died — students who died were necessarily young (many professions require age/experience to practice). The confound is **selection bias on age**.

---

**Example C: Bar Fight Survival**

*In 90% of bar fights ending in death, the person who started the fight died.*

**Interpretation problem:** The data comes from **survivors reporting the story**. The person who started the fight may have died, or the survivor may simply report the other person "started it." This is survivorship bias and reporting bias.

---

**Example D: Hormone Replacement Therapy (HRT)**

*Epidemiological studies showed women taking HRT had lower coronary heart disease (CHD) rates.*

**Initial (wrong) conclusion:** HRT protects against CHD.

**Randomized controlled trials:** HRT actually causes a **small increase** in CHD risk.

**What happened:** Women taking HRT were systematically from higher socioeconomic groups, with better diet and exercise habits. These habits — not HRT — were responsible for lower CHD rates. HRT and lower CHD were both effects of a **common cause** (high socioeconomic status), not a cause-and-effect relationship.

---

### 10.2 Edward Tufte's Rule

> *"Empirically observed covariation is a necessary but not sufficient condition for causality."*

**Translation:** You need correlation to claim causation, but correlation alone is never enough. You also need:

1. **Temporal ordering** — cause precedes effect
2. **Mechanism** — a plausible causal pathway
3. **Elimination of confounders** — ruling out common causes
4. **Randomized experiment** (gold standard) — random assignment eliminates confounders

---

## 11. Hospital Problem — Extended Application

### 11.1 Problem Setup

- **Larger hospital:** ~45 babies born per day
- **Smaller hospital:** ~15 babies born per day
- Each hospital recorded for 1 year how many days had **>60% boys born**

**Question (a):** Which hospital recorded more such days?

**Initial intuition of most people:** About the same (55 of 97 undergraduates thought this).

**Correct answer:** The **smaller hospital**, by a wide margin.

---

### 11.2 Why Smaller Samples Show More Extreme Deviations

By the CLT/LLN, the fraction of boys $\bar{X}$ in a sample of size $n$ has standard deviation:

$$\sigma_{\bar{X}} = \sqrt{\frac{p(1-p)}{n}} = \sqrt{\frac{0.25}{n}}$$

| Hospital | $n$ | $\sigma_{\bar{X}}$ |
|---|---|---|
| Larger | 45 | $\sqrt{0.25/45} \approx 0.0745$ |
| Smaller | 15 | $\sqrt{0.25/15} \approx 0.129$ |

The smaller hospital has nearly **twice** the standard deviation. More variability means more days where the fraction exceeds 60%.

---

### 11.3 Part (b): Computing Daily Probabilities

**Larger hospital:** $X_L \sim \text{Bin}(45, 0.5)$

$$p_L = P(X_L > 27) = \sum_{k=28}^{45} \binom{45}{k} 0.5^{45} \approx 0.068$$

(>60% of 45 babies means >27 boys)

**Smaller hospital:** $X_S \sim \text{Bin}(15, 0.5)$

$$p_S = P(X_S > 9) = \sum_{k=10}^{15} \binom{15}{k} 0.5^{15} \approx 0.151$$

(>60% of 15 babies means >9 boys)

$p_S \approx 0.151 > p_L \approx 0.068$ — the smaller hospital has **more than twice** the daily probability of seeing >60% boys.

---

### 11.4 Part (c): Annual Distributions

$L$ = number of days (out of 365) with >60% boys in larger hospital:

$$L \sim \text{Bin}(365, p_L) \qquad E[L] = 365 \cdot 0.068 \approx 25 \qquad \text{Var}(L) \approx 23$$

$S$ = number of days in smaller hospital:

$$S \sim \text{Bin}(365, p_S) \qquad E[S] = 365 \cdot 0.151 \approx 55 \qquad \text{Var}(S) \approx 47$$

---

### 11.5 Part (d): 0.84 Quantiles via CLT

By the CLT, $L \approx N(25, 23)$ and $S \approx N(55, 47)$.

The 0.84 quantile of $N(\mu, \sigma^2)$ is approximately $\mu + \sigma$ (since $\Phi(1) \approx 0.84$).

$$q_{0.84}(L) \approx 25 + \sqrt{23} \approx 25 + 4.8 = 29.8$$
$$q_{0.84}(S) \approx 55 + \sqrt{47} \approx 55 + 6.9 = 61.9$$

So in about 84% of years, the larger hospital will see fewer than 30 such days, while the smaller hospital will see fewer than 62 such days. The smaller hospital typically has more than double the count.

---

### 11.6 Part (e): Joint Distribution and $P(L > S)$

**Are $L$ and $S$ correlated?** No — different babies are born at different hospitals, so $L$ and $S$ are **independent**. $\text{Cor}(L,S) = 0$.

**Joint pmf:**

$$P(L = i,\; S = j) = \binom{365}{i} p_L^i (1-p_L)^{365-i} \cdot \binom{365}{j} p_S^j (1-p_S)^{365-j}$$

**$P(L > S)$:**

$$P(L > S) = \sum_{i=0}^{364} \sum_{j=0}^{i-1} P(L=i, S=j) \approx 0.0000916$$

The probability that the larger hospital had **more** extreme days than the smaller is extremely small — less than 0.01%. This confirms overwhelmingly that the smaller hospital will consistently see more extreme deviations.

**Lesson:** Sample size determines how often you see extreme deviations from the true mean. Small samples are "noisy" and show more extreme values. This is critical for understanding why medical studies need large sample sizes to be reliable.

---

## 12. Quick Reference Summary

### Core Formulas

| Concept | Formula |
|---|---|
| Joint PMF validity | $\sum_i \sum_j p(x_i, y_j) = 1$ |
| Joint PDF validity | $\int\int f(x,y)\, dx\, dy = 1$ |
| Joint CDF (continuous) | $F(x,y) = \int \int f(u,v)\, du\, dv$ |
| PDF from CDF | $f(x,y) = \partial^2 F / \partial x \partial y$ |
| Marginal PDF | $f_X(x) = \int f(x,y)\, dy$ |
| Marginal CDF | $F_X(x) = F(x, d)$ |
| Independence condition | $f(x,y) = f_X(x) \cdot f_Y(y)$ |
| Covariance (definition) | $\text{Cov}(X,Y) = E[(X-\mu_X)(Y-\mu_Y)]$ |
| Covariance (shortcut) | $\text{Cov}(X,Y) = E[XY] - \mu_X \mu_Y$ |
| Variance of sum | $\text{Var}(X+Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X,Y)$ |
| Correlation | $\rho = \text{Cov}(X,Y) / (\sigma_X \sigma_Y)$ |
| Correlation range | $-1 \leq \rho \leq 1$ |

### Independence Checklist

| Test | What to Check |
|---|---|
| Discrete | $p(x_i, y_j) = p_X(x_i) \cdot p_Y(y_j)$ for ALL cells |
| Continuous | $f(x,y)$ factors as $g(x) \cdot h(y)$ |
| Support | Joint range must be a rectangle |
| Covariance | $\text{Cov}=0$ is necessary but not sufficient |

### Covariance Properties at a Glance

```
Cov(X, X)    = Var(X)                        [Property 3]
Cov(aX, bY)  = ab·Cov(X,Y)                  [Property 1]
Cov(X+Z, Y)  = Cov(X,Y) + Cov(Z,Y)          [Property 2]
Cov(X,Y)     = E[XY] - E[X]E[Y]             [Property 4]
X⊥Y  ⟹  Cov(X,Y)=0,  but NOT vice versa   [Property 6]
```

### The Three Ways to Compute Covariance

1. **Direct definition:** $\sum_{i,j} p(x_i,y_j)(x_i-\mu_X)(y_j-\mu_Y)$
2. **Shortcut:** $E[XY] - \mu_X\mu_Y$
3. **Linearity (for sums):** Expand into sum of $\text{Cov}(X_i, Y_j)$, keep only shared terms

---
# FAANG Interview Q&A — MIT 18.05 Class 7
## Joint Distributions · Independence · Covariance · Correlation

> **Scope:** Based on MIT 18.05 Class 7 (Spring 2022) — Orloff & Bloom
> **Target:** Data Science · ML Engineering · Quantitative Research · Applied Scientist · SWE ML Track
> **Companies:** Google · Meta · Amazon · Apple · Netflix · Jane Street · Two Sigma · Citadel · Stripe

---

## Master Index

| # | Question | Difficulty | Topic |
|---|---|---|---|
| Q1 | What is a joint distribution and why does it matter in ML? | 🟢 Easy | Joint Distributions |
| Q2 | How do you test whether two discrete random variables are independent? | 🟢 Easy | Independence |
| Q3 | How do you test independence for continuous random variables? | 🟢 Easy | Independence |
| Q4 | Given $f(x,y) = x+y$ on $[0,1]^2$, verify it's a valid pdf and find $P(X > 0.3, Y > 0.5)$ | 🟡 Medium | Joint PDF |
| Q5 | What is a marginal distribution? How do you compute it? | 🟢 Easy | Marginals |
| Q6 | How do you find the marginal CDF from the joint CDF? | 🟡 Medium | Marginal CDF |
| Q7 | Define covariance. What are its key properties? | 🟢 Easy | Covariance |
| Q8 | Prove that $\text{Cov}(X,Y) = E[XY] - \mu_X \mu_Y$ | 🟡 Medium | Covariance Proof |
| Q9 | Prove that $\text{Var}(X+Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X,Y)$ | 🟡 Medium | Variance of Sum |
| Q10 | Flip a fair coin 11 times. $X$ = heads in first 6, $Y$ = heads in last 6. Find $\text{Cov}(X,Y)$ and $\text{Cor}(X,Y)$ | 🟡 Medium | Covariance Application |
| Q11 | Can $\text{Cov}(X,Y) = 0$ but $X$ and $Y$ not be independent? Prove it with an example. | 🔴 Hard | Covariance Trap |
| Q12 | What is correlation? Why is it preferred over covariance? | 🟢 Easy | Correlation |
| Q13 | Prove $-1 \leq \rho \leq 1$ | 🔴 Hard | Correlation Bound |
| Q14 | What is the bivariate normal distribution? When is $\rho=0$ sufficient for independence? | 🟡 Medium | Bivariate Normal |
| Q15 | Two hospitals: large (45 births/day) and small (15 births/day). Which records more days with >60% boys? | 🟡 Medium | Sampling / CLT |
| Q16 | Explain how correlation arises from overlapping coin flips. Generalize the formula. | 🟡 Medium | Correlation via Linearity |
| Q17 | "Ice cream sales correlate with drowning deaths. Does ice cream cause drowning?" | 🟢 Easy | Causation |
| Q18 | A feature has zero correlation with the label. Should you drop it from your ML model? | 🔴 Hard | ML Application |
| Q19 | How do you compute $F(3.5, 4)$ from a discrete joint probability table? | 🟡 Medium | Joint CDF Discrete |
| Q20 | Why does a smaller hospital show more extreme deviations from 50% boys than a larger one? Connect to variance. | 🔴 Hard | CLT + Variance + Sampling |

---

## Questions & Full Answers

---

### Q1 🟢 — What Is a Joint Distribution and Why Does It Matter in ML?

**Question:** What is a joint distribution? Why is it important in machine learning and data science?

---

**Answer:**

A **joint distribution** of two random variables $X$ and $Y$ is a complete description of the probability of every possible pair of outcomes $(x, y)$ simultaneously.

**Discrete:** The joint PMF is $p(x_i, y_j) = P(X = x_i \text{ and } Y = y_j)$, organized as a table.

**Continuous:** The joint PDF $f(x,y)$ satisfies $P((X,Y) \in R) = \iint_R f(x,y)\, dx\, dy$.

Both must satisfy: (1) non-negativity, and (2) total probability = 1.

**Why it matters for ML/DS:**

| Application | Why Joint Distributions Matter |
|---|---|
| **Feature engineering** | Understand correlations between features; remove redundant ones |
| **Generative models** | Learn the joint distribution $p(x, y)$ of data and labels |
| **Bayesian inference** | Posterior $\propto$ likelihood × prior is a joint distribution over parameters and data |
| **A/B testing** | Model joint behavior of test metrics to control false discovery rate |
| **Anomaly detection** | A point may be normal marginally but anomalous in the joint space |
| **Causal inference** | Joint distributions encode correlations; causal models encode interventions |

> **FAANG context:** At Meta, the joint distribution of (user engagement, ad click rate) is key to ad ranking models. At Google, the joint distribution of (query, document) determines ranking scores. Understanding joint behavior — not just marginals — drives product quality.

---

### Q2 🟢 — Testing Independence for Discrete Variables

**Question:** How do you check whether two discrete random variables are independent from their joint probability table?

---

**Answer:**

**Definition:** $X$ and $Y$ are independent if and only if for **every** pair $(x_i, y_j)$:

$$p(x_i, y_j) = p_X(x_i) \cdot p_Y(y_j)$$

**Step-by-step procedure:**

1. Compute all **row marginals** $p_X(x_i) = \sum_j p(x_i, y_j)$
2. Compute all **column marginals** $p_Y(y_j) = \sum_i p(x_i, y_j)$
3. For every cell, check whether $p(x_i, y_j) = p_X(x_i) \cdot p_Y(y_j)$
4. If **any** cell fails → **not independent**

**Fast shortcut to show dependence:** Find any single cell where the product rule fails. You only need ONE counterexample.

**Example 1 — Independent (Two dice $X$, $Y$):**

Every cell = $1/36$, every marginal = $1/6$, and $1/36 = (1/6)(1/6)$. ✓ Independent.

**Example 2 — Dependent ($X$, $T$ = sum):**

$P(X=1, T=8) = 0$ but $P(X=1) \cdot P(T=8) = (1/6)(5/36) \neq 0$. ✗ Not independent.

> **Intuition:** If knowing $X$ restricts which values $T$ can take (it does — if $X=1$, then $T \leq 7$), they cannot be independent.

**Common Mistake:** Checking only a few cells and concluding independence. You must verify the product rule holds at **all** cells, or equivalently show the joint table can be reconstructed as an outer product of the two marginal vectors.

---

### Q3 🟢 — Testing Independence for Continuous Variables

**Question:** How do you test independence for continuous random variables given their joint PDF?

---

**Answer:**

**Definition:** Continuous $X$ and $Y$ are independent if and only if:

$$f(x,y) = f_X(x) \cdot f_Y(y) \quad \text{for all } (x,y)$$

**Fastest test — the factorization shortcut:**

$X$ and $Y$ are independent if and only if $f(x,y)$ can be written as:

$$f(x,y) = g(x) \cdot h(y)$$

for **any** functions $g$ and $h$ (they don't need to be normalized individually).

**Three cases from MIT 18.05:**

| PDF | Factors? | Independent? |
|---|---|---|
| $4x^2 y^3$ | $= (4x^2)(y^3)$ → yes | ✓ Yes |
| $\frac{1}{2}(x^3 y + xy^3) = \frac{xy}{2}(x^2+y^2)$ | Cannot separate $(x^2+y^2)$ | ✗ No |
| $6e^{-3x-2y} = 6e^{-3x} \cdot e^{-2y}$ | Yes — exponential of sum | ✓ Yes |

**Critical requirement:** The joint **support** (range) must be a **rectangle**. If the support is a triangle, disk, or any non-rectangular region, $X$ and $Y$ are automatically dependent, regardless of whether $f$ appears to factor.

> **Example of support dependence:** $f(x,y) = 2$ on the triangle $\{0 < x < 1, 0 < y < x\}$. Even though $f$ is constant (trivially a product), $X$ and $Y$ are NOT independent because the support is not rectangular — knowing $X = 0.2$ tells you $Y < 0.2$.

---

### Q4 🟡 — Validate a Joint PDF and Compute a Probability

**Question:** Given $f(x,y) = x + y$ on $[0,1]^2$:
(a) Verify this is a valid joint PDF.
(b) Find $P(X > 0.3, Y > 0.5)$.
(c) Find the marginal PDF $f_X(x)$.

---

**Answer:**

**(a) Validity:**

**Non-negativity:** $x + y \geq 0$ for $x, y \in [0,1]$. ✓

**Total probability:**
$$\int_0^1 \int_0^1 (x+y)\, dx\, dy = \int_0^1 \left[\frac{x^2}{2} + xy\right]_0^1 dy = \int_0^1 \left(\frac{1}{2} + y\right) dy = \frac{1}{2} + \frac{1}{2} = 1 \checkmark$$

**(b) $P(X > 0.3, Y > 0.5)$:**

$$P(A) = \int_{0.5}^{1} \int_{0.3}^{1} (x+y)\, dx\, dy$$

**Inner integral** (over $x$):
$$\int_{0.3}^{1} (x+y)\, dx = \left[\frac{x^2}{2} + xy\right]_{0.3}^{1} = \left(\frac{1}{2} + y\right) - \left(\frac{0.09}{2} + 0.3y\right) = 0.455 + 0.7y$$

**Outer integral** (over $y$):
$$\int_{0.5}^{1} (0.455 + 0.7y)\, dy = \left[0.455y + 0.35y^2\right]_{0.5}^{1} = 0.805 - 0.315 = 0.49$$

**Answer: $P(X > 0.3, Y > 0.5) = 0.49$**

**(c) Marginal PDF $f_X(x)$:**

Integrate out $y$:
$$f_X(x) = \int_0^1 (x+y)\, dy = \left[xy + \frac{y^2}{2}\right]_0^1 = x + \frac{1}{2}$$

**Verification:** $\int_0^1 (x + 1/2)\, dx = [x^2/2 + x/2]_0^1 = 1/2 + 1/2 = 1$ ✓

> **FAANG context:** This type of problem — validating a density and computing probabilities — appears in ML inference pipelines when you need to verify that a learned density model is properly normalized, and when computing likelihoods over regions of feature space.

---

### Q5 🟢 — What Is a Marginal Distribution?

**Question:** What is a marginal distribution? How do you compute it from a joint distribution? Give an intuition for what it represents.

---

**Answer:**

The **marginal distribution** of $X$ is the distribution of $X$ alone, obtained by "averaging out" (integrating or summing over) all values of $Y$.

**Discrete:**
$$p_X(x_i) = \sum_j p(x_i, y_j)$$

**Continuous:**
$$f_X(x) = \int_{-\infty}^{\infty} f(x,y)\, dy$$

**Why "marginal"?** In a joint probability table, the marginals appear in the **margins** (border rows and columns) when you add up rows or columns.

**Intuition:** The marginal is what you'd observe if you ran the experiment but **only recorded** $X$ and ignored $Y$. It collapses the 2D distribution down to 1D.

**Example:** If $(X, Y)$ models (height, weight) of people, $f_X(x)$ is just the height distribution — the distribution you'd get if you measured everyone's height and forgot about weight.

**Key property:** The marginals do NOT tell you the joint distribution unless $X \perp Y$. Two very different joint distributions can have the same marginals.

> **ML implication:** A model trained on marginal distributions of features cannot capture feature interactions. That's why joint modeling (e.g., tree-based models, attention mechanisms) outperforms models that treat features as independent.

---

### Q6 🟡 — Marginal CDF from Joint CDF

**Question:** If the joint CDF of $(X,Y)$ on $[a,b] \times [c,d]$ is $F(x,y)$, how do you extract the marginal CDF of $X$?

---

**Answer:**

$$F_X(x) = F(x,\; d)$$

Set $y$ equal to the **upper boundary** of $Y$'s range.

**Why this works:** $F_X(x) = P(X \leq x) = P(X \leq x \text{ and } Y \leq \infty)$. Setting $y = d$ (or $y \to \infty$) means "no restriction on $Y$", which gives the marginal.

Similarly: $F_Y(y) = F(b, y)$ — set $x$ to the right boundary.

**Example:** Given $F(x,y) = \dfrac{x^2 y + xy^2}{2}$ on $[0,1]^2$:

$$F_X(x) = F(x, 1) = \frac{x^2(1) + x(1)^2}{2} = \frac{x^2 + x}{2}$$

$$P(X < 0.5) = F_X(0.5) = \frac{0.25 + 0.5}{2} = \frac{3}{8}$$

**Sanity check:** $F_X(0) = 0$ ✓ and $F_X(1) = (1+1)/2 = 1$ ✓

---

### Q7 🟢 — Covariance: Definition and Properties

**Question:** Define covariance. List its key properties. What does it measure?

---

**Answer:**

**Definition:**

$$\text{Cov}(X,Y) = E\bigl[(X - \mu_X)(Y - \mu_Y)\bigr]$$

**Intuition:** Covariance measures whether $X$ and $Y$ deviate from their means **in the same direction** (positive covariance) or **opposite directions** (negative covariance).

- When $X > \mu_X$ and $Y > \mu_Y$ simultaneously → product $(X-\mu_X)(Y-\mu_Y) > 0$ → positive contribution
- When $X > \mu_X$ but $Y < \mu_Y$ → product $< 0$ → negative contribution

**Six Key Properties:**

| # | Property | Formula | When Used |
|---|---|---|---|
| 1 | Scaling | $\text{Cov}(aX+b, cY+d) = ac\,\text{Cov}(X,Y)$ | Unit conversion |
| 2 | Bilinearity | $\text{Cov}(X_1+X_2, Y) = \text{Cov}(X_1,Y) + \text{Cov}(X_2,Y)$ | Expanding sums |
| 3 | Self = Variance | $\text{Cov}(X,X) = \text{Var}(X)$ | Diagonal of covariance matrix |
| 4 | Shortcut | $\text{Cov}(X,Y) = E[XY] - \mu_X\mu_Y$ | Computation |
| 5 | Var of sum | $\text{Var}(X+Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X,Y)$ | Portfolio variance, error propagation |
| 6 | Independence → 0 | $X \perp Y \Rightarrow \text{Cov}(X,Y)=0$ | Simplifying calculations |

**Warning:** Property 6 does NOT reverse. Zero covariance does NOT imply independence.

**Units:** $\text{Cov}(X,Y)$ has units of (unit of $X$) × (unit of $Y$). This is why raw covariance is hard to interpret across different problems.

---

### Q8 🟡 — Prove $\text{Cov}(X,Y) = E[XY] - \mu_X \mu_Y$

**Question:** Derive the shortcut covariance formula from the definition.

---

**Answer:**

**Starting point:** $\text{Cov}(X,Y) = E[(X-\mu_X)(Y-\mu_Y)]$

**Step 1:** Expand the product inside the expectation.

$$\text{Cov}(X,Y) = E[XY - \mu_X Y - \mu_Y X + \mu_X \mu_Y]$$

**Step 2:** Use linearity of expectation.

$$= E[XY] - \mu_X E[Y] - \mu_Y E[X] + \mu_X \mu_Y$$

**Step 3:** Substitute $E[X] = \mu_X$ and $E[Y] = \mu_Y$.

$$= E[XY] - \mu_X \mu_Y - \mu_Y \mu_X + \mu_X \mu_Y$$

**Step 4:** Simplify.

$$= E[XY] - \mu_X \mu_Y \quad \square$$

**Why this matters:** The shortcut is computationally faster. Instead of computing the joint expectation of $(X-\mu_X)(Y-\mu_Y)$, you compute $E[XY]$ (from the joint distribution) and subtract the product of the means.

**Analogy to variance:** Recall $\text{Var}(X) = E[X^2] - \mu^2$. This formula is exactly that formula with $Y$ replacing the second $X$.

---

### Q9 🟡 — Prove $\text{Var}(X+Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X,Y)$

**Question:** Derive the formula for the variance of a sum of two (possibly dependent) random variables.

---

**Answer:**

**Step 1:** Use Property 3: $\text{Var}(Z) = \text{Cov}(Z, Z)$.

$$\text{Var}(X+Y) = \text{Cov}(X+Y,\; X+Y)$$

**Step 2:** Apply bilinearity (Property 2) to expand.

$$= \text{Cov}(X, X+Y) + \text{Cov}(Y, X+Y)$$
$$= \text{Cov}(X,X) + \text{Cov}(X,Y) + \text{Cov}(Y,X) + \text{Cov}(Y,Y)$$

**Step 3:** Use Property 3 and symmetry ($\text{Cov}(X,Y) = \text{Cov}(Y,X)$).

$$= \text{Var}(X) + \text{Cov}(X,Y) + \text{Cov}(X,Y) + \text{Var}(Y)$$

$$= \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X,Y) \quad \square$$

**Corollaries:**

- If $X \perp Y$: $\text{Var}(X+Y) = \text{Var}(X) + \text{Var}(Y)$ (covariance term vanishes)
- If $X$ and $Y$ are positively correlated: $\text{Var}(X+Y) > \text{Var}(X) + \text{Var}(Y)$
- If $X$ and $Y$ are negatively correlated: $\text{Var}(X+Y) < \text{Var}(X) + \text{Var}(Y)$ — **diversification effect**

> **FAANG context:** In portfolio theory (used in quant finance at Jane Street, Two Sigma), combining two negatively correlated assets reduces total variance — this is mathematical diversification. The formula shows exactly why: the $2\text{Cov}(X,Y)$ term is negative when $\rho < 0$.

---

### Q10 🟡 — Coin Flip Covariance and Correlation

**Question:** Flip a fair coin 11 times. Let $X$ = number of heads in the first 6 flips, $Y$ = number of heads in the last 6 flips. Note: flip 6 appears in both. Compute $\text{Cov}(X,Y)$ and $\text{Cor}(X,Y)$.

---

**Answer:**

**Step 1: Decompose into individual flips.**

Let $X_i = $ result of flip $i$ ($X_i \sim \text{Bernoulli}(0.5)$, $\text{Var}(X_i) = 1/4$).

$$X = X_1 + X_2 + X_3 + X_4 + X_5 + X_6$$
$$Y = X_6 + X_7 + X_8 + X_9 + X_{10} + X_{11}$$

**Step 2: Apply bilinearity of covariance.**

$$\text{Cov}(X,Y) = \sum_{i=1}^{6} \sum_{j=6}^{11} \text{Cov}(X_i, X_j)$$

**Step 3: Use independence of different flips.**

$\text{Cov}(X_i, X_j) = 0$ for $i \neq j$ (independent flips).

The only nonzero term is $i = j = 6$:

$$\text{Cov}(X,Y) = \text{Cov}(X_6, X_6) = \text{Var}(X_6) = \frac{1}{4}$$

**Step 4: Compute standard deviations.**

$X$ = sum of 6 independent Bernoulli(0.5): $\text{Var}(X) = 6 \cdot \frac{1}{4} = \frac{3}{2}$, so $\sigma_X = \sqrt{\frac{3}{2}} = \frac{\sqrt{6}}{2}$.

Similarly $\sigma_Y = \frac{\sqrt{6}}{2}$.

**Step 5: Compute correlation.**

$$\text{Cor}(X,Y) = \frac{1/4}{({\sqrt{6}}/{2})({\sqrt{6}}/{2})} = \frac{1/4}{6/4} = \frac{1}{6} \approx 0.167$$

**Interpretation:** The mild positive correlation of $1/6$ comes entirely from the single shared flip. Out of 6 flips in each group, 1 is shared — so the theoretical correlation is exactly $1/6$. This matches the general formula $\rho = \text{overlap}/\text{group size} = 1/6$.

---

### Q11 🔴 — Zero Covariance But Not Independent

**Question:** Provide a concrete example where $\text{Cov}(X,Y) = 0$ but $X$ and $Y$ are NOT independent. Explain why this doesn't contradict the rule "independence implies zero covariance."

---

**Answer:**

**Example:** Let $X$ take values $\{-2, -1, 0, 1, 2\}$, each with probability $1/5$. Let $Y = X^2$.

**Step 1: Compute $\text{Cov}(X,Y) = E[XY] - E[X]E[Y]$.**

$E[X] = \frac{1}{5}(-2 - 1 + 0 + 1 + 2) = 0$ (symmetric distribution)

$E[XY] = E[X \cdot X^2] = E[X^3] = \frac{1}{5}(-8 - 1 + 0 + 1 + 8) = 0$

$$\text{Cov}(X,Y) = 0 - 0 \cdot E[Y] = 0$$

**Step 2: Show they are NOT independent.**

$P(X = -2, Y = 0) = 0$ (since if $X = -2$ then $Y = 4 \neq 0$)

But $P(X = -2) \cdot P(Y = 0) = \frac{1}{5} \cdot \frac{1}{5} = \frac{1}{25} \neq 0$

Product rule fails → **NOT independent**. In fact, $Y$ is **completely determined** by $X$; they are as dependent as two variables can be.

**Why doesn't this contradict the rule?**

The rule is: $X \perp Y \Rightarrow \text{Cov}(X,Y) = 0$.

The **converse** is: $\text{Cov}(X,Y) = 0 \Rightarrow X \perp Y$ — this is **FALSE** in general.

Covariance only captures **linear** relationships. The relationship $Y = X^2$ is **quadratic** (perfectly nonlinear), which covariance cannot detect because the symmetric distribution makes all odd moments zero.

**Practical implication:**

| Measure | What It Detects |
|---|---|
| Covariance / Correlation | Linear relationships only |
| Mutual information | Any statistical dependence |
| Distance correlation | Any non-trivial dependence |
| Spearman correlation | Monotonic (not just linear) relationships |

> **FAANG context:** In feature selection at Google or Meta, using only Pearson correlation to filter features can miss nonlinear dependencies. This is why mutual information or model-based feature importance (e.g., random forest feature importance) is often preferred over correlation for feature selection.

---

### Q12 🟢 — What Is Correlation and Why Is It Preferred?

**Question:** What is the Pearson correlation coefficient? Why is it preferred over raw covariance?

---

**Answer:**

**Definition:**

$$\rho = \text{Cor}(X,Y) = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}$$

**Properties:**

1. **Dimensionless:** No units — always a pure number in $[-1, 1]$
2. **Scale-invariant:** $\text{Cor}(aX, bY) = \text{Cor}(X, Y)$ for $a, b > 0$
3. **Bounded:** $-1 \leq \rho \leq 1$
4. **Extremes:** $\rho = 1$ iff $Y = aX + b$ ($a > 0$); $\rho = -1$ iff $Y = aX + b$ ($a < 0$)
5. **Measures linear association:** $\rho = 0$ means no linear relationship (but may have nonlinear)

**Why preferred over covariance:**

| Feature | Covariance | Correlation |
|---|---|---|
| Units | Unit$_X$ × Unit$_Y$ | Dimensionless |
| Scale-invariant | No | Yes |
| Interpretable magnitude | No | Yes ($[-1,1]$) |
| Comparable across problems | No | Yes |

**Example:** Height (cm) and weight (kg) have a very different covariance than height (inches) and weight (lbs), even though they measure the same relationship. Correlation is identical in both cases.

**Interpretation guide:**

| $|\rho|$ range | Interpretation |
|---|---|
| 0.0 – 0.1 | Negligible |
| 0.1 – 0.3 | Weak |
| 0.3 – 0.5 | Moderate |
| 0.5 – 0.7 | Strong |
| 0.7 – 0.9 | Very strong |
| 0.9 – 1.0 | Near-perfect linear |

---

### Q13 🔴 — Prove $-1 \leq \rho \leq 1$

**Question:** Prove that the correlation coefficient is always between $-1$ and $+1$.

---

**Answer:**

**Proof using variance non-negativity:**

**Step 1:** Variance is always non-negative. Consider:

$$0 \leq \text{Var}\!\left(\frac{X}{\sigma_X} - \frac{Y}{\sigma_Y}\right)$$

**Step 2:** Expand using $\text{Var}(A - B) = \text{Var}(A) + \text{Var}(B) - 2\text{Cov}(A,B)$:

$$0 \leq \text{Var}\!\left(\frac{X}{\sigma_X}\right) + \text{Var}\!\left(\frac{Y}{\sigma_Y}\right) - 2\text{Cov}\!\left(\frac{X}{\sigma_X}, \frac{Y}{\sigma_Y}\right)$$

**Step 3:** Evaluate each term:

- $\text{Var}(X/\sigma_X) = \text{Var}(X)/\sigma_X^2 = \sigma_X^2/\sigma_X^2 = 1$
- $\text{Var}(Y/\sigma_Y) = 1$
- $\text{Cov}(X/\sigma_X, Y/\sigma_Y) = \text{Cov}(X,Y)/(\sigma_X \sigma_Y) = \rho$

$$0 \leq 1 + 1 - 2\rho = 2 - 2\rho$$

$$\Rightarrow \rho \leq 1$$

**Step 4:** Repeat with the sum:

$$0 \leq \text{Var}\!\left(\frac{X}{\sigma_X} + \frac{Y}{\sigma_Y}\right) = 2 + 2\rho$$

$$\Rightarrow \rho \geq -1$$

**When is $\rho = 1$?**

$\rho = 1$ iff $\text{Var}(X/\sigma_X - Y/\sigma_Y) = 0$, meaning $X/\sigma_X - Y/\sigma_Y = c$ (constant a.s.), i.e., $Y = \frac{\sigma_Y}{\sigma_X} X - c\sigma_Y$ — a perfect positive linear relationship. $\square$

---

### Q14 🟡 — Bivariate Normal and When $\rho = 0$ Means Independence

**Question:** What is the bivariate normal distribution? In what special case does zero correlation imply independence?

---

**Answer:**

**Definition:**

$$f(x,y) = \frac{1}{2\pi\sigma_X\sigma_Y\sqrt{1-\rho^2}} \exp\!\left\{-\frac{1}{2(1-\rho^2)}\left[\frac{(x-\mu_X)^2}{\sigma_X^2} - \frac{2\rho(x-\mu_X)(y-\mu_Y)}{\sigma_X\sigma_Y} + \frac{(y-\mu_Y)^2}{\sigma_Y^2}\right]\right\}$$

**Key properties:**

- Marginal of $X$: $N(\mu_X, \sigma_X^2)$
- Marginal of $Y$: $N(\mu_Y, \sigma_Y^2)$
- The parameter $\rho$ in the formula = the Pearson correlation between $X$ and $Y$

**When does $\rho = 0 \Rightarrow$ independence?**

For the bivariate normal specifically:

$$\rho = 0 \iff X \text{ and } Y \text{ are independent}$$

When $\rho = 0$, the exponent simplifies to $-\frac{1}{2}\left[\frac{(x-\mu_X)^2}{\sigma_X^2} + \frac{(y-\mu_Y)^2}{\sigma_Y^2}\right]$, and the joint PDF factors as $f_X(x) \cdot f_Y(y)$ — the definition of independence.

**In general (non-normal distributions):** $\rho = 0$ does NOT imply independence. This is a special property of the Gaussian family.

**Why this matters for ML:** In Gaussian graphical models and Gaussian processes, testing whether correlation is zero is equivalent to testing independence. This property does not hold for other distributions, which is why nonparametric independence tests (like HSIC or mutual information) are needed in general.

---

### Q15 🟡 — The Hospital Problem

**Question:** A large hospital has ~45 births per day; a small one has ~15. Each records days when >60% of babies born are boys. Which hospital records more such days? Why?

---

**Answer:**

**Intuition first:** Most people guess "about the same" — but the correct answer is the **smaller hospital**.

**Key formula:** The fraction of boys $\hat{p}$ on any given day has standard deviation:

$$\sigma_{\hat{p}} = \sqrt{\frac{p(1-p)}{n}} = \sqrt{\frac{0.25}{n}}$$

| Hospital | $n$ | $\sigma_{\hat{p}}$ |
|---|---|---|
| Large | 45 | $\sqrt{0.25/45} \approx 0.074$ |
| Small | 15 | $\sqrt{0.25/15} \approx 0.129$ |

The small hospital has **nearly twice** the daily standard deviation. More variability = more frequent extreme outcomes.

**Numerical computation:**

$P(\text{>60\% boys in large hospital}) = P(X > 27)$ where $X \sim \text{Bin}(45, 0.5) \approx 0.068$

$P(\text{>60\% boys in small hospital}) = P(X > 9)$ where $X \sim \text{Bin}(15, 0.5) \approx 0.151$

**Expected days per year:**

$$E[L] = 365 \times 0.068 \approx 25 \text{ days (large)}$$
$$E[S] = 365 \times 0.151 \approx 55 \text{ days (small)}$$

The small hospital records more than **double** the number of extreme days.

**The deeper lesson — Law of Large Numbers:**

As $n \to \infty$, the sample fraction converges to the true probability 0.5. Larger samples are more "accurate." Smaller samples are "noisy" — they produce more extreme deviations routinely.

> **FAANG context:** This is exactly why A/B testing at Amazon or Netflix requires minimum sample sizes before declaring significance. A test with 100 users showing 60% conversion is much less surprising than 10,000 users showing 60%. Small sample sizes produce extreme-looking results by chance — this is why experiment platforms enforce power analysis before launch.

---

### Q16 🟡 — Correlation from Overlapping Sums

**Question:** $X_1, \ldots, X_{2n+1}$ are i.i.d. Bernoulli(0.5). Let $X = \sum_{i=1}^{n+1} X_i$ and $Y = \sum_{i=n+1}^{2n+1} X_i$. Derive $\text{Cov}(X,Y)$, $\text{Cor}(X,Y)$, and explain the behavior as $n \to \infty$.

---

**Answer:**

**Step 1: Identify the shared variable.**

$X$ uses flips $1$ through $n+1$; $Y$ uses flips $n+1$ through $2n+1$. Only flip $n+1$ is shared.

**Step 2: Apply bilinearity of covariance.**

$$\text{Cov}(X,Y) = \sum_{i=1}^{n+1}\sum_{j=n+1}^{2n+1} \text{Cov}(X_i, X_j) = \text{Cov}(X_{n+1}, X_{n+1}) = \text{Var}(X_{n+1}) = \frac{1}{4}$$

(All other terms are zero by independence.)

**Step 3: Compute standard deviations.**

$$\text{Var}(X) = (n+1) \cdot \frac{1}{4} \implies \sigma_X = \frac{\sqrt{n+1}}{2}$$

Similarly $\sigma_Y = \frac{\sqrt{n+1}}{2}$.

**Step 4: Compute correlation.**

$$\text{Cor}(X,Y) = \frac{1/4}{\frac{\sqrt{n+1}}{2} \cdot \frac{\sqrt{n+1}}{2}} = \frac{1/4}{(n+1)/4} = \frac{1}{n+1}$$

**Behavior as $n \to \infty$:**

As $n$ grows, $\text{Cor}(X,Y) \to 0$. The one shared flip becomes a smaller and smaller fraction of each sum, so its influence vanishes.

| $n$ | $n+1$ flips each | $\rho$ |
|---|---|---|
| 0 | 1 | 1.0 (identical) |
| 1 | 2 | 0.50 |
| 4 | 5 | 0.20 |
| 9 | 10 | 0.10 |
| 99 | 100 | 0.01 |

**General formula for $m$ shared terms out of $k$ per group:**

$$\rho = \frac{m}{k}$$

> **ML connection:** This is the theoretical basis for **feature correlation from shared subsets** in ensemble models. If two trees in a random forest use many of the same features, their predictions are correlated. Increasing tree diversity (using fewer shared features via max_features) reduces this correlation and improves ensemble performance.

---

### Q17 🟢 — Correlation vs. Causation: Ice Cream and Drowning

**Question:** Ice cream sales and drowning deaths are positively correlated. Does this mean ice cream causes drowning? How would you explain this in a data science interview?

---

**Answer:**

**Answer: No. This is a textbook case of a confounding variable.**

**Causal diagram:**

```
         Hot Weather
        ↙            ↘
Ice Cream Sales    Pool Attendance → Drownings
```

Both ice cream sales and drowning deaths are caused by **hot summer weather**. When it's hot, people buy more ice cream AND swim more. The two variables are correlated as a **shared effect of a common cause**, not because one causes the other.

**Edward Tufte's principle:** *"Empirically observed covariation is a necessary but not sufficient condition for causality."*

**What you need to establish causality:**

1. **Temporal ordering** — the cause must precede the effect
2. **Mechanism** — a plausible biological/physical pathway
3. **Elimination of confounders** — controlling for all common causes
4. **Randomized controlled experiment** — gold standard (randomly assign ice cream to see if drowning changes)

**Other classic examples of confounded correlations:**

| Correlation | True explanation |
|---|---|
| Nicolas Cage movies & pool drownings | Spurious (both trend over time) |
| Student profession & early death (1685) | Students are young by definition |
| HRT & lower heart disease | Both caused by high socioeconomic status |
| Shoe size & reading ability in children | Both caused by age |

> **FAANG context:** This comes up constantly in product analytics. "Users who see feature X have 20% higher retention" — is that because feature X causes retention, or because users who seek out feature X were already highly engaged? Answering this requires causal inference methods (A/B tests, instrumental variables, difference-in-differences), not just correlation.

---

### Q18 🔴 — Zero Correlation Feature: Should You Drop It from ML Model?

**Question:** You compute the Pearson correlation between a feature $X$ and a binary label $Y$ and find $\rho = 0$. Should you drop this feature? What are the risks?

---

**Answer:**

**Answer: Not necessarily. Zero Pearson correlation does NOT mean the feature is useless.**

**Why you might keep it:**

**1. Nonlinear relationship:** $\rho$ only measures linear association. $X$ could have a strong nonlinear relationship with $Y$.

*Example:* If $Y = 1$ when $|X| > 2$ and $Y = 0$ otherwise (a threshold effect), Pearson correlation is 0, but $X$ perfectly predicts $Y$.

**2. Interaction effects:** $X$ alone may not predict $Y$, but $X \times Z$ (interaction) might. Univariate correlation misses multivariate relationships.

**3. Quadratic/polynomial effects:** $X$ may correlate with $Y^2$ or $\log(Y)$ but not $Y$ directly.

**4. Class-conditional structure:** Even if overall $\text{Cor}(X,Y) = 0$, within each class the feature may be highly discriminative.

**Better approaches than Pearson correlation for feature selection:**

| Method | Detects |
|---|---|
| Pearson correlation | Linear relationships only |
| Spearman correlation | Monotonic relationships |
| Mutual information | Any statistical dependence |
| ANOVA F-test | Linear separation between class means |
| Random forest importance | Nonlinear + interaction effects |
| Permutation importance | Model-agnostic importance |

**When is it safe to drop?**

Only drop a feature based on correlation alone if:
- You have verified linearity of the relationship through residual plots
- You have confirmed no interaction effects with other features
- You have checked mutual information and it is also near zero
- Computational/latency constraints require aggressive feature reduction

> **FAANG context:** At Netflix, recommender system features with zero Pearson correlation with ratings have been found to significantly improve CTR models because of nonlinear effects. At Google Ads, features uncorrelated with click-through rates marginally can still have strong interactive effects with bidder behavior. Rule: test empirically before dropping.

---

### Q19 🟡 — Computing $F(3.5, 4)$ from a Discrete Joint Table

**Question:** Given the joint PMF table of two dice (each cell = 1/36), compute $F(3.5, 4)$.

---

**Answer:**

**Definition:** $F(3.5, 4) = P(X \leq 3.5 \text{ and } Y \leq 4)$

**Step 1:** Identify which values satisfy each condition.

- $X \leq 3.5$ means $X \in \{1, 2, 3\}$ (since $X$ takes integer values 1–6)
- $Y \leq 4$ means $Y \in \{1, 2, 3, 4\}$

**Step 2:** Count the cells in the shaded rectangle (rows 1–3, columns 1–4).

Number of cells = $3 \times 4 = 12$, each with probability $1/36$.

**Step 3:**

$$F(3.5, 4) = 12 \times \frac{1}{36} = \frac{12}{36} = \frac{1}{3}$$

**Conceptual note:** For discrete distributions, the CDF at a non-integer value like 3.5 just "rounds down" to the nearest value in the support. $F(3.5, 4) = F(3, 4)$ here.

**Comparison with continuous case:** For continuous $(X,Y)$ with $F(x,y) = x^2y^2$:
$F(0.35, 0.4) = (0.35)^2 (0.4)^2 = 0.1225 \times 0.16 = 0.0196$

The visual structure is the same (a "lower-left rectangle" of probability), just evaluated differently.

---

### Q20 🔴 — Why Do Smaller Samples Show More Extreme Deviations?

**Question:** Explain rigorously, using variance and the CLT, why a hospital with 15 daily births records more days with >60% boys than one with 45 daily births, even though the true probability of a boy is 50%.

---

**Answer:**

**The core principle:** The standard error of a sample proportion is $\sigma_{\hat{p}} = \sqrt{p(1-p)/n}$, which **decreases** as $n$ increases. Smaller $n$ → larger $\sigma_{\hat{p}}$ → more probability in the tails.

**Step 1: Derive the distribution of the daily fraction.**

Let $\hat{p}_n$ = fraction of boys on a day with $n$ births. By CLT:

$$\hat{p}_n \approx N\!\left(0.5,\; \frac{0.25}{n}\right)$$

**Step 2: Compute the probability of exceeding 60%.**

$$P(\hat{p}_n > 0.6) = P\!\left(Z > \frac{0.6 - 0.5}{\sqrt{0.25/n}}\right) = P\!\left(Z > \frac{0.1\sqrt{n}}{0.5}\right) = P(Z > 0.2\sqrt{n})$$

**Step 3: Evaluate for each hospital.**

| Hospital | $n$ | $0.2\sqrt{n}$ | $P(\hat{p} > 0.6)$ |
|---|---|---|---|
| Small | 15 | $0.2\sqrt{15} \approx 0.775$ | $P(Z > 0.775) \approx 0.219$ |
| Large | 45 | $0.2\sqrt{45} \approx 1.342$ | $P(Z > 1.342) \approx 0.090$ |

(Note: exact values use binomial, giving $\approx 0.151$ and $\approx 0.068$; CLT approximation is close.)

**Step 4: Annual expected days.**

$$E[\text{small}] \approx 365 \times 0.151 \approx 55 \text{ days}$$
$$E[\text{large}] \approx 365 \times 0.068 \approx 25 \text{ days}$$

**The mathematical intuition:**

As $n$ increases, $\sigma_{\hat{p}} = \sqrt{0.25/n}$ shrinks at rate $1/\sqrt{n}$. The threshold 0.6 is always $0.1$ above the mean. Expressed in standard deviations:

$$\frac{0.6 - 0.5}{\sigma_{\hat{p}}} = \frac{0.1}{\sqrt{0.25/n}} = 0.2\sqrt{n}$$

This **grows** with $n$. Larger $n$ → the 60% threshold is **more standard deviations away** from the mean → smaller tail probability → fewer extreme days.

**Connection to LLN:** The Law of Large Numbers guarantees $\hat{p}_n \to 0.5$ as $n \to \infty$. For large $n$, deviations of 10+ percentage points become vanishingly rare. For small $n$, they are routine.

> **FAANG context:** This is the mathematical foundation of **statistical power** in A/B testing. Experiments with small traffic have high variance and frequently show large-looking (but unreliable) effects. This is why Netflix and Google enforce minimum detectable effect sizes and minimum traffic thresholds before running experiments. Reacting to noisy small-sample results is "peeking" — it inflates false positive rates. The hospital problem is the canonical illustration of why.

---

## Quick Reference: FAANG Cheat Sheet

### Must-Know Formulas

$$p(x_i,y_j) = p_X(x_i)\cdot p_Y(y_j) \iff X \perp Y \text{ (discrete)}$$

$$f(x,y) = g(x)\cdot h(y) \iff X \perp Y \text{ (continuous, rectangular support)}$$

$$\text{Cov}(X,Y) = E[XY] - E[X]E[Y]$$

$$\text{Var}(X+Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X,Y)$$

$$\rho = \frac{\text{Cov}(X,Y)}{\sigma_X\sigma_Y}, \quad -1 \leq \rho \leq 1$$

$$\rho_{\text{overlap}} = \frac{\text{shared terms}}{\text{total terms per group}}$$

$$\sigma_{\hat{p}} = \sqrt{\frac{p(1-p)}{n}} \quad \text{(decreases as } n \text{ increases)}$$

---

### The Three Most Dangerous Misconceptions

| Misconception | Truth |
|---|---|
| Zero covariance → independent | False. Zero covariance only rules out **linear** dependence |
| Correlation → causation | False. Need mechanism, temporal order, no confounders |
| Large sample shows extreme results as often as small | False. $\sigma_{\hat{p}} \propto 1/\sqrt{n}$ — extremes are rarer with larger $n$ |

---

### Topic-to-Company Mapping

| Topic | Where It Comes Up |
|---|---|
| Joint distributions, marginals | Bayesian inference, generative models (Google Brain, DeepMind) |
| Independence testing, covariance | Feature engineering, PCA, recommendation systems (Netflix, Spotify) |
| Correlation interpretation | A/B testing, experimentation platforms (Meta, Amazon, Airbnb) |
| Zero cov ≠ independence | Feature selection, nonlinear ML (any DS role) |
| Hospital/sampling problem | Experiment design, power analysis (all FAANG) |
| Causation vs. correlation | Product analytics, causal inference (Meta CausalML, Uber) |
| Bivariate normal, $\rho$ | Gaussian processes, Kalman filters (robotics, quant finance) |
| Var of sum with covariance | Portfolio optimization, risk (Jane Street, Two Sigma, Citadel) |

---
