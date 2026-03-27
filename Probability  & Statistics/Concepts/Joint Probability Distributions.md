# Chapter 5 — Joint Probability Distributions


## Chapter Overview (Plain-English Explanation)

we dealt with **one random variable at a time**. But in engineering and science, measurements rarely come alone. The diameter of a shaft AND its length are both random. The current in a circuit AND the voltage across a resistor vary together. here we extend probability theory to handle **two or more random variables simultaneously**.

The central new idea is the **joint distribution** — describing probabilities for combinations of values. From the joint distribution you can always recover the individual ("marginal") distributions, and you can quantify the **relationship** between variables using covariance and correlation.

The chapter also introduces two powerful results:
- **Functions of random variables** — if X is random and Y = h(X), what is the distribution of Y?
- **Linear combinations** — if $Y = c_1X_1 + c_2X_2 + \cdots$, what are E(Y) and V(Y)?

These results are the mathematical foundation for all statistical inference methods in later chapters (confidence intervals, hypothesis tests, regression).

---

## Chapter Outline

| Section | Topic |
|---------|-------|
| 5.1 | Joint Probability Distributions (two discrete RVs) |
| 5.2 | Marginal Probability Distributions |
| 5.3 | Conditional Probability Distributions |
| 5.4 | Independence |
| 5.5 | Multiple Discrete Random Variables (Multinomial) |
| 5.6 | Covariance and Correlation |
| 5.7 | Common Joint Distributions (Bivariate Normal, Multinomial) |
| 5.8 | Linear Functions of Random Variables |
| 5.9 | General Functions of Random Variables |

---

## 5.1 Joint Probability Distributions



### 5.1.1 Joint Probability Mass Function (Discrete Case)

>**Definition:** The **joint probability mass function** (joint PMF) of the discrete random variables X and Y is a function $f_{XY}(x, y)$ such that:

>1. $f_{XY}(x, y) \geq 0$ for all (x, y)
>2. $\displaystyle\sum_x \sum_y f_{XY}(x, y) = 1$
>3. $f_{XY}(x, y) = P(X = x, Y = y)$ for each pair (x, y) in the range

**Probability of a region R:**
$$P[(X, Y) \in R] = \sum_{(x,y) \in R} f_{XY}(x, y)$$

---

# Example Joint Probability Table

## Variables
- **X** = Number of Bars of Signal Strength  
- **Y** = Response Time (nearest second)



## Joint Probability Distribution (X, Y)

| Y \ X | 1    | 2    | 3    | Marginal P(Y) |
|------|------|------|------|---------------|
| 4    | 0.15 | 0.10 | 0.05 | 0.30          |
| 3    | 0.02 | 0.10 | 0.05 | 0.17          |
| 2    | 0.02 | 0.03 | 0.20 | 0.25          |
| 1    | 0.01 | 0.02 | 0.25 | 0.28          |



## Marginal Probability Distribution of X

| X | P(X) |
|---|------|
| 1 | 0.20 |
| 2 | 0.25 |
| 3 | 0.55 |

## Example Calculation

### Finding \( f_X(3) = P(X = 3) \)

\[
P(X = 3) = P(X = 3, Y = 1) + P(X = 3, Y = 2) + P(X = 3, Y = 3) + P(X = 3, Y = 4)
\]

\[
= 0.25 + 0.20 + 0.05 + 0.05 = 0.55
\]

## Key Insight
- Marginal probability is obtained by **summing over the other variable**.
- Here, to find \( P(X = 3) \), we sum across all values of **Y**.

---

### 5.1.2 Joint Probability Density Function (Continuous Case)

**Definition:** A **joint probability density function** (joint PDF) for continuous random variables X and Y is a function $f_{XY}(x, y)$ such that:

1. $f_{XY}(x, y) \geq 0$ for all (x, y)
2. $\displaystyle\int_{-\infty}^{\infty}\int_{-\infty}^{\infty} f_{XY}(x, y)\,dx\,dy = 1$
3. For any region R in the xy-plane:
$$P[(X, Y) \in R] = \iint_R f_{XY}(x, y)\,dx\,dy$$

**Key:** For continuous variables, probability only makes sense over regions, never at a single point. $P(X = x, Y = y) = 0$ for any specific (x, y).

---

### Example  — Exponential Joint PDF

**Problem:** $f_{XY}(x, y) = e^{-x-y}$ for $x > 0$, $y > 0$; 0 otherwise. Verify validity and find $P(X < 1, Y < 1)$.

**Verification:**
$$\int_0^{\infty}\int_0^{\infty} e^{-x-y}\,dx\,dy = \left(\int_0^{\infty} e^{-x}\,dx\right)\left(\int_0^{\infty} e^{-y}\,dy\right) = 1 \times 1 = 1 \checkmark$$

**Probability:**
$$P(X < 1, Y < 1) = \int_0^1\int_0^1 e^{-x-y}\,dx\,dy = (1-e^{-1})^2 = (0.6321)^2 = 0.3996$$



# Probability that \( Y > 2000 \)

We are given a joint PDF:

\[
f_{XY}(x, y) = 6 \times 10^{-6} e^{-0.001x - 0.002y}
\]

We want to compute:

\[
P(Y > 2000)
\]

---

## Method 1: Using Joint PDF (Double Integration)

The region is split into two parts:

### Region 1: \( 0 \le x \le 2000, \; y \ge 2000 \)

\[
\int_{0}^{2000} \left( \int_{2000}^{\infty} 6 \times 10^{-6} e^{-0.001x - 0.002y} \, dy \right) dx
\]

#### Inner Integral:

\[
\int_{2000}^{\infty} e^{-0.002y} dy 
= \left[ \frac{e^{-0.002y}}{-0.002} \right]_{2000}^{\infty}
= \frac{e^{-4}}{0.002}
\]

#### Outer Integral:

\[
6 \times 10^{-6} \cdot \frac{e^{-4}}{0.002} \int_{0}^{2000} e^{-0.001x} dx
\]

\[
\int_{0}^{2000} e^{-0.001x} dx 
= \left[ \frac{e^{-0.001x}}{-0.001} \right]_{0}^{2000}
= \frac{1 - e^{-2}}{0.001}
\]

#### Final Value (Region 1):

\[
= 0.0475
\]

---

### Region 2: \( x \ge 2000, \; y \ge x \)

\[
\int_{2000}^{\infty} \left( \int_{x}^{\infty} 6 \times 10^{-6} e^{-0.001x - 0.002y} \, dy \right) dx
\]

#### Inner Integral:

\[
\int_{x}^{\infty} e^{-0.002y} dy 
= \frac{e^{-0.002x}}{0.002}
\]

#### Outer Integral:

\[
6 \times 10^{-6} \cdot \frac{1}{0.002} \int_{2000}^{\infty} e^{-0.003x} dx
\]

\[
\int_{2000}^{\infty} e^{-0.003x} dx 
= \left[ \frac{e^{-0.003x}}{-0.003} \right]_{2000}^{\infty}
= \frac{e^{-6}}{0.003}
\]

#### Final Value (Region 2):

\[
= 0.0025
\]

### Final Answer

\[
P(Y > 2000) = 0.0475 + 0.0025 = 0.05
\]


## Method 2: Using Marginal PDF of \( Y \)

### Step 1: Compute Marginal PDF

\[
f_Y(y) = \int_{0}^{y} 6 \times 10^{-6} e^{-0.001x - 0.002y} dx
\]

\[
= 6 \times 10^{-6} e^{-0.002y} \int_{0}^{y} e^{-0.001x} dx
\]

\[
= 6 \times 10^{-6} e^{-0.002y} \cdot \frac{1 - e^{-0.001y}}{0.001}
\]

\[
= 6 \times 10^{-3} e^{-0.002y}(1 - e^{-0.001y}), \quad y > 0
\]


### Step 2: Compute Probability

\[
P(Y > 2000) = \int_{2000}^{\infty} 6 \times 10^{-3} e^{-0.002y}(1 - e^{-0.001y}) dy
\]

Split into two integrals:

\[
= 6 \times 10^{-3} \left[
\int_{2000}^{\infty} e^{-0.002y} dy 
- \int_{2000}^{\infty} e^{-0.003y} dy
\right]
\]

\[
= 6 \times 10^{-3} \left[
\frac{e^{-4}}{0.002} - \frac{e^{-6}}{0.003}
\right]
\]

### Final Answer

\[
P(Y > 2000) = 0.05
\]

## Key Takeaways

- Always **split the region carefully** when limits depend on variables.
- Sometimes **marginalization simplifies the problem significantly**.
- Exponential PDFs often lead to **clean closed-form integrals**.
---

### Example 5.3 — Joint PDF on Restricted Range

**Problem:** Given $f_{XY}(x,y) = \frac{6}{5}(x + y^2)$ for $0 < x < 1$, $0 < y < 1$. Find $P(X < 0.5, Y < 0.5)$.

**Solution:**
$$P(X < 0.5, Y < 0.5) = \int_0^{0.5}\int_0^{0.5} \frac{6}{5}(x + y^2)\,dx\,dy$$
$$= \frac{6}{5}\int_0^{0.5}\left[\frac{x^2}{2} + xy^2\right]_0^{0.5}dy = \frac{6}{5}\int_0^{0.5}\left(0.125 + 0.5y^2\right)dy$$
$$= \frac{6}{5}\left[0.125y + \frac{0.5y^3}{3}\right]_0^{0.5} = \frac{6}{5}(0.0625 + 0.02083) = \frac{6}{5}(0.08333) = 0.10$$

---

## 5.2 Marginal Probability Distributions

### Definition

Given a joint distribution, the **marginal probability distribution** of X alone is obtained by summing (discrete) or integrating (continuous) over all values of Y.

**Discrete:**
$$f_X(x) = \sum_y f_{XY}(x, y) \qquad f_Y(y) = \sum_x f_{XY}(x, y)$$

**Continuous:**
$$f_X(x) = \int_{-\infty}^{\infty} f_{XY}(x, y)\,dy \qquad f_Y(y) = \int_{-\infty}^{\infty} f_{XY}(x, y)\,dx$$

**Intuition:** To get the marginal of X, "sum out" (marginalize over) Y. You are collapsing the joint table into a single row of probabilities for X alone.

---

### Example 5.4 — Marginal from Continuous Joint PDF

**Problem:** Find $f_X(x)$ and $f_Y(y)$ for $f_{XY}(x,y) = e^{-x-y}$, $x > 0$, $y > 0$.

**Solution:**
$$f_X(x) = \int_0^{\infty} e^{-x-y}\,dy = e^{-x}\int_0^{\infty} e^{-y}\,dy = e^{-x}, \quad x > 0$$
$$f_Y(y) = \int_0^{\infty} e^{-x-y}\,dx = e^{-y}, \quad y > 0$$

Both marginals are **exponential with λ = 1**. The joint PDF equals the product of marginals — this signals **independence** (see Section 5.4).

---

### Example 5.5 — Marginal from Mixed Joint PDF

**Problem:** $f_{XY}(x,y) = \frac{6}{5}(x + y^2)$ for $0 < x < 1$, $0 < y < 1$. Find $f_X(x)$.

**Solution:**
$$f_X(x) = \int_0^1 \frac{6}{5}(x + y^2)\,dy = \frac{6}{5}\left[xy + \frac{y^3}{3}\right]_0^1 = \frac{6}{5}\left(x + \frac{1}{3}\right), \quad 0 < x < 1$$

Similarly: $f_Y(y) = \frac{6}{5}\left(\frac{1}{2} + y^2\right)$, $0 < y < 1$.

---

### Mean and Variance from Marginals

Once marginals are found, compute means and variances using only the marginal distribution, exactly as in Chapters 3–4:
$$\mu_X = E(X) = \int_{-\infty}^{\infty} x\,f_X(x)\,dx \qquad \sigma_X^2 = E(X^2) - [E(X)]^2$$

---

## 5.3 Conditional Probability Distributions

### Definition

The **conditional probability distribution** of Y given X = x is:

**Discrete:**
$$f_{Y|X}(y|x) = \frac{f_{XY}(x, y)}{f_X(x)}, \quad \text{provided } f_X(x) > 0$$

**Continuous:**
$$f_{Y|X}(y|x) = \frac{f_{XY}(x, y)}{f_X(x)}, \quad \text{provided } f_X(x) > 0$$

**Interpretation:** $f_{Y|X}(y|x)$ is a valid probability distribution for Y when we *know* that X = x. It integrates/sums to 1 over all y.

### Conditional Mean and Variance

$$E(Y|X = x) = \int_{-\infty}^{\infty} y\,f_{Y|X}(y|x)\,dy$$
$$V(Y|X = x) = E(Y^2|X=x) - [E(Y|X=x)]^2$$

---

### Example 5.6 — Conditional Distribution (Continuous)

**Problem:** $f_{XY}(x,y) = \frac{6}{5}(x + y^2)$ for $0 < x < 1$, $0 < y < 1$. Find $f_{Y|X}(y|x)$ and $E(Y|X = 0.5)$.

**Solution:**

From Example 5.5: $f_X(x) = \frac{6}{5}(x + 1/3)$.

$$f_{Y|X}(y|x) = \frac{(6/5)(x+y^2)}{(6/5)(x+1/3)} = \frac{x + y^2}{x + 1/3}, \quad 0 < y < 1$$

At X = 0.5:
$$f_{Y|X}(y|0.5) = \frac{0.5 + y^2}{0.5 + 0.333} = \frac{0.5 + y^2}{0.833}$$

$$E(Y|X=0.5) = \int_0^1 y \cdot \frac{0.5 + y^2}{0.833}\,dy = \frac{1}{0.833}\int_0^1 (0.5y + y^3)\,dy$$
$$= \frac{1}{0.833}\left[\frac{0.5y^2}{2} + \frac{y^4}{4}\right]_0^1 = \frac{0.25 + 0.25}{0.833} = \frac{0.50}{0.833} = 0.60$$

---

### Example 5.7 — Conditional Distribution (Discrete)

**Problem:** From a joint PMF table of (X, Y), given $f_X(1) = 0.20$ and the row for X = 1 has entries $f(1,0) = 0.05$, $f(1,1) = 0.10$, $f(1,2) = 0.05$. Find $f_{Y|X}(y|1)$ and $E(Y|X=1)$.

**Solution:**
$$f_{Y|X}(0|1) = 0.05/0.20 = 0.25$$
$$f_{Y|X}(1|1) = 0.10/0.20 = 0.50$$
$$f_{Y|X}(2|1) = 0.05/0.20 = 0.25$$

$$E(Y|X=1) = 0(0.25) + 1(0.50) + 2(0.25) = 1.0$$

---

## 5.4 Independence

### Definition

Two random variables X and Y are **independent** if and only if:
$$f_{XY}(x, y) = f_X(x) \cdot f_Y(y) \quad \text{for ALL } (x, y)$$

**Equivalently:** $f_{Y|X}(y|x) = f_Y(y)$ for all x, y (knowing X gives no new information about Y).

### Practical Test for Independence

1. Find $f_X(x)$ and $f_Y(y)$.
2. Check if $f_{XY}(x,y) = f_X(x) \cdot f_Y(y)$ for **every** (x, y).
3. If the support region is **non-rectangular** (e.g., $0 < x < y < 1$), then X and Y are automatically **NOT** independent — the limits of one depend on the other.

---

### Example 5.8 — Independent (Continuous)

**Problem:** Is $f_{XY}(x,y) = e^{-x-y}$, $x > 0$, $y > 0$, independent?

**Check:** $f_X(x) \cdot f_Y(y) = e^{-x} \cdot e^{-y} = e^{-x-y} = f_{XY}(x,y)$ ✓

**Yes, X and Y are independent.**

---

### Example 5.9 — Not Independent (Continuous)

**Problem:** Is $f_{XY}(x,y) = \frac{6}{5}(x + y^2)$, $0 < x < 1$, $0 < y < 1$, independent?

**Check:** $f_X(x) \cdot f_Y(y) = \frac{6}{5}(x+1/3) \cdot \frac{6}{5}(1/2 + y^2) \neq \frac{6}{5}(x + y^2)$

**No, X and Y are NOT independent.**

---

### Independence and Probability

If X and Y are independent:
$$P(X \in A,\; Y \in B) = P(X \in A) \cdot P(Y \in B)$$

This extends to any number of variables: $X_1, X_2, \ldots, X_n$ are **mutually independent** if:
$$f(x_1, x_2, \ldots, x_n) = f_{X_1}(x_1) \cdot f_{X_2}(x_2) \cdots f_{X_n}(x_n)$$

---

## 5.5 Multiple Discrete Random Variables

### Extension to More Than Two Variables

All definitions extend naturally. For $X_1, X_2, \ldots, X_p$:

**Joint PMF:** $f(x_1, \ldots, x_p) = P(X_1 = x_1, \ldots, X_p = x_p)$

**Marginal of $X_1$:** Sum over all other variables.

**Mutual independence:** $f(x_1, \ldots, x_p) = \prod_{i=1}^p f_{X_i}(x_i)$ for all $(x_1, \ldots, x_p)$.

---

### Multinomial Distribution

A generalization of the binomial to k > 2 outcomes. In n **independent** trials, each trial results in outcome i with probability $p_i$ where $\sum_{i=1}^k p_i = 1$. Let $X_i$ = number of times outcome i occurs. Then $(X_1, \ldots, X_k)$ has a **multinomial distribution**:

$$f(x_1, x_2, \ldots, x_k) = \frac{n!}{x_1!\,x_2!\cdots x_k!}\,p_1^{x_1} p_2^{x_2} \cdots p_k^{x_k}$$

where $x_1 + x_2 + \cdots + x_k = n$, each $x_i \geq 0$.

**Marginal distributions:** Each $X_i \sim \text{Binomial}(n, p_i)$.

**Mean and Variance of each $X_i$:**
$$E(X_i) = np_i \qquad V(X_i) = np_i(1-p_i)$$

**Covariance between $X_i$ and $X_j$ (i ≠ j):**
$$\text{Cov}(X_i, X_j) = -np_ip_j$$

(Negative because if more trials result in outcome i, fewer are available for outcome j.)

---

### Example 5.10 — Multinomial

**Problem:** In a manufacturing process, components are acceptable (p₁ = 0.70), have minor defects (p₂ = 0.20), or major defects (p₃ = 0.10). In n = 5 components, find the probability that 3 are acceptable, 1 has minor defects, 1 has major defects.

**Solution:**
$$P(X_1=3, X_2=1, X_3=1) = \frac{5!}{3!\,1!\,1!}(0.70)^3(0.20)^1(0.10)^1$$
$$= 20 \times 0.343 \times 0.20 \times 0.10 = 0.1372$$

---

### Example 5.11 — Multinomial Probability

**Problem:** In a digital communication system, bits are received correctly (p₁ = 0.90), with minor errors (p₂ = 0.08), or with catastrophic errors (p₃ = 0.02). In a packet of n = 10 bits, what is the probability of exactly 9 correct, 1 minor error, 0 catastrophic?

**Solution:**
$$P = \frac{10!}{9!\,1!\,0!}(0.90)^9(0.08)^1(0.02)^0 = 10 \times 0.3874 \times 0.08 = 0.3097$$

---

## 5.6 Covariance and Correlation

### Motivation

How do we measure whether X and Y are related, and how strongly? Covariance and correlation provide the answer for **linear** relationships.

---

### Definition: Covariance

The **covariance** between X and Y is:
$$\sigma_{XY} = \text{Cov}(X, Y) = E[(X - \mu_X)(Y - \mu_Y)]$$

**Computational formula (easier to use):**
$$\text{Cov}(X, Y) = E(XY) - \mu_X\mu_Y$$

where:
- Discrete: $E(XY) = \sum_x \sum_y xy\,f_{XY}(x,y)$
- Continuous: $E(XY) = \int\!\!\int xy\,f_{XY}(x,y)\,dx\,dy$

**Interpretation:**
- $\text{Cov}(X,Y) > 0$: X and Y tend to be simultaneously above or below their means (positive co-movement)
- $\text{Cov}(X,Y) < 0$: When X is above its mean, Y tends to be below its mean (opposite movement)
- $\text{Cov}(X,Y) = 0$: No **linear** relationship

**Critical warning:** If X and Y are **independent**, then $\text{Cov}(X,Y) = 0$. But the **converse is NOT true** in general — zero covariance does NOT guarantee independence. (Exception: bivariate normal, where it does.)

---

### Definition: Correlation Coefficient

$$\rho_{XY} = \text{Corr}(X, Y) = \frac{\text{Cov}(X, Y)}{\sigma_X \cdot \sigma_Y}$$

**Properties:**
- $-1 \leq \rho_{XY} \leq 1$ always
- $\rho_{XY} = +1$: perfect positive linear relationship ($Y = a + bX$, $b > 0$)
- $\rho_{XY} = -1$: perfect negative linear relationship ($Y = a + bX$, $b < 0$)
- $\rho_{XY} = 0$: no linear relationship
- **Dimensionless** — independent of units of X and Y
- Advantage over covariance: allows comparison across different pairs of variables

---

### Example 5.12 — Covariance and Correlation (Discrete)

**Problem:** Joint PMF of (X, Y):

| | Y = 0 | Y = 1 | Y = 2 |
|--|--|--|--|
| **X = 0** | 1/8 | 2/8 | 1/8 |
| **X = 1** | 2/8 | 1/8 | 1/8 |

Find Cov(X, Y) and Corr(X, Y).

**Solution:**

Marginals:
- $f_X(0) = (1+2+1)/8 = 4/8$, $f_X(1) = (2+1+1)/8 = 4/8$
- $f_Y(0) = 3/8$, $f_Y(1) = 3/8$, $f_Y(2) = 2/8$

Means:
$$\mu_X = 0(1/2) + 1(1/2) = 0.5$$
$$\mu_Y = 0(3/8) + 1(3/8) + 2(2/8) = 3/8 + 4/8 = 7/8 = 0.875$$

E(XY):
$$E(XY) = \sum_{x,y} xy\cdot f(x,y) = 0 + 0 + 0 + 0 + 1\cdot1\cdot(1/8) + 1\cdot2\cdot(1/8) = \frac{1+2}{8} = 0.375$$

Covariance:
$$\text{Cov}(X,Y) = 0.375 - (0.5)(0.875) = 0.375 - 0.4375 = -0.0625$$

Variances:
$$\sigma_X^2 = E(X^2) - \mu_X^2 = 0.5 - 0.25 = 0.25 \implies \sigma_X = 0.5$$
$$E(Y^2) = 0(3/8) + 1(3/8) + 4(2/8) = 11/8 \implies \sigma_Y^2 = 11/8 - (7/8)^2 = 88/64 - 49/64 = 39/64$$
$$\sigma_Y = \sqrt{39/64} = 0.7806$$

Correlation:
$$\rho = \frac{-0.0625}{(0.5)(0.7806)} = \frac{-0.0625}{0.3903} = -0.160$$

Weak negative linear relationship: when X = 1, Y tends to be slightly lower.

---

### Example 5.13 — Zero Covariance (Independent Variables)

**Problem:** $f_{XY}(x,y) = e^{-x-y}$, $x > 0$, $y > 0$. Compute Cov(X, Y).

**Solution:** Since X and Y are independent (shown in 5.4), $E(XY) = E(X) \cdot E(Y) = 1 \times 1 = 1$.

$$\text{Cov}(X,Y) = E(XY) - E(X)E(Y) = 1 - 1 = 0$$

Independence implies zero covariance (as expected).

---

## 5.7 Common Joint Distributions

### 5.7.1 Bivariate Normal Distribution

The most important continuous joint distribution. If (X, Y) are jointly bivariate normal, each marginal is individually normal, and the parameter ρ captures their linear relationship.

**Joint PDF:**
$$f_{XY}(x,y) = \frac{1}{2\pi\sigma_X\sigma_Y\sqrt{1-\rho^2}}\exp\!\left\{-\frac{1}{2(1-\rho^2)}\!\left[\frac{(x-\mu_X)^2}{\sigma_X^2} - \frac{2\rho(x-\mu_X)(y-\mu_Y)}{\sigma_X\sigma_Y} + \frac{(y-\mu_Y)^2}{\sigma_Y^2}\right]\right\}$$

**Parameters:** $\mu_X$, $\mu_Y$, $\sigma_X^2$, $\sigma_Y^2$, and $\rho$ (correlation), where $-1 < \rho < 1$.

**Key properties:**
- Marginal: $X \sim N(\mu_X, \sigma_X^2)$ and $Y \sim N(\mu_Y, \sigma_Y^2)$
- $\text{Corr}(X,Y) = \rho$
- **If ρ = 0: X and Y are INDEPENDENT** (this is special to bivariate normal — elsewhere, zero correlation does not mean independence)

**Conditional distribution of Y given X = x:**
$$Y|X=x \sim N\!\left(\mu_Y + \rho\frac{\sigma_Y}{\sigma_X}(x-\mu_X),\;\; \sigma_Y^2(1-\rho^2)\right)$$

The conditional mean is a **linear function of x** — this is the basis of linear regression (Chapter 11).
The conditional variance $\sigma_Y^2(1-\rho^2)$ is constant (does not depend on x).

---

### Example 5.14 — Bivariate Normal

**Problem:** X = current (mA), Y = voltage (V) follow bivariate normal with $\mu_X=10$, $\mu_Y=120$, $\sigma_X=2$, $\sigma_Y=10$, $\rho=0.8$. Find $P(X>13)$ and the conditional distribution $Y|X=13$.

**P(X > 13):** Marginal $X \sim N(10, 4)$:
$$P(X > 13) = P\!\left(Z > \frac{13-10}{2}\right) = P(Z > 1.5) = 0.0668$$

**Conditional distribution Y | X = 13:**
$$E(Y|X=13) = 120 + 0.8\cdot\frac{10}{2}(13-10) = 120 + 12 = 132 \text{ V}$$
$$V(Y|X=13) = 100(1-0.64) = 36 \implies \sigma = 6 \text{ V}$$

So $Y|X=13 \sim N(132, 36)$.

---

### 5.7.2 Multinomial Distribution

Already covered in Section 5.5. The multinomial generalizes the binomial to k outcome categories. Each $X_i \sim \text{Binomial}(n, p_i)$ as a marginal, and $\text{Cov}(X_i, X_j) = -np_ip_j$ for $i \neq j$.

---

## 5.8 Linear Functions of Random Variables

### Single Variable: Y = aX + b

$$E(aX+b) = aE(X) + b \qquad V(aX+b) = a^2 V(X)$$

### Two Variables: Y = aX + bW

$$E(aX + bW) = aE(X) + bE(W)$$
$$V(aX + bW) = a^2V(X) + b^2V(W) + 2ab\,\text{Cov}(X,W)$$

If X and W are **independent**:
$$V(aX + bW) = a^2V(X) + b^2V(W)$$

---

### General Linear Combination

For $Y = c_1X_1 + c_2X_2 + \cdots + c_pX_p$:

$$\boxed{E(Y) = \sum_{i=1}^p c_i\mu_i} \quad \text{(always true)}$$

$$V(Y) = \sum_{i=1}^p c_i^2\sigma_i^2 + 2\sum_{i<j} c_ic_j\,\text{Cov}(X_i, X_j)$$

**If all $X_i$ are mutually independent:**
$$\boxed{V(Y) = \sum_{i=1}^p c_i^2\sigma_i^2}$$

---

### Mean and Variance of the Sample Mean

One of the most critical results in all of statistics. Let $X_1, X_2, \ldots, X_n$ be **independent and identically distributed (i.i.d.)** with mean μ and variance σ². Define:
$$\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$$

Applying the linear combination formula with all $c_i = 1/n$:

$$\boxed{E(\bar{X}) = \mu} \qquad \boxed{V(\bar{X}) = \frac{\sigma^2}{n}} \qquad \boxed{\sigma_{\bar{X}} = \frac{\sigma}{\sqrt{n}}}$$

**Interpretations:**
- $E(\bar{X}) = \mu$: The sample mean is an **unbiased estimator** of the population mean — on average it hits the truth.
- $V(\bar{X}) = \sigma^2/n$: Variance of $\bar{X}$ **decreases** as n increases — larger samples give more precise estimates.
- $\sigma/\sqrt{n}$ is called the **standard error of the mean**.

---

### Reproductive Property of the Normal Distribution

If $X_1, X_2, \ldots, X_n$ are **independent** with $X_i \sim N(\mu_i, \sigma_i^2)$, then any linear combination is also **normal**:
$$Y = c_1X_1 + \cdots + c_nX_n \sim N\!\left(\sum c_i\mu_i,\; \sum c_i^2\sigma_i^2\right)$$

**Critical corollary:** If $X_i \sim N(\mu, \sigma^2)$ i.i.d., then:
$$\bar{X} \sim N\!\left(\mu,\; \frac{\sigma^2}{n}\right)$$

This is the **exact** distribution of $\bar{X}$ when sampling from a normal population. No approximation — this is exact for any n.

---

### Example 5.15 — Linear Combination (Resistors in Series)

**Problem:** X = resistance of resistor 1, $E(X) = 100$ Ω, $V(X) = 25$ Ω²; Y = resistance of resistor 2, $E(Y) = 120$ Ω, $V(Y) = 36$ Ω². They are independent and connected in series. Total resistance T = X + Y. Find E(T) and σ(T).

**Solution:**
$$E(T) = 100 + 120 = 220 \text{ Ω}$$
$$V(T) = 25 + 36 = 61 \text{ Ω}^2 \implies \sigma_T = \sqrt{61} = 7.81 \text{ Ω}$$

---

### Example 5.16 — Standard Error

**Problem:** Assembly time of a circuit board: μ = 54 min, σ = 3 min. Sample of n = 16 boards. Find E($\bar{X}$) and $\sigma_{\bar{X}}$.

**Solution:**
$$E(\bar{X}) = 54 \text{ min}, \qquad \sigma_{\bar{X}} = \frac{3}{\sqrt{16}} = 0.75 \text{ min}$$

The average assembly time for a batch of 16 has a standard error of only 0.75 min.

---

### Example 5.17 — Sum of Normals (Shaft and Hole Clearance)

**Problem:** Shaft diameter $X \sim N(2.000, 0.0004)$ cm, hole diameter $Y \sim N(2.010, 0.0009)$ cm, independent. Clearance C = Y − X. Find the distribution of C and $P(C > 0)$.

**Solution:**
$$E(C) = 2.010 - 2.000 = 0.010 \text{ cm}$$
$$V(C) = V(Y) + (-1)^2 V(X) = 0.0009 + 0.0004 = 0.0013 \text{ cm}^2$$

(Note: $V(aX+bW) = a^2V(X) + b^2V(W)$ for independent X, W; here a = −1 for X, b = 1 for Y.)

$$C \sim N(0.010, 0.0013)$$

$$P(C > 0) = P\!\left(Z > \frac{0 - 0.010}{\sqrt{0.0013}}\right) = P(Z > -0.277) = \Phi(0.277) = 0.609$$

*Practical Interpretation:* About 60.9% of shaft-hole pairs have positive clearance (fit correctly). The rest would have interference — a quality concern.

---

## 5.9 General Functions of Random Variables

### CDF Technique

To find the distribution of Y = h(X) when h is a monotone function:

**Step 1:** Write $F_Y(y) = P(Y \leq y) = P(h(X) \leq y)$

**Step 2:** Solve for X (find the region in terms of X that corresponds to $Y \leq y$)

**Step 3:** Compute the probability using $F_X$ or $f_X$

**Step 4:** Differentiate $F_Y(y)$ to get $f_Y(y)$

---

### Change of Variables Formula

If Y = h(X) where h is **monotone and differentiable** with inverse $x = h^{-1}(y)$, then:
$$f_Y(y) = f_X\!\left(h^{-1}(y)\right) \cdot \left|\frac{d}{dy}h^{-1}(y)\right|$$

The Jacobian $|dx/dy|$ accounts for the "stretching" of the probability under the transformation.

---

### Example 5.18 — Generating Exponential from Uniform

**Problem:** $X \sim \text{Uniform}[0,1]$, $f_X(x) = 1$, $0 < x < 1$. Let $Y = -\ln(X)$. Find the distribution of Y.

**Solution:** When $x \in (0,1)$, $y = -\ln(x) \in (0, \infty)$.

CDF of Y:
$$F_Y(y) = P(Y \leq y) = P(-\ln X \leq y) = P(X \geq e^{-y}) = 1 - e^{-y}, \quad y > 0$$

Differentiating:
$$f_Y(y) = e^{-y}, \quad y > 0$$

**Y has an Exponential(λ=1) distribution.** This is used in Monte Carlo simulation: generate U ~ Uniform[0,1], then $Y = -\ln(U)/\lambda$ is exponentially distributed with mean $1/\lambda$.

---

### Example 5.19 — Square of a Standard Normal → Chi-Square

**Problem:** $X \sim N(0,1)$. Find the distribution of $Y = X^2$.

**Solution:** For y > 0:
$$F_Y(y) = P(X^2 \leq y) = P(-\sqrt{y} \leq X \leq \sqrt{y}) = 2\Phi(\sqrt{y}) - 1$$

$$f_Y(y) = \frac{d}{dy}[2\Phi(\sqrt{y}) - 1] = 2\phi(\sqrt{y}) \cdot \frac{1}{2\sqrt{y}} = \frac{1}{\sqrt{2\pi y}}\,e^{-y/2}, \quad y > 0$$

This is a **chi-square distribution with 1 degree of freedom** ($\chi^2_1$). Used extensively in hypothesis testing.

**Extension:** If $X_1, \ldots, X_n$ are i.i.d. $N(0,1)$, then $\sum_{i=1}^n X_i^2 \sim \chi^2_n$ (chi-square with n degrees of freedom).

---

### Example 5.20 — Lognormal Derivation

**Problem:** $W \sim N(\theta, \omega^2)$. Let $X = e^W$. Show X has a lognormal distribution.

**Solution:** Inverse: $w = \ln x$, so $|dw/dx| = 1/x$. Change of variables:
$$f_X(x) = f_W(\ln x) \cdot \frac{1}{x} = \frac{1}{\sqrt{2\pi}\,\omega}\,e^{-(\ln x - \theta)^2/(2\omega^2)} \cdot \frac{1}{x}, \quad x > 0$$

This confirms the lognormal PDF from Chapter 4.

---

## Summary of Key Formulas

### Joint, Marginal, Conditional

| Concept | Discrete | Continuous |
|---|---|---|
| Joint P | $P(X=x,Y=y) = f(x,y)$ | $P[(X,Y)\in R] = \iint_R f\,dA$ |
| Marginal $f_X$ | $\sum_y f(x,y)$ | $\int f(x,y)\,dy$ |
| Conditional $f_{Y\|X}$ | $f(x,y)/f_X(x)$ | $f(x,y)/f_X(x)$ |
| Independence | $f(x,y) = f_X(x)f_Y(y)$ | Same |

### Covariance and Correlation

$$\text{Cov}(X,Y) = E(XY) - E(X)E(Y) \qquad \rho = \frac{\text{Cov}(X,Y)}{\sigma_X\sigma_Y}, \quad -1 \leq \rho \leq 1$$

### Linear Combinations

$$E\!\left(\sum c_iX_i\right) = \sum c_i\mu_i \quad\text{(always)}$$

$$V\!\left(\sum c_iX_i\right) = \sum c_i^2\sigma_i^2 \quad\text{(independent only)}$$

$$E(\bar{X}) = \mu, \quad V(\bar{X}) = \frac{\sigma^2}{n}, \quad \bar{X}\sim N\!\left(\mu, \frac{\sigma^2}{n}\right)\text{ if }X_i\sim N(\mu,\sigma^2)$$

---

## Chapter 5 — Clear Conceptual Explanation

### What is this chapter really about?

Chapter 5 is the **mathematical bridge** between single-variable probability (Chapters 3–4) and the statistical inference of later chapters. It builds three things you will use constantly:

**1. Modeling multiple variables together (Sections 5.1–5.5)**

The joint distribution is the complete probabilistic description of (X, Y). The marginal recovers each variable's individual distribution. The conditional answers: "given I know X = x, what can I say about Y?" Independence is the special case where the answer is: "nothing new — knowing X is irrelevant to Y."

**2. Measuring association (Section 5.6)**

Covariance measures the direction of linear co-movement between X and Y. Correlation standardizes it to [−1, +1] and removes the effect of units. Zero covariance means no *linear* relationship — but variables can still be nonlinearly dependent. The exception: for bivariate normal (Section 5.7), zero correlation is equivalent to independence.

**3. Distributions of functions and combinations (Sections 5.8–5.9)**

This is the real payoff. The most important results:

- **$E(\bar{X}) = \mu$:** The sample mean is an unbiased estimator. It does not systematically over- or underestimate.
- **$V(\bar{X}) = \sigma^2/n$:** More data = smaller variance = more precise estimate. This is *why* larger samples are better.
- **$\bar{X} \sim N(\mu, \sigma^2/n)$ when $X_i$ are normal:** The sample mean of a normal population is exactly normally distributed. This enables exact confidence intervals and hypothesis tests (Chapters 8–10).
- **CDF technique:** Transforms like $Y = X^2$ producing chi-square, or $Y = -\ln(U)$ producing exponential from uniform, arise throughout statistics.

### Why Does $V(\bar{X}) = \sigma^2/n$?

This is perhaps the most important formula in all of applied statistics. Here is the intuition:

$\bar{X}$ is the average of n independent measurements, each with variance σ². When you average n independent sources of randomness, the spread decreases — not by a factor of n (that would be dividing by n), but by $\sqrt{n}$ in standard deviation terms, because variance adds under independence:

$$V\!\left(\frac{X_1+\cdots+X_n}{n}\right) = \frac{1}{n^2}\left(V(X_1)+\cdots+V(X_n)\right) = \frac{n\sigma^2}{n^2} = \frac{\sigma^2}{n}$$

Doubling n cuts $V(\bar{X})$ in half (standard error decreases by $1/\sqrt{2}$). Quadrupling n cuts the standard error in half. This is the mathematical reason why **"averaging reduces uncertainty"** and why larger samples are more informative.

### Key Warnings

1. **Zero covariance ≠ independence** (in general). It only means no linear relationship. Nonlinear dependence can still exist.
2. **Non-rectangular support → not independent.** If the bounds of y depend on x (or vice versa), X and Y cannot be independent.
3. **Variance of a difference:** $V(X - Y) = V(X) + V(Y)$ (not minus!) when X and Y are independent. This surprises many students — subtracting two random variables *adds* their variances because variability always compounds.
4. **The multinomial covariance is negative:** $\text{Cov}(X_i, X_j) = -np_ip_j < 0$. This makes intuitive sense — if more trials result in category i, fewer are left for category j.

### Important Terms (Glossary)

| Term | Meaning |
|---|---|
| Joint PDF/PMF | Complete probability description of (X, Y) together |
| Marginal distribution | Individual distribution of X (or Y) ignoring the other |
| Conditional distribution | Distribution of Y given specific value of X |
| Independence | $f(x,y) = f_X(x)f_Y(y)$; knowing X gives no info about Y |
| Covariance | $E(XY) - E(X)E(Y)$; measures linear co-movement |
| Correlation ρ | Standardized covariance; always in [−1, +1] |
| Bivariate normal | Joint normal; ρ = 0 implies independence |
| Multinomial | Joint distribution of counts in k categories (k > 2) |
| Linear combination | $Y = c_1X_1 + \cdots + c_nX_n$; mean and variance follow linear rules |
| Sample mean $\bar{X}$ | Average of n i.i.d. observations; $E=\mu$, $V=\sigma^2/n$ |
| Standard error | $\sigma_{\bar{X}} = \sigma/\sqrt{n}$; measure of precision of $\bar{X}$ |
| i.i.d. | Independent and identically distributed — the standard assumption |
| CDF technique | Method for finding distribution of Y = h(X) via $F_Y$ |
| Change of variables | $f_Y(y) = f_X(h^{-1}(y))\cdot|dx/dy|$ |
| Chi-square distribution | Distribution of $\sum Z_i^2$ where $Z_i \sim N(0,1)$; used in hypothesis tests |

---

# Probability with Independent Normal Random Variables

## Given

- \( X \sim N(10.5, 0.0025) \)
  - Mean \( \mu_X = 10.5 \)
  - Variance \( \sigma_X^2 = 0.0025 \Rightarrow \sigma_X = 0.05 \)

- \( Y \sim N(3.2, 0.0036) \)
  - Mean \( \mu_Y = 3.2 \)
  - Variance \( \sigma_Y^2 = 0.0036 \Rightarrow \sigma_Y = 0.06 \)

- \( X \) and \( Y \) are **independent**

---

## Problem

Find:

\[
P(10.4 < X < 10.6,\; 3.15 < Y < 3.25)
\]

---

## Step 1: Use Independence

Since \( X \) and \( Y \) are independent:

\[
P(A \cap B) = P(A) \cdot P(B)
\]

\[
P(10.4 < X < 10.6,\; 3.15 < Y < 3.25)
= P(10.4 < X < 10.6) \times P(3.15 < Y < 3.25)
\]

---

## Step 2: Standardize (Convert to Z)

### For \( X \)

\[
Z = \frac{X - \mu_X}{\sigma_X}
\]


::contentReference[oaicite:0]{index=0}


\[
\frac{10.4 - 10.5}{0.05} = -2,\quad
\frac{10.6 - 10.5}{0.05} = 2
\]

\[
P(10.4 < X < 10.6) = P(-2 < Z < 2)
\]

---

### For \( Y \)

\[
\frac{3.15 - 3.2}{0.06} = -0.833,\quad
\frac{3.25 - 3.2}{0.06} = 0.833
\]

\[
P(3.15 < Y < 3.25) = P(-0.833 < Z < 0.833)
\]

---

## Step 3: Use Standard Normal Table

- \( P(-2 < Z < 2) \approx 0.954 \)
- \( P(-0.833 < Z < 0.833) \approx 0.596 \)

---

## Step 4: Multiply

\[
P = 0.954 \times 0.596 \approx 0.568
\]

---

## Final Answer

\[
P(10.4 < X < 10.6,\; 3.15 < Y < 3.25) = 0.568
\]

---

## Intuition 💡

- Independence lets us **break a 2D probability into two 1D problems**.
- Without independence, you'd need a **joint distribution** (much harder).
- Standardization converts everything to a **single universal distribution (Z-table)**.

---

## Exam Tips 🚀

- Always check: **Are variables independent?**
- Convert to Z **carefully (common mistake: wrong σ)**.
- Use symmetry:
  - \( P(-a < Z < a) = 2P(0 < Z < a) \)

---

## Common Pitfalls ⚠️

- Using variance instead of standard deviation.
- Forgetting independence → multiplying incorrectly.
- Wrong Z conversion signs.

---

## Practical Interpretation

If dimensions are independent, you can compute tolerance probabilities easily — this is heavily used in:
- Manufacturing quality control
- Reliability engineering
- Statistical process control (SPC)
