#  Discrete Random Variables and Probability Distribution

---

## Chapter Outline
- 3.1 Probability Distributions and Probability Mass Functions
- 3.2 Cumulative Distribution Functions
- 3.3 Mean and Variance of a Discrete Random Variable
- 3.4 Discrete Uniform Distribution
- 3.5 Binomial Distribution
- 3.6 Geometric and Negative Binomial Distributions
- 3.7 Hypergeometric Distribution
- 3.8 Poisson Distribution

---

## 3.1 Probability Distributions and Probability Mass Functions

> **Definition — Probability Distribution:**
> The **probability distribution** of a random variable X is a description of the probabilities associated with the possible values of X.

For a discrete random variable, the distribution is specified by:
- A **list** of possible values along with the probability of each, or
- A **formula** (for more complex cases)

---

### Example — Flash Recharge Time

Three cell-phone cameras are tested. The probability that a camera meets the recharge specification is **0.8**, and cameras perform independently. The random variable X = number of cameras that pass. 

**Table  — Camera Flash Tests:**

| Camera 1 | Camera 2 | Camera 3 | Probability | X |
|----------|----------|----------|-------------|---|
| Pass | Pass | Pass | 0.512 | 3 |
| Fail | Pass | Pass | 0.128 | 2 |
| Pass | Fail | Pass | 0.128 | 2 |
| Fail | Fail | Pass | 0.032 | 1 |
| Pass | Pass | Fail | 0.128 | 2 |
| Fail | Pass | Fail | 0.032 | 1 |
| Pass | Fail | Fail | 0.032 | 1 |
| Fail | Fail | Fail | 0.008 | 0 |

For example, P(ppf) = (0.8)(0.8)(0.2) = 0.128.

In Example 3.1, we might summarize the experiment in terms of the three possible values of X: {0, 1, 2}. This simplifies description and analysis.



---

### Example  — Digital Channel

Let X = number of bits received in error in the next four bits transmitted through a digital communication channel. Possible values: {0, 1, 2, 3, 4}.

Suppose the probabilities are:
- P(X = 0) = 0.6561
- P(X = 1) = 0.2916
- P(X = 2) = 0.0486
- P(X = 3) = 0.0036
- P(X = 4) = 0.0001

The probability distribution of X is specified by these values along with the probability of each.

> **Practical Interpretation:** A random experiment can often be summarized with a random variable and its distribution. The details of the sample space can often be omitted.

---

### Probability Mass Function (PMF)

> **Definition — Probability Mass Function:**
> For a discrete random variable X with possible values x₁, x₂, ..., xₙ, a **probability mass function** is a function such that:
>
> 1. f(xᵢ) ≥ 0
> 2. $\sum_{i=1}^{n} f(x_i) = 1$
> 3. f(xᵢ) = P(X = xᵢ) &emsp; **(Eq. 3.1)**

For Example 3.3: f(0) = 0.6561, f(1) = 0.2916, f(2) = 0.0486, f(3) = 0.0036, f(4) = 0.0001. Check: probabilities sum to 1. ✓

**Analogy:** Just as loading on a long thin beam is described by a function specifying mass at each discrete point, a discrete random variable's distribution is described by a PMF specifying probability at each possible value.
![1](./images/3.1.png)
---

### Example  — Wafer Contamination (PMF with Infinite Range)

Let X = number of semiconductor wafers that need to be analyzed to detect a large particle of contamination. Probability a wafer contains a large particle = 0.01; wafers are independent.

Let p = wafer with large particle present, a = wafer where it is absent. Sample space:
$$s = \{p, ap, aap, aaap, aaaap, aaaaap, \ldots\}$$

Special cases:
- P(X = 1) = P(p) = 0.01
- P(X = 2) = P(ap) = 0.99(0.01) = 0.0099 (using independence)

General formula:
$$P(X = x) = 0.99^{x-1}(0.01), \quad x = 1, 2, 3, \ldots$$

Clearly f(x) ≥ 0. That probabilities sum to 1 is left as an exercise. This is an example of a **geometric random variable** (discussed in Section 3.6).

> **Practical Interpretation:** Even though the experiment has an unbounded number of outcomes, it can still be conveniently modeled with a discrete random variable with a (countably) infinite range.

---

## 3.2 Cumulative Distribution Functions

An alternate method for describing a probability distribution is using cumulative probabilities like P(X ≤ x).

> **Definition — Cumulative Distribution Function (CDF):**
> The **cumulative distribution function** of a discrete random variable X, denoted as F(x), is:
>
> $$F(x) = P(X \leq x) = \sum_{x_i \leq x} f(x_i)$$

**Properties of the CDF for a discrete random variable X:** (Eq. 3.2)

1. F(x) = P(X ≤ x) = Σ f(xᵢ) for all xᵢ ≤ x
2. 0 ≤ F(x) ≤ 1
3. If x ≤ y, then F(x) ≤ F(y) — (F is non-decreasing)

**Key properties explained:**
- Property 3 holds because if x ≤ y, the event {X ≤ x} is contained in {X ≤ y}.
- Even if X can only assume integer values, F(x) is defined at non-integer values.
- F(x) is **piecewise constant** between values x₁, x₂, ...

**Recovering the PMF from the CDF:**
$$P(X = x_i) = F(x_i) - \lim_{x \uparrow x_i} F(x)$$
i.e., the PMF at xᵢ equals the **jump** in F at xᵢ.

---

### Example  — Digital Channel (CDF)

From Example 3.3, find P(X ≤ 3):
The event that {X ≤ 3} is the union of the events
{X = 0}, {X = 1}, {X = 2}, and {X = 3}
$$P(X \leq 3) = P(X=0) + P(X=1) + P(X=2) + P(X=3)$$
$$= 0.6561 + 0.2916 + 0.0486 + 0.0036 = 0.9999$$

Also: P(X = 3) = P(X ≤ 3) − P(X ≤ 2) = 0.9999 − 0.9963 = 0.0036

The full CDF for this example is:

$$F(x) = \begin{cases} 0 & x < 0 \\ 
0.6561 & 0 \leq x < 1 \\ 
0.9477 & 1 \leq x < 2 \\ 
0.9963 & 2 \leq x < 3 \\ 
0.9999 & 3 \leq x < 4 \\ 
1 & 4 \leq x \end{cases}$$

For example: F(1.5) = P(X ≤ 1.5) = P(X = 0) + P(X = 1) = 0.6561 + 0.2916 = 0.9477.

---

### Example 3.6 — Recovering PMF from CDF

Given the cumulative distribution function:

$$F(x) = \begin{cases} 0 & x < -2 \\ 
0.2 & -2 \leq x < 0 \\ 
0.7 & 0 \leq x < 2 \\ 
1 & 2 \leq x \end{cases}$$

The only points with nonzero probability are −2, 0, and 2 (the jump points). The PMF at each point = jump size:

- f(−2) = 0.2 − 0 = **0.2**
- f(0) = 0.7 − 0.2 = **0.5**
- f(2) = 1.0 − 0.7 = **0.3**

---

## 3.3 Mean and Variance of a Discrete Random Variable

Two numbers often used to summarize a probability distribution:
- **Mean:** Measures the center or middle of the distribution
- **Variance:** Measures the dispersion or variability

Note: Two different distributions can have the same mean and variance — these measures do not uniquely identify a distribution.

> **Definition — Mean, Variance, and Standard Deviation:**
>
> The **mean** or **expected value** of the discrete random variable X, denoted μ or E(X), is:
> $$\mu = E(X) = \sum_x x \cdot f(x) \quad \text{(Eq. 3.3)}$$
>
> The **variance** of X, denoted σ² or V(X), is:
> $$\sigma^2 = V(X) = E(X - \mu)^2 = \sum_x (x - \mu)^2 f(x) = \sum_x x^2 f(x) - \mu^2$$
>
> The **standard deviation** of X is $\sigma = \sqrt{\sigma^2}$.

**Physical analogy:** The mean of a discrete random variable is a weighted average of the possible values, with weights equal to the probabilities. If f(x) is the PMF of loading on a long thin beam, E(X) is the **balance point** of the beam.

**Derivation of the alternative variance formula:**
$$V(X) = \sum_x (x-\mu)^2 f(x) = \sum_x x^2 f(x) - 2\mu \sum_x xf(x) + \mu^2 \sum_x f(x)$$
$$= \sum_x x^2 f(x) - 2\mu^2 + \mu^2 = \sum_x x^2 f(x) - \mu^2$$

---

### Example  — Digital Channel (Mean and Variance)

From Example 3.3, X = number of bits in error in next four bits transmitted.

**Mean:**
$$\mu = E(X) = 0f(0) + 1f(1) + 2f(2) + 3f(3) + 4f(4)$$
$$= 0(0.6561) + 1(0.2916) + 2(0.0486) + 3(0.0036) + 4(0.0001) = 0.4$$

Although X never assumes the value 0.4, the weighted average of the possible values is 0.4.

**Variance computation table:**

| x | x − 0.4 | (x − 0.4)² | f(x) | f(x)(x − 0.4)² |
|---|---------|------------|------|-----------------|
| 0 | −0.4 | 0.16 | 0.6561 | 0.104976 |
| 1 | 0.6 | 0.36 | 0.2916 | 0.104976 |
| 2 | 1.6 | 2.56 | 0.0486 | 0.124416 |
| 3 | 2.6 | 6.76 | 0.0036 | 0.024336 |
| 4 | 3.6 | 12.96 | 0.0001 | 0.001296 |

$$V(X) = \sigma^2 = \sum_{i=1}^{5} f(x_i)(x_i - 0.4)^2 = 0.36$$

> **Practical Interpretation:** The mean and variance summarize the distribution. The mean is a weighted average of the values; variance measures the dispersion of values from the mean.

---

### Example  — Marketing (Comparing Two Designs)

Two new product designs are compared on the basis of revenue potential.

- **Design A:** Revenue predicted accurately to be $3 million (certain). So X ~ deterministic: E(X) = $3 million, V(X) = 0.
- **Design B:** Revenue Y = $7 million with P = 0.3, or $2 million with P = 0.7.

$$E(Y) = \$7(0.3) + \$2(0.7) = \$3.5 \text{ million}$$

Since E(Y) > E(X), we might prefer Design B. However:

$$\sigma^2 = (7 - 3.5)^2(0.3) + (2 - 3.5)^2(0.7) = 5.25 \text{ million dollars squared}$$

$$\sigma = \sqrt{5.25} = 2.29 \text{ million dollars}$$

The standard deviation σ is large relative to μ, indicating high variability in Design B. Units of σ² are (millions of dollars)², which are hard to interpret — this is why standard deviation (in the same units as X) is preferred.

---

### Expected Value of a Function of a Discrete Random Variable

> **Definition — Expected Value of h(X):**
> If X is a discrete random variable with PMF f(x),
> $$E[h(X)] = \sum_x h(x) f(x) \quad \text{(Eq. 3.4)}$$

**Important:** In general, E[h(X)] ≠ h[E(X)]. For example, E[X²] ≠ [E(X)]².

**Special case:** For h(X) = aX + b:
- E(aX + b) = aE(X) + b
- V(aX + b) = a²V(X)

#### Example  — Digital Channel (Function of RV)

From Example 3.7, X = bits in error. Find E[X²]:

$$E[X^2] = 0^2(0.6561) + 1^2(0.2916) + 2^2(0.0486) + 3^2(0.0036) + 4^2(0.0001)$$
$$= 0 + 0.2916 + 0.1944 + 0.0324 + 0.0016 = 0.52$$

Note: E[X²] = 0.52 ≠ [E(X)]² = (0.4)² = 0.16.

**Continuing Example :** Suppose revenue for Design B is increased by 10%. New revenue U = h(Y) = 1.1Y.

$$E(U) = 1.1E(Y) = 1.1(3.5) = 3.85 \text{ million dollars}$$
$$V(U) = 1.1^2 V(Y) = 1.21(5.25) = 6.35 \text{ million dollars squared}$$

---

## 3.4 Discrete Uniform Distribution

The simplest discrete random variable assumes only a finite number of possible values, each with equal probability.

> **Definition — Discrete Uniform Distribution:**
> A random variable X has a **discrete uniform distribution** if each of the n values in its range x₁, x₂, ..., xₙ has equal probability. Then:
> $$f(x_i) = \frac{1}{n} \quad \text{(Eq. 3.5)}$$

---

### Example 3.10 — Serial Number

The first digit of a part's serial number is equally likely to be any digit 0 through 9. If X is the first digit of a randomly selected part: R = {0, 1, 2, ..., 9}, n = 10.

$$f(x) = 0.1 \quad \text{for each value in } R$$

---

### Mean and Variance of Discrete Uniform Distribution

Suppose X is a discrete uniform random variable on consecutive integers a, a+1, ..., b (for a ≤ b). The range contains b − a + 1 values, each with probability 1/(b − a + 1).

Using the algebraic identity $\sum_{k=a}^{b} k = \frac{b(b+1) - (a-1)a}{2}$:

> **Mean and Variance (Discrete Uniform — Eq. 3.6):**
> $$\mu = E(X) = \frac{b + a}{2}$$
> $$\sigma^2 = \frac{(b - a + 1)^2 - 1}{12}$$

---

### Example 3.11 — Number of Voice Lines

Let X = number of 48 voice lines in use at a particular time. Assume X ~ Discrete Uniform on [0, 48].

$$E(X) = \frac{48 + 0}{2} = 24$$

$$\sigma = \sqrt{\frac{(48 - 0 + 1)^2 - 1}{12}} = \sqrt{\frac{2400}{12}} = \sqrt{200} = 14.14$$

> **Practical Interpretation:** Average lines in use = 24, but the dispersion (σ = 14.14) is large, meaning at many times far more or fewer than 24 lines are in use.

**Note:** Eq. 3.6 is useful in transformation. If discrete uniform Y has range 5, 10, ..., 30, then Y = 5X where X has range 1, 2, ..., 6. Using E(aX + b) = aE(X) + b and V(aX + b) = a²V(X), results can be derived from Eq. 3.6.

---

## 3.5 Binomial Distribution

The binomial distribution is one of the most important discrete distributions. It arises whenever a random experiment consists of a fixed number of independent trials, each with only two possible outcomes.

### Bernoulli Trial

A single trial with two outcomes (success/failure) is called a **Bernoulli trial**.

> **Definition — Binomial Distribution:**
> A random experiment consists of n **Bernoulli trials** such that:
> 1. The trials are **independent**
> 2. Each trial results in only two possible outcomes: success (S) or failure (F)
> 3. The **probability of success** in each trial is constant, denoted p; probability of failure = 1 − p
>
> The random variable X = number of successes in n trials is a **binomial random variable** with parameters n and p.

> **PMF — Binomial Distribution:**
> $$f(x) = \binom{n}{x} p^x (1-p)^{n-x}, \quad x = 0, 1, 2, \ldots, n \quad \text{(Eq. 3.7)}$$

**Reasoning:** Any specific sequence of x successes and (n − x) failures has probability p^x(1−p)^(n−x) (by independence). The number of such sequences is C(n, x). So summing over all sequences gives the formula.

**Verify:** $\sum_{x=0}^{n} \binom{n}{x} p^x (1-p)^{n-x} = [p + (1-p)]^n = 1^n = 1$ ✓

**Notation:** X ~ B(n, p) or X ~ Bin(n, p).

---

### Mean and Variance — Binomial

> **Mean and Variance (Binomial — Eq. 3.8):**
> $$\mu = E(X) = np$$
> $$\sigma^2 = V(X) = np(1-p)$$

**Interpretation:** If p = 0.8 and n = 3 (Example 3.1), then E(X) = 3(0.8) = 2.4 and V(X) = 3(0.8)(0.2) = 0.48.

---

### Example 3.12 — Digital Channel (Binomial)

From Example 3.3, each bit is in error with probability 0.1, independently. X = number of bits in error in next 4 bits. X ~ B(4, 0.1).

$$P(X = 2) = \binom{4}{2}(0.1)^2(0.9)^2 = 6 \times 0.01 \times 0.81 = 0.0486$$

$$E(X) = 4(0.1) = 0.4, \quad V(X) = 4(0.1)(0.9) = 0.36$$

These match the earlier calculations in Examples 3.3 and 3.7.

---

### Example 3.13 — Semiconductor Wafers (Binomial)

Suppose that 10% of bits transmitted through a digital communications channel are received in error. Let X = number of bits in error in the next 5 bits transmitted. X ~ B(5, 0.1).

$$P(X \leq 1) = P(X=0) + P(X=1)$$
$$= \binom{5}{0}(0.1)^0(0.9)^5 + \binom{5}{1}(0.1)^1(0.9)^4$$
$$= 0.59049 + 5(0.1)(0.6561) = 0.59049 + 0.32805 = 0.91854$$

---

### Example 3.14 — Organic Pollution

Samples of water are analyzed for organic pollution. The probability of obtaining a positive test for organic pollution is 0.3. Ten samples are taken independently. Let X = number of positive tests. X ~ B(10, 0.3).

$$P(X \leq 2) = P(X=0) + P(X=1) + P(X=2)$$
$$= \binom{10}{0}(0.3)^0(0.7)^{10} + \binom{10}{1}(0.3)^1(0.7)^9 + \binom{10}{2}(0.3)^2(0.7)^8$$
$$= 0.0282 + 0.1211 + 0.2335 = 0.3828$$

$$E(X) = 10(0.3) = 3, \quad \sigma = \sqrt{10(0.3)(0.7)} = \sqrt{2.1} = 1.449$$

---

## 3.6 Geometric and Negative Binomial Distributions

### Geometric Distribution

Consider Bernoulli trials repeated until the **first success** occurs.

> **Definition — Geometric Distribution:**
> In a sequence of Bernoulli trials with success probability p, let X = number of trials until the **first success**. Then X is a **geometric random variable** and:
> $$f(x) = (1-p)^{x-1} p, \quad x = 1, 2, 3, \ldots \quad \text{(Eq. 3.9)}$$

Note: The sum of probabilities equals 1 because $\sum_{x=1}^{\infty}(1-p)^{x-1}p = p \cdot \frac{1}{1-(1-p)} = 1$.

The PMF for Example 3.4 (wafer contamination) is geometric with p = 0.01:
$$P(X = x) = 0.99^{x-1}(0.01)$$

> **Mean and Variance (Geometric — Eq. 3.10):**
> $$\mu = E(X) = \frac{1}{p}$$
> $$\sigma^2 = V(X) = \frac{1-p}{p^2}$$

**Interpretation:** If p = 0.01, E(X) = 100, meaning on average 100 wafers need to be inspected to find the first contaminated one.

**Memoryless property:** The geometric distribution has the property that:
$$P(X > s + t \mid X > s) = P(X > t)$$
meaning the number of additional trials needed is independent of how many have already been conducted without success.

---

### Example 3.15 — Geometric Distribution

The probability that a part from a manufacturing process is nonconforming is 0.1. Parts are selected sequentially. Let X = number of parts selected until first nonconforming part. X ~ Geometric(0.1).

$$P(X = 5) = (0.9)^4(0.1) = 0.0656$$
$$E(X) = \frac{1}{0.1} = 10, \quad \sigma^2 = \frac{0.9}{0.01} = 90$$

On average, 10 parts must be selected until the first nonconforming one.

---

### Negative Binomial Distribution

Generalize: instead of stopping after the **first** success, stop after the **r-th** success.

> **Definition — Negative Binomial Distribution:**
> In a sequence of Bernoulli trials with success probability p, let X = number of trials until the **r-th success**. Then X is a **negative binomial random variable** and:
> $$f(x) = \binom{x-1}{r-1}(1-p)^{x-r}p^r, \quad x = r, r+1, r+2, \ldots \quad \text{(Eq. 3.11)}$$

**Reasoning:** For the r-th success to occur on trial x, there must be exactly r−1 successes in the first x−1 trials AND a success on trial x. The binomial coefficient C(x−1, r−1) counts the ways to arrange r−1 successes among the first x−1 trials.

The geometric distribution is the special case of the negative binomial with r = 1.

> **Mean and Variance (Negative Binomial — Eq. 3.12):**
> $$\mu = E(X) = \frac{r}{p}$$
> $$\sigma^2 = V(X) = \frac{r(1-p)}{p^2}$$

---

### Example 3.16 — Negative Binomial

The probability that a bit is in error is 0.1. What is the probability that the 3rd error occurs on the 10th bit transmitted? X ~ Negative Binomial(r = 3, p = 0.1).

$$P(X = 10) = \binom{9}{2}(0.9)^7(0.1)^3 = 36 \times 0.4783 \times 0.001 = 0.0172$$

$$E(X) = \frac{3}{0.1} = 30, \quad V(X) = \frac{3(0.9)}{0.01} = 270$$

---

## 3.7 Hypergeometric Distribution

The hypergeometric distribution arises when sampling **without replacement** from a finite population containing two types of objects (successes and failures).

**Key difference from Binomial:** In binomial, trials are **independent** (sampling with replacement). In hypergeometric, trials are **not independent** (sampling without replacement).

> **Definition — Hypergeometric Distribution:**
> A set of N objects contains:
> - K objects classified as successes
> - N − K objects classified as failures
>
> A sample of size n is selected randomly (without replacement) where K ≤ N and n ≤ N.
>
> The random variable X = number of successes in the sample is a **hypergeometric random variable** and:
> $$f(x) = \frac{\binom{K}{x}\binom{N-K}{n-x}}{\binom{N}{n}}, \quad x = \max\{0, n+K-N\} \text{ to } \min\{K, n\} \quad \text{(Eq. 3.13)}$$

**Why those bounds?**
- Lower bound: if n + K > N, at least n + K − N successes must occur.
- Upper bound: can't have more successes than K (total successes) or n (sample size).

> **Mean and Variance (Hypergeometric — Eq. 3.14):**
> $$\mu = E(X) = np, \quad \text{where } p = K/N$$
> $$\sigma^2 = V(X) = np(1-p)\left(\frac{N-n}{N-1}\right)$$

The factor $\frac{N-n}{N-1}$ is called the **finite population correction factor**. When n is small relative to N, this factor ≈ 1, and the hypergeometric ≈ binomial.

---

### Example 3.22 — Sampling Without Replacement (Setup)

A bin contains 850 parts: 50 nonconforming and 800 conforming. Two parts are selected without replacement. Let X = number of nonconforming parts in sample.

Using direct calculation (conditional probability):

$$P(X = 0) = P(\text{both conform}) = \frac{800}{850} \cdot \frac{799}{849} = 0.886$$

$$P(X = 1) = \frac{800}{850} \cdot \frac{50}{849} + \frac{50}{850} \cdot \frac{800}{849} = 0.111$$

$$P(X = 2) = \frac{50}{850} \cdot \frac{49}{849} = 0.003$$

> This experiment is fundamentally different from the binomial — the trials are **not independent**. If each unit were replaced before the next selection, trials would be independent and X would be binomial.

---

### Example 3.23 — Sampling Without Replacement (Hypergeometric Formula)

Using the general hypergeometric formula with N = 850, K = 50, n = 2:

$$P(X=0) = \frac{\binom{50}{0}\binom{800}{2}}{\binom{850}{2}} = \frac{1 \times 319,600}{360,825} = 0.886$$

$$P(X=1) = \frac{\binom{50}{1}\binom{800}{1}}{\binom{850}{2}} = \frac{50 \times 800}{360,825} = \frac{40,000}{360,825} = 0.111$$

$$P(X=2) = \frac{\binom{50}{2}\binom{800}{0}}{\binom{850}{2}} = \frac{1,225}{360,825} = 0.003$$

---

### Example 3.24 — Sampling Without Replacement (Mean and Variance)

Continuing Example 3.23: N = 850, K = 50, n = 2, p = 50/850.

$$E(X) = np = 2 \times \frac{50}{850} = 0.118$$

$$V(X) = np(1-p)\frac{N-n}{N-1} = 2 \times \frac{50}{850} \times \frac{800}{850} \times \frac{848}{849} = 0.1115$$

---

## 3.8 Poisson Distribution

The Poisson distribution is used to model the **number of events** occurring in a fixed interval of time, length, area, or volume when events occur at a constant rate and independently of one another.

### Poisson Process

A **Poisson process** satisfies:
1. The probability that an event occurs in a small interval Δt is approximately λΔt, where λ is the **rate** (events per unit).
2. The probability of more than one event in a small interval is negligible (proportional to (Δt)²).
3. Events in non-overlapping intervals are **independent**.

> **Definition — Poisson Distribution:**
> The random variable X that equals the number of events in a Poisson process is a **Poisson random variable** with parameter λ > 0, and:
> $$f(x) = \frac{e^{-\lambda}\lambda^x}{x!}, \quad x = 0, 1, 2, \ldots \quad \text{(Eq. 3.15)}$$

**Verify:** $\sum_{x=0}^{\infty} \frac{e^{-\lambda}\lambda^x}{x!} = e^{-\lambda} \sum_{x=0}^{\infty} \frac{\lambda^x}{x!} = e^{-\lambda} \cdot e^{\lambda} = 1$ ✓

**Note:** λ can refer to the **rate** per unit (e.g., flaws per meter) OR the **expected number** over the interval of interest (e.g., flaws in 100 meters = 100 × rate).

> **Mean and Variance (Poisson):**
> $$\mu = E(X) = \lambda$$
> $$\sigma^2 = V(X) = \lambda \quad \text{(Eq. 3.16)}$$

The mean and variance are both equal to λ. This is a distinctive property of the Poisson distribution.

---

### Example 3.25 — Web Server Hits

A web server receives 0.5 hits per millisecond on average, and hits occur according to a Poisson process. Let X = number of hits in 6 milliseconds. Then λ = 0.5 × 6 = 3 hits.

$$P(X = 5) = \frac{e^{-3}(3)^5}{5!} = \frac{0.0498 \times 243}{120} = 0.1008$$

$$P(X \leq 2) = P(X=0) + P(X=1) + P(X=2)$$
$$= e^{-3}\left(\frac{3^0}{0!} + \frac{3^1}{1!} + \frac{3^2}{2!}\right)$$
$$= e^{-3}(1 + 3 + 4.5) = 0.0498 \times 8.5 = 0.4232$$

$$E(X) = \lambda = 3, \quad \sigma = \sqrt{3} = 1.732$$

---

### Example 3.26 — Contamination Particles

Contamination particles in a semiconductor wafer occur at a rate of 1 per cm². Wafer has area 100 cm². Let X = number of particles on the wafer. X ~ Poisson(λ = 100).

$$P(X = 0) = e^{-100} \approx 3.7 \times 10^{-44} \approx 0$$

This shows that essentially every wafer will contain contamination particles when the rate is 1 per cm².

---

### Poisson Approximation to the Binomial

When n is large and p is small (so that np = λ is moderate), the binomial distribution can be approximated by the Poisson:

> **Approximation:** If X ~ B(n, p) with n large and p small such that np = λ:
> $$\binom{n}{x} p^x (1-p)^{n-x} \approx \frac{e^{-\lambda}\lambda^x}{x!}$$
>
> Rule of thumb: n ≥ 100 and p ≤ 0.01 (so np ≤ 10).

---

### Example 3.27 — Poisson Approximation

Suppose X ~ B(100, 0.01), so λ = np = 1.

Exact: P(X = 0) = (0.99)^100 = 0.366

Poisson approximation: P(X = 0) = e^{-1} = 0.368 ← very close.

This approximation avoids the computational burden of large binomial calculations.

---

## Summary: Key Distributions at a Glance

| Distribution | Parameters | PMF | Mean | Variance |
|---|---|---|---|---|
| Discrete Uniform | a, b (integers) | 1/(b−a+1) | (b+a)/2 | [(b−a+1)²−1]/12 |
| Binomial | n, p | C(n,x)pˣ(1−p)ⁿ⁻ˣ | np | np(1−p) |
| Geometric | p | (1−p)^(x−1)p | 1/p | (1−p)/p² |
| Negative Binomial | r, p | C(x−1,r−1)(1−p)^(x−r)p^r | r/p | r(1−p)/p² |
| Hypergeometric | N, K, n | [C(K,x)C(N−K,n−x)]/C(N,n) | n(K/N) | np(1−p)(N−n)/(N−1) |
| Poisson | λ | e^(−λ)λˣ/x! | λ | λ |

---

# SECTION 4: CONCEPTUAL EXPLANATIONS

---

## 4.1 Probability Distributions and Probability Mass Functions — Deep Explanation

### What is a Probability Distribution?

A probability distribution is a **complete description** of the probabilistic behavior of a random variable. It tells us not just what values a random variable can take, but **how likely each value is**.

Think of it like this: if X is the number of defective parts in a batch, the probability distribution of X answers every question of the form "what is the chance that exactly k parts are defective?"

### What is a Probability Mass Function?

The PMF is the **mathematical tool** used to express a probability distribution for a **discrete** random variable. It is a function f(x) that assigns a probability to every possible value of X.

**Three requirements of a valid PMF:**
1. **Non-negativity:** f(x) ≥ 0 for all x — probabilities can't be negative.
2. **Summation to 1:** Σ f(x) = 1 — the total probability of all outcomes must be 1 (something must happen).
3. **Direct probability:** f(x) = P(X = x) — the function value at x IS the probability of that outcome.

**How to recognize a PMF problem:**
- You have a discrete random variable (count of something)
- You want the probability of a specific value: "What is P(X = 3)?"
- You're given a formula or table that satisfies the three properties above

**How to use a PMF:**
- For single values: P(X = k) = f(k) — just plug in k
- For ranges: P(a ≤ X ≤ b) = Σ f(x) for all x from a to b
- For events like P(X > k): use the complement: 1 − P(X ≤ k)

**Example walk-through:**
Suppose X has PMF f(x) = 0.2 for x ∈ {1, 2, 3, 4, 5}. Then:
- f(3) = P(X = 3) = 0.2
- P(X ≤ 2) = f(1) + f(2) = 0.2 + 0.2 = 0.4
- P(X > 3) = 1 − P(X ≤ 3) = 1 − 0.6 = 0.4

---

## 4.2 Cumulative Distribution Functions — Deep Explanation

### What is a CDF?

The CDF F(x) = P(X ≤ x) gives the probability that the random variable X takes a value **less than or equal to x**. It "accumulates" probability as x increases.

**Mental model:** Imagine walking along the x-axis from left to right. The CDF F(x) is the total probability you've "collected" at position x.

### Key Properties and Their Meaning

| Property | Mathematical Statement | Meaning |
|---|---|---|
| Starts at 0 | F(x) → 0 as x → −∞ | Before any possible values, no probability accumulated |
| Ends at 1 | F(x) → 1 as x → +∞ | After all possible values, all probability accumulated |
| Non-decreasing | If x ≤ y, then F(x) ≤ F(y) | Probability can only accumulate, never decrease |
| Bounded | 0 ≤ F(x) ≤ 1 | Always a valid probability |
| Piecewise constant | F(x) is flat between jump points | No probability mass between discrete values |

### The Jump = The PMF

For a discrete random variable, the **CDF jumps** exactly at the values where X has positive probability. The size of each jump equals the probability of that value:

$$f(x_i) = P(X = x_i) = F(x_i) - \lim_{x \uparrow x_i} F(x) = \text{(jump at } x_i\text{)}$$

**Example:** If F(x) jumps from 0.3 to 0.7 at x = 2, then P(X = 2) = 0.4.

### How to Read a CDF

Given F(x), you can calculate any probability:

| Probability | Formula using F |
|---|---|
| P(X ≤ a) | F(a) |
| P(X > a) | 1 − F(a) |
| P(X < a) (discrete) | F(a) − f(a) = lim F(x) as x → a⁻ |
| P(a ≤ X ≤ b) | F(b) − F(a) + f(a) = F(b) − F(a⁻) |
| P(a < X ≤ b) | F(b) − F(a) |
| P(X = a) | F(a) − F(a⁻) = jump at a |

### CDF vs PMF — When to Use Which?

| Use PMF when... | Use CDF when... |
|---|---|
| You need P(X = specific value) | You need P(X ≤ some value) |
| You're building the distribution | You want to find probabilities of ranges quickly |
| You're told individual probabilities | You're given the CDF and need to extract the PMF |

---

## 4.3 Mean and Variance of a Discrete Random Variable — Deep Explanation

### What the Mean Really Means

The mean μ = E(X) = Σ x·f(x) is a **probability-weighted average** of all possible values.

**Intuition 1 — Long-run average:** If you repeated the random experiment thousands of times and averaged all the values of X, the average would approach μ. This is the Law of Large Numbers.

**Intuition 2 — Balance point:** Think of the PMF as a loading diagram on a beam. f(x) at each point is the weight placed there. The mean is the point at which the beam would balance.

**Key insight:** The mean does NOT have to be a value that X can actually take. In Example 3.7, X (bits in error) never equals 0.4, yet μ = 0.4. It is a **property of the distribution**, not a specific outcome.

### What the Variance Really Measures

The variance σ² = V(X) = Σ (x − μ)²f(x) measures how **spread out** the distribution is around the mean.

- Each term (x − μ)² is the squared distance from a value to the mean.
- f(x) weights each squared distance by the probability of that value.
- Result: a probability-weighted average of squared deviations from the mean.

**Why squared?**
- Raw deviations (x − μ) would sum to zero by definition of the mean.
- Squaring makes all deviations positive and emphasizes large deviations.

**Alternative formula:** σ² = Σ x²f(x) − μ²
- Often easier to compute because you don't need to subtract μ from each x first.
- Interpret as: "Average of X² minus the square of the average of X."

### Standard Deviation — Returning to Original Units

Variance has units of (X's units)², making it hard to interpret. The standard deviation σ = √σ² is in the **same units as X**, making it much more interpretable.

A rough rule: in many distributions, most values of X fall within 2σ of the mean.

### Summary of Steps to Compute Mean and Variance

**Step 1:** List all possible values x and their probabilities f(x).

**Step 2:** Compute μ = Σ x·f(x) (multiply each value by its probability, add up).

**Step 3 (Variance — Method 1):**
- Compute (x − μ)² for each x
- Multiply by f(x): (x − μ)²f(x)
- Add all: σ² = Σ(x − μ)²f(x)

**Step 3 (Variance — Method 2 — often easier):**
- Compute x²f(x) for each x
- Add all: Σx²f(x)
- Subtract μ²: σ² = Σx²f(x) − μ²

**Step 4:** σ = √σ²

### Linear Transformation Rules (Frequently Tested)

If U = aX + b for constants a and b:
$$E(U) = aE(X) + b$$
$$V(U) = a^2 V(X)$$
$$\sigma_U = |a|\sigma_X$$

**Note:** Adding a constant b shifts the distribution (changes mean) but does NOT change the spread (variance). Multiplying by a stretches the distribution (changes both mean and variance).

---

*End of Chapter 3 Notes*
