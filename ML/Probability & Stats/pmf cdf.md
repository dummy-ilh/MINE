Here’s a **structured, complete summary** of PMF, CDF, and Expected Value, along with examples, properties, and solutions to your given problems:

---

# **1. Probability Mass Function (PMF)**

A **PMF** gives the probability that a **discrete random variable (X)** takes a specific value (x):

[
p(x) = P(X = x)
]

**Properties of PMF:**

1. (0 \le p(x) \le 1) for all (x)
2. (\sum_{x} p(x) = 1) (sum over all possible values)

**Plot:** PMF is usually shown as **vertical bars** at each value (x_i) with height (p(x_i)).

---

# **2. Cumulative Distribution Function (CDF)**

A **CDF** of a random variable (X) is the probability that (X \le x):

[
F(x) = P(X \le x) = \sum_{x_i \le x} p(x_i)
]

**Properties of CDF:**

1. (F(x)) is **non-decreasing**: (F(x_1) \le F(x_2)) if (x_1 < x_2)
2. (0 \le F(x) \le 1)
3. (\lim_{x \to -\infty} F(x) = 0), (\lim_{x \to +\infty} F(x) = 1)

**Plot:** CDF is a **step function** for discrete random variables, jumping at each (x_i) by (p(x_i)).

---

# **Example: 4-sided die game**

**Payoff table:**

| Outcome ((\omega)) | 1   | 2   | 3   | 4   |
| ------------------ | --- | --- | --- | --- |
| (X) (Payoff)       | -2  | -1  | 0   | 4   |
| PMF (p(a))         | 1/4 | 1/4 | 1/4 | 1/4 |
| CDF (F(a))         | 1/4 | 2/4 | 3/4 | 4/4 |

**Questions:**

1. (F(-1) = P(X \le -1) = P(X=-2) = 1/4)
2. (F(0.5) = P(X \le 0.5) = P(X=-2,-1,0) = 3/4)
3. (F(-5) = 0) (no values ≤ -5)
4. (F(5) = 1) (all values ≤ 5)
5. (p(-1) = P(X=-1) = 1/4)
6. (p(0.5) = 0) (X never takes 0.5)

---

# **Concept Questions with CDF**

Suppose (X) has values (1, 3, 5, 7) and CDF (F(a) = 0.5, 0.75, 0.9, 1).

**Problem 1:** (P(X \le 3) = F(3) = 0.75) → **Answer: d**

**Problem 2:** (P(X = 3) = F(3) - F(1) = 0.75 - 0.5 = 0.25) → **Answer: b**

---

# **3. Expected Value (Mean)**

The **expected value** of a discrete random variable (X) is:

[
E[X] = \sum_i x_i , p(x_i)
]

* It is a **weighted average** of all possible values.
* Measures **central tendency**.

**Properties:**

1. **Linearity I:** (E[X+Y] = E[X] + E[Y])
2. **Linearity II:** (E[aX + b] = aE[X] + b)
3. (E[h(X)] = \sum_i h(x_i) , p(x_i))

---

# **Example: Expected Value**

Random variable (X) takes values (80, 1, 10) with PMF:

| Value (x)  | 80  | 1   | 10  |
| ---------- | --- | --- | --- |
| PMF (p(x)) | 1/5 | 1/5 | 3/5 |

[
E[X] = 80 \cdot \frac{1}{5} + 1 \cdot \frac{1}{5} + 10 \cdot \frac{3}{5} = 16 + 0.2 + 6 = 22.2
]

So the **expected average payoff** over many experiments is **22.2**.

---

# **PMF and CDF Plots (Visual Summary)**

* **PMF:** Vertical bars for each discrete value; height = probability.
* **CDF:** Step function; jumps at each value by PMF.

**Example: Die payoff (X = -2, -1, 0, 4)**

```
PMF:
X:   -2  -1   0   4
p:   0.25 0.25 0.25 0.25
```

```
CDF:
F:  0.25 0.5 0.75 1
Step jumps at each X value.
```

---

If you want, I can **draw the actual PMF and CDF plots for this die example**—it’s really helpful for visualizing and remembering the differences.

Do you want me to do that?
