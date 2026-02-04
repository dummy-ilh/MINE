Now we arrive at one of the most important concepts in probability:

> **Expected Value = long-run average = weighted center**

This idea connects probability, statistics, decision theory, and even physics.

Let’s build it carefully and deeply.

---

# 1️⃣ Motivation: Long-Run Average

When you simulated rolling a fair die many times, the average approached:

$[
3.5
]$

But 3.5 is not even a face of the die.

So what is it?

It is the **expected value** — the theoretical long-run average.

---

# 2️⃣ Example 1: Biased Die (Five 3’s, One 6)

Faces:

* 3 appears 5 times
* 6 appears once

So:

$[
P(3) = 5/6
]$
$[
P(6) = 1/6
]$

If we roll 6000 times:

Expected counts:

* 5000 threes
* 1000 sixes

Average:

$[
\frac{5000\cdot3 + 1000\cdot6}{6000}
]$

Factor:

$[
= \frac{5}{6}\cdot3 + \frac{1}{6}\cdot6
]$

$[
= 3.5
]$

Notice what happened:

We multiplied each value by its probability and added.

That’s the key structure.

---

# 3️⃣ Example 2: Risky Dice Game

Two dice.

* Win $1000 if sum = 2
* Lose $100 otherwise

Probability of sum = 2:

$[
1/36
]$

Probability of not 2:

$[
35/36
]$

Expected winnings per game:

$[
1000\cdot\frac{1}{36} - 100\cdot\frac{35}{36}
]$

$[
= \frac{1000 - 3500}{36}
]$

$[
= -69.44
]$

This means:

On average, you lose $69.44 per game.

Even though you could win $1000 once.

This is why expected value is central in gambling and finance.

---

# 4️⃣ Formal Definition

If (X) takes values (x_1, x_2, ..., x_n),

$[
E$[X]$ = \sum_{j=1}^n p(x_j)x_j
]$

This is a **weighted average**.

Each value is weighted by how likely it is.

---

# 5️⃣ Important Observations

### 1️⃣ It may not be a possible value.

Fair die:

$[
E$[X]$ = 3.5
]$

But no roll gives 3.5.

---

### 2️⃣ If all values are equally likely:

It reduces to the normal average.

---

### 3️⃣ It measures location (center).

It tells you where the distribution balances.

---

# 6️⃣ Example 3

Values:

1, 3, 5

Probabilities:

1/6, 1/6, 2/3

Compute:

$[
E$[X]$ = 1\cdot\frac{1}{6} + 3\cdot\frac{1}{6} + 5\cdot\frac{2}{3}
]$

$[
= \frac{1 + 3 + 20}{6}
]$

$[
= 4
]$

Even though 4 is not a possible value.

---

# 7️⃣ Example 4: Bernoulli(p)

This is extremely important.

A Bernoulli random variable:

$[
X =
\begin{cases}
1 & \text{with prob } p \
0 & \text{with prob } 1-p
\end{cases}
]$

Expected value:

$[
E$[X]$ = p\cdot1 + (1-p)\cdot0 = p
]$

This is foundational.

In statistics:

* The sample mean of 0–1 data estimates p.
* Logistic regression predicts probabilities (expected value of Bernoulli).

---

# 8️⃣ Think Question: Sum of Two Dice

Each die has mean 3.5.

So intuitively:

$[
E$[\text{sum}]$ = 3.5 + 3.5 = 7
]$

Later we prove:

$[
E$[X+Y]$ = E$[X]$ + E$[Y]$
]$

This linearity property is one of the most powerful results in probability.

---

# 9️⃣ Expected Value as Center of Mass

This is the deepest intuition.

Imagine:

Each possible value (x_j) is a location on a line.

At that location, you place a mass equal to (p(x_j)).

Then:

$[
E$[X]$ = \frac{\sum p(x_j)x_j}{\sum p(x_j)}
]$

Since total probability is 1:

$[
E$[X]$ = \sum p(x_j)x_j
]$

This is identical to the physics formula:

$[
x_{\text{center}} = \frac{m_1x_1 + m_2x_2}{m_1+m_2}
]$

Probability behaves like mass.

That’s why it’s called a **probability mass function**.

---

# 🔟 Deep Interpretation

Expected value is:

* Long-run average
* Weighted mean
* Center of balance
* Decision-making metric
* First moment of distribution

It summarizes location but **not spread**.

Two very different distributions can have the same expected value.

That’s why variance comes next.

---

# 🔎 Philosophical Insight

Expected value answers:

> If I repeat this experiment infinitely many times, what is the average outcome?

But it does *not* answer:

* How risky is it?
* How variable is it?
* How likely are extreme outcomes?

For that, we need variance.

---

Now we move into one of the most powerful results in probability:

> **Expectation is linear.**

This single fact makes otherwise horrible calculations beautifully simple.

Let’s understand it structurally, not just mechanically.

---

# 1️⃣ Linearity of Expectation

## Property 1 (Additivity)

If (X) and (Y) are random variables:

$[
E$[X + Y]$ = E$[X]$ + E$[Y]$
]$

This works **always** — independence is NOT required.

That’s extremely important.

---

## Property 2 (Scaling and Shifting)

For constants (a, b):

$[
E$[aX + b]$ = aE$[X]$ + b
]$

Interpretation:

* Multiply a variable → mean scales
* Add a constant → mean shifts

This mirrors ordinary algebra.

That’s why we say expectation is **linear**.

---

# 2️⃣ Example 6: Sum of Two Dice

Let:

$[
X_1 = \text{first die}
]$
$[
X_2 = \text{second die}
]$

Each has mean:

$[
E$[X_1]$ = E$[X_2]$ = 3.5
]$

The total sum:

$[
X = X_1 + X_2
]$

By linearity:

$[
E$[X]$ = E$[X_1]$ + E$[X_2]$ = 3.5 + 3.5 = 7
]$

Notice what we did NOT do:

* We did not compute the full pmf.
* We did not sum 36 terms.

Linearity collapses complexity.

---

# 3️⃣ Example 7: Binomial Distribution

Let:

$[
X \sim \text{Binomial}(n, p)
]$

Interpretation:

(X) = number of successes in (n) trials.

Write:

$[
X = X_1 + X_2 + \dots + X_n
]$

where each (X_i) is Bernoulli(p).

Since:

$[
E$[X_i]$ = p
]$

By linearity:

$[
E$[X]$ = p + p + \dots + p = np
]$

This avoids evaluating:

$[
\sum_{k=0}^n k \binom{n}{k} p^k (1-p)^{n-k}
]$

Linearity turns a complicated combinatorial identity into a trivial calculation.

This is one of the most important tricks in probability.

---

# 4️⃣ Infinite Series: Mean May Not Exist

Expectation only exists if the series converges.

Example:

Values:
$[
2^k
]$

Probabilities:
$[
1/2^k
]$

Compute:

$[
E$[X]$ = \sum_{k=1}^{\infty} 2^k \cdot \frac{1}{2^k}
]$

Each term equals 1.

So:

$[
E$[X]$ = 1 + 1 + 1 + \dots = \infty
]$

The mean does not exist.

This shows:

Even if probabilities sum to 1,
the expectation may diverge.

Heavy-tailed distributions behave like this.

---

# 5️⃣ Geometric Distribution Mean

Let:

$[
X \sim \text{Geometric}(p)
]$

Definition used here:

(X) = number of failures before first success.

$[
P(X = k) = (1-p)^k p
]$

We compute:

$[
E$[X]$ = \sum_{k=0}^{\infty} k(1-p)^k p
]$

Using geometric series differentiation trick, the result is:

$[
E$[X]$ = \frac{1-p}{p}
]$

---

# 6️⃣ Understanding the Geometric Mean Intuitively

If success probability is (p):

* Small (p) → wait longer
* Large (p) → wait shorter

Example:

If (p = 1/2):

$[
E$[X]$ = \frac{1/2}{1/2} = 1
]$

On average, you expect 1 failure before first success.

This surprises many people.

---

# 7️⃣ Example 10: Coin Until First Head

Fair coin:

$[
p = 1/2
]$

Expected tails before first head:

$[
E$[X]$ = 1
]$

Even though occasionally you may flip many tails.

This highlights:

Expectation is an average,
not a guarantee.

---

# 8️⃣ Example 11: Michael Jordan

Free throw success rate:

$[
0.8
]$

Failure probability:

$[
p = 0.2
]$

We want:

Number of successes before first failure.

So:

$[
X \sim \text{Geometric}(0.2)
]$

Mean:

$[
E$[X]$ = \frac{1-0.2}{0.2} = \frac{0.8}{0.2} = 4
]$

On average:

He would make 4 shots before missing one.

---

# 9️⃣ Deep Structural Understanding of Linearity

Linearity is powerful because:

$[
E\left$[\sum X_i\right]$ = \sum E$[X_i]$
]$

No independence needed.

This allows:

* Counting arguments
* Combinatorial expectations
* Random graph analysis
* Algorithm analysis
* Indicator variable tricks

It is one of the most used tools in probability theory.

---

# 🔟 Why Linearity Works (Intuition)

Expectation is defined as:

$[
E$[X]$ = \sum p(\omega) X(\omega)
]$

When you expand:

$[
E$[X + Y]$ = \sum p(\omega)(X(\omega) + Y(\omega))
]$

Distribute the sum:

$[
= \sum p(\omega)X(\omega) + \sum p(\omega)Y(\omega)
]$

$[
= E$[X]$ + E$[Y]$
]$

It works because summation is linear.

That’s it.

No deeper magic.

---

# 11️⃣ Big Conceptual Takeaways

* Expectation is linear.
* It simplifies complex distributions.
* It works even without independence.
* Infinite distributions may have infinite mean.
* Geometric and binomial expectations become trivial using linearity.

---

Now we are entering one of the **most important ideas in probability theory** — the expectation of a function of a random variable.

This concept is so fundamental that it powers variance, moment generating functions, risk modeling, machine learning loss functions, finance pricing — everything.

Let’s master it properly.

---

# 1️⃣ The Core Idea: Expectation of a Function

If:

* ( X ) is a discrete random variable
* It takes values ( x_1, x_2, x_3, \dots )
* With probabilities ( p(x_j) )

And you define a new random variable:

$[
Y = h(X)
]$

Then:

$[
E$[h(X)]$ = \sum_j h(x_j) p(x_j)
]$

### 🔥 Important Insight

You **do NOT** need to find the distribution of ( Y ).

You simply:

1. Take each value of ( X )
2. Apply the function
3. Multiply by the original probability
4. Add everything

This is sometimes called the **change of variables formula (for discrete case)**.

---

# 2️⃣ Example 12 — Square of a Die

Let:

$[
X = \text{value of one die}
]$

Each outcome has probability ( 1/6 ).

Define:

$[
Y = X^2
]$

### Step 1: Table

| X      | 1   | 2   | 3   | 4   | 5   | 6   |
| ------ | --- | --- | --- | --- | --- | --- |
| Y = X² | 1   | 4   | 9   | 16  | 25  | 36  |
| Prob   | 1/6 | 1/6 | 1/6 | 1/6 | 1/6 | 1/6 |

### Step 2: Compute expectation

$[
E$[X^2]$ = \frac{1+4+9+16+25+36}{6}
]$

$[
= \frac{91}{6}
]$

$[
= 15.167
]$

---

### 🔎 Notice Something Interesting

$[
E$[X]$ = 3.5
]$

But:

$[
E$[X]$^2 = 12.25
]$

That is NOT equal to:

$[
E$[X^2]$ = 15.167
]$

So in general:

$[
E$[h(X)]$ \neq h(E$[X]$)
]$

This is extremely important.

Only linear functions behave nicely.

---

# 3️⃣ Example 13 — Two Dice with a Payoff Function

Now we step into a more serious example.

Let:

$[
X = \text{sum of two dice}
]$

Distribution:

| X    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   | 11   | 12   |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Prob | 1/36 | 2/36 | 3/36 | 4/36 | 5/36 | 6/36 | 5/36 | 4/36 | 3/36 | 2/36 | 1/36 |

Payoff function:

$[
Y = X^2 - 6X + 1
]$

We compute:

$[
E$[Y]$ = \sum (x^2 - 6x + 1) P(X=x)
]$

They computed:

$[
E$[Y]$ = 13.833
]$

Since this is positive, the bet has **positive expected value**.

---

# 4️⃣ Why Is ( E$[h(X)]$ \neq h(E$[X]$) )?

Let’s understand deeply.

Expectation is a **weighted average**.

When you apply a nonlinear function:

* Squaring exaggerates larger values.
* Convex functions amplify extremes.
* Concave functions dampen extremes.

Because of this distortion:

$[
E$[h(X)]$ \neq h(E$[X]$)
]$

This idea leads to:

### 🔥 Jensen’s Inequality (very important in ML and optimization)

If ( h ) is convex:

$[
E$[h(X)]$ \ge h(E$[X]$)
]$

---

# 5️⃣ But Linear Functions Are Special

If:

$[
Y = aX + b
]$

Then:

$[
E$[Y]$ = aE$[X]$ + b
]$

Why?

Because expectation is **linear**.

---

# 6️⃣ Linearity of Expectation (Very Powerful)

Two key properties:

### Property 1:

$[
E$[X + Y]$ = E$[X]$ + E$[Y]$
]$

This is true:

* Even if X and Y are dependent
* Always true
* No assumptions required

This is incredibly powerful.

---

### Property 2:

$[
E$[aX + b]$ = aE$[X]$ + b
]$

---

# 7️⃣ Why Linearity Works (Intuition)

Expectation is just a weighted average:

$[
E$[X]$ = \sum x_i P(\omega_i)
]$

When adding random variables:

$[
E$[X + Y]$ = \sum (x_i + y_i)P(\omega_i)
]$

Break apart:

$[
= \sum x_i P(\omega_i) + \sum y_i P(\omega_i)
]$

$[
= E$[X]$ + E$[Y]$
]$

---

# 8️⃣ Deep Insight: Why This Matters

This idea powers:

* Variance:
  $[
  Var(X) = E$[X^2]$ - (E$[X]$)^2
  ]$
* Bias-variance tradeoff
* Loss functions in ML
* Risk-neutral pricing in finance
* Expected utility theory
* Reinforcement learning reward

---

# 9️⃣ Common Pitfalls

❌ Thinking expectation is “plug in the mean”

Wrong for nonlinear functions.

❌ Forgetting probabilities

Always multiply by probability.

❌ Confusing expectation of sum with sum of distributions

Linearity works without independence.

---

# 10️⃣ Big Picture Summary

### Core formula:

$[
E$[h(X)]$ = \sum h(x)p(x)
]$

### Not true in general:

$[
E$[h(X)]$ = h(E$[X]$)
]$

### Always true:

$[
E$[X + Y]$ = E$[X]$ + E$[Y]$
]$

$[
E$[aX + b]$ = aE$[X]$ + b
]$

---


