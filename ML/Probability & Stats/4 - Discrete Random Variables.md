# Random variable

## 1. The Core Definition

Formally, a **random variable (X)** is a function that assigns a real number to each outcome in a sample space ().

* **Sample Space ():** The set of all possible outcomes (e.g., ).
* **Random Variable ():** The numerical value we care about (e.g., , ).

---

## 2. Discrete vs. Continuous

Random variables are classified based on the types of numbers they can take:

### Discrete Random Variables

These have "gaps" between their values. They are usually **counts**.

* **Definition:** Can take on a countable number of distinct values ().
* **Examples:**
* Number of students in a room.
* Number of heads in 10 coin flips.
* Number of cars passing through a toll booth.


* **Math Tool:** We use a **Probability Mass Function (PMF)** to find the probability of a specific value.

### Continuous Random Variables

These can take any value within a range. They are usually **measurements**.

* **Definition:** Can take an infinite number of values within an interval.
* **Examples:**
* The exact height of a person (e.g.,  cm).
* The time it takes for a battery to die.
* The temperature outside.


* **Math Tool:** We use a **Probability Density Function (PDF)** because the probability of an *exact* point (like exactly  degrees) is technically zero; we measure probabilities over intervals instead.

[Image comparing a discrete probability mass function bar chart and a continuous probability density function curve]

---

## 3. Why is it called "Random"?

It is called **random** because you don't know the specific value the variable will take until the experiment is performed. However, you *do* know the "distribution"—the pattern of how likely each value is.

| Concept | Algebraic Variable () | Random Variable () |
| --- | --- | --- |
| **Value** | A specific fixed number (here, ). | Any value in a set (). |
| **Goal** | Solve for the unknown. | Describe the likelihood of outcomes. |

---

## 4. Key Properties

To fully describe a random variable, we look at:

1. **Expected Value ():** The "long-run average" or mean.
2. **Variance ():** How much the values typically spread out from the mean.
3. **Standard Deviation ():** The square root of variance, providing a measure of spread in the original units.

Got it 👍
Here is the **same content rewritten cleanly in `.md` format**, **without renumbering**, and written so that **everything naturally reads as a subsection of Section 2** (i.e., no new major section numbers).

---

### Probability Mass Function and Cumulative Distribution Function

Writing expressions such as
[
P(X = a)
]
repeatedly can be tedious. When it is clear from context which random variable we are referring to, we simplify the notation.

* We write **( p(a) )** to mean the probability that the random variable takes the value ( a ).
* When we want to explicitly specify the random variable, we write **( p_X(a) )**.

---

### Probability Mass Function (PMF)

**Definition**
The **probability mass function (pmf)** of a discrete random variable ( X ) is the function:
[
p(a) = P(X = a)
]

---

### Properties of the PMF

* ( 0 \le p(a) \le 1 ) for all ( a )
* The value ( a ) can be any real number

  * If ( X ) never takes the value ( a ), then ( p(a) = 0 )
* The total probability over all possible values is:
  [
  \sum_a p(a) = 1
  ]

📌 **Interpretation:**
The pmf completely describes how probability is distributed across the possible values of a discrete random variable.

---

### Example: Maximum of Two Dice

Let ( \Omega ) be the sample space for rolling two fair dice:
[
\Omega = {(i,j) \mid i,j = 1,2,\ldots,6}
]

Define the random variable ( M ) as the maximum of the two dice:
[
M(i,j) = \max(i,j)
]

For example, if the outcome is ( (3,5) ), then:
[
M(3,5) = 5
]

---

### PMF of the Maximum

We can describe a discrete random variable by listing its possible values and the corresponding probabilities.

| Value ( a ) | 1                | 2                | 3                | 4                | 5                | 6                 |
| ----------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | ----------------- |
| ( p(a) )    | ( \frac{1}{36} ) | ( \frac{3}{36} ) | ( \frac{5}{36} ) | ( \frac{7}{36} ) | ( \frac{9}{36} ) | ( \frac{11}{36} ) |

For example:
[
p(2) = \frac{3}{36}
]

---

### Question

What is ( p(8) )?

**Solution**
The maximum of two dice can never exceed 6, so:
[
p(8) = 0
]

---

### Think About It

Let ( Z(i,j) = i + j ) be the sum of two dice.

* What are the possible values of ( Z )?
* What does its pmf look like?
* Does it resemble a familiar distribution?

📌 *(This pmf forms the well-known triangular distribution for dice sums.)*

---

### Events and Inequalities

Inequalities involving random variables describe **events** in the sample space.

An expression such as:
[
X \le a
]
represents the event consisting of all outcomes ( \omega \in \Omega ) for which:
[
X(\omega) \le a
]

---

### Example: Sum of Two Dice

Let:

* ( \Omega ) be the set of all outcomes ( (i,j) ) from rolling two dice,
* ( Z(i,j) = i + j ) be the sum of the dice.

Consider the event:
[
Z \le 4
]

This event corresponds to the set:
[
{(1,1), (1,2), (1,3), (2,1), (2,2), (3,1)}
]

📌 **Interpretation:**
Inequalities on random variables allow us to describe meaningful events such as:

* “the sum is at most 4,”
* “the maximum is no larger than 5,”
* “the payoff is positive.”

These ideas are essential for defining and working with the **cumulative distribution function (CDF)** in the next subsection.

---
Here is a **clean, precise, and conceptually strong rewrite** in **Markdown (`.md`) format**, keeping everything as a **subsection of Section 2** (no new numbering), with emphasis on clarity and correct probabilistic interpretation.

---

### The Cumulative Distribution Function (CDF)

#### Definition

The **cumulative distribution function (CDF)** of a random variable ( X ) is the function ( F ) defined by:
[
F(a) = P(X \le a)
]

We often refer to ( F ) simply as the **distribution function**.

⚠️ **Important:**
The definition uses **“less than or equal to” (≤)**. This detail is crucial for obtaining correct probability calculations, especially when dealing with discrete random variables.

---

### Interpretation

The CDF ( F(a) ) gives the **total probability accumulated** by the random variable ( X ) up to the value ( a ). In other words, it sums all probability mass at values **less than or equal to** ( a ).

Formally, for a discrete random variable:
[
F(a) = \sum_{b \le a} p(b)
]

---

### Example: CDF of the Maximum of Two Dice

Recall the random variable ( M ), defined as the maximum of two dice.

#### PMF of ( M )

| Value ( a ) | 1                | 2                | 3                | 4                | 5                | 6                 |
| ----------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | ----------------- |
| ( p(a) )    | ( \frac{1}{36} ) | ( \frac{3}{36} ) | ( \frac{5}{36} ) | ( \frac{7}{36} ) | ( \frac{9}{36} ) | ( \frac{11}{36} ) |

---

#### CDF of ( M )

| Value ( a ) | 1                | 2                | 3                | 4                 | 5                 | 6                 |
| ----------- | ---------------- | ---------------- | ---------------- | ----------------- | ----------------- | ----------------- |
| ( F(a) )    | ( \frac{1}{36} ) | ( \frac{4}{36} ) | ( \frac{9}{36} ) | ( \frac{16}{36} ) | ( \frac{25}{36} ) | ( \frac{36}{36} ) |

---

### How the CDF Is Computed

Each value of the CDF is obtained by **adding up the PMF values** from the smallest possible outcome up to ( a ).

For example:
[
F(4) = P(M \le 4) = \frac{1}{36} + \frac{3}{36} + \frac{5}{36} + \frac{7}{36} = \frac{16}{36}
]

---

### Event-Based View

The inequality ( M \le 4 ) corresponds to the event:
[
{1, 2, 3, 4}
]

Thus:
[
F(4) = P(M \le 4)
]

📌 This highlights the close connection between **inequalities involving random variables** and **events in the sample space**.

---

### CDF Values Outside the Support

Like the PMF, the CDF is defined for **all real numbers**, not just the values that the random variable can take.

Examples:

* ( F(8) = 1 )
  *(All probability mass has already accumulated.)*
* ( F(-2) = 0 )
  *(No probability mass has accumulated yet.)*
* ( F(2.5) = \frac{4}{36} )
  *(Same as ( F(2) ), since ( M ) cannot take values between 2 and 3.)*
* ( F(\pi) = \frac{9}{36} )
  *(Same as ( F(3) ).)*

📌 **Key Insight:**
For discrete random variables, the CDF is a **step function**—it increases only at values where the PMF is nonzero.

---


Below is a **clear `.md` rewrite with ASCII graphs** and **conceptual explanation**, written so it fits naturally as a continuation of **Section 2**.

---

### Graphs of the PMF ( p(a) ) and the CDF ( F(a) )

Probability mass functions and cumulative distribution functions can be **visualized graphically**, which helps build intuition about how probability is distributed and accumulated.

Consider the following example.

---

### Example: Number of Heads in Three Coin Tosses

Let ( X ) be the number of heads obtained when a fair coin is tossed three times.

#### Possible Values and Distributions

| Value ( a )  | 0               | 1               | 2               | 3               |
| ------------ | --------------- | --------------- | --------------- | --------------- |
| PMF ( p(a) ) | ( \frac{1}{8} ) | ( \frac{3}{8} ) | ( \frac{3}{8} ) | ( \frac{1}{8} ) |
| CDF ( F(a) ) | ( \frac{1}{8} ) | ( \frac{4}{8} ) | ( \frac{7}{8} ) | ( 1 )           |

---
![PMF and CDF for number of heads in three coin tosses](pmf_cdf.jpeg)

![PMF and CDF for number of heads in three coin tosses](pmf_cdf2.jpeg)



Here is a **clean, well-explained rewrite in `.md` format**, aligned with your Section 2 flow (no new numbering), and with **intuition added for each property**.

---

### Properties of the Cumulative Distribution Function ( F )

The cumulative distribution function ( F ) of a random variable satisfies several fundamental properties. These properties follow directly from the fact that ( F(a) = P(X \le a) ).

---

### Monotonicity (Non-decreasing Property)

The function ( F ) is **non-decreasing**.

Formally, if ( a \le b ), then:
[
F(a) \le F(b)
]

📌 **Explanation:**
If ( a \le b ), then the event ( {X \le a} ) is a subset of the event ( {X \le b} ).
Adding more possible values can only **increase or preserve** the total probability—it can never reduce it.

Graphically, this means the CDF **never goes down**.

---

### Boundedness

For all real numbers ( a ):
[
0 \le F(a) \le 1
]

📌 **Explanation:**
Since ( F(a) ) is a probability, it must lie between 0 and 1.
It represents the probability that ( X ) takes a value **no larger than ( a )**.

---

### Limits at ( \pm\infty )

The CDF satisfies:
[
\lim_{a \to \infty} F(a) = 1
]
[
\lim_{a \to -\infty} F(a) = 0
]

📌 **Explanation:**

* As ( a ) becomes very large, the event ( {X \le a} ) eventually includes **all possible outcomes**, so the probability approaches 1.
* As ( a ) becomes very small (very negative), the event ( {X \le a} ) becomes **impossible**, so the probability approaches 0.

In words:

* For large ( a ), it is almost certain that ( X \le a ).
* For very small ( a ), it is almost certain that ( X > a ).

---

### Why These Properties Must Hold

These properties are not assumptions—they are **necessary consequences** of the definition:
[
F(a) = P(X \le a)
]

Every valid CDF must:

* Accumulate probability without losing it,
* Stay within valid probability bounds,
* Start at 0 and end at 1.

📌 **Key Insight:**
If a function violates any of these properties, it **cannot** be the CDF of a random variable.

---

### Think About It

* Why can the CDF remain flat over some intervals?
* Why does the CDF jump at certain values for discrete random variables?
* How would these properties change (or not change) for continuous random variables?

---

