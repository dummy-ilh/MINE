# Probability: Terminology and Examples



---

## 1. Learning Goals

By the end of this lesson, you should be able to:

1. Understand and define:

   * Sample space
   * Event
   * Probability function

2. Translate a real-world random scenario into:

   * A well-defined experiment
   * A corresponding sample space

3. Perform basic probability calculations using a probability function.

---

## 2. Core Terminology

### 2.1 Probability “Cast List”

* **Experiment**
  A repeatable procedure with clearly defined possible outcomes.

* **Sample Space (Ω or S)**
  The set of all possible outcomes of an experiment.

* **Event**
  A subset of the sample space.

* **Probability Function**
  A function that assigns a probability to each outcome in the sample space.

Later in the course, we will also study:

* **Probability Density**
  Used when outcomes form a continuous range.

* **Random Variable**
  A numerical quantity determined by the outcome of an experiment.

---

## 2.2 Simple Examples

---

### Example 1: Tossing a Fair Coin

**Experiment:**
Toss a coin and record whether it lands Heads or Tails.

**Sample Space:**
Ω = {H, T}

**Probability Function:**
P(H) = 0.5
P(T) = 0.5

Since the coin is fair, both outcomes are equally likely.

---

### Example 2: Tossing a Fair Coin Three Times

**Experiment:**
Toss the coin three times and record the sequence of outcomes.

**Sample Space:**
Ω = {
HHH, HHT, HTH, HTT,
THH, THT, TTH, TTT
}

There are 2³ = 8 possible outcomes.

**Probability Function:**
Each outcome has probability 1/8.

We can organize this in a probability table:

| Outcome     | HHH | HHT | HTH | HTT | THH | THT | TTH | TTT |
| ----------- | --- | --- | --- | --- | --- | --- | --- | --- |
| Probability | 1/8 | 1/8 | 1/8 | 1/8 | 1/8 | 1/8 | 1/8 | 1/8 |

Since the coin tosses are independent and fair, all sequences are equally likely.

---

### Example 3: Measuring the Mass of a Proton

**Experiment:**
Follow a defined procedure to measure the proton’s mass and report the result.

**Sample Space:**
Ω = $[0, ∞)

In principle, any positive real value is possible.

**Probability Function:**
Because there are infinitely many possible outcomes in a continuous range, we cannot assign probabilities to individual values.

Instead, we use a **probability density function (PDF)**. This will be introduced later in the course.

Key idea:
Discrete outcomes → probability function
Continuous outcomes → probability density

---

### Example 4: Counting Taxis (Infinite Discrete Sample Space)

**Experiment:**
Count how many taxis pass 77 Mass Ave during an 18.05 class.

**Sample Space:**
Ω = {0, 1, 2, 3, 4, ...}

This is an infinite but discrete set.

A common model for this scenario is the **Poisson distribution**:

$[
P(k) = \frac{e^{-\lambda} \lambda^k}{k!}
]$

where:

* k = number of taxis observed
* λ = average number of taxis

We can represent this as:

| Outcome     | 0      | 1       | 2           | 3           | ... | k           | ... |
| ----------- | ------ | ------- | ----------- | ----------- | --- | ----------- | --- |
| Probability | e^{-λ} | e^{-λ}λ | e^{-λ}λ²/2! | e^{-λ}λ³/3! | ... | e^{-λ}λᵏ/k! | ... |

---

### Important Question

What is:

$[
\sum_{k=0}^{\infty} \frac{e^{-\lambda} \lambda^k}{k!} , ?
]$

**Answer:**
The sum equals 1.

Why?

Because:

* It represents the total probability over all possible outcomes.
* A valid probability distribution must sum to 1.

Mathematically, this follows from the Taylor series expansion of the exponential function:

$[
e^\lambda = \sum_{n=0}^{\infty} \frac{\lambda^n}{n!}
]$

Multiplying both sides by ( e^{-\lambda} ) gives:

$[
\sum_{k=0}^{\infty} \frac{e^{-\lambda} \lambda^k}{k!} = 1
]$

---

## Big Picture Takeaways

1. A probability model always starts with:

   * A clear experiment
   * A well-defined sample space

2. Sample spaces can be:

   * Finite (coin toss)
   * Infinite discrete (taxi count)
   * Continuous (proton mass)

3. Probability functions must satisfy:

   * Each probability ≥ 0
   * Total probability = 1

---


## Example 5: Two Dice (Choice of Sample Space)

### Step 1: One Die

If we roll one die:

| Outcome     | 1   | 2   | 3   | 4   | 5   | 6   |
| ----------- | --- | --- | --- | --- | --- | --- |
| Probability | 1/6 | 1/6 | 1/6 | 1/6 | 1/6 | 1/6 |

---

### Step 2: Two Dice

Now suppose we roll two dice. What should the sample space be?

There are two natural choices.

---

### Option 1: Record the Ordered Pair (First Die, Second Die)

Sample space:

{(1,1), (1,2), …, (6,6)}

Why are there 36 outcomes?

Because:

* First die has 6 possibilities
* Second die has 6 possibilities
* Total outcomes = 6 × 6 = 36

Each outcome is equally likely with probability 1/36.

We can organize this in a two-dimensional table:

| Die 1 \ Die 2 | 1    | 2    | 3    | 4    | 5    | 6    |
| ------------- | ---- | ---- | ---- | ---- | ---- | ---- |
| 1             | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 |
| 2             | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 |
| 3             | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 |
| 4             | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 |
| 5             | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 |
| 6             | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 |

This is called the **product sample space**.

---

### Option 2: Record the Sum of the Dice

Now the sample space becomes:

{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

There are 11 possible sums.

But these outcomes are **not equally likely**.

We count how many ordered pairs produce each sum:

| Sum | Probability |
| --- | ----------- |
| 2   | 1/36        |
| 3   | 2/36        |
| 4   | 3/36        |
| 5   | 4/36        |
| 6   | 5/36        |
| 7   | 6/36        |
| 8   | 5/36        |
| 9   | 4/36        |
| 10  | 3/36        |
| 11  | 2/36        |
| 12  | 1/36        |

For example:

* Sum = 7 occurs in 6 ways:
  (1,6), (2,5), (3,4), (4,3), (5,2), (6,1)

So:
$[
P(7) = 6/36
]$

---

## Think: What Is the Relationship Between the Two Tables?

The second table (sums) is derived from the first table (ordered pairs).

Specifically:

* The ordered-pair table is the **more detailed sample space**.
* The sum table is obtained by **grouping together outcomes that produce the same sum**.
* The probabilities in the sum table are found by **adding the probabilities of the corresponding ordered pairs**.

In other words:

The second distribution is obtained by applying a function
(sum of dice) to the original sample space.

This idea is fundamental in probability:

A **random variable** transforms one sample space into another by grouping outcomes together.

---

## Big Conceptual Insight

When modeling randomness:

1. The experiment must be clearly defined.
2. The sample space depends on what you choose to record.
3. Different sample spaces can describe the same physical experiment.
4. More detailed spaces (like ordered pairs) can be “collapsed” into simpler ones (like sums).

---


