# FAANG & Quant Interview Questions — With Full Answers
## Chapter: Set Theory, Counting, P&C, Probability vs. Statistics, Frequentist vs. Bayesian

---

## How to Use This

Each question has three parts:
- **The question** — exactly as it would be asked
- **The answer** — what a top candidate actually says
- **What they're testing** — so you understand *why* it's asked

---

---

# PART 1 — SET THEORY

---

**Q1. Events A and B are mutually exclusive. Are they independent?**

**Answer:**
Almost certainly **no** — and this is a common trap.

Mutually exclusive means $A \cap B = \emptyset$, so $P(A \cap B) = 0$.

For independence we need $P(A \cap B) = P(A) \cdot P(B)$.

That would require $P(A) \cdot P(B) = 0$, meaning at least one of them has zero probability.

So unless one of the events is impossible, mutually exclusive events are actually **dependent** — knowing A occurred tells you B definitely did not. That's the definition of dependence.

*What they're testing:* Whether you conflate two concepts that sound related but mean opposite things.

---

**Q2. You have user data: 40% opened an email, 30% clicked an ad, 20% visited the site. 10% did both email + ad, 8% did both ad + site, 5% did both email + site, 2% did all three. What fraction of users did at least one of these things?**

**Answer:**
Use three-set inclusion-exclusion:

$$|A \cup B \cup C| = |A| + |B| + |C| - |A \cap B| - |A \cap C| - |B \cap C| + |A \cap B \cap C|$$

$$= 0.40 + 0.30 + 0.20 - 0.10 - 0.08 - 0.05 + 0.02 = 0.69$$

**69% of users** did at least one action.

*What they're testing:* Whether you know the three-set formula and can apply it cleanly. This exact structure appears in analytics, funnel analysis, and feature overlap problems constantly.

---

**Q3. How many subsets of {1, 2, 3, ..., n} contain no two consecutive integers?**

**Answer:**
Let $f(n)$ = number of valid subsets of $\{1, \ldots, n\}$.

Consider element $n$:
- If $n$ is **not** in the subset: any valid subset of $\{1,\ldots,n-1\}$ works → $f(n-1)$ options
- If $n$ **is** in the subset: then $n-1$ cannot be in it → any valid subset of $\{1,\ldots,n-2\}$ works → $f(n-2)$ options

So: $f(n) = f(n-1) + f(n-2)$

Base cases: $f(0) = 1$ (empty set), $f(1) = 2$ ($\emptyset$ and $\{1\}$)

This is the **Fibonacci sequence**. The answer is $F_{n+2}$ (the $(n+2)$th Fibonacci number).

For $n = 4$: valid subsets are $\emptyset, \{1\}, \{2\}, \{3\}, \{4\}, \{1,3\}, \{1,4\}, \{2,4\}, \{1,3\}$ — wait, let me count: $F_6 = 8$. ✓

*What they're testing:* Whether you recognize a set-counting problem hiding a Fibonacci recurrence. The transition from combinatorics to DP.

---

**Q4. $P(A) = 0.4$, $P(B) = 0.3$, $P(A \cup B) = 0.6$. Find $P(A \cap B)$. Are A and B independent?**

**Answer:**

From inclusion-exclusion:
$$P(A \cap B) = P(A) + P(B) - P(A \cup B) = 0.4 + 0.3 - 0.6 = 0.1$$

For independence, we need $P(A \cap B) = P(A) \cdot P(B) = 0.4 \times 0.3 = 0.12$

But $P(A \cap B) = 0.1 \neq 0.12$, so **A and B are not independent** — they are slightly negatively correlated (co-occurrence is less likely than if they were independent).

*What they're testing:* Direct application of inclusion-exclusion plus independence check in one clean problem.

---

---

# PART 2 — COUNTING & RULE OF PRODUCT

---

**Q5. How many integers from 1 to 1000 are divisible by 3 or 7?**

**Answer:**

Let $A$ = divisible by 3, $B$ = divisible by 7.

$$|A| = \lfloor 1000/3 \rfloor = 333$$
$$|B| = \lfloor 1000/7 \rfloor = 142$$
$$|A \cap B| = \lfloor 1000/21 \rfloor = 47 \quad \text{(divisible by lcm(3,7) = 21)}$$

$$|A \cup B| = 333 + 142 - 47 = \mathbf{428}$$

*What they're testing:* Inclusion-exclusion on number theory. Quants love this — it's fast, clean, and tests whether you know to use LCM for the intersection.

---

**Q6. How many ways can you arrange the letters of MISSISSIPPI?**

**Answer:**

Letters: M×1, I×4, S×4, P×2. Total = 11 letters.

$$\frac{11!}{1! \cdot 4! \cdot 4! \cdot 2!} = \frac{39{,}916{,}800}{1 \cdot 24 \cdot 24 \cdot 2} = \frac{39{,}916{,}800}{1{,}152} = \mathbf{34{,}650}$$

*What they're testing:* Repeated-elements permutation formula. Speed and accuracy.

---

**Q7. How many distinct paths exist from top-left to bottom-right of a 4×4 grid, moving only right or down?**

**Answer:**

To go from $(0,0)$ to $(3,3)$ you must make exactly 3 moves right and 3 moves down — 6 total moves.

The number of paths = the number of ways to choose which 3 of the 6 moves are "right":

$$\binom{6}{3} = \frac{6!}{3! \cdot 3!} = \mathbf{20}$$

In general for an $m \times n$ grid: $\binom{m+n-2}{m-1}$.

*What they're testing:* Whether you see this as a combinations problem (choose which steps are right) rather than grinding out a DP table. Both are valid but the combinatorial insight is faster and shows deeper understanding.

---

**Q8. How many 5-digit numbers have strictly increasing digits (using digits 1–9)?**

**Answer:**

Key insight: if the digits are strictly increasing, then the set of digits completely determines the number — there is only one way to arrange them (in increasing order).

So the question becomes: how many ways to choose 5 digits from {1, 2, ..., 9}?

$$\binom{9}{5} = \mathbf{126}$$

*What they're testing:* Structural insight. The candidate who reaches for $P(9,5)$ is thinking mechanically. The candidate who sees "strictly increasing = just choose the set" is thinking.

---

**Q9. A restaurant menu has 4 starters, 6 mains, 3 desserts. But two specific (starter, main) combinations are banned. How many valid meals (one from each)?**

**Answer:**

Without constraint: $4 \times 6 \times 3 = 72$

Each banned (starter, main) pair can be paired with any of 3 desserts → removes $2 \times 3 = 6$ meals.

$$72 - 6 = \mathbf{66} \text{ valid meals}$$

*What they're testing:* Rule of product with subtraction (complementary counting). Clean, fast, practical.

---

---

# PART 3 — PERMUTATIONS & COMBINATIONS

---

**Q10. 52-card deck, deal 2 cards. What is the probability both are aces?**

**Answer:**

$$P(\text{both aces}) = \frac{4}{52} \times \frac{3}{51} = \frac{12}{2652} = \frac{1}{221} \approx 0.0045$$

Or equivalently: $\dfrac{\binom{4}{2}}{\binom{52}{2}} = \dfrac{6}{1326} = \dfrac{1}{221}$ ✓

*What they're testing:* Whether you can compute a conditional/sequential probability quickly. Quants expect this in under 30 seconds mentally.

---

**Q11. You deal a 5-card poker hand. What is the probability of exactly one pair?**

**Answer:**

**Total hands:** $\binom{52}{5} = 2{,}598{,}960$

**One-pair hands** (two cards of one rank, three cards of three other distinct ranks):

| Step | Choose | Count |
|---|---|---|
| Rank for the pair | from 13 | $\binom{13}{1} = 13$ |
| 2 suits for the pair | from 4 | $\binom{4}{2} = 6$ |
| 3 other ranks (all different) | from remaining 12 | $\binom{12}{3} = 220$ |
| 1 suit for each kicker | each from 4 | $4^3 = 64$ |

$$\text{One-pair hands} = 13 \times 6 \times 220 \times 64 = 1{,}098{,}240$$

$$P(\text{one pair}) = \frac{1{,}098{,}240}{2{,}598{,}960} \approx \mathbf{0.423}$$

Greater than 40% — the answer to the MIT 18.05 concept question.

*What they're testing:* Multi-stage combination setup. Whether you correctly prevent double-counting (the 3 kicker ranks must all be different from the pair rank and from each other).

---

**Q12. From 6 men and 4 women, form a committee of 4 with at least 2 women.**

**Answer:**

Use cases (complement would require computing "0 women + 1 woman" — similar effort):

| Case | Count |
|---|---|
| 2W + 2M | $\binom{4}{2}\binom{6}{2} = 6 \times 15 = 90$ |
| 3W + 1M | $\binom{4}{3}\binom{6}{1} = 4 \times 6 = 24$ |
| 4W + 0M | $\binom{4}{4} = 1$ |
| **Total** | **115** |

$$P(\text{at least 2 women}) = \frac{115}{\binom{10}{4}} = \frac{115}{210} \approx 0.548$$

*What they're testing:* Case-splitting on "at least" problems. Recognizing when complement is not obviously simpler.

---

**Q13. A password is 8 characters: first 3 are distinct letters (A–Z), last 5 are digits (0–9) with repetition allowed. How many valid passwords?**

**Answer:**

- First 3 letters, no repeats, order matters → ${}_{26}P_3 = 26 \times 25 \times 24 = 15{,}600$
- Last 5 digits, repeats allowed → $10^5 = 100{,}000$

$$\text{Total} = 15{,}600 \times 100{,}000 = \mathbf{1{,}560{,}000{,}000}$$

*What they're testing:* Recognizing that one part of the problem is a permutation (ordered, no repeats) and another is rule of product (independent slots, repeats OK). Many candidates apply the same formula to both parts — wrong.

---

**Q14. 8 people sit around a circular table. Two specific people (Alice and Bob) must not sit adjacent. How many valid arrangements?**

**Answer:**

**Total circular arrangements:** $(8-1)! = 7! = 5{,}040$

**Arrangements where Alice and Bob ARE adjacent:**
Treat Alice+Bob as one unit → 7 units around a table → $(7-1)! = 6! = 720$ circular arrangements.
Alice and Bob can swap within the unit → multiply by 2.
So: $720 \times 2 = 1{,}440$

**Answer:**
$$5{,}040 - 1{,}440 = \mathbf{3{,}600}$$

*What they're testing:* Circular permutations + complementary counting. Two concepts in one problem.

---

**Q15. 10 servers, each fails independently with probability 0.01. What is the probability at least one fails?**

**Answer:**

Complement: probability that **no** server fails:

$$P(\text{none fail}) = (0.99)^{10} \approx 0.9044$$

$$P(\text{at least one fails}) = 1 - (0.99)^{10} \approx \mathbf{0.0956} \approx 9.6\%$$

*What they're testing:* Instinctive use of the complement for "at least one" problems. Also tests independence — you can only multiply the probabilities because failures are independent.

---

**Q16. Distribute 10 identical balls into 4 distinct boxes with each box having at least 1 ball.**

**Answer:**

Stars and bars with minimum constraint.

Let $x_i$ = balls in box $i$. We need: $x_1 + x_2 + x_3 + x_4 = 10$, each $x_i \geq 1$.

Substitute $y_i = x_i - 1$ (each $y_i \geq 0$):

$$y_1 + y_2 + y_3 + y_4 = 6$$

Stars and bars: $\binom{6 + 4 - 1}{4 - 1} = \binom{9}{3} = \mathbf{84}$

*What they're testing:* Stars and bars + the minimum-constraint substitution trick. If you forget the substitution, you get the wrong answer.

---

**Q17. How many binary strings of length N have no two consecutive 1s?**

**Answer:**

Let $f(N)$ = valid strings of length $N$.

- If the string ends in **0**: any valid string of length $N-1$ can precede it → $f(N-1)$
- If the string ends in **1**: the previous character must be 0 (otherwise two 1s are adjacent) → any valid string of length $N-2$ can precede "01" → $f(N-2)$

$$f(N) = f(N-1) + f(N-2)$$

Base cases: $f(1) = 2$ (strings: "0", "1"), $f(2) = 3$ (strings: "00", "01", "10" — "11" invalid)

This is Fibonacci. $f(N) = F_{N+2}$ where $F_1 = 1, F_2 = 1, \ldots$

| N | f(N) |
|---|---|
| 1 | 2 |
| 2 | 3 |
| 3 | 5 |
| 4 | 8 |
| 5 | 13 |

*What they're testing:* Whether you can translate a combinatorial constraint into a recurrence. This is the bridge between counting and dynamic programming.

---

---

# PART 4 — PROBABILITY vs. STATISTICS

---

**Q18. You train a classifier that outputs P(spam | email). Which part is probability and which is statistics?**

**Answer:**

- **Training the model** = statistics. You observe labeled emails (data) and infer the model's parameters — weights, thresholds — that best explain the data. You're going from data → model.

- **Running the model on a new email** = probability. The model is now fixed (parameters are known). You're computing the probability of an outcome (spam) given a known mechanism. You're going from model → prediction.

The same system uses both — statistics to build it, probability to use it.

*What they're testing:* Whether you understand the forward/backward distinction in a real ML context, not just as an abstract definition.

---

**Q19. What is the difference between MLE and MAP?**

**Answer:**

**MLE (Maximum Likelihood Estimation):**
Find $\theta$ that maximizes $P(\text{data} \mid \theta)$ — the probability of seeing the observed data given parameter $\theta$.
$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta P(\text{data} \mid \theta)$$
Purely data-driven. No prior beliefs.

**MAP (Maximum A Posteriori):**
Find $\theta$ that maximizes $P(\theta \mid \text{data}) \propto P(\text{data} \mid \theta) \cdot P(\theta)$.
$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta \left[ P(\text{data} \mid \theta) \cdot P(\theta) \right]$$
Data + prior beliefs. MAP = MLE + regularization from the prior.

**The ML connection:**
- L2 regularization (Ridge) = MAP with a **Gaussian prior** on weights
- L1 regularization (LASSO) = MAP with a **Laplace prior** on weights

With infinite data, the prior is overwhelmed and MAP → MLE.

*What they're testing:* Whether you understand that regularization has a probabilistic interpretation, not just a computational one. This separates candidates who've used these tools from those who understand them.

---

**Q20. Your model has 95% accuracy on test data. Can you say there is a 95% probability the next prediction is correct?**

**Answer:**

Not exactly — and the distinction matters.

**95% accuracy** is a frequentist statistic over the test set — it means the model was correct on 95% of those specific examples. It says nothing about any individual prediction.

For any specific new input, the probability of a correct prediction depends on:
- How similar that input is to the training distribution
- The model's **calibration** — whether its output probabilities actually match empirical frequencies
- Whether the test set was representative

A well-calibrated model that outputs $P(\text{correct}) = 0.95$ for a given input *is* making a direct probability claim. But raw accuracy across a test set is not the same statement.

*What they're testing:* Whether you conflate aggregate statistics with individual-level probabilities — a very common mistake in practice.

---

---

# PART 5 — FREQUENTIST vs. BAYESIAN

---

**Q21. What is a confidence interval? Most people get this wrong — what does it actually mean?**

**Answer:**

A 95% confidence interval is a statement about the **procedure**, not about this specific interval.

**What it means:** If we repeated the experiment many times and computed a 95% CI each time, 95% of those intervals would contain the true parameter.

**What it does NOT mean:** "There is a 95% probability the true parameter is in this interval." — In the frequentist framework, the parameter is fixed. This specific interval either contains it or doesn't. There's no probability about it.

The confusion arises because we *want* to make a direct probability statement about the parameter. The Bayesian credible interval actually does this: $P(\theta \in [a,b] \mid \text{data}) = 0.95$.

*What they're testing:* This is the single most commonly misinterpreted concept in statistics. Interviewers ask it specifically because most people (including practitioners) state the wrong interpretation. Getting this right immediately signals statistical maturity.

---

**Q22. Flip a coin 10 times, get 7 heads. The coin came from a factory that produces two types: P(heads) = 0.5 or P(heads) = 0.7, with equal probability. What is the posterior probability the coin has P(heads) = 0.7?**

**Answer:**

Let $H_{0.5}$ = "coin is fair", $H_{0.7}$ = "coin is biased".

Prior: $P(H_{0.5}) = P(H_{0.7}) = 0.5$

Likelihoods of observing 7 heads in 10 flips:

$$P(\text{data} \mid H_{0.5}) = \binom{10}{7}(0.5)^{10} = 120 \times \frac{1}{1024} = \frac{120}{1024}$$

$$P(\text{data} \mid H_{0.7}) = \binom{10}{7}(0.7)^7(0.3)^3 = 120 \times 0.0823 \times 0.027 = 120 \times 0.002224 \approx \frac{266.9}{1024}$$

(Using $120 \times 0.002224 \approx 0.2668$, and $120/1024 \approx 0.1172$)

By Bayes' theorem:

$$P(H_{0.7} \mid \text{data}) = \frac{P(\text{data} \mid H_{0.7}) \cdot P(H_{0.7})}{P(\text{data} \mid H_{0.5}) \cdot P(H_{0.5}) + P(\text{data} \mid H_{0.7}) \cdot P(H_{0.7})}$$

$$= \frac{0.2668 \times 0.5}{0.1172 \times 0.5 + 0.2668 \times 0.5} = \frac{0.2668}{0.1172 + 0.2668} = \frac{0.2668}{0.384} \approx \mathbf{0.695}$$

So after seeing 7 heads in 10 flips, the probability the coin is biased goes from the prior of 50% to a posterior of about **69.5%**.

*What they're testing:* Full Bayes calculation. Prior → likelihood → posterior. The ability to set this up and execute it cleanly under pressure.

---

**Q23. Your A/B test ran for 2 weeks. The p-value is 0.04. The business wants to ship. What do you recommend?**

**Answer:**

This is a judgment question, not a calculation question. A strong answer covers several layers:

**What p = 0.04 actually means:**
The probability of observing a result this extreme (or more extreme) *if the null hypothesis were true* (i.e., if the change had no effect). It is **not** the probability that the change is ineffective.

**Why I wouldn't immediately ship:**
- Was this a pre-registered hypothesis or did we check results during the experiment? If we peeked and stopped when p < 0.05, we have a multiple testing problem — the true false positive rate is higher than 5%.
- Is the effect size practically significant, or just statistically significant? A 0.01% improvement may be real but not worth the engineering cost.
- Did we test on a representative sample? Two weeks may miss seasonality.

**What I'd additionally consider:**
- Run for the pre-planned duration regardless
- Report the effect size and confidence interval, not just the p-value
- Consider a Bayesian approach: what is the posterior probability the change is positive? What is the expected lift?

**Conclusion:** p = 0.04 is evidence in favor of shipping, but it needs context. I'd want to see the effect size, confirm no early stopping, and ideally see a Bayesian posterior before recommending.

*What they're testing:* Statistical maturity. The ability to not just compute but interpret and apply statistical results to real decisions — recognizing limitations, not just reading a threshold.

---

**Q24. With infinite data, do frequentist and Bayesian methods give the same answer?**

**Answer:**

Yes — this is formalized as the **Bernstein–von Mises theorem**.

As sample size $n \to \infty$, the posterior distribution concentrates around the MLE regardless of the prior:

$$\theta \mid \text{data} \xrightarrow{d} \mathcal{N}\left(\hat{\theta}_{\text{MLE}},\; \frac{1}{n} I(\theta)^{-1}\right)$$

where $I(\theta)$ is the Fisher information.

The prior gets "washed out" by the data. Bayesian credible intervals and frequentist confidence intervals become numerically equivalent.

**Practical implication:** The two schools disagree most when data is scarce. With big data, use whichever is more computationally convenient. With small data, the Bayesian prior is doing real work and the choice of method matters.

*What they're testing:* Depth. Most candidates know the philosophical difference. Fewer know the asymptotic equivalence.

---

---

# PART 6 — THE ONES THAT LOOK SIMPLE BUT AREN'T

These are the questions that filter the strongest candidates.

---

**Q25. Expected number of fixed points in a random permutation of N elements.**

**Answer:**

A fixed point is an element that maps to itself: $\pi(i) = i$.

Define indicator variables: $X_i = 1$ if element $i$ is a fixed point, 0 otherwise.

$$P(X_i = 1) = \frac{1}{N} \quad \text{(by symmetry — each element is equally likely to be in any position)}$$

By **linearity of expectation**:

$$E\left[\sum_{i=1}^N X_i\right] = \sum_{i=1}^N E[X_i] = \sum_{i=1}^N \frac{1}{N} = \mathbf{1}$$

The expected number of fixed points is **always 1**, for any $N$. This is one of the most elegant results in combinatorics.

*What they're testing:* Linearity of expectation — the most powerful tool in probabilistic combinatorics. You don't need to know the distribution of the total count. You just add up the individual expectations.

---

**Q26. Non-transitive dice: Die A beats Die B 60% of the time. Die B beats Die C 60% of the time. What is the probability Die A beats Die C?**

**Answer:**

This is a **trick question** — you cannot determine the answer from the given information.

The relationship "beats" is not transitive. You might expect $P(A > C) > 0.5$, but it could be anything including less than 0.5. The MIT 18.05 notes show a concrete example: Blue beats White, White beats Orange, Orange beats Blue — a cycle.

The probability depends on the specific values on the dice faces, not just the pairwise win probabilities.

*What they're testing:* Whether you blindly apply transitivity or recognize the trap. The correct answer is "it cannot be determined."

---

**Q27. You have a fair coin. You flip until you get heads. What is the expected number of flips?**

**Answer:**

Let $X$ = number of flips. This is a **Geometric distribution** with $p = 0.5$.

$$P(X = k) = (0.5)^{k-1} \times 0.5 = (0.5)^k$$

$$E[X] = \frac{1}{p} = \frac{1}{0.5} = \mathbf{2}$$

Intuition: on average, 2 flips to get the first head.

*What they're testing:* Recognition of the geometric distribution and its expectation formula $E[X] = 1/p$.

---

**Q28. In a room of 23 people, what is the probability that at least two share a birthday?**

**Answer:**

Use the complement: probability all 23 birthdays are distinct.

$$P(\text{all distinct}) = \frac{365}{365} \times \frac{364}{365} \times \frac{363}{365} \times \cdots \times \frac{343}{365} = \prod_{k=0}^{22} \frac{365-k}{365}$$

$$P(\text{all distinct}) \approx 0.4927$$

$$P(\text{at least one shared}) = 1 - 0.4927 \approx \mathbf{0.5073}$$

With just 23 people, there's a greater than 50% chance two share a birthday.

**Why so few?** With 23 people there are $\binom{23}{2} = 253$ pairs — each with a $1/365 \approx 0.27\%$ chance of matching. The cumulative probability across 253 pairs crosses 50% quickly.

*What they're testing:* Complement technique, and the birthday problem intuition (which is a standard quant/FAANG question about surprising probability thresholds).

---

---

# The Pattern: What Every Question Is Really Asking

| Surface question | Deeper thing being tested |
|---|---|
| Mutually exclusive ≠ independent | Conceptual precision, not just definitions |
| Confidence interval meaning | Statistical maturity — most people get this wrong |
| MLE vs MAP | Whether you connect theory to practice (regularization) |
| "At least one" problems | Instinct to use complement |
| Counting questions | Setup skill — the formula is trivial once you see the structure |
| Fibonacci in counting | Recognizing recurrence structure |
| Non-transitive dice | Whether you blindly apply intuition or question it |
| Linearity of expectation | The most powerful probabilistic tool, used constantly |

---

*These questions are not hard because the math is hard. They are hard because the setup requires understanding, not pattern-matching.*
