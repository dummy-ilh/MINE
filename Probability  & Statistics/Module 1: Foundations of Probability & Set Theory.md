# Module 1: Foundations of Probability & Set Theory
---

## Table of Contents

1. [Probability vs. Statistics](#1-probability-vs-statistics)
2. [Frequentist vs. Bayesian Interpretations](#2-frequentist-vs-bayesian-interpretations)
3. [Set Theory Foundations](#3-set-theory-foundations)
4. [Sample Spaces, Events & Kolmogorov Axioms](#4-sample-spaces-events--kolmogorov-axioms)
5. [Counting Techniques](#5-counting-techniques)
6. [Conditional Probability & Independence](#6-conditional-probability--independence)
7. [Law of Total Probability](#7-law-of-total-probability)
8. [Bayes' Theorem](#8-bayes-theorem)
9. [Common Interview Traps & Pitfalls](#9-common-interview-traps--pitfalls)
10. [Worked Interview Problems](#10-worked-interview-problems)
11. [Cheat Sheet](#11-cheat-sheet)

---

## 1. Probability vs. Statistics

Probability and statistics are deeply connected — all statistical statements are, at bottom, statements about probability — but they run in **opposite directions**.

Probability is about predicting the future.
Statistics is about guessing the past.

**Probability example:**You know the rules (e.g., a coin is 50/50).
You want to know the odds of a specific result (e.g., what are the chances of getting 60 heads in 100 tosses?).
It is exact: there is only one right answer.

**Statistics example:** You have a coin of unknown fairness. You toss it 100 times and observe 60 heads.
You want to figure out the rules (e.g., is this coin fair or rigged?).
It is messy: different smart people might look at the same data and come to different conclusions. the same data — this is why statistics involves judgment, not just calculation.

**Where this shows up in ML (interview framing):** Training a classifier (fitting parameters from labeled data) is statistics — data → model. Running the trained classifier on a new input (computing P(spam | email) from fixed, known parameters) is probability — model → prediction. The same system uses both.

> **Trap:** 95% test-set accuracy is a frequentist statistic about the test set — it does not mean "95% probability this specific next prediction is correct." That claim requires the model to be well-*calibrated* (its output probabilities matching empirical frequencies), which is a separate property from raw accuracy.

---


### The One-Line Difference

| | You know... | You're finding... |
|---|---|---|
| **Probability** | The rules of the game | What will likely happen |
| **Statistics** | What happened | The rules of the game |

They are **mirror images** of each other.

---

### Probability: Working Forward

You have a fair coin. You know it's fair ($p = 0.5$). You flip it 100 times.

> *"What is the probability of getting 60 or more heads?"*

The process is known. You're computing what outcomes look like **coming out** of it.

$$P(\text{event}) = \; ?$$

---

### Statistics: Working Backward

You have a coin of unknown origin. You flip it 100 times and get 60 heads.

> *"Is this coin fair? What is the true probability of heads?"*

The outcome is known. You're inferring what process **produced** it.

$$\theta = \; ?$$

---

### A Simple Analogy

Imagine a bag with red and blue marbles.

- **Probability:** *"The bag has 30% red marbles. If I draw 10, how many reds do I expect?"*
- **Statistics:** *"I drew 10 marbles and got 7 red. How many red marbles are in the bag?"*

Same bag. Same marbles. Opposite direction of reasoning.

---

### Why This Matters

| | Probability | Statistics |
|---|---|---|
| What's known | The model & its parameters | The data |
| What's unknown | The outcome | The model / parameters |
| Reasoning style | Deductive ("given the rules, what happens?") | Inductive ("given what happened, what are the rules?") |
| Output | An exact probability | An estimate + some uncertainty |
| Real example | Insurance pricing | Clinical trial analysis |
| ML example | Computing $P(\text{class} \mid \text{input})$ at prediction time | Training a model (fitting parameters to data) |

---

### The Key Insight

> **Statistics is built on probability.** Every statistical method — hypothesis tests, confidence intervals, regression — uses probability under the hood. Probability is the engine. Statistics is what you build with it.

---

## 2. Frequentist vs. Bayesian Interpretations

## The Core Debate: What does "probability" actually mean?


### Frequentist: Probability = Repeatable Frequency

- **Definition:** Probability is just what happens in the long run. Flip a coin a million times, and heads will be about half. That's it.
- **Parameters (like the true chance of heads):** Fixed numbers. Not random. They exist, we just don't know them.
- **What they do:** Collect data, give a "best guess" (point estimate), and build a "confidence interval" (a range that works 95% of the time *across many repeated experiments*).
- **Key rule:** You **cannot** say "there's a 95% chance the true value is in this range." The true value is fixed—it's either in or out.

---

### Bayesian: Probability = Personal Belief

- **Definition:** Probability is how sure you are about something, given what you know right now.
- **Parameters:** Treated as random—not because they change, but because *your uncertainty* about them changes.
- **What they do:** 
  1. Start with a **prior** (your guess before seeing data).
  2. Collect data.
  3. Use math to update to a **posterior** (your new, improved guess).
- **Key rule:** You **can** say "there's a 95% chance the true value is in this range." That's called a "credible interval."



### The Coin Flip Example (10 flips, 8 heads)

| **Frequentist** | **Bayesian** |
|----------------|--------------|
| Your best guess: 80% heads. | Start with a prior: "I think it's probably fair (around 50%)." |
| Build a confidence interval around 80%. | Update after seeing 8 heads. Now you have a full curve showing what you believe about the true chance. |
| **Cannot** say "80% is likely correct." | **Can** say "there's a high chance the true chance is above 70%." |





### Where each is used

| **Frequentist** | **Bayesian** |
|----------------|--------------|
| Clinical trials | Machine Learning |
| Regulatory stats | A/B testing |
| Social sciences | Recommendation systems |
| | NLP (language models) |

---

### Machine Learning Connection (Interview Gold)

Many ML techniques are secretly Bayesian:

| **ML Technique** | **Bayesian Equivalent** |
|------------------|--------------------------|
| L2 regularization | Gaussian prior on weights |
| L1 regularization (LASSO) | Laplace prior on weights |
| MLE (Maximum Likelihood) | Frequentist approach |
| MAP (Maximum A Posteriori) | Bayesian with a prior |

---

### One-Sentence Summary

- **Frequentist:** Probability = repeated experiments. Parameters are fixed truths. You estimate them, but don't assign probabilities to them.
- **Bayesian:** Probability = how sure you are. Parameters have distributions because *you* are uncertain. You start with a guess, update with data, and get a full picture of what you believe.

---

### Interview Cheat Sheet

> **Q: What's the difference?**
> 
> Frequentists see probability as long-run frequency and parameters as fixed. Bayesians see probability as belief and parameters as random variables. The key difference: confidence intervals are about *the procedure*; credible intervals are about *the parameter*.

> **Q: What's a confidence interval?**
> 
> It's a range that would contain the true value in 95% of repeated experiments. It is **not** a 95% chance for this specific range.

> **Q: What's a prior?**
> 
> Your belief before seeing data. It gets updated with data to become the posterior.

> **Q: MLE vs MAP?**
> 
> MLE = best guess from data alone. MAP = best guess from data + your prior belief (it's MLE with a regularizer).



Two schools of thought, rooted in different interpretations of what probability *means*.

### Frequentist View
Probability measures the **long-run frequency** of outcomes in a repeated experiment. A fair coin having 50% probability of heads means: over many tosses, about half land heads. Probability is a property of the physical world, not of your mind. Historically dominant in biology, medicine, and the social sciences.

### Bayesian View
Probability is an abstract concept measuring a **state of knowledge or degree of belief**. A Bayesian doesn't assign one value for "the probability this coin is fair" — they maintain a distribution over possible values, updated as data arrives. Especially useful for incorporating new data into an existing model. Has surged with modern computing and big data (machine learning).

Neither is universally "correct" — modern statistics uses both in complementary ways.

### What Confidence Intervals Actually Mean (a common interview trap)

A 95% confidence interval is a statement about the **procedure**, not about any one interval.

- **What it means:** if you repeated the experiment many times and built a 95% CI each time, 95% of those intervals would contain the true parameter.
- **What it does NOT mean:** "there's a 95% probability the true parameter is in *this* interval." In the frequentist framework the parameter is fixed — this specific interval either contains it or it doesn't.
- The Bayesian **credible interval** *does* make the direct probability statement: P(θ ∈ [a,b] | data) = 0.95.

Getting this distinction right is one of the fastest ways to signal statistical maturity in an interview.

### Do the Two Schools Converge?

Yes — with infinite data, per the **Bernstein–von Mises theorem**, the posterior distribution concentrates around the MLE regardless of the prior:

$$\theta \mid \text{data} \xrightarrow{d} \mathcal{N}\left(\hat{\theta}_{\text{MLE}},\; \tfrac{1}{n} I(\theta)^{-1}\right)$$

The prior gets washed out by data, and Bayesian credible intervals become numerically equivalent to frequentist confidence intervals. **Practical implication:** the two schools disagree most when data is scarce; with big data, use whichever is computationally convenient.

### MLE vs. MAP

| | Objective | Notes |
|---|---|---|
| **MLE** | $\arg\max_\theta P(\text{data}\mid\theta)$ | Purely data-driven, no prior |
| **MAP** | $\arg\max_\theta P(\text{data}\mid\theta)\,P(\theta)$ | Data + prior; MAP = MLE + regularization from the prior |

Connection to ML: **L2 regularization (Ridge)** = MAP with a Gaussian prior on weights; **L1 (LASSO)** = MAP with a Laplace prior. With infinite data, the prior is overwhelmed and MAP → MLE.



### The Confidence Interval vs. Credible Interval Trap

This trips up almost everyone in interviews.

**Frequentist 95% Confidence Interval** — say, $[0.61, 0.99]$

What it means:
> "If I repeated this experiment many times and built an interval each time, 95% of those intervals would contain the true $p$."

What it does **NOT** mean:
> ~~"There is a 95% chance $p$ is in this interval."~~ ← This is wrong in the frequentist world. $p$ is fixed. It's either in the interval or it isn't.

**Bayesian 95% Credible Interval** — say, $[0.57, 0.95]$

What it means:
> "Given the data I observed, there is a 95% probability that $p$ lies in this interval."

This is the statement people *think* a confidence interval makes. Bayesian inference gives you that directly.

---

### The Full Comparison

| | Frequentist | Bayesian |
|---|---|---|
| **Probability is...** | Long-run frequency | Degree of belief |
| **Parameters are...** | Fixed unknowns | Random variables |
| **Prior beliefs** | Not allowed | Required (and explicit) |
| **What you get** | Point estimate + confidence interval | Full distribution over parameter |
| **Interval interpretation** | 95% of intervals contain true value | 95% probability parameter is in interval |
| **Subjectivity** | Objective — no assumptions beyond data | Prior introduces subjectivity (can be a feature) |
| **Strength** | No need to pick priors; widely accepted | Incorporates prior knowledge; handles small data well |
| **Weakness** | Confidence intervals are unintuitive; can't make probability claims about parameters | Choosing priors can be controversial |

---






### The One-Paragraph Summary

**Frequentist:** The world has fixed truths. Probability is about what happens if you repeat experiments forever. Parameters are unknown constants — we estimate them, build intervals around them, and test hypotheses about them. We do not assign probabilities to parameters.

**Bayesian:** Probability is about your state of knowledge. You start with a belief (prior), update it with data, and end with a refined belief (posterior). Parameters can have probability distributions — because you're uncertain about them, not because they're random. This gives you richer, more interpretable outputs.

---

---

## 3. Set Theory Foundations

Set theory is the language of probability. Every event is a set; every probability rule is ultimately a statement about set operations.

### 3.1 What Is a Set

A set is a well-defined collection of objects (**elements**). A set is defined only by its elements — **order doesn't matter, repetition doesn't matter**:

$$\{1,2,3\} = \{3,2,1\} = \{1,1,2,3\}$$

This is what distinguishes sets from *sequences* (order matters) and *multisets* (repetition matters) — relevant later when distinguishing permutations from combinations.

### 3.2 Core Set Operations

Running example — animals:

$$S = \{\text{Antelope, Bee, Cat, Dog, Elephant, Frog, Gnat, Hyena, Iguana, Jaguar}\}$$
$$M = \text{mammals} = \{\text{Antelope, Cat, Dog, Elephant, Hyena, Jaguar}\}$$
$$W = \text{wild animals} = \{\text{Antelope, Bee, Elephant, Frog, Gnat, Hyena, Iguana, Jaguar}\}$$

| Operation | Notation | Meaning | Logic | On the example |
|---|---|---|---|---|
| Subset | $A \subseteq S$ (or strict $A \subset S$) | Every element of $A$ is in $S$ | — | $M \subset S$ |
| Complement | $A^c = S - A$ | Elements of $S$ not in $A$ (always relative to a universe $S$) | NOT | $M^c = \{\text{Bee, Frog, Gnat, Iguana}\}$ |
| Union | $A \cup B$ | Elements in $A$ or $B$ (or both), no double-counting | OR | $M \cup W = S$ (every animal is a mammal or wild, or both) |
| Intersection | $A \cap B$ | Elements common to both | AND | $M \cap W = \{\text{Antelope, Elephant, Hyena, Jaguar}\}$ |
| Difference | $A - B$ | In $A$ but not $B$ | AND NOT | $M - W = \{\text{Cat, Dog}\}$ (domesticated mammals) |
| Empty set | $\emptyset$ | No elements; a subset of every set | — | — |
| Disjoint | $A \cap B = \emptyset$ | No shared elements | mutually exclusive | even vs. odd numbers |

**Key identities:**
$$(A^c)^c = A \qquad A - B = A \cap B^c \qquad A \cup \emptyset = A \qquad A \cap \emptyset = \emptyset$$
$$A \cup A^c = S \qquad A \cap A^c = \emptyset \quad \text{(law of excluded middle)}$$

**Disjoint ≠ independent** (important distinction, expanded in §6.4): disjoint events *cannot* co-occur; independent events simply don't *influence* each other. If $A,B$ are disjoint and $P(A)>0$, they are actually **dependent** — knowing $A$ occurred tells you $B$ definitely didn't.

### 3.3 Cardinality

If $S$ is finite, $|S|$ denotes its number of elements: $|S|=10,\ |M|=6,\ |W|=8,\ |M\cap W|=4$ in the example above. This is the bridge to probability:

$$P(\text{event}) = \frac{|\text{favorable outcomes}|}{|\text{total outcomes}|}$$

Computing probabilities reduces to counting — the reason Module 1 spends so much time on counting techniques (§5).

### 3.4 Cartesian Products

$$S \times T = \{(s,t) \mid s \in S,\ t \in T\}, \qquad |S \times T| = |S|\cdot|T|$$

Example: $\{1,2,3\}\times\{1,2,3,4\}$ has $3\times 4 = 12$ ordered pairs. This is the set-theoretic foundation of the **Rule of Product** (§5.2) — e.g., the sample space of "flip a coin, then roll a die" is $\{H,T\}\times\{1,\dots,6\}$, giving $2\times 6=12$ equally likely outcomes. Fact: if $A\subset S$ and $B\subset T$, then $A\times B \subset S\times T$.

### 3.5 DeMorgan's Laws

$$\boxed{(A \cup B)^c = A^c \cap B^c} \qquad \boxed{(A \cap B)^c = A^c \cup B^c}$$

**Plain English:** "not (A or B)" = "(not A) and (not B)"; "not (A and B)" = "(not A) or (not B)".

**Element-chasing proof of Law 1:** let $x \in (A\cup B)^c$. Then $x\notin A\cup B \Rightarrow x\notin A$ and $x\notin B \Rightarrow x\in A^c$ and $x \in A^c \cap B^c$. The argument reverses exactly, giving both directions. $\square$

**Numeric check** ($S=\{1..5\}$, $A=\{1,2,3\}$, $B=\{3,4\}$): $(A\cup B)^c=\{5\}$ and $A^c\cap B^c=\{4,5\}\cap\{1,2,5\}=\{5\}$ ✓. Similarly $(A\cap B)^c=\{1,2,4,5\}=A^c\cup B^c$ ✓.

### 3.6 Algebra of Sets

Parallels Boolean algebra; the **distributive laws** are most used in probability proofs:

$$A \cap (B \cup C) = (A \cap B) \cup (A \cap C) \qquad A \cup (B \cap C) = (A \cup B) \cap (A \cup C)$$

Also: identity ($A\cup\emptyset=A,\ A\cap S=A$), idempotent ($A\cup A=A$), commutative, associative, and absorption laws ($A\cup(A\cap B)=A$) — all standard Boolean-algebra analogues.

### 3.7 Venn Diagrams

A visual proof system where, in probability, **area = probability mass** and the full rectangle ($S$/$\Omega$) has total area 1.

| Expression | Shaded region |
|---|---|
| $A \cup B$ | Everything inside either circle |
| $A \cap B$ | Only the overlap |
| $A^c$ | Everything outside circle $A$ |
| $A - B$ | $A$'s circle excluding the overlap |
| Disjoint | Two non-touching circles |

### 3.8 Set Theory → Probability Translation

| Set Theory | Probability |
|---|---|
| Universal set $S$ (or $\Omega$) | Sample space — all possible outcomes |
| Element $\omega \in S$ | A single outcome |
| Subset $A \subseteq S$ | An event |
| $A \cup B$ | Event $A$ or $B$ occurs |
| $A \cap B$ | Event $A$ and $B$ both occur |
| $A^c$ | Event $A$ does not occur |
| $A \cap B = \emptyset$ | $A, B$ mutually exclusive |
| $|A|/|S|$ | $P(A)$, under equal likelihood |

The **complement trick** — computing $1-P(A^c)$ instead of $P(A)$ directly — is one of the most useful tools in probability (used repeatedly in §10: birthday problem, "at least one," server failures, etc.).

### 3.9 Advanced: σ-Algebras (Foundation for Measure Theory)

In measure-theoretic probability: sample space $\Omega$, a collection of events $\mathcal{F}$ (a **σ-algebra** of subsets of $\Omega$), and a probability measure $P$ on $(\Omega,\mathcal{F})$. A σ-algebra must be:

1. Closed under complement: $A \in \mathcal{F} \Rightarrow A^c \in \mathcal{F}$
2. Closed under countable union: $A_1, A_2, \ldots \in \mathcal{F} \Rightarrow \bigcup_{i=1}^\infty A_i \in \mathcal{F}$
3. Contain $\Omega$ itself

The union/intersection/complement operations you learn as "set theory" are literally the axioms underlying all of probability theory.

---

## 4. Sample Spaces, Events & Kolmogorov Axioms

- **Experiment**: any process with an uncertain outcome.
- **Sample space ($\Omega$)**: the set of *all* possible outcomes.
- **Event ($A$)**: any subset of $\Omega$.
- **Elementary outcome ($\omega$)**: a single outcome in $\Omega$.

### Kolmogorov's Three Axioms

1. **Non-negativity:** $P(A) \geq 0$
2. **Normalization:** $P(\Omega) = 1$
3. **Countable additivity:** if $A_1, A_2,\ldots$ are pairwise disjoint, $P(A_1\cup A_2\cup\cdots)=P(A_1)+P(A_2)+\cdots$

### Derived Properties

$$P(\emptyset)=0 \qquad P(A^c)=1-P(A) \qquad P(A\cup B)=P(A)+P(B)-P(A\cap B)$$
$$A\subseteq B \Rightarrow P(A)\leq P(B) \qquad 0\leq P(A)\leq 1$$

The inclusion-exclusion identity $P(A\cup B)=P(A)+P(B)-P(A\cap B)$ *is* Axiom 3 in disguise once $A\cap B=\emptyset$ is relaxed — full treatment in §5.3.

---

## 5. Counting Techniques

When all outcomes are equally likely: $P(A) = |A|/|\Omega|$.

### 5.1 The Core Principle

If an experiment has $n$ equally probable outcomes and $k$ are "desirable," $P(\text{desirable}) = k/n$.

**Example:** flip a fair coin 3 times. All 8 outcomes: $\{TTT, TTH, THT, THH, HTT, HTH, HHT, HHH\}$. Exactly one head: $\{TTH, THT, HTT\}$, so $P = 3/8$. (Listing becomes infeasible fast — 10 flips already gives $2^{10}=1024$ outcomes — which motivates the techniques below.)

### 5.2 Rule of Product (Multiplication Rule)

If there are $n$ ways to do action 1 and then $m$ ways to do action 2, there are $n\cdot m$ ways to do both in sequence. **This holds even if the specific options for action 2 depend on action 1's outcome, as long as the *number* of options stays the same.**

- **Shirts × pants:** 3 shirts, 4 pants → 12 outfits.
- **Olympic medals:** 5 competitors, gold/silver/bronze → $5\times4\times3=60$ ways (who wins silver depends on gold, but the *count* of remaining candidates is always 4).
- **When the count itself varies, split into cases** (a multiplication tree) and sum: e.g., a wardrobe where shirt color restricts compatible sweaters requires computing red/black/green branches separately and adding, rather than one multiplication.
- **DNA sequences of length 3** (alphabet size 4): with repetition $4^3=64$; with no repeats $4\times3\times2=24$.
- **Complementary counting:** menu with 4 starters × 6 mains × 3 desserts = 72 combos; if 2 specific (starter, main) pairs are banned, each removes 3 desserts' worth of meals: $72 - 2\times3 = 66$.

### 5.3 Inclusion-Exclusion Principle

**Two events:**
$$\boxed{|A\cup B| = |A|+|B|-|A\cap B|} \qquad \boxed{P(A\cup B)=P(A)+P(B)-P(A\cap B)}$$

Adding $|A|+|B|$ double-counts the overlap, so it's subtracted once. If disjoint, $P(A\cup B)=P(A)+P(B)$ — this is Kolmogorov's Axiom 3.

**Three events:**
$$P(A\cup B\cup C) = P(A)+P(B)+P(C)-P(A\cap B)-P(A\cap C)-P(B\cap C)+P(A\cap B\cap C)$$

(General pattern: add singles, subtract pairs, add triples, subtract quadruples, …)

**Worked examples:**
- *Band problem:* 7 sing, 4 play guitar, 2 do both → band size $=7+4-2=9$ (naively adding to get 11 double-counts the 2 who do both).
- *Divisibility:* integers 1–1000 divisible by 3 or 7: $\lfloor1000/3\rfloor+\lfloor1000/7\rfloor-\lfloor1000/21\rfloor = 333+142-47=428$ (intersection uses lcm(3,7)=21).
- *Three-set marketing example:* 40% opened an email, 30% clicked an ad, 20% visited the site; pairwise overlaps 10%, 8%, 5%; triple overlap 2%. At least one action: $0.40+0.30+0.20-0.10-0.08-0.05+0.02=0.69$, i.e. 69%.
- *User engagement (two-set):* 60% click Search, 45% click Ads, 30% click both → $0.60+0.45-0.30=0.75$, i.e. 75% click at least one.

### 5.4 Permutations

An ordered arrangement — **order matters**. The number of permutations of $k$ distinct elements is $k!$. Number of ordered selections of $k$ from $n$:

$$\boxed{{}_nP_k = \frac{n!}{(n-k)!} = n(n-1)\cdots(n-k+1)}$$

*Example:* permutations of 3 elements out of $\{a,b,c,d\}$: $4\times3\times2=24$.

**Repeated-element permutations** (letters of a word): divide by the factorial of each repeated letter's count. MISSISSIPPI (M×1, I×4, S×4, P×2, 11 letters total):

$$\frac{11!}{1!\,4!\,4!\,2!} = \frac{39{,}916{,}800}{1{,}152} = 34{,}650$$

**Circular permutations** of $n$ people around a table: $(n-1)!$ (rotations counted as identical).

### 5.5 Combinations

An unordered selection — **order doesn't matter**. Number of $k$-subsets of an $n$-set:

$$\boxed{{}_nC_k = \binom{n}{k} = \frac{n!}{k!\,(n-k)!} = \frac{{}_nP_k}{k!}}$$

read "$n$ choose $k$." Each $k$-subset can be internally arranged in $k!$ ways, which is why dividing $ {}_nP_k$ by $k!$ converts permutations to combinations. *Example:* combinations of 3 from $\{a,b,c,d\}$: only 4 (vs. 24 permutations).

**Structural-insight example:** how many 5-digit numbers (digits 1–9) have strictly increasing digits? Since increasing order is the *only* valid arrangement of any chosen digit set, this is just $\binom{9}{5}=126$ — not a permutation problem in disguise.

**Grid paths:** paths from top-left to bottom-right of an $m\times n$ grid (right/down moves only) = choose which steps are "right" out of the total moves: $\binom{m+n-2}{m-1}$. For a 4×4 grid: $\binom{6}{3}=20$.

### 5.6 Stars and Bars (with a minimum constraint)

Distribute $N$ identical items into $k$ distinct bins, each bin ≥ 1: substitute $y_i = x_i - 1 \geq 0$ to reduce to a $\geq 0$ problem, then

$$\binom{(N-k)+k-1}{k-1} = \binom{N-1}{k-1}$$

*Example:* 10 identical balls into 4 distinct boxes, each ≥ 1 ball: $y_1+\cdots+y_4=6 \Rightarrow \binom{9}{3}=84$.

### 5.7 Recurrences Hiding in Counting Problems

Some set-counting problems reduce to a Fibonacci-style recurrence — useful bridge to dynamic programming.

**Subsets of $\{1,\ldots,n\}$ with no two consecutive integers:** let $f(n)$ count them. Either $n$ is excluded ($f(n-1)$ options for the rest) or $n$ is included, forcing $n-1$ excluded ($f(n-2)$ options). So $f(n)=f(n-1)+f(n-2)$, with $f(0)=1,\ f(1)=2$ — the Fibonacci sequence, $f(n) = F_{n+2}$. Check for $n=4$: valid subsets are $\emptyset,\{1\},\{2\},\{3\},\{4\},\{1,3\},\{1,4\},\{2,4\}$ — 8 subsets, matching $F_6=8$.

**Binary strings of length $N$ with no two consecutive 1s:** identical recurrence. $f(N)=f(N-1)+f(N-2)$, $f(1)=2,\ f(2)=3$: gives $2,3,5,8,13,\ldots$ for $N=1..5$.

---

## 6. Conditional Probability & Independence

### 6.1 Conditional Probability

$$P(A\mid B) = \frac{P(A\cap B)}{P(B)}, \quad P(B)>0$$

Read: "probability of $A$ given $B$ has occurred." Intuition: you've restricted the universe from $\Omega$ to $B$; now what fraction of $B$ is also in $A$?

*Example:* draw a card; given it's a face card (12/52), probability it's a King (4/52): $P(K\mid\text{face}) = (4/52)/(12/52)=1/3$.

### 6.2 Multiplication Rule / Chain Rule

$$P(A\cap B) = P(A\mid B)P(B) = P(B\mid A)P(A)$$

Generalizes to:
$$P(A_1\cap\cdots\cap A_n) = P(A_1)\,P(A_2\mid A_1)\,P(A_3\mid A_1,A_2)\cdots P(A_n\mid A_1,\ldots,A_{n-1})$$

*Example (sequential draws without replacement):* bag with 3 red, 2 blue; draw 2. $P(\text{both red}) = \frac{3}{5}\times\frac{2}{4} = 0.30$.

*Example (both aces from a deck):* $P = \frac{4}{52}\times\frac{3}{51} = \frac{1}{221} \approx 0.0045$, equivalently $\binom{4}{2}/\binom{52}{2}$.

### 6.3 Independence

$$A, B \text{ independent} \iff P(A\cap B) = P(A)P(B) \iff P(A\mid B)=P(A)$$

**Pairwise vs. mutual independence:** pairwise independence (every *pair* independent) does **not** imply mutual independence (every *subset* independent). Classic counterexample: two fair coin tosses, $A=\{H \text{ on coin 1}\}$, $B=\{H\text{ on coin 2}\}$, $C=\{\text{both same}\}$. All three pairs are independent, but $P(A\cap B\cap C)=1/4 \ne P(A)P(B)P(C)=1/8$.

### 6.4 Mutually Exclusive vs. Independent — the critical distinction

These sound related but are nearly opposite.

| Property | Definition | Can both hold simultaneously (with positive probabilities)? |
|---|---|---|
| Independent | $P(A\cap B)=P(A)P(B)$ | — |
| Mutually exclusive | $P(A\cap B)=0$ | — |

If $A,B$ are mutually exclusive, $P(A\cap B)=0$. For independence we'd need $P(A)P(B)=0$, i.e. one event has zero probability. So **unless one event is impossible, mutually exclusive events are always dependent** — knowing $A$ occurred tells you $B$ definitely did not, which is maximal dependence, not independence. This single confusion is one of the most common interview traps in probability.

**Numeric check:** given $P(A)=0.4,\ P(B)=0.3,\ P(A\cup B)=0.6$: by inclusion-exclusion $P(A\cap B)=0.4+0.3-0.6=0.1$. Independence would require $P(A)P(B)=0.12 \ne 0.1$, so $A,B$ are **not** independent (slightly negatively correlated).

---

## 7. Law of Total Probability

**Partition:** events $B_1,\ldots,B_n$ partition $\Omega$ if they're mutually exclusive and exhaustive ($\bigcup B_i = \Omega$).

$$P(A) = \sum_i P(A\mid B_i)\,P(B_i)$$

Break a hard probability into cases, compute within each case, weight by how likely each case is.

*Example (spam filter):* 30% of emails are spam (S); filter catches 95% of spam and flags 2% of legit (L) mail as spam.
$$P(\text{Flagged}) = 0.95(0.30) + 0.02(0.70) = 0.285+0.014 = 0.299$$
About 29.9% of all emails get flagged — this sets up the Bayes' theorem examples below.

---

## 8. Bayes' Theorem

$$P(B\mid A) = \frac{P(A\mid B)\,P(B)}{P(A)}, \qquad P(B_i\mid A) = \frac{P(A\mid B_i)P(B_i)}{\sum_j P(A\mid B_j)P(B_j)}$$

| Term | Symbol | Meaning |
|---|---|---|
| Prior | $P(B)$ | Belief before evidence |
| Likelihood | $P(A\mid B)$ | How probable the evidence is if $B$ is true |
| Marginal | $P(A)$ | Total probability of the evidence |
| Posterior | $P(B\mid A)$ | Updated belief after evidence |

$$\text{Posterior} \propto \text{Likelihood} \times \text{Prior}$$

### Worked Examples

**Medical test (classic base-rate example):** disease affects 1% of population; test sensitivity 99% ($P(+\mid D)$), specificity 95% ($P(-\mid D^c)$, so false-positive rate 5%).
$$P(+) = 0.99(0.01)+0.05(0.99) = 0.0099+0.0495 = 0.0594$$
$$P(D\mid +) = \frac{0.0099}{0.0594} \approx 0.167$$
Only ~16.7% chance of actually having the disease despite testing positive — because false positives (5% of the large healthy population) vastly outnumber true positives (99% of the small sick population). **This is why base rates dominate rare-event detection** (fraud detection, anomaly detection, etc.).

**A/B test posterior (statistical significance ≠ "real effect"):** 20 experiments/quarter, historically only 10% truly improve the metric ($P(T)=0.10$); power $P(S\mid T)=0.80$; false-positive rate $P(S\mid T^c)=0.05$.
$$P(S) = 0.80(0.10)+0.05(0.90) = 0.125, \qquad P(T\mid S) = \frac{0.08}{0.125} = 0.64$$
Only a **64%** chance a "statistically significant" result reflects a real improvement — the positive predictive value of the testing process. This is why multiple-testing correction and effect-size reporting matter, not just $p<0.05$.

**Treatment vs. control conversion:** 50/50 split; control converts at 5%, treatment at 7%. Given a random converter, $P(\text{treatment}\mid\text{convert})$:
$$P(C) = 0.07(0.5)+0.05(0.5)=0.06, \qquad P(T\mid C) = \frac{0.035}{0.06} \approx 0.583$$

**Coin-type posterior (full Bayesian update):** a coin is fair (P=0.5) or biased (P=0.7) with equal prior; you observe 7 heads in 10 flips.
$$P(\text{data}\mid H_{0.5}) = \binom{10}{7}(0.5)^{10} \approx 0.1172, \qquad P(\text{data}\mid H_{0.7}) = \binom{10}{7}(0.7)^7(0.3)^3 \approx 0.2668$$
$$P(H_{0.7}\mid\text{data}) = \frac{0.2668(0.5)}{0.1172(0.5)+0.2668(0.5)} \approx 0.695$$
Prior 50% → posterior ~69.5% that the coin is biased.

**Prosecutor's fallacy check:** if $P(A\mid B)=0.9$ and $P(B)=0.01$, is $P(B\mid A)$ also high? Not necessarily — with $P(A)=0.10$: $P(B\mid A) = 0.9(0.01)/0.10 = 0.09$, i.e. **9%**, despite a 90% conditional in the other direction. The rarity of $B$ dominates. Always flip conditioning explicitly via Bayes rather than assuming symmetry.

---

## 9. Common Interview Traps & Pitfalls

1. **$P(A\mid B) \ne P(B\mid A)$** — the Prosecutor's Fallacy. Always use Bayes to flip conditioning.
2. **Mutually exclusive ≠ independent** — they're nearly opposite (§6.4).
3. **Forgetting base rates** — rare events produce counterintuitive posteriors (medical test example).
4. **Assuming independence** without it being stated or provable — "independently chosen" licenses multiplication; nothing else does.
5. **"At least one" problems** — almost always easier via the complement: $P(\text{at least one}) = 1-P(\text{none})$.
6. **Conditional ≠ causal** — $P(B\mid A) > P(B)$ shows association, not causation.
7. **Confidence interval misinterpretation** — see §2; the single most commonly misstated concept in statistics.
8. **Non-transitive relationships** — "A beats B 60%, B beats C 60%" does *not* let you compute $P(A \text{ beats } C)$; win relations need not be transitive (a cyclic die example: Blue beats White, White beats Orange, Orange beats Blue). The honest answer is "cannot be determined from the given information."

---

## 10. Worked Interview Problems

Organized by technique; each states what it's really testing.

### The Complement Trick ("at least one")

**Q: Room of 23 people — probability at least two share a birthday?**
$$P(\text{all distinct}) = \prod_{k=0}^{22}\frac{365-k}{365} \approx 0.4927 \quad\Rightarrow\quad P(\text{shared}) \approx 1-0.4927 = 0.5073$$
With just 23 people there's a >50% chance of a shared birthday — surprising because there are $\binom{23}{2}=253$ pairs, each with a small but compounding ~0.27% match chance. *Tests:* instinctive reach for the complement on "at least one" phrasing.

**Q: 10 servers, each fails independently with probability 0.01 — P(at least one fails)?**
$$P(\text{none fail}) = (0.99)^{10}\approx 0.9044 \quad\Rightarrow\quad P(\text{≥1 fails}) \approx 0.0956$$
*Tests:* complement instinct + correctly invoking independence to multiply.

**Q: Fair coin, 10 flips — P(at least one head)?**
$$P = 1-(1/2)^{10} = 1023/1024 \approx 0.999$$

### Combinatorics Requiring Careful Setup

**Q: Probability of a one-pair poker hand (5-card, standard deck)?**

| Step | Choose | Count |
|---|---|---|
| Rank for the pair | 1 of 13 | 13 |
| Suits for the pair | 2 of 4 | $\binom{4}{2}=6$ |
| 3 other distinct ranks | from remaining 12 | $\binom{12}{3}=220$ |
| Suit for each kicker | 4 options each | $4^3=64$ |

$$\text{One-pair hands} = 13\times6\times220\times64 = 1{,}098{,}240, \qquad P = \frac{1{,}098{,}240}{\binom{52}{5}} = \frac{1{,}098{,}240}{2{,}598{,}960} \approx 0.423$$

Surprisingly high (>40%). *Tests:* multi-stage combination setup and avoiding double-counting (kicker ranks must all differ from the pair rank and each other).

**Q: Committee of 4 from 6 men + 4 women, at least 2 women?**

| Case | Count |
|---|---|
| 2W+2M | $\binom{4}{2}\binom{6}{2}=6\times15=90$ |
| 3W+1M | $\binom{4}{3}\binom{6}{1}=4\times6=24$ |
| 4W+0M | $\binom{4}{4}=1$ |
| **Total** | **115** |

$$P = \frac{115}{\binom{10}{4}} = \frac{115}{210}\approx 0.548$$

*Tests:* case-splitting on "at least" when the complement isn't obviously simpler.

**Q: Password — first 3 chars distinct letters (A–Z), last 5 digits with repetition allowed?**
$${}_{26}P_3 \times 10^5 = 15{,}600 \times 100{,}000 = 1{,}560{,}000{,}000$$
*Tests:* recognizing one segment is a permutation (ordered, no repeats) and the other is independent-slot counting (repeats allowed) — many candidates wrongly apply one formula to both.

**Q: 8 people around a circular table — Alice and Bob must NOT sit adjacent?**
Total circular arrangements: $(8-1)! = 5{,}040$. Arrangements with them adjacent: treat as one unit → $(7-1)!=720$, times 2 internal orders $=1{,}440$. Answer: $5{,}040-1{,}440=3{,}600$. *Tests:* circular permutations + complementary counting together.

### Linearity of Expectation

**Q: Expected number of fixed points in a random permutation of $N$ elements?**
Let $X_i=1$ if element $i$ maps to itself; $P(X_i=1)=1/N$ by symmetry. By linearity of expectation (no need to know the joint distribution):
$$E\left[\sum_{i=1}^N X_i\right] = \sum_{i=1}^N \frac{1}{N} = 1$$
The expected number of fixed points is **always 1**, for any $N$. *Tests:* linearity of expectation — the single most powerful tool in probabilistic combinatorics.

### Distribution Recognition

**Q: Fair coin, flip until first heads — expected number of flips?**
Geometric distribution, $p=0.5$: $E[X] = 1/p = 2$.

### The Non-Transitivity Trap

**Q: Die A beats Die B 60% of the time; B beats C 60% of the time. P(A beats C)?**
**Cannot be determined** from the given information — "beats" need not be transitive (a concrete cyclic counterexample exists: Blue beats White, White beats Orange, Orange beats Blue). *Tests:* whether you blindly apply transitivity or recognize the trap.

---

## 11. Cheat Sheet

```
┌───────────────────────────────────────────────────────────────────┐
│                     SET THEORY                                     │
├───────────────────────────────────────────────────────────────────┤
│  A ∪ B  (OR)     A ∩ B  (AND)     Aᶜ  (NOT)     A − B = A ∩ Bᶜ    │
│  DeMorgan 1:  (A ∪ B)ᶜ = Aᶜ ∩ Bᶜ                                    │
│  DeMorgan 2:  (A ∩ B)ᶜ = Aᶜ ∪ Bᶜ                                    │
│  Disjoint:    A ∩ B = ∅   (≠ independent!)                         │
│  |S × T| = |S|·|T|                                                  │
├───────────────────────────────────────────────────────────────────┤
│                   PROBABILITY AXIOMS                                │
├───────────────────────────────────────────────────────────────────┤
│  P(A) ≥ 0        P(Ω) = 1        disjoint ⇒ P(A∪B) = P(A)+P(B)     │
├───────────────────────────────────────────────────────────────────┤
│                     KEY FORMULAS                                    │
├───────────────────────────────────────────────────────────────────┤
│  Complement:        P(Aᶜ) = 1 − P(A)                                │
│  Inclusion-Excl:    P(A∪B) = P(A)+P(B)−P(A∩B)                      │
│  Conditional:       P(A|B) = P(A∩B) / P(B)                          │
│  Multiplication:    P(A∩B) = P(A|B)·P(B)                            │
│  Independence:      P(A∩B) = P(A)·P(B)                              │
│  Total Prob:        P(A) = Σ P(A|Bᵢ)·P(Bᵢ)                          │
│  Bayes:             P(B|A) = P(A|B)·P(B) / P(A)                     │
│  Permutations:      ₙPₖ = n!/(n−k)!                                 │
│  Combinations:      ₙCₖ = n!/(k!(n−k)!) = ₙPₖ/k!                    │
│  Stars & bars (≥1): C(N−1, k−1)  for N items, k bins                │
├───────────────────────────────────────────────────────────────────┤
│                  BAYES COMPONENTS                                   │
├───────────────────────────────────────────────────────────────────┤
│  Prior       P(B)      what you believed before                     │
│  Likelihood  P(A|B)    how well B explains A                         │
│  Marginal    P(A)      total probability of the evidence            │
│  Posterior   P(B|A)    updated belief                                │
├───────────────────────────────────────────────────────────────────┤
│                       GO-TO TRICKS                                   │
├───────────────────────────────────────────────────────────────────┤
│  "At least one"       → 1 − P(none)                                  │
│  Need to flip P(A|B)? → use Bayes, never assume symmetry             │
│  Sequential events?   → chain rule / multiplication rule              │
│  Given a partition?   → law of total probability                     │
│  Counting w/ a twist? → check: does it hide a recurrence (Fibonacci)?│
├───────────────────────────────────────────────────────────────────┤
│                          TRAPS                                       │
├───────────────────────────────────────────────────────────────────┤
│  ✗ P(A|B) ≠ P(B|A)          Prosecutor's fallacy                    │
│  ✗ Exclusive ≠ Independent   They are nearly opposite                │
│  ✗ Ignoring base rates       Rare events crush posteriors            │
│  ✗ Assuming independence     Must be stated or proven                │
│  ✗ Assuming transitivity     "Beats" relations need not transfer     │
│  ✗ CI ≠ direct probability   95% CI is about the procedure, not θ    │
└───────────────────────────────────────────────────────────────────┘
```

### The Pattern Behind the Questions

| Surface question | What's really being tested |
|---|---|
| Mutually exclusive ≠ independent | Conceptual precision, not rote definitions |
| Confidence interval meaning | Statistical maturity |
| MLE vs. MAP | Connecting theory (regularization) to probabilistic reasoning |
| "At least one" problems | Reflex to use the complement |
| Multi-stage counting (poker, committees) | Setup skill — the arithmetic is trivial once structured correctly |
| Fibonacci-in-counting problems | Recognizing recurrence structure |
| Non-transitive dice | Whether you question intuition instead of pattern-matching |
| Linearity of expectation | The most powerful tool in probabilistic combinatorics |


![Set Operations](images/set.png)
