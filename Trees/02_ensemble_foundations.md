# Chapter 2 — Ensemble Foundations (Expanded Edition)

Chapter 1 ended by establishing that a single decision tree is a **high-variance, low-bias** learner. This chapter answers, rigorously but *slowly*, why combining many such learners helps. Every equation below is broken into small pieces, and every numerical example is worked one arithmetic step at a time — nothing is left for you to fill in.

---

## 2.1 Bias–Variance Decomposition, Revisited for a Single Model

### 2.1.1 Setting up the problem in plain language

Imagine the true relationship between input $x$ and output $y$ is some fixed function $f(x)$, plus a bit of random noise $\epsilon$ that you can never predict no matter how good your model is (measurement error, unmeasured factors, etc.). In symbols:

$$
y = f(x) + \epsilon
$$

- $\mathbb{E}[\epsilon] = 0$ — the noise doesn't systematically push predictions up or down, it's just as likely to be positive as negative.
- $\text{Var}(\epsilon) = \sigma_\epsilon^2$ — the noise has some fixed "spread," called $\sigma_\epsilon^2$.

Now you train a model on a dataset $D$ to get a prediction function $\hat{f}(x)$. Here's the key subtlety: **if you had collected a different random training set, you'd get a different $\hat f$.** So $\hat f$ itself is random — its randomness comes from *which* training set you happened to draw.

**Goal:** at one fixed test point $x_0$, with true outcome $y_0 = f(x_0) + \epsilon_0$, how big is the prediction error on average, across (a) all the different training sets you might have drawn, and (b) all the different noise draws you might see at $x_0$?

$$
\text{Expected Test Error} = \mathbb{E}_{D,\epsilon_0}\Big[(y_0 - \hat f(x_0))^2\Big]
$$

### 2.1.2 The derivation, one line at a time

**Step 1 — define a reference point.** Let $\bar f(x_0)$ be the *average prediction you'd get if you retrained on every possible training set and averaged the results*:

$$
\bar f(x_0) = \mathbb{E}_D[\hat f(x_0)]
$$

This is a fixed number (no randomness left — we've already averaged over all of $D$'s randomness).

**Step 2 — split the error into three pieces.** Add and subtract $\bar f(x_0)$:

$$
y_0 - \hat f(x_0) = \big(f(x_0) - \bar f(x_0)\big) + \big(\bar f(x_0) - \hat f(x_0)\big) + \epsilon_0
$$

Give each piece a name:

| Piece | Symbol | Meaning |
|---|---|---|
| $f(x_0) - \bar f(x_0)$ | $A$ | **Bias**: how far your model's *average* behavior is from the truth |
| $\bar f(x_0) - \hat f(x_0)$ | $B$ | **Variance source**: how far *this particular* fit strays from the average fit |
| $\epsilon_0$ | $C$ | **Irreducible noise**: randomness you can never model away |

**Step 3 — square the three-term sum.** For any three numbers, $(A+B+C)^2 = A^2 + B^2 + C^2 + 2AB + 2AC + 2BC$. So:

$$
(y_0-\hat f(x_0))^2 = A^2 + B^2 + C^2 + 2AB + 2AC + 2BC
$$

**Step 4 — take the expectation and watch three cross-terms disappear.** This is the part the original draft compressed too fast — here it is slowly:

- $A$ is a plain constant (no $D$ or $\epsilon_0$ randomness in it), so it just rides along.
- $\mathbb{E}_D[B] = \mathbb{E}_D[\bar f(x_0) - \hat f(x_0)] = \bar f(x_0) - \mathbb{E}_D[\hat f(x_0)] = \bar f(x_0) - \bar f(x_0) = 0$. So $B$ averages to zero over retrainings — makes sense, $\bar f$ *is* the average of $\hat f$.
- $\mathbb{E}[2AB] = 2A \cdot \mathbb{E}_D[B] = 2A \cdot 0 = 0$. Gone.
- $\epsilon_0$ has mean $0$ and is independent of which training set $D$ you happened to draw, so $\mathbb{E}[2AC] = 2A\cdot\mathbb{E}[\epsilon_0] = 2A \cdot 0 = 0$. Gone.
- Same logic: $\mathbb{E}[2BC] = 2\,\mathbb{E}[B]\cdot\mathbb{E}[\epsilon_0] = 2\cdot 0\cdot 0 = 0$. Gone.

**Step 5 — what's left.**

$$
\mathbb{E}_{D,\epsilon_0}\big[(y_0-\hat f(x_0))^2\big] = \underbrace{A^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}_D[B^2]}_{\text{Variance}} + \underbrace{\sigma_\epsilon^2}_{\text{Irreducible}}
$$

Or, in the simplest possible form:

$$
\boxed{\text{Test Error} = \text{Bias}^2 + \text{Variance} + \text{Noise}}
$$

In words: **your total error comes from being systematically off-target (bias), being inconsistent across different training runs (variance), and randomness you could never predict (noise) — and these three add up cleanly with nothing left over.**

### 2.1.3 Worked numerical — every arithmetic step shown

Setup: true value $f(x_0) = 10$. You retrain your model on 5 different training sets and record the prediction each time at $x_0$:

$$
8,\ 9,\ 11,\ 12,\ 10
$$

**Step A — compute the average prediction $\bar f(x_0)$:**

$$
\bar f(x_0) = \frac{8+9+11+12+10}{5} = \frac{50}{5} = 10
$$

**Step B — compute bias:**

$$
\text{Bias} = f(x_0) - \bar f(x_0) = 10 - 10 = 0 \quad\Rightarrow\quad \text{Bias}^2 = 0
$$

The model is *unbiased on average* — its average guess lands exactly on the truth.

**Step C — compute variance, one squared deviation at a time.** Variance is the average squared distance of each individual prediction from the *average* prediction ($10$), not from the truth:

| Prediction | Deviation from $\bar f=10$ | Squared |
|---|---|---|
| 8 | $10-8=2$ | $4$ |
| 9 | $10-9=1$ | $1$ |
| 11 | $10-11=-1$ | $1$ |
| 12 | $10-12=-2$ | $4$ |
| 10 | $10-10=0$ | $0$ |

Sum of squares: $4+1+1+4+0 = 10$. Divide by 5 (number of draws):

$$
\text{Variance} = \frac{10}{5} = 2.0
$$

**Step D — add the noise term.** Suppose $\sigma_\epsilon^2 = 1.0$ (given/assumed for this dataset).

**Step E — total error:**

$$
\text{Test Error} = \underbrace{0}_{\text{Bias}^2} + \underbrace{2.0}_{\text{Variance}} + \underbrace{1.0}_{\text{Noise}} = 3.0
$$

**Takeaway:** even though this model is bias-free *on average*, it's still generating real error, because any single fit can be off by a lot (swinging from 8 to 12 depending on the training data). That "swing" is exactly the high-variance, low-bias signature described in Chapter 1. **This swing — variance — is the quantity ensembling directly attacks in Section 2.2.**

---

## 2.2 Why Averaging Reduces Variance — Full Derivation, Simplified

### 2.2.1 The setup

Train $M$ separate models $\hat f_1, \hat f_2, \dots, \hat f_M$ — think of $M$ trees, each trained on its own bootstrap sample (the full mechanism is formalized in Chapter 3). Assume, for simplicity:

- Every model has the **same variance**, call it $\sigma^2$.
- Every pair of models has the **same correlation** with each other, call it $\rho$ (a number between 0 and 1 — how similarly two models tend to move together across different training draws).

Average their predictions:

$$
\hat f_{\text{avg}}(x_0) = \frac{1}{M}\sum_{m=1}^{M} \hat f_m(x_0)
$$

### 2.2.2 Bias: unaffected by averaging (short proof)

$$
\mathbb{E}[\hat f_{\text{avg}}] = \frac{1}{M}\sum_{m=1}^M \mathbb{E}[\hat f_m] = \frac{1}{M}\cdot M \cdot \mathbb{E}[\hat f_1] = \mathbb{E}[\hat f_1]
$$

(The middle step just uses "every model has the same bias," so each of the $M$ terms in the sum is identical, and dividing by $M$ cancels the $M$ you just multiplied by.) **Averaging cannot fix a biased model — a biased model averaged with itself many times is still biased by the exact same amount.**

### 2.2.3 Variance: the part that actually changes — derived slowly

We want $\text{Var}(\hat f_{\text{avg}})$. Start from the definition and pull the constant $\frac{1}{M}$ out (constants come out of a variance *squared*):

$$
\text{Var}(\hat f_{\text{avg}}) = \text{Var}\left(\frac{1}{M}\sum_m \hat f_m\right) = \frac{1}{M^2}\,\text{Var}\left(\sum_m \hat f_m\right)
$$

**Now expand the variance of a sum.** For any collection of random variables, the variance of their sum equals the sum of their individual variances *plus* all the pairwise covariances:

$$
\text{Var}\left(\sum_{m=1}^M \hat f_m\right) = \sum_{m=1}^M \text{Var}(\hat f_m) \;+\; \sum_{m \ne m'} \text{Cov}(\hat f_m, \hat f_{m'})
$$

**Count the terms carefully — this is the step usually rushed:**

- The first sum has exactly $M$ terms (one per model), and each equals $\sigma^2$ by assumption. Total: $M\sigma^2$.
- The second sum runs over every *ordered* pair $(m, m')$ with $m \ne m'$ — for $M$ models there are $M \times (M-1)$ such ordered pairs (pick any of $M$ for the first slot, any of the remaining $M-1$ for the second). By definition of correlation, $\text{Cov}(\hat f_m,\hat f_{m'}) = \rho\sigma^2$ (correlation times the two standard deviations, which are both $\sigma$). Total: $M(M-1)\rho\sigma^2$.

Putting them together:

$$
\text{Var}\left(\sum_m \hat f_m\right) = M\sigma^2 + M(M-1)\rho\sigma^2
$$

Now divide by $M^2$ (from the step above) and simplify by splitting into two separate fractions:

$$
\text{Var}(\hat f_{\text{avg}}) = \frac{M\sigma^2}{M^2} + \frac{M(M-1)\rho\sigma^2}{M^2} = \frac{\sigma^2}{M} + \frac{M-1}{M}\,\rho\sigma^2
$$

This is already the exact, finite-$M$ formula. As $M$ gets very large, $\frac{M-1}{M} \to 1$ (e.g., at $M=100$, $\frac{99}{100}=0.99$; at $M=1000$, $\frac{999}{1000}=0.999$ — it creeps toward 1 but never quite gets there for finite $M$). So the clean **limiting form** is:

$$
\boxed{\text{Var}(\hat f_{\text{avg}}) \approx \rho\sigma^2 + \frac{(1-\rho)\sigma^2}{M} \quad \text{for large } M}
$$

### 2.2.4 Reading the formula in plain terms

Think of total variance as splitting into two buckets:

1. **A floor you can never get below:** $\rho\sigma^2$. This depends only on how correlated your models are — adding more models does *nothing* to shrink this piece.
2. **A shrinking piece:** $\frac{(1-\rho)\sigma^2}{M}$. This is the part more models actually helps with, and it shrinks toward 0 as $M\to\infty$.

Three concrete cases:

- **$\rho=1$ (models are carbon copies of each other):** floor $=\sigma^2$, shrinking piece $=0$. Formula gives $\text{Var}=\sigma^2$ — identical to a single model's variance. **Averaging clones buys you literally nothing.** This is why plain bagging has a ceiling, and exactly why Random Forests (Ch. 4) add feature subsampling — specifically to push $\rho$ down further than bootstrap sampling alone can.
- **$\rho=0$ (models are totally independent):** floor $=0$, shrinking piece $=\sigma^2/M$. As $M\to\infty$, variance $\to 0$. Averaging independent models can drive variance arbitrarily close to zero.
- **Realistic case, $0<\rho<1$:** you get *some* benefit from adding models, but it tapers off — you're always stuck at or above the floor $\rho\sigma^2$.

### 2.2.5 Worked numerical — three values of $M$, every arithmetic step shown

Given: single-tree variance $\sigma^2 = 4.0$, pairwise correlation $\rho = 0.3$.

**At $M=10$ trees:**

$$
\text{Var} = \underbrace{(0.3)(4.0)}_{\text{floor}} + \underbrace{\frac{(1-0.3)(4.0)}{10}}_{\text{shrinking piece}}
$$

Compute the floor: $0.3 \times 4.0 = 1.2$.

Compute the shrinking piece: $1 - 0.3 = 0.7$, then $0.7 \times 4.0 = 2.8$, then $2.8 / 10 = 0.28$.

Add: $1.2 + 0.28 = 1.48$.

**At $M=100$ trees:**

Floor is unchanged: $1.2$ (it never depends on $M$).

Shrinking piece: $2.8 / 100 = 0.028$.

Add: $1.2 + 0.028 = 1.228$.

**At $M\to\infty$:**

Shrinking piece $\to 0$. Variance floors at exactly $1.2$.

**Compare the gains directly:**

| Step | Variance | Improvement from previous |
|---|---|---|
| $M=1$ (single tree) | $4.0$ | — |
| $M=10$ | $1.48$ | $4.0 - 1.48 = 2.52$ |
| $M=100$ | $1.228$ | $1.48 - 1.228 = 0.252$ |
| $M\to\infty$ | $1.2$ | $1.228 - 1.2 = 0.028$ (max possible remaining gain) |

**What this table is telling you:** going from 1 tree to 10 trees bought you a variance drop of $2.52$ — huge. Going from 10 to 100 trees (a 10x increase in compute) only bought $0.252$ — about a tenth as much. Going from 100 trees all the way to infinity can buy you at most another $0.028$. **The returns are not just diminishing — they collapse by roughly a factor of 10 every time you multiply $M$ by 10, while the floor set by $\rho$ stays completely fixed.** This is the mathematical reason Random Forest error curves flatten out well before hundreds of trees, and why in practice tuning `max_features` (which controls $\rho$, the correlation between trees) tends to move the needle more than cranking `n_estimators` (which only controls $M$) past the point where the curve has already gone flat.

### 2.2.6 Why the simplifying assumptions are there

The derivation assumes every model has *the same* variance $\sigma^2$ and *every pair* has *the same* correlation $\rho$. In reality, some bootstrap samples happen to be "luckier" than others, so this symmetry is only approximate. It's kept because:

1. It keeps the algebra tractable enough to derive by hand.
2. It still captures the dominant real-world effect — decorrelating models matters more than piling on more of them.
3. It's the standard justification given in Hastie, Tibshirani & Friedman's *Elements of Statistical Learning* (Ch. 15) for why Random Forests work, and the qualitative conclusion holds up even when the symmetry assumption is only roughly true.

---

## 2.3 The Three Ensembling Families — What Each One Actually Attacks

| | Bagging | Boosting | Stacking |
|---|---|---|---|
| Model training | **Parallel** — every base model trained independently and simultaneously, no communication between them | **Sequential** — each new model is explicitly built to fix the errors the current ensemble is still making | Base models can train in parallel; a separate meta-learner is trained afterward, using their outputs as its inputs |
| What it primarily reduces | **Variance** (this is exactly the mechanism derived in 2.2) | **Bias** (each new weak learner targets whatever error still remains) | Neither directly — it learns the *best way to combine* diverse models, which indirectly can improve both |
| Best base learner | Low-bias, high-variance (e.g., deep, unpruned trees) — bagging a high-bias model barely helps, since 2.2.2 showed averaging leaves bias completely untouched | High-bias, low-variance ("weak learners" — shallow trees, sometimes called "stumps") — boosting a low-bias learner risks fast overfitting, because sequential correction has no built-in brake on chasing noise | A mix of anything — ideally models that are each reasonably good but tend to make *different* mistakes from one another |
| Overfitting behavior | Adding more base models does **not** raise overfitting risk (proved formally in Ch. 3) | Adding more rounds **can** raise overfitting risk — needs explicit regularization such as a learning rate or early stopping (Ch. 5) | Depends heavily on the meta-learner; the main danger is leakage from improperly validated base-model predictions (Ch. 6) |
| Chapter | 3–4 | 5 | 6 |

### 2.3.1 Why this pairing holds — spelled out step by step

**Why bagging pairs with high-variance learners:** Section 2.2 proved two things: (1) averaging shrinks variance toward the floor $\rho\sigma^2$, and (2) averaging leaves bias completely unchanged. Put those two facts together: if you bag a model that already has *low* variance, there's very little variance left for averaging to remove — you're applying a variance-reduction tool to a problem that barely has variance. And if that same low-variance model also has *high* bias (a common combination, e.g. a shallow linear model), bagging can't touch the bias either. Net result: almost no improvement. This is exactly why bagging is paired with deep, unpruned, high-variance trees — that's where the tool actually has something to work with.

**Why boosting pairs with high-bias learners:** Boosting's whole mechanism is to look at what the ensemble currently gets wrong and add a new model that specifically targets those errors — it is, by construction, a bias-reduction machine. If you feed it an already-low-bias learner (like a deep tree), that single learner already fits the training data closely on round 1. There's nothing systematic left for boosting to correct — whatever residual is left over is mostly noise, not signal. Boosting has no built-in way to tell the difference between "real pattern I should fit" and "just noise" in those residuals, so it fits the noise too, and overfits. Feeding it weak, high-bias learners instead means there's always a real, systematic pattern left to correct at each step, which is exactly the fuel boosting needs.

---

## 2.4 Bootstrap Sampling Theory

### 2.4.1 The mechanism in plain language

Bagging (fully detailed in Ch. 3) needs a way to manufacture $M$ *different* training sets out of one original dataset. It does this via **bootstrap sampling**: from an original dataset of $n$ samples, draw $n$ samples **with replacement** — meaning after you draw a sample, you put it back before drawing again, so it could be drawn more than once.

Result: each bootstrap sample is the same size ($n$) as the original, but because of the "with replacement" rule, some original samples show up multiple times and others don't show up at all.

### 2.4.2 The question: how much data gets left out?

This matters because whatever gets left out of a given bootstrap sample becomes free validation data for that tree (Out-of-Bag estimation, Ch. 3.3).

**Setting up the probability, one piece at a time:**

- The original dataset has $n$ samples, each equally likely to be drawn on any single draw.
- Probability that a *specific* sample $i$ **is** picked on one draw: $\frac{1}{n}$.
- Probability that sample $i$ is **not** picked on one draw: $1 - \frac{1}{n}$.
- We draw $n$ times total, and — because it's *with replacement* — each draw is completely independent of the others.
- Probability that sample $i$ is missed on **every single one** of the $n$ draws: multiply the "not picked" probability by itself $n$ times.

$$
P(\text{sample } i \text{ never picked}) = \left(1-\frac{1}{n}\right)^n
$$

### 2.4.3 Worked numerical for small $n=10$ — every multiplication shown

We need $(0.9)^{10}$. Rather than multiplying 0.9 by itself ten times in a row, build it up by repeated squaring, which is faster and easy to double check:

$$
(0.9)^2 = 0.9 \times 0.9 = 0.81
$$

$$
(0.9)^4 = (0.9)^2 \times (0.9)^2 = 0.81 \times 0.81 = 0.6561
$$

$$
(0.9)^8 = (0.9)^4 \times (0.9)^4 = 0.6561 \times 0.6561 = 0.43046721
$$

Now combine $(0.9)^8$ with $(0.9)^2$ to get $(0.9)^{10}$, since $8+2=10$:

$$
(0.9)^{10} = (0.9)^8 \times (0.9)^2 = 0.43046721 \times 0.81 \approx 0.3487
$$

**So with $n=10$, about 34.87% (roughly 35%) of the original samples are left out of any single bootstrap draw.**

### 2.4.4 The limiting case as $n \to \infty$

As datasets get larger, this percentage settles down to a fixed constant. The tool for finding it is a well-known calculus limit: $\lim_{n\to\infty}\left(1+\frac{x}{n}\right)^n = e^x$. Plugging in $x=-1$:

$$
\lim_{n\to\infty}\left(1-\frac{1}{n}\right)^n = e^{-1} \approx 0.3679
$$

**Sanity check against the $n=10$ number above:** we got $34.87\%$ at $n=10$, and the limiting value is $36.79\%$ — close, and the gap shrinks further as $n$ grows (try $n=100$ or $n=1000$ mentally: it creeps closer to $36.79\%$ each time). This is a good way to confirm the formula makes sense: small-$n$ approximation should converge toward the limit, not jump around randomly.

### 2.4.5 The headline number to memorize

$$
\boxed{\approx 36.8\% \text{ of samples excluded} \;\Longleftrightarrow\; \approx 63.2\% \text{ of unique samples included}}
$$

(The "63.2% included" side follows because $1 - 0.368 = 0.632$ — whatever isn't excluded is included. Note the *included* 63.2% is measured in "unique samples," not slots: since sampling is with replacement, some of the $n$ slots in the bootstrap sample are duplicates of those same included samples, not new distinct ones.)

This ratio is essentially fixed regardless of dataset size (once $n$ is reasonably large) — it's a common interview number-check: *"what fraction of data is in a bootstrap sample on average?"* → ~63%. *"What fraction is out-of-bag?"* → ~37%.

### 2.4.6 Why this matters beyond trivia

That consistent ~37% "held-out" chunk per tree is exactly the mechanism Chapter 3.3 turns into **Out-of-Bag (OOB) error estimation**: an honest, validation-like error estimate that costs you nothing in training data, because each original sample was, on average, excluded from about 37% of the trees in the bagged ensemble — so you can evaluate each sample using exactly the subset of trees that never saw it during training, with no separate holdout set required.

---

## Quick Interview Q&A

**Q: Does averaging more models always reduce test error?**
A: Averaging always reduces *variance* — and it does so monotonically (strictly decreasing as $M$ grows, as long as $\rho<1$, per the formula in 2.2.3), and it *never* changes bias (proved in 2.2.2). So total test error only drops if variance was actually the dominant problem in the first place. If your base model is high-bias/low-variance to begin with, averaging more of them barely moves total error at all, because you're shrinking a term (variance) that was already small, while the large term (bias) sits untouched.

**Q: Why can't boosting just use the same "average many high-variance models" trick that bagging uses?**
A: Because boosting's whole error-reduction mechanism is about sequentially targeting *what's currently wrong* — residuals or reweighted misclassifications — and that requires a controlled, incremental correction signal. That signal only makes sense if each individual step is a weak, high-bias adjustment. Feed boosting a high-variance base learner (like an unpruned tree) instead, and round 1 already overfits hard; every subsequent round is then "correcting" residuals that are mostly noise, which compounds overfitting rather than steadily reducing bias.

**Q: If OOB gives a free validation estimate, why does anyone still use k-fold CV with bagging or Random Forests?**
A: OOB error is a valid, essentially-free approximation of leave-one-out CV, but *only for bagged models specifically* — it depends entirely on the bootstrap resampling structure to even define "out-of-bag" in the first place. For hyperparameter searches that go beyond "how many trees," or for comparing against non-bagged models on a level playing field, an explicit k-fold CV setup is still the more standard, broadly comparable protocol.

---

**Next up: Chapter 3 — Bagging in full detail (the algorithm, OOB error walked through numerically, and precisely when it does and doesn't help).**
