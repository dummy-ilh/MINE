# Ensemble Foundations — Master Notes

## 1. The Big Idea (in one paragraph)

Chapter 1 established that a single decision tree is low-bias, high-variance — it can fit almost any pattern, but small changes in training data swing its predictions a lot. This chapter proves, with the actual math, *why* combining many such models helps: averaging kills variance (and does nothing to bias), while the *amount* of benefit depends entirely on how correlated your models are with each other — not just how many you have. That one formula is the mathematical foundation for bagging, Random Forests, and (in reverse) explains why boosting needs the opposite kind of base model.

---

## 2. Bias–Variance Decomposition, for One Model

### 2.1 The setup, in plain language

The true relationship is $y = f(x) + \epsilon$, where $\epsilon$ is unpredictable noise (mean 0, some fixed spread $\sigma_\epsilon^2$). You train a model on a dataset $D$ to get a prediction function $\hat f(x)$ — but **if you'd drawn a different training set, you'd get a different $\hat f$.** So $\hat f$ is itself random, and its randomness comes entirely from *which* training set you happened to see.

**Question:** at a test point $x_0$, how big is the expected squared error, averaged over every training set you might have drawn and every noise draw you might see?

### 2.2 The result (skipping the algebra — full derivation below if you want it)

$$\boxed{\text{Test Error} = \text{Bias}^2 + \text{Variance} + \text{Noise}}$$

- **Bias** = how far your model's *average* behavior (averaged across all possible training sets) is from the truth.
- **Variance** = how much any *one* particular fit wobbles around that average.
- **Noise** = randomness no model can ever remove.

These three terms add up cleanly with nothing left over — no cross-terms, no double-counting.

<details>
<summary>Full derivation (click to expand — worth knowing for a rigor-focused interview)</summary>

Define $\bar f(x_0) = \mathbb{E}_D[\hat f(x_0)]$ — the average prediction you'd get if you retrained on every possible training set. Add and subtract it:

$$y_0 - \hat f(x_0) = \underbrace{(f(x_0)-\bar f(x_0))}_{A:\text{ bias}} + \underbrace{(\bar f(x_0)-\hat f(x_0))}_{B:\text{ variance source}} + \underbrace{\epsilon_0}_{C:\text{ noise}}$$

Square it: $(A+B+C)^2 = A^2+B^2+C^2+2AB+2AC+2BC$. Take the expectation — every cross term vanishes:
- $\mathbb{E}_D[B]=0$ (by definition, $\bar f$ *is* the average of $\hat f$), so $\mathbb{E}[2AB]=0$.
- $\epsilon_0$ has mean 0 and is independent of $D$, so $\mathbb{E}[2AC]=0$ and $\mathbb{E}[2BC]=0$.

What's left: $A^2 + \mathbb{E}_D[B^2] + \sigma_\epsilon^2$ = Bias² + Variance + Noise.
</details>

### 2.3 Worked numerical

True value $f(x_0)=10$. Retrain on 5 different training sets, predictions at $x_0$: $8, 9, 11, 12, 10$.

**Average prediction:** $\bar f(x_0) = \frac{8+9+11+12+10}{5} = 10$

**Bias:** $10 - 10 = 0$ → Bias² = 0 (unbiased *on average*)

**Variance** (average squared distance of each prediction from the average, 10):

| Prediction | Deviation from 10 | Squared |
|---|---|---|
| 8 | 2 | 4 |
| 9 | 1 | 1 |
| 11 | −1 | 1 |
| 12 | −2 | 4 |
| 10 | 0 | 0 |

Sum = 10, ÷5 → **Variance = 2.0**

**Noise:** assume $\sigma_\epsilon^2 = 1.0$

**Total:** $0 + 2.0 + 1.0 = 3.0$

**Takeaway:** this model is bias-free *on average*, yet still has real error — because any single fit can swing from 8 to 12 depending on which training data it saw. That swing **is** variance, and it's exactly what ensembling attacks next.

---

## 3. Why Averaging Reduces Variance

### 3.1 The setup

Train $M$ models $\hat f_1,\dots,\hat f_M$ (think: $M$ trees on $M$ bootstrap samples, formalized in the Bagging notes). Assume every model has the same variance $\sigma^2$, and every pair has the same correlation $\rho$ (how similarly two models move together across different training draws, 0 to 1). Average them:

$$\hat f_{\text{avg}}(x_0) = \frac{1}{M}\sum_{m=1}^M \hat f_m(x_0)$$

### 3.2 Bias is untouched

$$\mathbb{E}[\hat f_{\text{avg}}] = \mathbb{E}[\hat f_1]$$

Averaging $M$ equally-biased models just gives you back the same bias — **a biased model averaged with itself many times is still exactly as biased.**

### 3.3 Variance — the part that changes

$$\text{Var}(\hat f_{\text{avg}}) = \frac{\sigma^2}{M} + \frac{M-1}{M}\rho\sigma^2 \;\xrightarrow[M\to\infty]{}\; \boxed{\rho\sigma^2 + \frac{(1-\rho)\sigma^2}{M}}$$

<details>
<summary>Full derivation</summary>

Pull the constant out: $\text{Var}(\hat f_{\text{avg}}) = \frac{1}{M^2}\text{Var}(\sum_m \hat f_m)$.

Variance of a sum = sum of variances + all pairwise covariances:
$$\text{Var}\left(\sum_m \hat f_m\right) = \sum_m \text{Var}(\hat f_m) + \sum_{m\ne m'}\text{Cov}(\hat f_m,\hat f_{m'})$$

- $M$ variance terms, each $\sigma^2$ → total $M\sigma^2$.
- $M(M-1)$ ordered pairs, each covariance $=\rho\sigma^2$ → total $M(M-1)\rho\sigma^2$.

Divide by $M^2$: $\frac{\sigma^2}{M} + \frac{M-1}{M}\rho\sigma^2$. As $M\to\infty$, $\frac{M-1}{M}\to 1$, giving the boxed limiting form.
</details>

### 3.4 Reading the formula

Two buckets:

1. **A floor you can't get below:** $\rho\sigma^2$ — depends only on correlation. More models does *nothing* to shrink this.
2. **A shrinking piece:** $\frac{(1-\rho)\sigma^2}{M}$ — this is what more models actually helps with, shrinking toward 0.

| $\rho$ | What happens |
|---|---|
| $\rho=1$ (models are carbon copies) | Variance = $\sigma^2$, identical to one model. Averaging clones buys nothing. |
| $\rho=0$ (models fully independent) | Variance → 0 as $M\to\infty$. |
| $0<\rho<1$ (realistic) | Some benefit, but it tapers off — always stuck at or above the $\rho\sigma^2$ floor. |

### 3.5 Worked numerical — diminishing returns, concretely

$\sigma^2=4.0$, $\rho=0.3$. Floor $= 0.3\times4.0 = 1.2$ (fixed, regardless of $M$).

| $M$ | Variance | Improvement from previous step |
|---|---|---|
| 1 (single tree) | 4.0 | — |
| 10 | $1.2 + \frac{0.7\times4.0}{10} = 1.48$ | 2.52 |
| 100 | $1.2 + \frac{0.7\times4.0}{100} = 1.228$ | 0.252 |
| ∞ | 1.2 | 0.028 (max remaining gain) |

**Reading it:** 1→10 trees bought a 2.52 drop. 10→100 trees (10x the compute) bought only 0.252 — about a tenth as much. This is why Random Forest error curves flatten out well before hundreds of trees, and why tuning `max_features` (which lowers $\rho$) tends to matter more than cranking `n_estimators` past the point the curve has gone flat.

### 3.6 Why the simplifying assumptions are fine to use

The derivation assumes every model has *identical* variance and every pair has *identical* correlation — not exactly true in practice (some bootstrap samples are luckier than others). It's used anyway because it keeps the math tractable, still captures the dominant real effect (decorrelating matters more than piling on more models), and is the standard textbook justification (Hastie, Tibshirani & Friedman, *Elements of Statistical Learning*, Ch. 15) — the qualitative conclusion holds even when the symmetry is only approximate.

---

## 4. The Three Ensembling Families

| | Bagging | Boosting | Stacking |
|---|---|---|---|
| Training | Parallel — models trained independently | Sequential — each model fixes the current ensemble's errors | Base models parallel; a meta-learner trained afterward on their outputs |
| Reduces | **Variance** (Section 3) | **Bias** (targets remaining error each round) | Neither directly — learns the best way to *combine* diverse models |
| Best base learner | Low-bias, high-variance (deep trees) | High-bias, low-variance ("weak learners," shallow trees/stumps) | Anything reasonably good that tends to make *different* mistakes |
| More rounds/models = more overfitting? | No | Yes — needs a learning rate or early stopping | Depends on meta-learner; main risk is leakage from poorly validated base predictions |

**Why bagging needs high-variance learners:** averaging shrinks variance toward the $\rho\sigma^2$ floor but leaves bias completely untouched (Section 3.2). Bag a model that's already low-variance, and there's barely any variance left to remove — you're using a variance tool on a problem that doesn't have much variance. If that same model is also high-bias, bagging can't fix that either. Net: little improvement. This is exactly why bagging pairs with deep, unpruned trees.

**Why boosting needs high-bias learners:** boosting's mechanism is "look at what's currently wrong, add a model that fixes it" — a bias-reduction machine by construction. Feed it an already-low-bias learner (a deep tree), and round 1 already fits closely — whatever's "left over" to correct is mostly noise, not signal. Boosting has no way to tell noise from signal in the residuals, so it fits the noise too and overfits. Weak, high-bias learners always leave a *real* systematic pattern to correct — exactly the fuel boosting needs.

---

## 5. Bootstrap Sampling Theory

### 5.1 The mechanism

To manufacture $M$ different training sets from one original dataset, draw $n$ samples **with replacement** (put each one back before drawing again). Result: same size ($n$) as the original, but some rows appear multiple times and others don't appear at all.

### 5.2 What fraction gets left out?

Probability a specific sample $i$ is missed on one draw: $1-\frac{1}{n}$. Missed on all $n$ independent draws:

$$P(\text{sample } i \text{ never picked}) = \left(1-\frac{1}{n}\right)^n$$

**Worked numerical, $n=10$:** $(0.9)^{10}$. Build it by squaring: $(0.9)^2=0.81$, $(0.9)^4=0.6561$, $(0.9)^8=0.4305$, then $(0.9)^{10}=(0.9)^8\times(0.9)^2 = 0.4305\times0.81 \approx 0.3487$ → **~35% left out** at $n=10$.

**As $n\to\infty$:** using $\lim_{n\to\infty}(1+\frac{x}{n})^n = e^x$ with $x=-1$:
$$\left(1-\frac{1}{n}\right)^n \to e^{-1} \approx 0.3679$$

The $n=10$ estimate (34.87%) is already close to the limit (36.79%) — a good sanity check that the formula's right.

### 5.3 The number to memorize

$$\boxed{\approx 36.8\% \text{ excluded} \iff \approx 63.2\% \text{ included (unique samples)}}$$

This ratio is essentially fixed once $n$ is reasonably large. Common interview check: *"what fraction of data is in a bootstrap sample?"* → ~63%. *"Out-of-bag?"* → ~37%.

**Why it matters:** that consistent ~37% held-out chunk per tree is exactly the mechanism behind Out-of-Bag (OOB) error estimation — free, honest validation with no separate holdout set required, since each sample is naturally excluded from roughly 37% of the trees.

---

## 6. Quick Q&A (general)

**Q: Does averaging more models always reduce test error?**
A: It always reduces variance (monotonically, as long as $\rho<1$) and never changes bias. So total error only drops meaningfully if variance was the dominant problem to begin with. If your base model is already high-bias/low-variance, averaging more of them barely moves total error — you're shrinking the small term, leaving the large one untouched.

**Q: Why can't boosting just use bagging's "average many high-variance models" trick?**
A: Boosting's correction mechanism needs a controlled, incremental signal — that only makes sense with weak, high-bias adjustments. Feed it a high-variance base learner instead, and round 1 already overfits; every later round "corrects" residuals that are mostly noise, compounding overfitting instead of steadily reducing bias.

**Q: If OOB gives free validation, why does anyone still use k-fold CV with bagged models?**
A: OOB only exists *because of* bootstrap resampling — it's specific to bagged models. For hyperparameter searches beyond "how many trees," or comparing against non-bagged models on equal footing, explicit k-fold CV is still the more standard, broadly comparable protocol.

---

## 7. Google MLE Interview Q&A

**Q: You're asked to derive, on a whiteboard, why the bias-variance decomposition has no cross-terms. What's the one-sentence reason, and where exactly does it come from?**
A: Every cross-term involves $\mathbb{E}_D[B]$ or $\mathbb{E}[\epsilon_0]$ as a factor, and both of those are exactly zero by definition — $B=\bar f(x_0)-\hat f(x_0)$ averages to zero over retrainings because $\bar f$ *is* defined as that average, and $\epsilon_0$ is defined to have mean zero and be independent of the training set. So the cross-terms don't cancel by some fortunate coincidence — they're forced to zero directly by how bias and noise were defined in the first place.

**Q: A teammate proposes increasing `n_estimators` from 200 to 2000 to squeeze out more accuracy on a Random Forest that's already performing well. Using the variance formula, how do you evaluate that proposal before running it?**
A: The formula tells you the ceiling on possible gain *before* you spend the compute: the maximum remaining improvement is bounded by $\frac{(1-\rho)\sigma^2}{200} - \rho\sigma^2$'s already-realized gain versus the floor $\rho\sigma^2$ that $M$ alone can never cross. Concretely, per the worked example, going from 100→∞ trees only ever recovers a small fraction of what 1→10 trees recovered — so before burning 10x the training/serving cost, it's worth estimating $\rho$ (e.g., from OOB or held-out variance at a couple of $M$ values) to see whether you're already near the floor. If so, the better lever is reducing correlation (lower `max_features`) rather than raising $M$.

**Q: Explain why a boosting model and a bagging model can have identical training error but very different generalization behavior, using the bias/variance split from this chapter.**
A: Training error mixes bias and variance together and doesn't distinguish them — a bagged ensemble of deep trees and a boosted ensemble of shallow trees can both drive training error to a similar low number, but by fundamentally different routes: bagging gets there mainly by having low-bias base learners to begin with (variance was reduced by averaging, but bias was never the issue), while boosting gets there by sequentially eliminating bias round by round. Generalization diverges because boosting's mechanism has no natural brake on continuing to reduce "error" even after the true signal is exhausted — it will keep fitting noise as bias-reduction, unlike bagging where more base models provably can't increase variance (Section 4).

---

## 8. Apple MLE Interview Q&A (on-device / practical flavor)

**Q: You're deciding between shipping a bagged ensemble vs. a single boosted model to run on-device. The variance-reduction formula in this chapter is about training-time statistics — does it tell you anything about the on-device inference trade-off?**
A: Not directly — the formula describes *statistical* variance reduction, not compute cost, so it doesn't by itself argue for one deployment over the other. What it does tell you indirectly: bagging's variance-reduction benefit comes specifically from evaluating *all* $M$ models and averaging, so you can't get the statistical benefit described in Section 3 without paying for $M$ forward passes at inference — there's no way to "bank" the variance reduction into a single cheaper model the way boosting's sequential corrections end up baked into one final additive function. That's a deployment-cost distinction the math doesn't cover but that follows directly from *how* the math's benefit is realized.

**Q: A teammate wants to prune a trained Random Forest down to fewer trees after training, to save on-device inference cost. Using the diminishing-returns table in this chapter, how would you frame the trade-off to them?**
A: Frame it as reading the table backwards — cutting from, say, 100 trees to 10 doesn't cost you the full gap between "1 tree" and "100 trees," it costs you specifically the *shrinking piece* that hadn't yet flattened, which per the worked numerical is a small slice (roughly 0.25 out of a 4.0 starting variance in that example) compared to what the first 10 trees bought you. In other words: because returns diminish so sharply after the first handful of trees, pruning from a large $M$ down to a moderate one is often a very favorable trade for on-device latency/memory, whereas pruning from 10 trees down to 2 gives back a much larger share of the original variance-reduction benefit. The right move is checking where on that curve your current $M$ sits, not assuming the cost is linear in trees removed.

**Q: How does the ~63.2% / 36.8% bootstrap sampling split matter if you're training a bagged model on a small, privacy-constrained on-device dataset (e.g., a per-user personalization model with very limited local data)?**
A: With small $n$, the exact exclusion probability sits measurably below the asymptotic 36.8% limit (the chapter's own $n=10$ example gives ~34.87%, not 36.8%), and more importantly, small $n$ means each bootstrap sample is a much less faithful re-creation of the underlying distribution — a handful of duplicated/missing rows has an outsized effect when there are only a few dozen local examples to begin with. That pushes correlation $\rho$ between trees *up* (bootstrap samples end up looking more alike than the asymptotic theory assumes), which — per the Section 3 formula — directly raises the variance floor $\rho\sigma^2$ that no amount of averaging can get below. Practically: bagging's benefit is weaker on small local datasets than the standard large-$n$ intuition suggests, which is a real consideration for per-device or per-user models trained on limited local data.

---

**One-line summary to remember:** *Test error = Bias² + Variance + Noise, additively with no cross-terms → averaging $M$ models leaves bias untouched but shrinks variance toward a correlation-set floor $\rho\sigma^2$, with diminishing returns from $M$ alone → bagging (variance-focused) wants high-variance base learners, boosting (bias-focused) wants high-bias ones → bootstrap sampling naturally leaves ~36.8% of rows out of each sample, which is what makes OOB error possible for free.*
