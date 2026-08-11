# Chapter 2 — Ensemble Foundations

Chapter 1 ended by establishing that a single decision tree is a **high-variance, low-bias** learner. This chapter answers, rigorously, *why* combining many such learners helps — the math here is the foundation everything from Chapter 3 (Bagging) onward builds on.

---

## 2.1 Bias–Variance Decomposition, Revisited for a Single Model

Before touching ensembles, let's re-derive the decomposition itself in full, since every later result is just this formula applied to a *combination* of models instead of one.

Assume the true data-generating process is $y = f(x) + \epsilon$, where $\epsilon$ is irreducible noise with $\mathbb{E}[\epsilon]=0$, $\text{Var}(\epsilon) = \sigma^2_\epsilon$, independent of $x$.

Let $\hat{f}(x)$ be our model's prediction, trained on a random training set $D$ (so $\hat{f}$ is itself a random variable — different draws of $D$ give different $\hat{f}$). For a fixed test point $x_0$ with true value $y_0 = f(x_0) + \epsilon_0$, the expected squared prediction error, averaged over draws of $D$ and over noise $\epsilon_0$, is:

$$
\mathbb{E}_{D,\epsilon_0}\left[(y_0 - \hat{f}(x_0))^2\right]
$$

**Step-by-step derivation — no steps skipped:**

Add and subtract $\mathbb{E}_D[\hat f(x_0)]$ (call this $\bar f(x_0)$, the *average prediction across all possible training sets*):

$$
y_0 - \hat f(x_0) = \underbrace{(f(x_0) - \bar f(x_0))}_{\text{Bias}} + \underbrace{(\bar f(x_0) - \hat f(x_0))}_{\text{Variance term}} + \underbrace{\epsilon_0}_{\text{noise}}
$$

Square this three-term sum. When you expand $(A+B+C)^2 = A^2+B^2+C^2+2AB+2AC+2BC$ and take the expectation over $D$ and $\epsilon_0$, every cross term vanishes:

- $A = f(x_0)-\bar f(x_0)$ is a **constant** (no randomness left — $\bar f$ already averaged out $D$).
- $B = \bar f(x_0) - \hat f(x_0)$ has $\mathbb{E}_D[B] = \bar f(x_0) - \mathbb{E}_D[\hat f(x_0)] = \bar f(x_0)-\bar f(x_0) = 0$, so $\mathbb{E}[2AB] = 2A\cdot\mathbb{E}[B] = 0$.
- $C = \epsilon_0$ is independent of $D$ and has mean 0, so $\mathbb{E}[2AC]=2A\cdot\mathbb{E}[\epsilon_0]=0$ and $\mathbb{E}[2BC] = 2\mathbb{E}[B]\mathbb{E}[\epsilon_0] = 0$ (independence lets the expectation factor).

What survives:

$$
\mathbb{E}_{D,\epsilon_0}[(y_0-\hat f(x_0))^2] = \underbrace{(f(x_0)-\bar f(x_0))^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}_D[(\bar f(x_0)-\hat f(x_0))^2]}_{\text{Variance}} + \underbrace{\sigma_\epsilon^2}_{\text{Irreducible}}
$$

$$
\boxed{\text{Expected Test Error} = \text{Bias}^2 + \text{Variance} + \sigma_\epsilon^2}
$$

**Worked numerical:** suppose across 5 different training sets, a model's prediction at a fixed $x_0$ (true $f(x_0)=10$) comes out as: 8, 9, 11, 12, 10.

$\bar f(x_0) = (8+9+11+12+10)/5 = 50/5 = 10$

$\text{Bias} = f(x_0)-\bar f(x_0) = 10-10=0 \Rightarrow \text{Bias}^2=0$

$\text{Variance} = \frac{1}{5}[(10-8)^2+(10-9)^2+(10-11)^2+(10-12)^2+(10-10)^2]$
$= \frac{1}{5}[4+1+1+4+0] = \frac{10}{5}=2.0$

If $\sigma_\epsilon^2 = 1.0$, expected test error $= 0 + 2.0 + 1.0 = 3.0$.

Notice: this model is **unbiased on average** (bias² = 0) but still has real test error because individual fits swing around that average — exactly the high-variance, low-bias signature from Chapter 1.5. This is the number ensembling attacks.

---

## 2.2 Why Averaging Reduces Variance — Full Derivation

Suppose we train $M$ models $\hat f_1, \dots, \hat f_M$ (think: $M$ trees on $M$ different bootstrap samples — bagging, previewed here, formalized in Ch.3), each with the same variance $\sigma^2 = \text{Var}(\hat f_m(x_0))$ and the same bias, and average their predictions:

$$
\hat f_{\text{avg}}(x_0) = \frac{1}{M}\sum_{m=1}^{M} \hat f_m(x_0)
$$

**Averaging never changes bias** (bias is linear, so the average of $M$ equally-biased predictors has the same bias as any one of them):
$$
\mathbb{E}[\hat f_{\text{avg}}(x_0)] = \frac{1}{M}\sum_m \mathbb{E}[\hat f_m(x_0)] = \mathbb{E}[\hat f_1(x_0)] \quad \text{(same bias as a single model)}
$$

**Averaging's effect on variance — this is the part that matters.** Let $\rho$ be the average pairwise correlation between any two models' predictions (across the randomness in how they were trained — e.g., across different bootstrap draws), and $\sigma^2$ the variance of a single model. Using the general formula for variance of a sum/average of correlated random variables:

$$
\text{Var}(\hat f_{\text{avg}}) = \text{Var}\left(\frac{1}{M}\sum_m \hat f_m\right) = \frac{1}{M^2}\text{Var}\left(\sum_m \hat f_m\right)
$$

Expand $\text{Var}(\sum_m \hat f_m) = \sum_m \text{Var}(\hat f_m) + \sum_{m \neq m'} \text{Cov}(\hat f_m, \hat f_{m'})$. There are $M$ variance terms (each $=\sigma^2$) and $M(M-1)$ ordered covariance terms (each $=\rho\sigma^2$, using $\text{Cov}=\rho\sigma^2$ since both variables have the same variance $\sigma^2$):

$$
\text{Var}\left(\sum_m \hat f_m\right) = M\sigma^2 + M(M-1)\rho\sigma^2
$$

Divide by $M^2$:

$$
\text{Var}(\hat f_{\text{avg}}) = \frac{M\sigma^2 + M(M-1)\rho\sigma^2}{M^2} = \frac{\sigma^2}{M} + \frac{(M-1)}{M}\rho\sigma^2
$$

As $M \to \infty$, $\frac{(M-1)}{M} \to 1$, giving the clean limiting form:

$$
\boxed{\text{Var}(\hat f_{\text{avg}}) \;\xrightarrow{M\to\infty}\; \rho\sigma^2 + \frac{(1-\rho)\sigma^2}{M}}
$$

**This single formula is the theoretical core of every variance-reduction ensemble method in this curriculum.** Read it carefully:

- If $\rho = 1$ (models are perfectly correlated — e.g., you trained $M$ identical trees on the identical data with no randomization): $\text{Var}(\hat f_{\text{avg}}) = \sigma^2$. **No benefit at all** from averaging — this is why bagging without any decorrelation mechanism has a ceiling, and exactly why Random Forests (Ch.4) add feature-subsampling on top of bootstrap sampling: to push $\rho$ down further than bagging alone can.
- If $\rho = 0$ (models are fully independent): $\text{Var}(\hat f_{\text{avg}}) = \sigma^2/M \to 0$ as $M\to\infty$. Variance can be driven arbitrarily close to zero.
- Realistically $0 < \rho < 1$: you get a floor at $\rho\sigma^2$ that no amount of additional trees can push below — **adding more trees only kills the $\frac{(1-\rho)\sigma^2}{M}$ term, never the $\rho\sigma^2$ term.** This is precisely why Random Forest performance plateaus with enough trees rather than continuing to improve indefinitely, and why decorrelating the trees (lowering $\rho$) is a more powerful lever than simply adding more of them.

**Worked numerical:** single-tree variance $\sigma^2 = 4.0$, pairwise correlation between bagged trees $\rho=0.3$.

At $M=10$ trees:
$$
\text{Var} = 0.3(4.0) + \frac{(1-0.3)(4.0)}{10} = 1.2 + \frac{2.8}{10} = 1.2+0.28=1.48
$$
At $M=100$ trees:
$$
\text{Var} = 1.2 + \frac{2.8}{100} = 1.2+0.028=1.228
$$
At $M\to\infty$: variance floors at exactly $1.2$ ($=\rho\sigma^2$). Going from 10→100 trees only bought a variance reduction of $1.48-1.228=0.252$; going from 100→$\infty$ trees can buy at most another $0.028$. **Diminishing returns are baked into the formula itself** — this is why in practice Random Forest/bagging performance curves flatten hard well before hundreds of trees, and why tuning `max_features` (which controls $\rho$) matters more than cranking `n_estimators` past the flattening point.

**Why does this derivation assume equal variance $\sigma^2$ and equal pairwise correlation $\rho$ across all model pairs?** It's a simplifying symmetry assumption (all trees trained the "same way," just on different resampled data) that keeps the algebra clean; it's not exactly true in practice (some bootstrap samples are luckier than others), but it captures the dominant real effect well enough that this formula (from Hastie, Tibshirani & Friedman's *Elements of Statistical Learning*, Ch. 15) is the standard textbook justification for Random Forests, and the qualitative conclusion — decorrelating matters more than raw count — holds even when the symmetry is only approximate.

---

## 2.3 The Three Ensembling Families — What Each One Actually Attacks

| | Bagging | Boosting | Stacking |
|---|---|---|---|
| Model training | **Parallel** — each base model trained independently, simultaneously | **Sequential** — each new model explicitly corrects the previous ensemble's errors | Base models can be parallel; meta-learner trained after, on their outputs |
| What it primarily reduces | **Variance** (per 2.2's derivation) | **Bias** (each new weak learner is fit specifically to what's still wrong) | Neither directly — it learns the *optimal way to combine* diverse models, which can improve on both |
| Base learner requirement | Should be low-bias, high-variance (e.g., deep/unpruned trees) — bagging a high-bias model barely helps, since averaging doesn't touch bias | Should be high-bias, low-variance ("weak learners," e.g., shallow trees/"stumps") — boosting a low-bias model risks overfitting fast, since sequential correction has no natural brake on chasing noise | Mix of anything, ideally models that are individually decent but make *different kinds* of errors |
| Overfitting behavior | Adding more base models does **not** increase overfitting risk (Ch.3 will show this formally) | Adding more rounds **can** increase overfitting risk — needs explicit regularization (learning rate, early stopping, Ch.5) | Depends on meta-learner; leakage from improperly-validated base predictions is the main risk (Ch.6) |
| Chapter | 3–4 | 5 | 6 |

**Why does this pairing (bagging↔high-variance learners, boosting↔high-bias learners) hold, precisely?** It falls directly out of 2.2's math plus the mirror-image logic for boosting: bagging's variance-reduction mechanism has nothing to offer a model that's already low-variance (there's little variance left to average away), and if that same low-variance model is high-bias, bagging leaves the bias completely untouched (as shown in 2.2 — averaging preserves bias exactly) — so bagging a high-bias/low-variance model like a shallow linear model produces almost no improvement. Boosting, conversely, is explicitly built to iteratively *reduce residual error* (i.e., attack bias) — it works best when each individual base learner is weak/high-bias precisely because a low-bias base learner would already fit the training data closely on round 1, leaving boosting's sequential correction mechanism nothing meaningful to do (and everything left to correct would just be noise, which boosting would then overfit to, since it has no built-in mechanism to distinguish signal from noise in the residuals).

---

## 2.4 Bootstrap Sampling Theory

Bagging's mechanism (Ch.3 in full) depends on generating $M$ different training sets from one original dataset via **bootstrap sampling**: draw $n$ samples **with replacement** from the original $n$-sample dataset, so each bootstrap sample is the same size as the original but contains duplicates and omissions.

**The question that matters for Out-of-Bag estimation (Ch.3.3): what fraction of the original data is *not* selected into a given bootstrap sample?**

For a dataset of size $n$, the probability that any *specific* sample $i$ is **not** picked on a single draw is $\left(1-\frac{1}{n}\right)$ (there are $n$ equally likely samples to draw, only 1 of which is sample $i$). Since we draw $n$ times independently (with replacement, so each draw is independent of the others):

$$
P(\text{sample } i \text{ never picked in } n \text{ draws}) = \left(1-\frac{1}{n}\right)^n
$$

**Worked numerical, small $n$:** for $n=10$:
$$
\left(1-\frac{1}{10}\right)^{10} = (0.9)^{10}
$$
$(0.9)^2=0.81$, $(0.9)^4 = 0.81^2=0.6561$, $(0.9)^8=0.6561^2=0.4305$, $(0.9)^{10}=(0.9)^8\times(0.9)^2=0.4305\times0.81=0.3487$

So with $n=10$, about **34.9%** of samples are left out of a given bootstrap draw.

**The limiting case, $n\to\infty$:** this is a textbook calculus limit —
$$
\lim_{n\to\infty}\left(1-\frac{1}{n}\right)^n = e^{-1} \approx 0.3679
$$
(This follows from the standard identity $\lim_{n\to\infty}(1+\frac{x}{n})^n = e^x$ with $x=-1$.)

**So for any reasonably large dataset, ~36.8% of the original samples are excluded from any given bootstrap sample — equivalently, each bootstrap sample contains, in expectation, only about 63.2% of the unique original samples** (the rest of its $n$ slots filled with duplicates of the ones that *were* selected). This ~63.2%/36.8% split is a fixed, dataset-size-independent constant (for reasonably large $n$) — worth memorizing directly, since it's a very common interview number-check ("what fraction of data is in a bootstrap sample on average?" → ~63%, "what fraction is out-of-bag?" → ~37%).

**Why does this matter beyond trivia?** That consistent ~37% "held out" chunk per tree is exactly what Chapter 3.3 turns into **Out-of-Bag (OOB) error estimation** — a way to get an honest validation-like error estimate *without* sacrificing any training data to a separate holdout set, because each original sample is, on average, left out of about 37% of the trees in a bagged ensemble and can be validated on exactly that subset of trees that never saw it during training.

---

## Quick Interview Q&A

**Q: Does averaging more models always reduce test error?**
A: Averaging always reduces *variance* (monotonically, per the 2.2 formula — variance strictly decreases as $M$ increases as long as $\rho<1$), and never changes bias. So total error only goes down if variance was actually the dominant problem to begin with. If your base model is high-bias/low-variance, averaging more of them barely moves total error, since you're shrinking a term that was already small.

**Q: Why can't boosting just use the same "average many high-variance models" trick that bagging uses?**
A: Because boosting's error-reduction mechanism is fundamentally about sequentially targeting *what's currently wrong* (residuals/reweighted misclassifications) — that requires a controlled, incremental correction signal, which needs each step to be a weak, high-bias adjustment. Feeding boosting a high-variance base learner (like an unpruned tree) means round 1 already overfits hard, and every subsequent round is then "correcting" residuals that are mostly noise, compounding overfitting rather than reducing bias in a controlled way.

**Q: If OOB gives a free validation estimate, why does anyone still use k-fold CV with bagging/Random Forests?**
A: OOB error is a valid, nearly-free approximation of leave-one-out CV for bagged models specifically, but it's tied to the bagging mechanism itself (you need the bootstrap resampling structure to define "out-of-bag" at all) — for hyperparameter search across configurations that aren't just "how many trees," or for comparing against non-bagged models on a level playing field, an explicit k-fold CV setup is still the more standard, comparable protocol.

---

**Next up: Chapter 3 — Bagging in full detail (the algorithm, OOB error walked through numerically, and precisely when it does/doesn't help).**
