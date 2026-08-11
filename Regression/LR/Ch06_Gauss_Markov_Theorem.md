# Chapter 6 — The Gauss-Markov Theorem (Why OLS Is BLUE) — Interview-Boosted Edition

*Synthesized from Kutner, Montgomery, Sheather, and ESL/ISL.*

---

## 6.0 The 60-Second Answer (say this if put on the spot)

> "Gauss-Markov says OLS has the smallest variance among all estimators that are **linear in $y$** and **unbiased**. It needs four conditions: linearity, zero-mean errors, constant error variance (homoscedasticity), and no correlation between errors. It does **not** need normal errors — normality only matters later, for exact t-tests and F-tests. And it's a 'best in its weight class' result, not 'best ever' — a *biased* estimator like ridge regression can still beat OLS on total error by trading a bit of bias for a lot less variance."

That's the whole chapter in one paragraph. Everything below is the "why" behind each sentence, plus drills to make it automatic.

---

## 6.1 The Motivating Question

Every chapter so far just *used* OLS without asking: **out of every way to estimate $\beta_0,\beta_1,...$, why is "minimize squared error" the right choice?**

You've been fitting the best-fit line by minimizing squared errors — but why is that the *correct* rule, instead of some other formula someone could invent? The Gauss-Markov theorem is the proof that settles this: it tells you exactly when OLS is guaranteed best, and exactly what "best" means. Without it, OLS is just "a reasonable-sounding idea," not a provably optimal method.

---

## 6.2 What "BLUE" Means

| Letter | Meaning | In one line |
|---|---|---|
| **B** | Best | Lowest variance — most precise |
| **L** | Linear | The estimator is a weighted sum of the $y_i$'s: $\sum c_iy_i$ |
| **U** | Unbiased | On average, over repeated sampling, it hits the true value: $E[\hat\beta]=\beta$ |
| **E** | Estimator | It's estimating a number, not running a hypothesis test |

- **Best = tightest cluster around the truth.** Re-run the experiment many times, re-estimate $\beta$ each time — "best" means those estimates bounce around the least.
- **Linear = a weighted sum of your data.** $c_1y_1+c_2y_2+...+c_ny_n$ for fixed weights. OLS itself has this shape (weights come from $(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$).
- **Unbiased = correct on average, not necessarily on any one sample.**

**The precise claim:** among *all* linear, unbiased estimators, OLS has the smallest variance. It does **not** claim OLS beats every conceivable estimator — a biased estimator (ridge, Chapter 16) can win on total error.

**One-line summary:** OLS is the champion of one weight class (linear + unbiased) — other weight classes have different champions.

---

## 6.3 The Four Required Assumptions

| # | Formal statement | Plain meaning |
|---|---|---|
| 1 | Linearity: $E[y_i]=\beta_0+\beta_1x_{i1}+...$ | The true relationship really is a straight line, not secretly curved |
| 2 | Zero-mean errors: $E[\varepsilon_i]=0$ | Mistakes don't systematically lean positive or negative |
| 3 | Homoscedasticity: $\text{Var}(\varepsilon_i)=\sigma^2$ for all $i$ | Every point is equally "noisy" — the scatter around the line is equally thick everywhere |
| 4 | No autocorrelation: $\text{Cov}(\varepsilon_i,\varepsilon_j)=0$ | One point's error doesn't leak into or predict another's |

**The trap interviewers set:** "regression assumes normal errors" is only half-true. **Normality is not one of the four Gauss-Markov conditions.** You only need it later, to get *exact* t- and F-distributions for hypothesis testing (Chapters 2, 5). For the *best point estimate* alone, the four conditions above are all you need — no bell curve required.

---

## 6.4 Proof — Compressed to the Essential Moves

**The one-paragraph strategy:** take any other linear, unbiased estimator. Rewrite it as "whatever OLS does, plus a correction term." Unbiasedness forces that correction to be blind to the predictor data — it can't lean on $\mathbf{X}$ at all. And once it's blind to $\mathbf{X}$, that correction can only ever *add* variance, never remove it. So the best you can do is make the correction exactly zero — i.e., just use OLS.

**Step 1 — relabel any rival estimator as "OLS plus a deviation."** Let $\tilde{\boldsymbol\beta}=\mathbf{C}\mathbf{y}$ be any linear estimator. Write $\mathbf{C}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T+\mathbf{D}$. This is pure bookkeeping — $\mathbf{D}$ is just defined as "whatever's left after subtracting OLS's part," so it always works.

**Step 2 — unbiasedness forces $\mathbf{D}\mathbf{X}=0$.** If $\mathbf{D}$ picked up any signal from $\mathbf{X}$, it would tilt the estimate away from the truth on average. So the only way $\mathbf{D}$ can exist without breaking unbiasedness is if it's completely orthogonal to $\mathbf{X}$.

**Step 3 — the variance splits into two nonnegative pieces:**

$$ \text{Var}(\tilde{\boldsymbol\beta}) = \text{Var}(\hat{\boldsymbol\beta}) + \sigma^2\mathbf{D}\mathbf{D}^T $$

$\mathbf{D}\mathbf{D}^T$ is built from squared terms, so it's always positive semi-definite — it can only add variance, never subtract it. So $\text{Var}(\tilde{\boldsymbol\beta})\ge\text{Var}(\hat{\boldsymbol\beta})$ always, with equality only when $\mathbf{D}=0$, i.e., when the "rival" is just OLS in disguise.

**One-line version for a whiteboard:** *any competitor to OLS is secretly "OLS + extra noise," and extra noise can never be negative — so nothing can beat it while staying linear and unbiased.*

---

## 6.5 A Concrete Illustration

Recall Chapter 1's simple regression: $\hat\beta_1=S_{xy}/S_{xx}=7.5$, with $SE(\hat\beta_1)=0.5$, so $\text{Var}(\hat\beta_1)=0.25$ (Chapter 2, §2.4).

Suppose someone proposes weighting the middle student ($x=3$) more heavily, arguing "the middle student is most typical." You could build this so it's still unbiased. But Gauss-Markov guarantees, **without checking this specific idea numerically**, that its variance is $\ge0.25$ — strictly larger unless it collapses back to being identical to OLS.

**The payoff:** you never have to hand-check a competing linear unbiased estimator's variance. Gauss-Markov already guarantees OLS wins. A formula that "feels smarter" because it trusts one point more doesn't actually buy you anything — the theorem rules it out in advance.

---

## 6.6 Where the Theorem Stops Applying (the three escape hatches)

| If this happens... | OLS is still... | But it's no longer BLUE — instead... |
|---|---|---|
| You accept a little bias on purpose | biased, on purpose | **Ridge regression** (Ch. 16) can win on total error (bias² + variance) by cutting variance a lot for a little bias — especially valuable under multicollinearity, where OLS's variance can already be large (Ch. 5, §5.4) |
| Errors have unequal variance (heteroscedasticity) | unbiased | **Weighted Least Squares** (Ch. 19) becomes BLUE — it trusts less-noisy points more |
| Errors are correlated across observations (e.g. time series) | unbiased | **Generalized Least Squares** (Ch. 19) becomes BLUE instead |

**The unifying interview insight:** Gauss-Markov isn't a permanent crown — it's a conditional guarantee that holds only while its four assumptions hold. Every later diagnostics/generalized-methods chapter is really asking: *which assumption broke, and what's the new BLUE estimator once it did?*

**The one-liner to end on:** *Gauss-Markov is a "best in class" trophy, not a "best overall" trophy — and the class only exists as long as its four rules are being followed.*

---

## 6.7 Where the Textbooks Differ

- **Kutner** gives the fullest formal proof, structured like §6.4, right after the matrix-formulation chapter — the theoretical capstone of estimation.
- **Montgomery** gives a lighter proof sketch, spending more time on which assumptions tend to break in real engineering/QC data.
- **Sheather** treats it as a known result to cite, leaning on simulation demonstrations rather than algebra.
- **ESL/ISL** invoke it briefly, mainly to set up the bias-variance tradeoff that motivates ridge/lasso — for them it's a departure point, not a destination.

---

## 6.8 Common Interview Traps — Watch For These

- **"Does OLS need normal errors?"** → No. That's the single most common trap in this chapter. Normality is for exact hypothesis tests, not for BLUE.
- **"Is OLS always the best possible estimator?"** → No — only best *within linear + unbiased*. Say this explicitly; don't let "BLUE" sound like "unconditionally best."
- **"If a biased estimator has lower variance, does that violate Gauss-Markov?"** → No — Gauss-Markov never claims biased estimators can't have lower variance. It only compares OLS to *other unbiased* estimators.
- **"Does heteroscedasticity make OLS wrong?"** → No — OLS stays unbiased. It just stops being *minimum variance*. Don't say "OLS breaks"; say "OLS loses its BLUE property, but the WLS alternative exists."
- **"What's the actual mechanism in the proof?"** → If asked to sketch it, don't recite matrix algebra from memory — use the compressed one-liner from §6.4: any rival is "OLS plus noise," and noise can't be negative.

---

## 6.9 Rapid-Fire Flashcards (drill these out loud)

| Q | A |
|---|---|
| What does BLUE stand for? | Best Linear Unbiased Estimator |
| What are the 4 Gauss-Markov assumptions? | Linearity, zero-mean errors, homoscedasticity, no autocorrelation |
| Is normality required? | No — only for exact t/F tests, not for BLUE |
| Can a biased estimator beat OLS? | Yes, on total MSE (bias²+variance) — e.g. ridge regression |
| What happens under heteroscedasticity? | OLS stays unbiased, loses BLUE status; WLS becomes BLUE |
| What happens under autocorrelated errors? | OLS stays unbiased, loses BLUE status; GLS becomes BLUE |
| One-line proof intuition? | Any rival estimator = OLS + noise; noise variance can't be negative, so it can only hurt |
| What does "linear" mean in BLUE? | The estimator is a fixed weighted sum of the $y_i$'s |
| What's the practical payoff of the theorem? | You never need to hand-check a rival linear unbiased estimator — Gauss-Markov already guarantees OLS wins |

---

## 6.10 Interview Q&A (full answers)

**Q: What does BLUE stand for, and what exactly does it guarantee?**
A: Best Linear Unbiased Estimator — OLS has minimum variance among all estimators that are linear in $y$ and unbiased for $\beta$. It doesn't claim OLS beats every possible estimator, only ones within that linear-unbiased class.

**Q: Does Gauss-Markov require normally distributed errors?**
A: No. It needs zero mean, constant variance, and no autocorrelation. Normality is needed for exact hypothesis testing (t-tests, F-tests), not for BLUE itself.

**Q: Can a biased estimator ever outperform OLS?**
A: Yes, in total Mean Squared Error (bias² + variance) — the justification for ridge regression, which trades a bit of bias for a large variance reduction, especially valuable under multicollinearity.

**Q: What happens to BLUE if errors are heteroscedastic?**
A: OLS stays unbiased but is no longer minimum-variance — Weighted Least Squares becomes BLUE instead, since it down-weights noisier observations.

**Q: In one sentence, what's the core trick in the proof?**
A: Any competing linear unbiased estimator can be written as "OLS's weights plus a deviation"; unbiasedness forces that deviation to be orthogonal to the design matrix, which forces its contribution to variance to be nonnegative — so it can only increase variance, never decrease it below OLS's.

---

*End of Chapter 6 (interview-boosted). Next: Chapter 7 — Diagnostics I: Residual Analysis (standardized and studentized residuals, the four-panel residual plot diagnostic, and connecting each diagnostic pattern back to which Gauss-Markov assumption it signals a violation of).*
