# Chapter 6 — The Gauss-Markov Theorem (Why OLS Is BLUE)

*Synthesized from Kutner, Montgomery, Sheather, and ESL/ISL.*

---

## 6.1 The Motivating Question

Every chapter so far has just *used* OLS without asking a foundational question: **out of every possible way to estimate $\beta_0, \beta_1, ...$ from data, why is "minimize squared error" the right choice — is there some cleverer estimator that would do better?**

The Gauss-Markov theorem answers this precisely, within a specific, well-defined class of alternatives. It's the theoretical justification for everything you've computed so far — without it, OLS would just be "a reasonable-sounding idea," not a provably optimal method.

---

## 6.2 What "BLUE" Means — Every Letter Explained

The theorem states OLS is **BLUE**:

| Letter | Meaning | Plain-English translation |
|---|---|---|
| **B** | Best | Lowest variance among the class described below — "most precise" |
| **L** | Linear | The estimator is a linear function of the observed $y_i$'s — i.e., it can be written as $\sum c_i y_i$ for some fixed weights $c_i$ |
| **U** | Unbiased | On average, across repeated sampling, the estimator equals the true parameter — $E[\hat{\beta}]=\beta$ |
| **E** | Estimator | It's an estimator (of $\beta_0,\beta_1,...$), not a hypothesis test or anything else |

**The precise claim:** among *all* estimators that are both linear in $y$ and unbiased, OLS has the smallest possible variance. Note carefully what this does **not** claim: it does not say OLS beats every conceivable estimator (biased estimators, like ridge regression in Chapter 16, can have *lower* total error by deliberately trading a little bias for a lot less variance). It's optimal only *within the linear-and-unbiased category*.

---

## 6.3 The Required Assumptions

Gauss-Markov requires exactly:

1. **Linearity** — $E[y_i]=\beta_0+\beta_1x_{i1}+...$ (the mean structure is correctly specified as linear)
2. **Zero-mean errors** — $E[\varepsilon_i]=0$
3. **Homoscedasticity** — $\text{Var}(\varepsilon_i)=\sigma^2$, the same for every observation
4. **No autocorrelation** — $\text{Cov}(\varepsilon_i,\varepsilon_j)=0$ for $i\neq j$

**Critical point, frequently tested in interviews:** **normality of the errors is *not* required for Gauss-Markov.** This theorem is purely about mean and variance — it holds regardless of the error distribution's shape. Normality only becomes necessary later, when you want exact t-distributions and F-distributions for hypothesis testing (Chapters 2 and 5) rather than just point estimates. This is the same estimation-vs-inference distinction from Chapter 1, §1.8, now formalized at the theorem level.

---

## 6.4 Proof Sketch — Necessary Steps Only

The full proof (found nearly identically across Kutner, Montgomery, and any mathematical statistics text) proceeds in three stages. We show the structure and key algebraic moves without every intermediate line.

**Setup:** Let $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ be the OLS estimator. Consider any *other* linear unbiased estimator $\tilde{\boldsymbol{\beta}} = \mathbf{C}\mathbf{y}$ for some matrix $\mathbf{C}$ (this is what "linear in y" means — a fixed weighting matrix applied to the data).

**Step 1 — Express $\mathbf{C}$ relative to the OLS weighting.** Write $\mathbf{C} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T + \mathbf{D}$ for some matrix $\mathbf{D}$ — i.e., any other linear estimator is "OLS's weighting plus some deviation $\mathbf{D}$." This is the key trick: instead of comparing two unrelated estimators, express the competitor as OLS-plus-a-correction, so the correction term is what we need to show is unhelpful (must be exactly zero for unbiasedness).

**Step 2 — Enforce unbiasedness.** For $\tilde{\boldsymbol{\beta}}=\mathbf{C}\mathbf{y}$ to be unbiased for every possible true $\boldsymbol{\beta}$, algebra forces $\mathbf{D}\mathbf{X}=0$ — the deviation matrix must be **orthogonal to the design matrix**. This is a strong constraint: $\mathbf{D}$ can't correlate at all with the predictors.

**Step 3 — Compare variances.** Using $\mathbf{D}\mathbf{X}=0$, the variance of $\tilde{\boldsymbol{\beta}}$ works out to:

$$ \text{Var}(\tilde{\boldsymbol{\beta}}) = \text{Var}(\hat{\boldsymbol{\beta}}) + \sigma^2\mathbf{D}\mathbf{D}^T $$

Since $\mathbf{D}\mathbf{D}^T$ is a matrix of squared terms, it's always **positive semi-definite** — meaning it can only ever *add* nonnegative variance, never subtract any. Therefore:

$$ \text{Var}(\tilde{\boldsymbol{\beta}}) \geq \text{Var}(\hat{\boldsymbol{\beta}}) $$

for every possible choice of $\mathbf{D}$, with equality only when $\mathbf{D}=0$ (i.e., $\tilde{\boldsymbol{\beta}}$ **is** OLS). **This is the entire proof:** any deviation from OLS's weighting either breaks unbiasedness or strictly increases variance — there's no way to do better while staying linear and unbiased.

---

## 6.5 A Concrete Illustration Using the Running Dataset

Go back to Chapter 1's simple regression ($\hat{\beta}_1=S_{xy}/S_{xx}=7.5$). Suppose someone proposed an alternative, *also* linear and unbiased estimator — for instance, weighting the middle observation ($x=3$) more heavily than the OLS formula does, on the theory that "the middle student is most typical." This modified weighting can still be constructed to remain unbiased (as long as the weights satisfy the constraints from Step 2), but Gauss-Markov guarantees, *without needing to check this specific alternative numerically*, that its variance must be at least as large as $SE(\hat{\beta}_1)^2 = 0.5^2 = 0.25$ (from Chapter 2, §2.4) — and strictly larger unless the alternative weighting collapses back to being identical to OLS's own formula. This is the practical payoff of the theorem: **you never need to hand-check a competing linear unbiased estimator's variance — Gauss-Markov guarantees OLS already wins.**

---

## 6.6 Why This Theorem Has Limits (Setting Up Later Chapters)

Three important boundaries of the theorem, each previewing later material:

- **"Best" only within linear + unbiased.** A *biased* estimator can have lower total error (measured as Mean Squared Error = variance + bias²) if it trades a small, controlled amount of bias for a large reduction in variance. This is exactly the philosophy behind **ridge regression** (Chapter 16) — deliberately biased, but often lower total error, especially under multicollinearity (which, per Chapter 5, inflates OLS variance substantially).
- **Requires homoscedasticity.** If errors have unequal variance (heteroscedasticity, Chapter 10), OLS remains *unbiased* but is **no longer BLUE** — a different linear unbiased estimator (Weighted Least Squares, previewed in Chapter 19) achieves lower variance in that setting instead.
- **Requires no autocorrelation.** If errors are correlated across observations (common in time-series data, Chapter 11), Generalized Least Squares (also Chapter 19) becomes the new BLUE estimator instead of ordinary OLS.

**The unifying interview insight:** Gauss-Markov isn't a permanent crown for OLS — it's a conditional guarantee that only holds while its four assumptions hold. Every later chapter on diagnostics and generalized methods is really asking "which Gauss-Markov assumption is violated here, and what's the new BLUE estimator once it is?"

---

## 6.7 Where the Textbooks Differ

- **Kutner** gives the most complete formal proof, nearly identical in structure to §6.4 above, typically presented immediately after the matrix formulation chapter — treating it as the theoretical capstone of the estimation section.
- **Montgomery** states the theorem with a lighter proof sketch and spends more relative time on the *practical implications* — when each assumption is likely violated in real engineering/quality-control data, and what to do about it.
- **Sheather** treats Gauss-Markov almost as a known result to cite rather than re-derive, spending more space on simulation-based demonstrations (showing empirically, via repeated simulated datasets, that OLS's variance is smallest among several candidate estimators) rather than the algebraic proof.
- **ESL/ISL** invoke Gauss-Markov briefly, mainly to set up the bias-variance tradeoff discussion that motivates regularization — for them, the theorem's real purpose is as the *departure point* for justifying ridge/lasso, not a destination in itself.

---

## 6.8 Interview Q&A

**Q: What does BLUE stand for, and what exactly does it guarantee?**
A: Best Linear Unbiased Estimator — OLS has the minimum variance among all estimators that are both linear functions of $y$ and unbiased for $\beta$. It does not guarantee OLS beats every possible estimator, only ones within that linear-unbiased class.

**Q: Does Gauss-Markov require the errors to be normally distributed?**
A: No. It only requires zero mean, constant variance, and no autocorrelation among errors. Normality is needed for exact-distribution hypothesis testing (t-tests, F-tests), not for the BLUE property itself.

**Q: Can a biased estimator ever outperform OLS?**
A: Yes, in total Mean Squared Error (bias² + variance) — this is exactly the justification for ridge regression, which deliberately introduces bias to substantially reduce variance, especially valuable under multicollinearity where OLS's variance (Chapter 5, §5.4) can be very large.

**Q: What happens to the BLUE property if errors are heteroscedastic?**
A: OLS remains unbiased but is no longer the minimum-variance linear unbiased estimator — Weighted Least Squares becomes BLUE instead, since it appropriately down-weights noisier observations.

**Q: In one sentence, what's the core trick in the Gauss-Markov proof?**
A: Write any competing linear unbiased estimator as "OLS's weights plus a deviation," show unbiasedness forces that deviation to be orthogonal to the design matrix, and show that orthogonality condition makes the deviation's contribution to variance always nonnegative — so it can only ever increase variance, never decrease it below OLS's own.

---

*End of Chapter 6. Next: Chapter 7 — Diagnostics I: Residual Analysis (standardized and studentized residuals, the four-panel residual plot diagnostic, and formally connecting each diagnostic pattern back to which Gauss-Markov assumption it signals a violation of).*
