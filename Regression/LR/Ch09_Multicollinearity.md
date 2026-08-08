# Chapter 9 — Multicollinearity

*Synthesized from Kutner, Montgomery, Sheather, and ESL/ISL. Formalizes the instability first observed in Chapter 5, §5.4 and Chapter 3, §3.8, using the same $x_1$ (hours studied), $x_2$ (practice tests) predictors.*

---

## 9.1 The Motivating Question

Chapter 5 found something strange: the overall F-test was overwhelmingly significant ($F\approx279.5$), yet the individual t-test for $\hat{\beta}_2$ failed to reject at $\alpha=0.05$. Chapter 3 (§3.8) predicted mathematically *why* this can happen: when predictors are correlated with each other, $\mathbf{X}^T\mathbf{X}$ becomes close to non-invertible, inflating the variance of individual coefficient estimates.

This chapter builds the formal diagnostic tools to **detect and quantify** this problem — turning "the coefficients seemed unstable" into a precise, standard, interview-ready number.

---

## 9.2 What Multicollinearity Actually Is (and Isn't)

**Multicollinearity** is a strong linear relationship among two or more predictors in $\mathbf{X}$. **Perfect multicollinearity** (one predictor is an exact linear function of others) makes $\mathbf{X}^T\mathbf{X}$ singular — no unique OLS solution exists at all. **Near-multicollinearity** (strong but imperfect correlation) is the far more common real-world problem, and the one this chapter's diagnostics target.

**What it does NOT affect:** predictions and overall model fit ($R^2$, $\hat{y}$) remain reliable even under severe multicollinearity — the model as a whole can still predict well. **What it DOES affect:** the precision and interpretability of *individual* coefficients — exactly the asymmetry observed in Chapter 5. This distinction is worth stating explicitly in an interview, since it's commonly misunderstood as "multicollinearity ruins your model," when it more precisely "ruins your ability to interpret individual coefficients confidently."

---

## 9.3 Variance Inflation Factor (VIF) — Definition and Intuition

For each predictor $x_j$, regress it on **all the other predictors** in the model, and take the resulting $R_j^2$:

$$ VIF_j = \frac{1}{1-R_j^2} $$

**Plain-English reading:** $R_j^2$ measures how well the *other* predictors can already predict $x_j$ on their own. If $x_j$ is almost entirely explainable by the other predictors ($R_j^2$ near 1), then $x_j$ contributes very little *new, independent* information — and the model has to work hard (with correspondingly higher variance) to tease apart $x_j$'s own unique contribution. $VIF_j$ measures exactly how much the variance of $\hat{\beta}_j$ is "inflated" relative to a hypothetical world where $x_j$ were uncorrelated with everything else: $VIF_j=1$ means no inflation at all; $VIF_j=10$ means $\hat{\beta}_j$'s variance is 10 times larger than it would be if $x_j$ were independent of the other predictors.

---

## 9.4 Worked Example — Computing VIF by Hand

With only two predictors, $R_j^2$ (regressing one predictor on the other) is the same in both directions and equals the squared correlation between them.

**Step 1 — regress $x_2$ on $x_1$** (same mechanics as Chapter 1's simple regression, applied to the predictors instead of $y$):

$\bar{x}_1=3$, $\bar{x}_2=1.8$, $S_{x_1x_1}=10$

$$ S_{x_1x_2} = \sum(x_1-\bar{x}_1)(x_2-\bar{x}_2) = 1.6+0.8+0+0.2+2.4 = 5.0 $$

Slope $=S_{x_1x_2}/S_{x_1x_1}=5/10=0.5$; intercept $=1.8-0.5(3)=0.3$. Fitted: $\hat{x}_2=0.3+0.5x_1$.

**Step 2 — compute $R_2^2$:**

$$ SSE = \sum(x_2-\hat{x}_2)^2 = 0.2^2+(-0.3)^2+0.2^2+(-0.3)^2+0.2^2 = 0.30 $$

$$ SST = \sum(x_2-\bar{x}_2)^2 = 2.8 $$

$$ R_2^2 = 1-\frac{0.30}{2.8} = 1-0.1071 = 0.8929 $$

**Step 3 — compute VIF:**

$$ VIF_2 = \frac{1}{1-0.8929} = \frac{1}{0.1071} \approx 9.33 $$

By symmetry (with exactly two predictors, the $R^2$ from regressing either one on the other is identical, both equal to the squared correlation between them), $VIF_1=VIF_2\approx9.33$ as well.

**Interpretation against common thresholds:** $VIF > 5$ is often treated as worth attention; $VIF > 10$ as a clear red flag. At **9.33**, this dataset sits right at the edge of serious concern — directly confirming Chapter 5's individual-t-test symptom with a precise, standard number instead of just an observed anomaly.

---

## 9.5 Condition Number — A Complementary Diagnostic

VIF diagnoses one predictor at a time. **Condition number** looks at the *overall* stability of $\mathbf{X}^T\mathbf{X}$ using its eigenvalues:

$$ \kappa = \sqrt{\frac{\lambda_{max}}{\lambda_{min}}} $$

where $\lambda_{max}, \lambda_{min}$ are the largest and smallest eigenvalues (of the standardized/correlation form of $\mathbf{X}^T\mathbf{X}$, to keep the result unit-free).

**Worked example:** using the $2\times2$ correlation matrix between $x_1,x_2$ (correlation $r=S_{x_1x_2}/\sqrt{S_{x_1x_1}S_{x_2x_2}}=5/\sqrt{10\times2.8}=5/5.29\approx0.945$):

$$ \mathbf{R} = \begin{bmatrix}1 & 0.945\\0.945&1\end{bmatrix}, \qquad \lambda = 1\pm0.945 = \{1.945,\ 0.055\} $$

$$ \kappa = \sqrt{1.945/0.055} = \sqrt{35.4} \approx 5.95 $$

**A caveat worth stating plainly in an interview:** different textbooks compute "condition number" with different conventions (this simplified correlation-matrix version for two predictors, vs. Belsley's more involved scaled-design-matrix approach including the intercept, which is standard software's default and typically flags severe multicollinearity above roughly 30). These conventions don't always agree on an exact numeric cutoff — the VIF diagnostic in §9.4 is the more standardized, more commonly interview-tested tool for this reason; condition number is worth recognizing conceptually and being able to compute in the simplified two-predictor case, but treat absolute thresholds with some caution unless you know which convention is being used.

---

## 9.6 Remedies for Multicollinearity

In rough order of preference:

1. **Drop one of the correlated predictors**, if theoretically justified (e.g., if $x_2$ is nearly redundant given $x_1$, and both aren't independently essential to the research question).
2. **Combine correlated predictors** into a single composite (e.g., a combined "study effort" index from hours studied and practice tests), if that's a defensible construct.
3. **Center the predictors** (subtract their means) before creating interaction or polynomial terms (Chapter 13) — this specifically reduces a form of *artificial* multicollinearity introduced by the modeling choice itself, not the underlying data.
4. **Collect more data**, ideally more varied in the predictors — multicollinearity is fundamentally a property of the *observed sample*, and a differently-sampled dataset with more spread in $x_1,x_2$ independently could reduce $R_j^2$.
5. **Ridge regression** (Chapter 16) — directly designed to stabilize coefficient estimates under multicollinearity by trading a small amount of bias for a large reduction in variance, exactly the scenario previewed in Chapter 6's Gauss-Markov discussion.

**What NOT to do:** don't simply drop a predictor purely because its individual t-test wasn't significant (as with $\hat{\beta}_2$ in Chapter 5) without first checking VIF — an insignificant t-test under high multicollinearity doesn't mean the predictor is truly unimportant, only that its *individual* effect is hard to isolate given the current data.

---

## 9.7 Where the Textbooks Differ

- **Kutner** derives VIF's connection to the variance-covariance matrix most rigorously, directly tying it back to the $(\mathbf{X}^T\mathbf{X})^{-1}$ diagonal entries from Chapter 3.
- **Montgomery** is the strongest source on condition number and eigenvalue-based diagnostics, being an industrial-statistics text where design matrices are often deliberately structured (design of experiments) to avoid collinearity in the first place.
- **Sheather** emphasizes reading VIF directly from software (`vif()` in R), and demonstrates multicollinearity's effects via simulation — showing coefficient estimates swinging wildly across simulated resamples of correlated predictors.
- **ESL/ISL** treat multicollinearity mainly as *motivation* for regularization — their multicollinearity discussion is brief and exists primarily to set up why ridge regression's $\lambda(\mathbf{X}^T\mathbf{X}+\lambda\mathbf{I})^{-1}$ correction (Chapter 16) directly addresses the near-singularity problem at its algebraic root.

---

## 9.8 Interview Q&A

**Q: What does a VIF of 10 mean, precisely?**
A: The variance of that predictor's coefficient estimate is 10 times larger than it would be if that predictor were completely uncorrelated with the other predictors in the model — a direct measure of how much multicollinearity is inflating your uncertainty about that specific coefficient.

**Q: Does multicollinearity bias your coefficient estimates?**
A: No — OLS remains unbiased under multicollinearity (Gauss-Markov, Chapter 6, still holds). The problem is inflated variance, not bias — coefficients become unstable and imprecise, not systematically wrong on average.

**Q: If VIF is high for a predictor, should you always drop it?**
A: Not automatically — first consider whether the predictor is theoretically essential, whether combining it with the correlated predictor makes sense, or whether ridge regression better serves the goal of stable prediction without discarding information.

**Q: Can a model have severe multicollinearity and still predict well?**
A: Yes — overall predictive accuracy and $R^2$ are largely unaffected by multicollinearity; only individual coefficient interpretation and precision suffer. If prediction (not coefficient interpretation) is the sole goal, multicollinearity is often much less of a practical concern.

**Q: How does VIF relate to the variance-covariance matrix from Chapter 3?**
A: $VIF_j$ is exactly the diagonal entry of $(\mathbf{X}^T\mathbf{X})^{-1}$ for a *standardized* design matrix (predictors centered and scaled) — it's a direct, interpretable rescaling of the same quantity that determines $\text{Var}(\hat{\boldsymbol{\beta}})=\sigma^2(\mathbf{X}^T\mathbf{X})^{-1}$.

---

*End of Chapter 9. Next: Chapter 10 — Heteroscedasticity (Breusch-Pagan and White tests, Weighted Least Squares, and robust/sandwich standard errors as three different ways of handling unequal error variance).*
