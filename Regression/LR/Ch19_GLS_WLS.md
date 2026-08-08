# Chapter 19 — Generalized & Weighted Least Squares

*Synthesized from Kutner, Montgomery, Sheather, and ESL/ISL. This chapter unifies Chapter 10's WLS and Chapter 11's AR(1)/GLS material under one general derivation, using the matrix machinery from Chapter 3.*

---

## 19.1 The Motivating Question — One Formula, Two Chapters' Worth of Remedies

Chapters 10 and 11 each introduced a fix for a broken Gauss-Markov assumption: WLS for heteroscedasticity (unequal variance), GLS/Cochrane-Orcutt for autocorrelation (correlated errors). Both are actually the **same** underlying idea, differing only in what structure they assume for the error covariance matrix $\boldsymbol{\Sigma}=\text{Var}(\boldsymbol{\varepsilon})$. This chapter derives the single general formula properly and shows exactly how both prior chapters' remedies fall out of it as special cases.

---

## 19.2 The General GLS Estimator — Full Derivation

Suppose $\text{Var}(\boldsymbol{\varepsilon})=\boldsymbol{\Sigma}$, some known, possibly non-diagonal, positive-definite matrix (instead of OLS's assumed $\sigma^2\mathbf{I}$). The GLS objective **reweights** the squared-error criterion by $\boldsymbol{\Sigma}^{-1}$:

$$ RSS_{GLS}(\boldsymbol{\beta}) = (\mathbf{y}-\mathbf{X}\boldsymbol{\beta})^T\boldsymbol{\Sigma}^{-1}(\mathbf{y}-\mathbf{X}\boldsymbol{\beta}) $$

Following the identical matrix-calculus steps as Chapter 3, §3.4 (just with $\boldsymbol{\Sigma}^{-1}$ inserted between the two residual-vector factors), setting the derivative to zero gives the **generalized normal equations**:

$$ \mathbf{X}^T\boldsymbol{\Sigma}^{-1}\mathbf{X}\,\boldsymbol{\beta} = \mathbf{X}^T\boldsymbol{\Sigma}^{-1}\mathbf{y} $$

$$ \boxed{\hat{\boldsymbol{\beta}}_{GLS} = (\mathbf{X}^T\boldsymbol{\Sigma}^{-1}\mathbf{X})^{-1}\mathbf{X}^T\boldsymbol{\Sigma}^{-1}\mathbf{y}} $$

**Setting $\boldsymbol{\Sigma}=\sigma^2\mathbf{I}$ recovers ordinary OLS exactly** ($\hat{\boldsymbol{\beta}}_{GLS}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$, since the constant $\sigma^2$ cancels) — confirming GLS is a strict generalization of everything in Chapters 1–9, not a separate technique.

---

## 19.3 The "Whitening" Transformation — Why This Formula Is Really Just OLS in Disguise

Since $\boldsymbol{\Sigma}$ is positive-definite, it has a matrix square root: $\boldsymbol{\Sigma}^{-1}=\mathbf{P}^T\mathbf{P}$ for some matrix $\mathbf{P}$ (e.g., via Cholesky decomposition). Define **transformed variables** $\mathbf{y}^*=\mathbf{P}\mathbf{y}$, $\mathbf{X}^*=\mathbf{P}\mathbf{X}$. Then:

$$ \hat{\boldsymbol{\beta}}_{GLS} = (\mathbf{X}^{*T}\mathbf{X}^*)^{-1}\mathbf{X}^{*T}\mathbf{y}^* $$

**This is exactly the ordinary OLS formula, applied to the transformed data $\mathbf{y}^*,\mathbf{X}^*$.** The transformation $\mathbf{P}$ is called **whitening** because it converts the correlated/heteroscedastic errors $\boldsymbol{\varepsilon}$ into new errors $\mathbf{P}\boldsymbol{\varepsilon}$ with covariance $\mathbf{P}\boldsymbol{\Sigma}\mathbf{P}^T=\mathbf{P}(\mathbf{P}^T\mathbf{P})^{-1}\mathbf{P}^T=\mathbf{I}$ (up to a scalar) — i.e., plain, homoscedastic, uncorrelated "white noise," exactly satisfying OLS's original assumptions. **This is precisely why Chapter 11's Cochrane-Orcutt transformation ($y_t^*=y_t-\hat{\rho}y_{t-1}$) worked** — it's a specific, concrete instance of this same general whitening trick, applied to the particular AR(1) structure of $\boldsymbol{\Sigma}$.

---

## 19.4 Special Case 1 — Recovering WLS (Chapter 10)

When errors are heteroscedastic but **uncorrelated** with each other, $\boldsymbol{\Sigma}=\text{diag}(\sigma_1^2,...,\sigma_n^2)$ — diagonal, but not constant on the diagonal. Then $\boldsymbol{\Sigma}^{-1}=\text{diag}(1/\sigma_1^2,...,1/\sigma_n^2)=\mathbf{W}$, exactly the weight matrix from Chapter 10:

$$ \hat{\boldsymbol{\beta}}_{GLS} = (\mathbf{X}^T\mathbf{W}\mathbf{X})^{-1}\mathbf{X}^T\mathbf{W}\mathbf{y} = \hat{\boldsymbol{\beta}}_{WLS} $$

**Numerical check against Chapter 10.** Recall Chapter 10, §10.4 assumed $\text{Var}(\varepsilon_i)\propto x_{1i}$ (simple regression, $y$ on $x_1$ alone) and computed $\hat{\beta}_{1,WLS}\approx7.70$, $\hat{\beta}_{0,WLS}\approx41.5$ using weighted-mean formulas. Since $\boldsymbol{\Sigma}=\text{diag}(x_{1i})$ here, $\boldsymbol{\Sigma}^{-1}=\text{diag}(1/x_{1i})=\mathbf{W}$ exactly matches the weights $w_i=1/x_{1i}$ used there — **confirming Chapter 10's WLS computation was, all along, a special case of this chapter's general GLS formula, just worked through with simpler weighted-average arithmetic instead of full matrix notation.**

---

## 19.5 Special Case 2 — Recovering AR(1) GLS (Chapter 11)

When errors follow an AR(1) process ($\varepsilon_t=\rho\varepsilon_{t-1}+u_t$), $\boldsymbol{\Sigma}$ takes the banded Toeplitz form shown in Chapter 11, §11.4. Applying $\boldsymbol{\Sigma}^{-1}$ directly is what the Cochrane-Orcutt/Prais-Winsten transformation ($y_t^*=y_t-\hat{\rho}y_{t-1}$) implements in practice, avoiding the need to explicitly build and invert the full $n\times n$ Toeplitz matrix — a concrete instance of §19.3's whitening trick, using a computationally cheap shortcut specific to the AR(1) structure rather than a generic matrix square root.

---

## 19.6 GLS Is BLUE — The Aitken Theorem

Just as Gauss-Markov (Chapter 6) established OLS as BLUE under $\text{Var}(\boldsymbol{\varepsilon})=\sigma^2\mathbf{I}$, the **Aitken theorem** is the direct generalization: **under a known, general covariance structure $\boldsymbol{\Sigma}$, GLS is BLUE** — the minimum-variance linear unbiased estimator, among *all* linear unbiased estimators, not just among those that ignore the covariance structure. Plain OLS remains unbiased even when $\boldsymbol{\Sigma}\neq\sigma^2\mathbf{I}$, but it is no longer minimum-variance — GLS is the estimator that reclaims BLUE status once the true covariance structure is properly accounted for. This is the same logical pattern as Chapter 16's ridge regression discussion, but pointing the opposite direction: ridge deliberately steps *outside* the unbiased class to reduce variance; GLS stays *inside* the unbiased class and reclaims minimum variance by correctly modeling $\boldsymbol{\Sigma}$ rather than ignoring it.

---

## 19.7 Feasible GLS (FGLS) — What Happens When $\boldsymbol{\Sigma}$ Is Unknown

In practice, $\boldsymbol{\Sigma}$ is essentially never known exactly — it must be **estimated** from the data. **Feasible GLS** proceeds in two steps:

1. Fit an initial OLS model, and use its residuals to estimate the structure of $\boldsymbol{\Sigma}$ (e.g., regress squared residuals on predictors to estimate a heteroscedasticity pattern, as in Chapter 10's Breusch-Pagan auxiliary regression; or estimate $\hat{\rho}$ from residuals, as in Chapter 11's Durbin-Watson-based approach).
2. Plug the estimated $\hat{\boldsymbol{\Sigma}}$ into the GLS formula from §19.2 and refit.

**Important caveat worth stating in an interview:** FGLS is only **asymptotically** BLUE — with an *estimated* rather than *known* $\boldsymbol{\Sigma}$, the exact finite-sample optimality guarantee from the Aitken theorem no longer strictly holds, though FGLS is generally still a substantial practical improvement over plain OLS when the assumed error structure is reasonably close to correct. This mirrors the robust-standard-errors caveat from Chapters 10–11: there's often a genuine choice between "model the error structure and gain efficiency" (WLS/GLS/FGLS) versus "don't model it, but correct standard errors regardless" (sandwich/Newey-West estimators) — and which is preferable depends on how confident you are in the assumed structure.

---

## 19.8 Where the Textbooks Differ

- **Kutner** presents GLS as the unifying unifying framework precisely as done here, deriving WLS as an explicit special case within the same chapter — the closest match to this chapter's structure.
- **Montgomery** treats WLS and time-series-correlated-error remedies in genuinely separate chapters (mirroring this curriculum's Chapters 10–11), with GLS as an underlying concept mentioned but less central than in Kutner.
- **Sheather** emphasizes FGLS in practice, since known $\boldsymbol{\Sigma}$ is rare in applied work — the two-step estimate-then-refit procedure is presented as the practical default.
- **ESL/ISL** essentially don't cover GLS as such — it's a classical-inference-focused topic outside their more prediction/ML-centric scope; weighted loss functions do appear in some ML contexts (e.g., weighted regression for imbalanced data) but without the classical GLS/Aitken-theorem framing.

---

## 19.9 Interview Q&A

**Q: Write the general GLS estimator and show how it reduces to OLS when $\boldsymbol{\Sigma}=\sigma^2\mathbf{I}$.**
A: $\hat{\boldsymbol{\beta}}_{GLS}=(\mathbf{X}^T\boldsymbol{\Sigma}^{-1}\mathbf{X})^{-1}\mathbf{X}^T\boldsymbol{\Sigma}^{-1}\mathbf{y}$. When $\boldsymbol{\Sigma}=\sigma^2\mathbf{I}$, the scalar $\sigma^2$ cancels in the formula, giving exactly $(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ — ordinary OLS.

**Q: What is the "whitening" transformation, and why is it useful?**
A: Writing $\boldsymbol{\Sigma}^{-1}=\mathbf{P}^T\mathbf{P}$ and transforming $\mathbf{y}^*=\mathbf{P}\mathbf{y}$, $\mathbf{X}^*=\mathbf{P}\mathbf{X}$ turns GLS into ordinary OLS applied to the transformed variables — because the transformed errors $\mathbf{P}\boldsymbol{\varepsilon}$ have covariance proportional to the identity matrix ("white noise"). It's useful because it lets you reuse all of OLS's machinery and software rather than needing a fundamentally different estimation method.

**Q: How is WLS a special case of GLS?**
A: WLS assumes $\boldsymbol{\Sigma}$ is diagonal (heteroscedastic but uncorrelated errors); its weight matrix $\mathbf{W}$ is exactly $\boldsymbol{\Sigma}^{-1}$ in that diagonal case, making the WLS formula from Chapter 10 an exact instance of the general GLS formula.

**Q: Is GLS still BLUE if $\boldsymbol{\Sigma}$ has to be estimated rather than known exactly?**
A: Not in the strict finite-sample sense of the Aitken theorem — Feasible GLS (FGLS), using an estimated $\hat{\boldsymbol{\Sigma}}$, is only asymptotically BLUE. It's still typically a substantial practical improvement over plain OLS, but the exact optimality guarantee requires a truly known covariance structure.

**Q: When would you prefer FGLS over just using OLS with robust/sandwich standard errors?**
A: FGLS improves both the point estimates' efficiency and the validity of inference, if the assumed error structure is reasonably accurate — worth the extra modeling effort when you have good reason to believe a specific structure (e.g., known proportional heteroscedasticity, or a clear AR(1) pattern). Robust standard errors are the safer default when you're unsure of the true structure, since they only fix inference, without attempting to improve efficiency.

---

*End of Chapter 19. Next: Chapter 20 — Outliers & Robust Regression (M-estimators, Huber loss, and RANSAC as alternatives to OLS's sensitivity to extreme observations, building on the influence diagnostics from Chapter 8).*
