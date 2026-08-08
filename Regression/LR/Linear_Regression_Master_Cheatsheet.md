# Linear Regression — Complete Interview Cheatsheet

*One document, everything you need. Formulas paired with one-line meanings and the numbers from your curriculum that anchor each concept.*

---

## 1. THE MODEL & OLS FOUNDATIONS

**Model:** $y_i=\beta_0+\beta_1x_i+\varepsilon_i$ — $\beta_0$=intercept (predicted $y$ at $x=0$), $\beta_1$=slope (Δ$y$ per unit $x$), $\varepsilon_i$=unobservable true error. **Error vs. residual:** $\varepsilon_i$ is theoretical/unobservable; $e_i=y_i-\hat y_i$ is real/computable.

**OLS objective:** minimize $RSS=\sum(y_i-\beta_0-\beta_1x_i)^2$. Squared (not absolute) loss because: differentiable everywhere → closed form; penalizes large errors more; = MLE under Gaussian errors.

**Closed form (simple):**
$$\hat\beta_1=\frac{S_{xy}}{S_{xx}}=\frac{\sum(x_i-\bar x)(y_i-\bar y)}{\sum(x_i-\bar x)^2}, \qquad \hat\beta_0=\bar y-\hat\beta_1\bar x$$
**Fact:** line always passes through $(\bar x,\bar y)$ exactly (from the $\partial RSS/\partial\beta_0=0$ normal equation).
**Slope–correlation link:** $\hat\beta_1=r\cdot s_y/s_x$. If $x,y$ standardized, slope = $r$ exactly.

**Matrix form:** $\mathbf y=\mathbf X\boldsymbol\beta+\boldsymbol\varepsilon$; $\mathbf X$ has a leading column of 1's for the intercept.
$$\boxed{\hat{\boldsymbol\beta}=(\mathbf X^T\mathbf X)^{-1}\mathbf X^T\mathbf y}$$
This is THE formula — works for any # of predictors, no new math needed as $p$ grows.

**Hat matrix:** $\mathbf H=\mathbf X(\mathbf X^T\mathbf X)^{-1}\mathbf X^T$, $\hat{\mathbf y}=\mathbf H\mathbf y$, $\mathbf e=(\mathbf I-\mathbf H)\mathbf y$.
- Symmetric, **idempotent** ($HH=H$) — it's an orthogonal projection matrix onto the column space of $\mathbf X$.
- Diagonal $h_{ii}$ = **leverage** (depends only on $x$, not $y$).
- $\text{trace}(\mathbf H)=p$ (# of parameters) — **always**, exact identity.
- Residuals ⟂ fitted values ⟂ every column of $\mathbf X$ → $\sum e_i=0$, $\sum e_ix_i=0$ **always**, by construction.

**Anchor numbers (5-student dataset, x=hours, y=score):** $\hat\beta_0=41.5,\hat\beta_1=7.5$.

---

## 2. INFERENCE (SE, CI, TESTS, ANOVA, R²)

$\hat\beta_1\sim N(\beta_1,\sigma^2/S_{xx})$ — unbiased, variance shrinks as predictor spread ($S_{xx}$) grows (more "leverage" to pin down slope).

**Estimate $\sigma^2$:** $MSE=\dfrac{SSE}{n-2}$ — divide by $n-2$ because 2 parameters ($\beta_0,\beta_1$) were estimated first (lost 2 df). General: $n-p$ for $p$ parameters.

$$SE(\hat\beta_1)=\sqrt{MSE/S_{xx}}$$

**t-test:** $t=\hat\beta_1/SE(\hat\beta_1)$, df$=n-2$. Uses **t**, not z, because $\sigma$ is estimated (extra uncertainty fattens tails).

**CI:** $\hat\beta_1\pm t^*_{(\alpha/2,n-2)}SE(\hat\beta_1)$. **Correct reading:** "95% of intervals built this way, across repeated samples, contain the true $\beta_1$" — NOT "95% probability $\beta_1$ is in this interval" (true $\beta_1$ is fixed, not random).

**ANOVA decomposition (exact identity, always):**
$$\underbrace{\sum(y_i-\bar y)^2}_{SST}=\underbrace{\sum(\hat y_i-\bar y)^2}_{SSR}+\underbrace{\sum(y_i-\hat y_i)^2}_{SSE}$$

$$R^2=SSR/SST$$
In **simple** regression only: $R^2=r^2$ (breaks in multiple regression). High $R^2$ ≠ correct model, says nothing about assumptions/causality. $R^2$ **mechanically never decreases** when adding any predictor, even noise.

**Mean-response CI vs. Prediction Interval** (classic trap):
$$\text{CI (mean): } \hat y_0\pm t^*s\sqrt{\tfrac1n+\tfrac{(x_0-\bar x)^2}{S_{xx}}} \qquad \text{PI (new obs): } \hat y_0\pm t^*s\sqrt{1+\tfrac1n+\tfrac{(x_0-\bar x)^2}{S_{xx}}}$$
PI always wider (extra "+1" = the new point's own irreducible error). PI never shrinks to 0 even as $n\to\infty$; CI does.

**Anchors:** $SE(\hat\beta_1)=0.5$, $t=15$, 95% CI $=(5.91,9.09)$, $R^2\approx0.987$.

---

## 3. MATRIX INFERENCE (VARIANCE-COVARIANCE)

$$\text{Var}(\hat{\boldsymbol\beta})=\sigma^2(\mathbf X^T\mathbf X)^{-1}$$
Diagonal = each coefficient's variance (→ SEs). Off-diagonal = covariance between coefficient estimates → large negative/positive off-diag = multicollinearity symptom.

---

## 4. MULTIPLE REGRESSION & PARTIAL EFFECTS

Coefficient meaning shift: simple-regression $\hat\beta_1$ = total effect; multiple-regression $\hat\beta_1$ = **partial effect, holding other predictors fixed**. If predictors correlated, adding one changes the others' coefficients (credit reallocation) — NOT a red flag, expected.

**Anchor:** hours-only slope 7.5 → drops to 4.6 once practice-tests (7) added (correlated predictors sharing credit).

---

## 5. THREE HYPOTHESIS TESTS — DON'T CONFUSE THEM

| Test | Question | Formula |
|---|---|---|
| **Overall F** | Do ALL predictors jointly matter? | $F=\dfrac{SSR/p}{SSE/(n-p-1)}$ |
| **Individual t** | Does ONE predictor matter, holding others fixed? | $t=\hat\beta_j/SE(\hat\beta_j)$ |
| **Partial F** | Does adding a GROUP of predictors help? | $F=\dfrac{(SSE_{red}-SSE_{full})/(\#\text{added})}{SSE_{full}/df_{full}}$ |

**Identity:** for exactly ONE added predictor, partial-F $=t^2$ exactly. **Always check overall F first** — if model isn't significant overall, individual coefficients are close to meaningless to interpret.

**Classic trap:** overall F can be huge/significant while one individual t-test fails → multicollinearity signature (below), not a broken model.

**Anchor:** $F=279.5$ (huge) but $t_{\beta_2}=3.5<t^*=4.303$ (individually insignificant) — correlation between $\hat\beta_1,\hat\beta_2$ estimates $=-0.945$.

---

## 6. GAUSS-MARKOV (BLUE)

**OLS is BLUE** = Best (min variance) Linear Unbiased Estimator, among linear+unbiased estimators only.
- Requires: linearity, zero-mean errors, homoscedasticity+uncorrelated errors ($\text{Var}(\boldsymbol\varepsilon)=\sigma^2\mathbf I$). **Does NOT require normality** (normality only needed for exact t/F inference, not for BLUE/point estimates).
- **Does NOT claim** OLS beats every possible estimator — only unbiased ones. A **biased** estimator (ridge) can have lower *total* error (bias²+variance) — this is why regularization can win.
- Proof skeleton: any other linear unbiased $\tilde\beta=\mathbf C\mathbf y$; unbiasedness forces $\mathbf D\mathbf X=0$ where $\mathbf C=(\mathbf X^T\mathbf X)^{-1}\mathbf X^T+\mathbf D$; then $\text{Var}(\tilde\beta)=\text{Var}(\hat\beta_{OLS})+\sigma^2\mathbf{DD}^T\succeq\text{Var}(\hat\beta_{OLS})$.

---

## 7. DIAGNOSTICS — RESIDUALS

**Standardized:** $d_i=e_i/s$ — ignores leverage, quick check only.

**Studentized (internal):** $r_i=\dfrac{e_i}{s\sqrt{1-h_{ii}}}$ — accounts for leverage; ~t-distributed.

**Studentized (external/deleted):** $t_i=\dfrac{e_i}{s_{(i)}\sqrt{1-h_{ii}}}$, where $s_{(i)}^2=\dfrac{(n-p-1)MSE-e_i^2/(1-h_{ii})}{n-p-2}$ — refits excluding point $i$; the statistically correct outlier test (exact t-dist, $n-p-2$ df). Internal version can "hide" its own outlier effect on $s$.

**Four-panel plot → assumption map:**
| Panel | Checks | Bad sign |
|---|---|---|
| Residuals vs. Fitted | Linearity | Curve/U-shape |
| Normal Q-Q | Normality | Off the 45° line |
| Scale-Location ($\sqrt{|r_i|}$ vs. fitted) | Equal variance | Funnel/trend |
| Residuals vs. Leverage | Influential points | Outside Cook's contours |

---

## 8. LEVERAGE vs. INFLUENCE (NOT THE SAME THING)

- **Leverage** ($h_{ii}$) — depends ONLY on $x$-values (unusual predictor position). $\sum h_{ii}=p$.
- **Influence** — leverage AND residual combined; does removing the point actually change the fit?
- **A big residual alone ≠ high influence** if leverage is low. **High leverage alone ≠ high influence** if the point's $y$ agrees with prediction.

**Cook's Distance:** $D_i=\dfrac{r_i^2}{p}\cdot\dfrac{h_{ii}}{1-h_{ii}}$. Flag: $D_i>4/n$ (common) or $D_i>1$ (classical). Requires BOTH factors large.

**DFBETAS:** $\dfrac{\hat\beta_j-\hat\beta_{j(i)}}{s_{(i)}\sqrt{[(\mathbf X^T\mathbf X)^{-1}]_{jj}}}$ — influence on ONE specific coefficient. Flag: $|{\cdot}|>2/\sqrt n$.

**Other:** DFFITS (influence on fitted value, threshold $2\sqrt{p/n}$), COVRATIO (influence on estimation precision).

**What to do with a flagged point:** check data error first → consider if it signals model misspecification → report with/without → delete only as last resort, justified.

**Anchor:** $D_5=1.833$ (dwarfs all others: 0.11–0.38) vs. $4/n=0.8$ threshold. Point 3 had the LARGEST raw residual but LOW Cook's D (0.21) — low leverage. Removing point 5 leaves an exact-fit (SSE=0) on the rest — DFBETAS denominator → 0/undefined, meaning point 5 was solely responsible for ALL residual variance.

---

## 9. MULTICOLLINEARITY

**Definition:** strong correlation among predictors. Does NOT bias coefficients (still unbiased/Gauss-Markov holds) — inflates their **variance**, destabilizing individual interpretation. Overall prediction/$R^2$ unaffected.

**VIF:** $VIF_j=\dfrac{1}{1-R_j^2}$, where $R_j^2$ = $R^2$ from regressing $x_j$ on all other predictors. $VIF=1$: no inflation. $VIF>5$: caution. $VIF>10$: red flag.

**Condition number:** $\kappa=\sqrt{\lambda_{max}/\lambda_{min}}$ of (standardized) $\mathbf X^T\mathbf X$ eigenvalues — conventions vary (correlation-matrix eigenvalues vs. Belsley's scaled design-matrix approach, threshold ~30), so treat absolute cutoffs cautiously.

**Remedies (in order):** drop a predictor → combine into composite → center before polynomial/interaction terms → collect more/varied data → ridge regression. **Don't** drop a predictor solely because its t-test is insignificant without checking VIF first.

**Anchor:** VIF(x1)=VIF(x2)$\approx9.33$ — right at the red-flag line, directly explaining Section 5's F/t mismatch.

---

## 10. HETEROSCEDASTICITY

**What breaks:** OLS still unbiased; loses BLUE status; SEs/t/F/CI become **invalid** (often falsely too small → inflated significance).

**Breusch-Pagan:** regress $e_i^2$ on predictors → $BP=n\times R^2_{aux}\sim\chi^2_p$. **White's test:** adds squares+cross-products of predictors — more general, needs more data/df.

**WLS:** $\hat{\boldsymbol\beta}_{WLS}=(\mathbf X^T\mathbf W\mathbf X)^{-1}\mathbf X^T\mathbf W\mathbf y$, $w_i=1/\hat\sigma_i^2$ — down-weight noisy points. WLS = new BLUE once weight structure is known.

**Robust/sandwich SEs:** keeps $\hat\beta_{OLS}$ unchanged, fixes SE only:
$$\widehat{\text{Var}}_{robust}=(\mathbf X^T\mathbf X)^{-1}(\mathbf X^T\text{diag}(e_i^2)\mathbf X)(\mathbf X^T\mathbf X)^{-1}$$
"Sandwich" = bread ($(\mathbf X^T\mathbf X)^{-1}$) both sides, filling (observed $e_i^2$) in middle. Use WLS when you know the variance structure; robust SEs when you don't want to commit to one.

**Anchor:** $BP\approx1.85$ vs. $\chi^2_2$ critical $5.99$ → no signal (small-n, low power caveat applies).

---

## 11. AUTOCORRELATION

**What breaks:** unbiased still, but SEs typically look artificially SMALL → false confidence (often more dangerous than heteroscedasticity).

**Durbin-Watson:** $DW=\dfrac{\sum_{i=2}^n(e_i-e_{i-1})^2}{\sum e_i^2}$. $DW\approx2$: no autocorrelation. $DW\to0$: strong positive. $DW\to4$: strong negative. Relation: $\hat\rho\approx1-DW/2$.

**GLS (general remedy):** $\hat{\boldsymbol\beta}_{GLS}=(\mathbf X^T\boldsymbol\Sigma^{-1}\mathbf X)^{-1}\mathbf X^T\boldsymbol\Sigma^{-1}\mathbf y$ — WLS is the special case where $\boldsymbol\Sigma$ is diagonal.

**Cochrane-Orcutt/Prais-Winsten transform:** $y_t^*=y_t-\hat\rho y_{t-1}$ (same for $x$) then run plain OLS — a concrete "whitening" trick.

**Newey-West (HAC) SEs:** like sandwich SEs but also robust to autocorrelation up to a chosen lag — doesn't require specifying the exact AR structure.

**Anchor:** $DW\approx2.02$ → no autocorrelation (expected: cross-sectional data, arbitrary order).

---

## 12. TRANSFORMATIONS

**Log transform:** fixes multiplicative/exponential curvature. $\ln y=\beta_0+\beta_1x$ → $e^{\hat\beta_1}$ = multiplicative change in $y$ per unit $x$ (NOT additive %, that's an approximation for small $\hat\beta_1$).
**Log-log:** $\hat\beta_1$ = **elasticity** (1% change in $x$ → $\hat\beta_1$% change in $y$).
**Back-transform trap:** $E[e^Z]\neq e^{E[Z]}$ (Jensen's inequality) — naive exponentiation gives the median not mean; need a correction (e.g., Duan's smearing, or $e^{\hat\sigma^2/2}$).

**Box-Cox family:** $y^{(\lambda)}=(y^\lambda-1)/\lambda$ ($\lambda\neq0$), $=\ln y$ ($\lambda=0$). Choose $\lambda$ by maximizing profile likelihood over a grid.

**Polynomial terms:** still LINEAR regression (linear in parameters, not in $x$). Alternative to transforms when curvature isn't multiplicative.

---

## 13. CATEGORICAL PREDICTORS & INTERACTIONS

**Dummy coding:** $k$-level category → $k-1$ dummies (never $k$ — the **dummy variable trap**: all $k$ + intercept = perfect collinearity, singular $\mathbf X^T\mathbf X$).

**With interaction:** $y=\beta_0+\beta_1x+\beta_2D+\beta_3(xD)$. Reference group ($D=0$): intercept $\beta_0$, slope $\beta_1$. Other group ($D=1$): intercept $\beta_0+\beta_2$, slope $\beta_1+\beta_3$.

**BIGGEST TRAP:** with an interaction present, $\hat\beta_1$ alone is ONLY the reference group's slope — not "the" effect of $x$. True marginal effect: $\partial y/\partial x=\beta_1+\beta_3D$.

**Hierarchy/marginality principle:** if interaction is kept, keep both main effects regardless of their own individual significance.

**Effect coding** ($-1/+1$, Montgomery/DOE convention) vs. **dummy coding** ($0/1$, default): effect coding = deviations from grand mean; dummy = relative to reference category.

---

## 14. MODEL SELECTION

| Criterion | Formula | Better = |
|---|---|---|
| Adjusted $R^2$ | $1-(1-R^2)\dfrac{n-1}{n-p-1}$ | Higher |
| AIC | $n\ln(SSE/n)+2p$ | Lower |
| BIC | $n\ln(SSE/n)+p\ln(n)$ | Lower |
| Mallows' $C_p$ | $SSE_p/MSE_{full}+2p-n$ | $C_p\approx p$ is good; $C_p\gg p$ = underfitting/bias |

**BIC penalizes complexity more than AIC once $n>7$** ($\ln n>2$) → BIC favors smaller models as $n$ grows. **Identity:** $C_p=p$ exactly when a model is evaluated against itself as the reference.

**Stepwise selection (forward/backward/both):** criticized for implicit multiple testing (no correction) → overoptimistic fit stats, unstable across small data changes. Modern preference: regularization or CV instead.

**Anchor:** all four criteria agreed to KEEP $x_2$ despite its insignificant individual t-test — model-selection criteria and single-coefficient significance are different questions.

---

## 15. OVERFITTING & CROSS-VALIDATION

In-sample fit is **always optimistic** — model has "seen the answers."

**k-fold CV:** split into $k$ folds, train on $k-1$, test on 1, average. **LOOCV** = $k=n$ (extreme case).
**Bias-variance in choosing $k$:** LOOCV = low bias, high variance (correlated folds); small $k$ (5/10) = more bias, less variance — standard default for larger data.

**PRESS shortcut (linear regression only), avoids refitting $n$ times:**
$$CV_{(n)}=\frac1n\sum\left(\frac{e_i}{1-h_{ii}}\right)^2$$

**Anchor:** LOOCV MSE $\approx3.26$ vs. in-sample MSE $0.48$ (7× gap). The point with the WORST leave-one-out error is the SAME point with the highest Cook's distance — influence and predictability are the same underlying fact.

---

## 16. RIDGE REGRESSION

$$RSS_{ridge}=RSS+\lambda\sum\beta_j^2, \qquad \hat{\boldsymbol\beta}_{ridge}=(\mathbf X^T\mathbf X+\lambda\mathbf I)^{-1}\mathbf X^T\mathbf y$$
- $\lambda=0$→OLS. $\lambda\to\infty$→ all slopes →0 (never exactly).
- Adding $\lambda\mathbf I$ guarantees invertibility even under severe multicollinearity — numerically stabilizes the normal equations.
- **Geometry:** constraint region = circle (L2 ball). Never produces exact zeros.
- **Biased** for $\lambda>0$ (violates Gauss-Markov's unbiasedness on purpose) but variance-reduction can lower TOTAL error.
- Must **standardize** predictors first (penalty is scale-sensitive).
- $\lambda$ chosen via cross-validation.

**Anchor:** at $\lambda=0$: (4.6,7) exact match to OLS. At $\lambda=5$: shrinks to (4.55, 2.54) — the unstable/insignificant $\hat\beta_2$ shrinks hardest.

---

## 17. LASSO REGRESSION

$$RSS_{lasso}=RSS+\lambda\sum|\beta_j|$$
- **Geometry:** constraint region = diamond (L1 ball) — has corners ON the axes → solutions land there → **exact zeros** (sparsity/feature selection).
- **No closed form** (not differentiable at 0) → fit via **coordinate descent** + soft-thresholding:
$$\hat\beta_j\leftarrow\text{sign}(z_j)\max(|z_j|-\lambda/(2S_{x_jx_j}),0), \quad z_j=\frac{\sum x_{ij}r_i^{(j)}}{S_{x_jx_j}}$$
- Under correlated predictors: arbitrarily keeps ONE, zeros the rest (contrast with ridge's "shrink together").

**Anchor:** at $\lambda=10$: converges to $(\hat\beta_1,\hat\beta_2)=(7.6,0)$ EXACTLY — critical threshold for sparsity derived as $\lambda\geq8.4$.

---

## 18. ELASTIC NET

$$RSS+\lambda_1\sum|\beta_j|+\lambda_2\sum\beta_j^2$$
$\lambda_2=0$→lasso. $\lambda_1=0$→ridge. Update: $\hat\beta_j\leftarrow S(z_j,\lambda_1/2)/(S_{x_jx_j}+\lambda_2)$.

**Grouping effect:** correlated predictors tend to enter/exit together (L2 component "rescues" what lasso alone would zero out) — more stable than lasso under correlated groups.

**Anchor:** same $\lambda_1=10$ that gave lasso exact zero for $\hat\beta_2$ → adding $\lambda_2=2$ rescues it to $\approx2.18$ (both nonzero).

**Decision table (Ch16-18):**
| Situation | Method |
|---|---|
| Mostly uncorrelated, all relevant | Ridge (or OLS if $n\gg p$) |
| Few predictors truly matter | Lasso |
| Correlated groups, want group-level in/out | Elastic Net |
| $p>n$ | Any of the three — OLS is impossible ($\mathbf X^T\mathbf X$ guaranteed singular) |

---

## 19. GENERALIZED LEAST SQUARES

$$\hat{\boldsymbol\beta}_{GLS}=(\mathbf X^T\boldsymbol\Sigma^{-1}\mathbf X)^{-1}\mathbf X^T\boldsymbol\Sigma^{-1}\mathbf y$$
$\boldsymbol\Sigma=\sigma^2\mathbf I$ → OLS. Diagonal $\boldsymbol\Sigma$ → WLS. Toeplitz/AR(1) $\boldsymbol\Sigma$ → autocorrelation case.

**Whitening trick:** $\boldsymbol\Sigma^{-1}=\mathbf P^T\mathbf P$; transform $\mathbf y^*=\mathbf P\mathbf y,\mathbf X^*=\mathbf P\mathbf X$ → run plain OLS on transformed data. Explains WHY Cochrane-Orcutt works.

**Aitken's theorem:** GLS is BLUE under general known $\boldsymbol\Sigma$ (generalizes Gauss-Markov).

**FGLS:** $\boldsymbol\Sigma$ usually unknown → estimate it (2-step), refit. Only **asymptotically** BLUE, not exactly.

---

## 20. OUTLIERS & ROBUST REGRESSION

**Huber loss:** $L_\delta(e)=\frac12e^2$ if $|e|\le\delta$; $\delta(|e|-\frac12\delta)$ if $|e|>\delta$. Fit via **IRLS**: weight $w(e)=1$ if $|e|\le\delta$, else $\delta/|e|$.

**KEY LIMITATION:** Huber weights by residual size only — NOT by leverage. A **bad leverage point** (unusual BOTH in $x$ and $y$) still distorts the fit even after down-weighting.

**RANSAC:** repeatedly fit on random minimal subsets → count inliers within threshold → keep best-consensus fit → final refit on inlier set only. Cleanly EXCLUDES bad leverage points (hard exclusion) vs. Huber's soft down-weighting.

**Redescending M-estimators (Tukey biweight):** weight → 0 for extreme residuals — smooth analogue of RANSAC's hard exclusion.

**Anchor:** true slope 7.5 → OLS on corrupted data: 21.5 (wrecked) → Huber (1 iteration): ≈19.6 (partial, still leverage-vulnerable) → RANSAC: 7.0 (fully recovered).

---

## 21. POLYNOMIAL & NONLINEAR REGRESSION

Still **linear regression** — linear in parameters, not in $x$.

**Collinearity trap:** raw $x,x^2$ are highly correlated unless $x$ is symmetric around 0 (e.g., VIF≈26 uncentered in a worked example). **Fix: center $x$ first** ($x_c=x-\bar x$) → for symmetric spacing, $\text{Corr}(x_c,x_c^2)=0$ EXACTLY. For degree≥3, use orthogonal polynomials.

**Risk:** degree-$(n-1)$ polynomial fits $n$ points perfectly trivially — meaningless overfitting. **Runge's phenomenon:** wild oscillation near data edges at high degree. Choose degree via CV/AIC/BIC, not max in-sample fit.

**Splines:** piecewise polynomials (usually cubic) joined smoothly at **knots** (continuity in value + 1st + 2nd derivative). **Natural cubic spline:** linear beyond boundary knots (tames edge behavior). **Smoothing spline:** knot at every point + roughness penalty (integrated squared 2nd derivative) — structurally identical idea to ridge's L2 penalty, just penalizing curvature instead of coefficient size.

---

## 22. GLM BRIDGE (LINEAR → LOGISTIC)

**Why not OLS on 0/1 outcome:** predictions can exceed $[0,1]$; variance mechanically depends on mean ($p(1-p)$, guaranteed heteroscedastic); true relationship is S-shaped not linear.

**GLM = 3 components:** random component (distribution: Gaussian/Binomial/Poisson), systematic component ($\eta=\mathbf x^T\boldsymbol\beta$, unchanged), link function $g(\mu)=\eta$.

| Model | Link | Inverse |
|---|---|---|
| Linear | Identity | $\mu=\eta$ |
| Logistic | Logit: $\ln(\mu/(1-\mu))$ | $\mu=1/(1+e^{-\eta})$ |
| Poisson | Log | $\mu=e^\eta$ |

**Fitting = IRLS**, literally the SAME WLS machinery as Ch.10, iterated: weight $w_i=\hat p_i(1-\hat p_i)$; working response $z_i=\hat\eta_i+\dfrac{y_i-\hat p_i}{\hat p_i(1-\hat p_i)}$; run WLS of $z$ on $x$; repeat.

**Coefficient interpretation:** $\hat\beta_1$=change in log-odds per unit $x$; $e^{\hat\beta_1}$=multiplicative change in odds.

**Perfect separation:** if classes perfectly separable, MLE diverges to $\pm\infty$ — huge unstable coefficients, not a strong effect.

---

## 23. INTEGRATED WORKFLOW (WHAT ORDER TO DO THINGS)

1. Fit, look at residual plot visually FIRST (cheap, guides everything else).
2. Check leverage/influence BEFORE formal tests (an influential point can corrupt downstream diagnostics).
3. Run formal tests only for what the visual inspection suggested (don't test everything reflexively).
4. Weigh test results against POWER — non-significant in small samples ≠ confirmed assumption.
5. Remedy proportional to evidence: robust SEs (cheap safeguard) → WLS/GLS/transform (if structure known) → point removal (last resort, justified).
6. Report honestly, including sensitivity (with/without flagged points).

---

## 24. CAUSAL INFERENCE

**Omitted Variable Bias (exact formula):**
$$E[\hat\alpha_1]=\beta_1+\beta_2\delta_1, \qquad \delta_1=S_{x_1x_2}/S_{x_1x_1}$$
($\delta_1$ = regressing the omitted var on the included var). **Anchor: EXACT match** — $4.6+7(0.5)=8.1$, the actual reduced-model coefficient.

**The math is the SAME regardless of causal structure — the correct ACTION differs completely:**

| Structure | Definition | Action |
|---|---|---|
| **Confounder** | Common cause of $x_1$ AND $y$ | **DO control for it** (omitting biases the estimate) |
| **Mediator** | $x_1\to x_2\to y$ (part of the causal pathway) | **DO NOT control for it** ("bad control" — blocks the true effect) |
| **Collider** | Common EFFECT of $x_1$ and $y$ | **DO NOT control for it** — controlling INTRODUCES spurious association that wasn't there |

Which one is true = **domain knowledge / DAG reasoning**, NOT something any statistical test can determine.

**Beyond regression for causal claims:** RCTs (gold standard — randomization breaks confounding by design); instrumental variables (affects $x_1$, not $y$ directly); difference-in-differences; regression discontinuity.

---

## 25. HIGH-DENSITY "GOTCHA" LIST (READ THIS TWICE)

- Squared error chosen for: differentiability + MLE-under-Gaussian + penalizes big errors more — NOT because "it's more accurate."
- $\sum e_i=0$ and $\sum e_ix_i=0$ ALWAYS hold for OLS — geometric fact (orthogonality), not an empirical coincidence.
- BLUE requires NO normality assumption — normality is for exact-sample t/F inference only.
- $t^2=F$ for single-parameter partial F-tests — exact identity, not approximation.
- $R^2=r^2$ ONLY in simple regression — breaks in multiple regression.
- Overall F significant + individual t insignificant = classic multicollinearity, not a broken model.
- High leverage ≠ high influence. Big residual ≠ high influence. Need BOTH (Cook's D).
- Multicollinearity inflates VARIANCE, not bias.
- Heteroscedasticity/autocorrelation: unbiased still, but SE/inference invalid — NOT a bias problem.
- Ridge: shrinks toward 0, NEVER exactly 0. Lasso: CAN hit exactly 0 (geometry: diamond corners).
- Regularization (ridge) is deliberately BIASED — trades bias for variance reduction, stepping outside Gauss-Markov's "unbiased" category on purpose.
- $C_p=p$ exactly for the full/reference model — always, by identity.
- BIC penalty $p\ln n$ > AIC penalty $2p$ once $n>7$ (since $\ln n>2$) → BIC picks smaller models as $n$ grows.
- In-sample fit stats are ALWAYS optimistic — LOOCV/CV needed for honest assessment.
- Huber M-estimator ≠ leverage-robust — only helps with vertical outliers, not bad leverage points. RANSAC handles both.
- Polynomial regression is LINEAR regression (linear in parameters).
- Centering removes ARTIFICIAL collinearity between $x$ and $x^2$ introduced by the modeling choice itself — not present in the underlying relationship.
- With an interaction term, the "main effect" coefficient is ONLY the reference group's effect, not an overall effect.
- Confidence interval (mean response) vs. Prediction interval (new observation) — PI always wider, PI never shrinks to 0.
- Non-rejection of a null (e.g., Breusch-Pagan, Durbin-Watson) is NOT confirmation of the null — especially in small samples (power).
- WLS is a special case of GLS (diagonal $\Sigma$). GLS is a special case of nothing — it's the general form.
- Correlation/good fit/significance ≠ causation, ever — regardless of diagnostics passed.
- A variable can improve $R^2$ and be individually significant and STILL be the wrong thing to control for (mediator/collider) if the goal is a causal estimate.
