# Chapter 14 — Logistic Regression, Poisson Regression, and Generalized Linear Models
### (Kutner, Nachtsheim, Neter, Li — *Applied Linear Statistical Models*)

This is the capstone chapter — where everything built across Chapters 1–13 (least squares, MLE, the matrix framework, iterative nonlinear fitting) generalizes into the single unifying framework used throughout modern applied ML for classification and count problems: the **Generalized Linear Model (GLM)**.

---

## 14.1 Why Ordinary Least Squares Fails for a Binary Response

**The setup:** Y is binary (0/1) — e.g., did a customer churn. **Plain English reasons OLS breaks down, all worth knowing explicitly:**

1. **Fitted values aren't constrained to [0,1].** $\hat Y=b_0+b_1X$ can produce predictions like $-0.3$ or $1.4$ — meaningless as probabilities.
2. **The error variance is mechanically non-constant.** For a binary Y with true probability $p_i$, $\text{Var}(Y_i)=p_i(1-p_i)$ — which *depends on $X_i$* (since $p_i$ does), directly and unavoidably violating Chapter 1's constant-variance assumption. This isn't a fixable "sometimes" problem (like Chapter 3's heteroscedasticity) — it's baked into what a binary variable *is*.
3. **The true relationship between X and $P(Y=1)$ is naturally S-shaped** (probabilities compress near 0 and 1, can't keep changing linearly forever), not a straight line.

**Interview trap:** the "linear probability model" (literally just running OLS with a 0/1 outcome) does still get taught and used informally in some fields (particularly economics, for interpretability of average marginal effects) — but it's essential to know its specific failure modes above and why logistic regression is the standard, principled alternative rather than a mere stylistic choice.

---

## 14.2 The Logistic Response Function and the Logit Transformation

**The fix:** model $P(Y_i=1|X_i)=p_i$ using a function that's naturally bounded in $(0,1)$ and S-shaped:
$$
p_i = \frac{e^{\beta_0+\beta_1X_i}}{1+e^{\beta_0+\beta_1X_i}} = \frac{1}{1+e^{-(\beta_0+\beta_1X_i)}}
$$

**The logit (log-odds) transformation** linearizes this:
$$
\text{logit}(p_i) = \ln\left(\frac{p_i}{1-p_i}\right) = \beta_0+\beta_1X_i
$$

**Why this specific transformation, and not some other S-curve.** The ratio $p/(1-p)$ is the **odds** — a natural, interpretable quantity (odds of 3 means "3-to-1 in favor"). Taking its log makes it symmetric around zero (odds of 3 and odds of 1/3 become $+\ln3$ and $-\ln3$ — equal magnitude, opposite sign) and, crucially, **makes the right-hand side linear in the parameters** — bringing us back to a linear-in-parameters framework, just for the *log-odds* rather than for $Y$ or $p$ directly.

---

## 14.3 Maximum Likelihood Estimation via Iteratively Reweighted Least Squares

### Why We Need MLE Rather Than Least Squares Here

**Recall Chapter 1.11:** OLS and MLE coincide *only* under the Normal-errors assumption. Binary Y follows a **Bernoulli**, not Normal, distribution — so we go directly to maximum likelihood, without an intervening least-squares detour.

$$
L(\boldsymbol\beta) = \prod_i p_i^{Y_i}(1-p_i)^{1-Y_i}, \qquad \ell(\boldsymbol\beta)=\sum_i\left[Y_i\ln p_i+(1-Y_i)\ln(1-p_i)\right]
$$

**The score equations** (setting $\partial\ell/\partial\boldsymbol\beta=0$) turn out to have a strikingly familiar form:
$$
\mathbf{U}(\boldsymbol\beta) = \mathbf{X}'(\mathbf{Y}-\mathbf{p}) = \mathbf{0}
$$
**This looks exactly like Chapter 1's normal equations** ($\mathbf{X}'\mathbf{e}=\mathbf{0}$, with $\mathbf{e}=\mathbf{Y}-\hat{\mathbf{Y}}$) — except $p_i$ is a *nonlinear* function of $\boldsymbol\beta$, so this can't be solved in one closed-form step. **We're back to Chapter 13's iterative machinery.**

### The Newton-Raphson / IRLS Update — Directly Recalling Chapters 11 and 13

$$
\boldsymbol\beta^{(t+1)} = \boldsymbol\beta^{(t)} + (\mathbf{X}'\mathbf{W}\mathbf{X})^{-1}\mathbf{X}'(\mathbf{Y}-\mathbf{p})
$$
where $\mathbf{W}=\text{diag}(p_i(1-p_i))$ — **exactly Chapter 13's Gauss-Newton update, with the Jacobian replaced by $\mathbf{X}$ weighted by $p_i(1-p_i)$**, and **exactly Chapter 11's Iteratively Reweighted Least Squares**, with weights now representing the Bernoulli variance function rather than a heteroscedasticity correction. This is the same algorithmic skeleton reappearing for the third time in this course — a genuinely unifying thread worth being able to articulate explicitly in an interview.

### Worked Example: One Full Newton-Raphson Iteration by Hand

**Data** (X = number of prior complaints, Y = churned): 

| X | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Y | 0 | 0 | 1 | 0 | 1 | 1 |

**Starting guess:** $\boldsymbol\beta^{(0)}=[0,0]$ — a convenient starting point, since $\text{logit}=0\Rightarrow p_i=0.5$ for every observation.

**Step 1 — score vector** at $p_i=0.5$ for all $i$ (so $Y_i-p_i = Y_i-0.5$):
$$
U_0=\sum(Y_i-0.5) = -0.5-0.5+0.5-0.5+0.5+0.5=-0.5
$$
$$
U_1=\sum X_i(Y_i-0.5) = 0-0.5+1.0-1.5+2.0+2.5=3.5
$$

**Step 2 — weight matrix**, since $p_i(1-p_i)=0.25$ for every observation at this starting point: $\mathbf{W}=0.25\mathbf{I}$, so $\mathbf{X}'\mathbf{W}\mathbf{X}=0.25\,\mathbf{X}'\mathbf{X}$:
$$
\mathbf{X}'\mathbf{X}=\begin{bmatrix}6&15\\15&55\end{bmatrix} \quad(\Sigma X_i=15,\ \Sigma X_i^2=55) \quad\Rightarrow\quad \mathbf{X}'\mathbf{W}\mathbf{X}=\begin{bmatrix}1.5&3.75\\3.75&13.75\end{bmatrix}
$$

**Step 3 — solve for the update** (identical 2×2 method as every prior chapter):
$$
\det=1.5(13.75)-3.75^2=6.5625
$$
$$
\delta_0 = \frac{13.75(-0.5)-3.75(3.5)}{6.5625}=\frac{-6.875-13.125}{6.5625}=-3.048
$$
$$
\delta_1 = \frac{-3.75(-0.5)+1.5(3.5)}{6.5625}=\frac{1.875+5.25}{6.5625}=1.086
$$

**Updated:** $\boldsymbol\beta^{(1)}=[-3.048,\ 1.086]$.

**Repeating this exact process** (recompute $p_i$, then $\mathbf{W}$, then the score, then solve again) for two more iterations converges to approximately:
$$
\hat\beta_0\approx-2.97, \qquad \hat\beta_1\approx1.18
$$
(Verified via the same iterative mechanics shown above, carried one step further — by this point the score vector is close to $\mathbf{0}$, the convergence criterion.)

### Interpreting the Fitted Model

$$
\text{logit}(\hat p_i) = -2.97+1.18X_i
$$

**The odds-ratio interpretation — the standard, expected way to communicate a logistic coefficient:**
$$
e^{\hat\beta_1}=e^{1.18}=3.25
$$
**"Each additional prior complaint multiplies the odds of churning by about 3.25×, holding nothing else in the model constant."** This is the single most important sentence-template for communicating a logistic coefficient in an interview — always in terms of the **multiplicative** effect on **odds**, never a direct percentage-point effect on probability (which varies depending on where you are on the S-curve — a crucial distinction from linear regression's constant-slope interpretation).

**Interview trap, extremely common:** people say "a one-unit increase in X increases the probability of Y by $\beta_1$" — this is simply wrong for logistic regression. The correct statement is about **log-odds** (which do change linearly) or **odds** (which change multiplicatively) — the *probability* itself changes by a varying amount depending on the current probability level (the derivative of the logistic curve, $p(1-p)\beta_1$, is largest near $p=0.5$ and shrinks near $p=0$ or $1$).

---

## 14.6 Inference: the Wald Test and the Likelihood Ratio Test

### The Wald Test

Just as $t^*=b_1/s(b_1)$ tested $\beta_1$ in linear regression, the analogous **Wald test** here uses the asymptotic standard error from $(\mathbf{X}'\mathbf{W}\mathbf{X})^{-1}$ at convergence (recall Chapter 13's identical asymptotic-inference caveat for nonlinear parameters):
$$
z^* = \frac{\hat\beta_1}{s(\hat\beta_1)}, \qquad s^2(\hat\beta_1) \approx \left[(\mathbf{X}'\mathbf{W}\mathbf{X})^{-1}\right]_{22}
$$

Using our converged weight structure: $s^2(\hat\beta_1)\approx0.691$, $s(\hat\beta_1)\approx0.831$:
$$
z^* = \frac{1.18}{0.831}=1.42
$$
Since $|1.42|<1.96$, **not statistically significant at the 0.05 level** — unsurprising given the tiny sample ($n=6$).

### The Likelihood Ratio Test — Often Preferred in Practice

**Deviance**, the GLM generalization of SSE: $D=-2\ell(\hat{\boldsymbol\beta})$.

**Worked example:** using our converged fitted probabilities, the full model's log-likelihood computes to $\ell_{\text{full}}\approx-2.479$, so $D_{\text{full}}=4.958$.

The **null model** (intercept only, $\hat p=\bar Y=0.5$ for all): $\ell_{\text{null}}=n\ln(0.5)=6(-0.693)=-4.159$, so $D_{\text{null}}=8.318$.

**Likelihood ratio test statistic:**
$$
G^2 = D_{\text{null}}-D_{\text{full}} = 8.318-4.958=3.360
$$
Compared to $\chi^2_{(0.95,1)}=3.841$: since $3.360<3.841$, **also not significant** — consistent with the Wald test's conclusion, and a reassuring cross-check (the two tests are asymptotically equivalent but can differ somewhat in small samples, as seen here — both point the same direction, though).

**Interview question:** *"When might the Wald test and the likelihood ratio test for the same logistic regression coefficient disagree, and which should you trust more?"*
**Ideal answer:** Both are asymptotically equivalent, but the Wald test relies on the local quadratic approximation of the log-likelihood around the MLE being accurate — which can be poor with small samples, coefficients far from zero, or near-separation in the data. The likelihood ratio test directly compares the actual likelihood at two nested models and tends to be more reliable in exactly those problematic small-sample or extreme-coefficient situations — generally the safer choice when the two disagree.

---

## 14.7 Poisson Regression: Modeling Count Data

### Why Not Just Use Ordinary Regression for Counts

Counts (number of complaints, number of clicks, number of defects) are non-negative integers, typically **right-skewed**, with **variance that grows with the mean** (for a true Poisson process, $\text{Var}(Y)=E[Y]$ exactly) — violating both the constant-variance and (for small counts) normality assumptions outright.

### The Poisson Regression Model

$$
\ln(\mu_i) = \beta_0+\beta_1X_i \quad\Leftrightarrow\quad \mu_i=e^{\beta_0+\beta_1X_i}
$$

**Why the log link specifically:** it guarantees $\mu_i>0$ for any values of $\beta_0,\beta_1,X_i$ — exactly analogous to why the logit link guarantees $p_i\in(0,1)$ in logistic regression.

**The same IRLS machinery applies** (score $\mathbf{U}=\mathbf{X}'(\mathbf{Y}-\boldsymbol\mu)$, weights $\mathbf{W}=\text{diag}(\mu_i)$, since $\text{Var}(Y_i)=\mu_i$ for Poisson data) — structurally identical to the logistic case, just with a different weight function reflecting a different variance-mean relationship.

### Worked Example

**Data** (X = months as a customer, Y = number of complaints):

| X | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Y | 1 | 2 | 2 | 4 | 6 |

**Smart starting point:** $\beta_0^{(0)}=\ln(\bar Y)=\ln(3)=1.0986$, $\beta_1^{(0)}=0$ (so $\mu_i=3$ for all observations initially — a much better-behaved starting point than $[0,0]$, which for a log-link model can produce wildly unstable first steps; starting at the log of the sample mean is standard practice).

**Iteration 1:** at $\mu_i=3$ for all, $U_0=\sum(Y_i-3)=0$ (automatically, since starting at the sample mean), $U_1=\sum X_i(Y_i-3)=12$. With $\mathbf{W}=3\mathbf{I}$:
$$
\mathbf{X}'\mathbf{W}\mathbf{X}=3\begin{bmatrix}5&15\\15&55\end{bmatrix}=\begin{bmatrix}15&45\\45&165\end{bmatrix}, \quad\det=450
$$
$$
\delta_0=\frac{165(0)-45(12)}{450}=-1.2, \qquad \delta_1=\frac{-45(0)+15(12)}{450}=0.4
$$
$$
\boldsymbol\beta^{(1)}=[1.0986-1.2,\ 0+0.4]=[-0.101,\ 0.4]
$$

**Continuing this same process** for two further iterations (recomputing $\mu_i$, $\mathbf{W}$, and the score at each new $\boldsymbol\beta$, exactly as in the logistic example above) converges to approximately:
$$
\hat\beta_0\approx-0.5, \qquad \hat\beta_1\approx0.45
$$

### Interpretation: Incidence Rate Ratios

$$
e^{\hat\beta_1}=e^{0.45}\approx1.57
$$
**"Each additional month as a customer is associated with about a 57% increase in the expected number of complaints, holding nothing else constant."** This multiplicative interpretation (the **Incidence Rate Ratio**, IRR) is the Poisson-regression analog of the logistic model's odds ratio — same underlying reason (log link ⟹ additive effects on the log scale ⟹ multiplicative effects on the original scale).

**Interview trap — overdispersion:** real count data very often has $\text{Var}(Y)>E[Y]$ (**overdispersion**), violating the strict Poisson assumption that variance equals the mean exactly. This inflates the true standard errors beyond what the plain Poisson model reports, leading to overconfident inference — the standard fixes are a **quasi-Poisson** model (same coefficients, corrected standard errors via an estimated dispersion parameter) or switching to a **Negative Binomial** model (which explicitly includes an extra parameter for the mean-variance relationship). Always worth mentioning overdispersion checks when discussing Poisson regression in an interview — it's a near-universal real-data caveat.

---

## 14.8 The Generalized Linear Model (GLM): the Unifying Framework

**Every model in this entire course** — from Chapter 1's simple linear regression through this chapter's logistic and Poisson models — is a special case of one framework, defined by three components:

1. **A random component:** the assumed distribution of $Y$, from the **exponential family** (Normal, Bernoulli, Poisson, Gamma, etc.).
2. **A systematic component:** the linear predictor $\eta_i=\beta_0+\beta_1X_{i1}+\cdots$ — always linear in the parameters, exactly as in every chapter of this course.
3. **A link function** $g(\cdot)$ connecting the mean of Y to the linear predictor: $g(\mu_i)=\eta_i$.

| Model | Distribution | Link function $g(\mu)$ | Variance function |
|---|---|---|---|
| Linear regression (Ch. 1–13) | Normal | Identity: $g(\mu)=\mu$ | Constant: $\sigma^2$ |
| Logistic regression | Bernoulli | Logit: $g(\mu)=\ln\frac{\mu}{1-\mu}$ | $\mu(1-\mu)$ |
| Poisson regression | Poisson | Log: $g(\mu)=\ln(\mu)$ | $\mu$ |

**Why this table is worth memorizing directly.** Every single row shares the exact same estimation machinery: IRLS, with weights derived from the variance function, converging (via Newton-Raphson) to the MLE — Chapter 1's ordinary least squares is simply the special case where the link is the identity function and the weights are constant (so IRLS converges in exactly **one** step, which is precisely why closed-form OLS exists at all — it's not a fundamentally different algorithm, just the one-iteration special case of the general GLM-fitting procedure).

**Interview question:** *"How would you explain to someone that linear regression, logistic regression, and Poisson regression are 'the same model' in some deep sense?"*
**Ideal answer:** All three are Generalized Linear Models: they share the same linear predictor structure (a linear combination of coefficients and predictors) and are fit via the same maximum-likelihood/IRLS machinery, differing only in **which distribution** is assumed for Y (Normal, Bernoulli, Poisson) and **which link function** connects the linear predictor to the mean of that distribution. Linear regression is the special case where the link is the identity function, which happens to make the IRLS weights constant — collapsing the general iterative algorithm down to a single, closed-form step. That's genuinely *why* OLS has a closed-form solution and logistic/Poisson regression don't: it's not a different kind of model, just the one-step-convergence special case of the same underlying framework.

---

## Python Implementation

```python
import numpy as np
import statsmodels.api as sm

# --- Logistic regression ---
X_logit = np.array([0,1,2,3,4,5], dtype=float)
Y_logit = np.array([0,0,1,0,1,1], dtype=float)
X_design = sm.add_constant(X_logit)

logit_model = sm.Logit(Y_logit, X_design).fit()
print(logit_model.summary())
print("Odds ratio for X:", np.exp(logit_model.params[1]))

# --- Manual IRLS for logistic regression ---
def irls_logistic(X, Y, n_iter=10):
    n = len(Y)
    X_d = np.column_stack([np.ones(n), X])
    beta = np.zeros(2)
    for _ in range(n_iter):
        eta = X_d @ beta
        p = 1/(1+np.exp(-eta))
        W = np.diag(p*(1-p))
        U = X_d.T @ (Y-p)
        XtWX = X_d.T @ W @ X_d
        beta = beta + np.linalg.inv(XtWX) @ U
    return beta

print("Manual IRLS logistic coefficients:", irls_logistic(X_logit, Y_logit))
```

```python
# --- Poisson regression ---
X_pois = np.array([1,2,3,4,5], dtype=float)
Y_pois = np.array([1,2,2,4,6], dtype=float)
X_design_p = sm.add_constant(X_pois)

pois_model = sm.GLM(Y_pois, X_design_p, family=sm.families.Poisson()).fit()
print(pois_model.summary())
print("Incidence Rate Ratio for X:", np.exp(pois_model.params[1]))

# Check for overdispersion
print("Deviance / df_resid (should be ~1 if no overdispersion):",
      pois_model.deviance / pois_model.df_resid)
```

```python
# --- GLM framework: same call structure across families ---
import statsmodels.api as sm

# Linear regression as a GLM (identity link, Gaussian family)
sm.GLM(Y_pois, X_design_p, family=sm.families.Gaussian()).fit()
# Logistic regression as a GLM (logit link, Binomial family)
sm.GLM(Y_logit, X_design, family=sm.families.Binomial()).fit()
# Poisson regression as a GLM (log link, Poisson family)
sm.GLM(Y_pois, X_design_p, family=sm.families.Poisson()).fit()
```

---

## Interview Question Bank — Chapter 14

**Conceptual:**
1. Name three specific reasons ordinary least squares is inappropriate for a binary response.
2. Why is a logistic regression coefficient interpreted multiplicatively on the odds scale, rather than additively on the probability scale?
3. What does "overdispersion" mean in Poisson regression, and why does it matter for inference?

**Derivation:**
4. Derive the IRLS/Newton-Raphson update for logistic regression from the score equations of the Bernoulli log-likelihood.
5. Explain why ordinary least squares is the special case of GLM/IRLS that converges in exactly one iteration.

**ML/Statistics:**
6. Compare the Wald test and likelihood ratio test for a logistic regression coefficient — when might they give meaningfully different conclusions?
7. What's the practical fix for overdispersed count data, and how does it change the model (coefficients vs. standard errors)?
8. Explain, in one unifying sentence, how linear regression, logistic regression, and Poisson regression relate to each other.

**Coding:**
9. Implement IRLS from scratch in NumPy for logistic regression, and verify it matches `statsmodels`' `Logit`.
10. Fit a Poisson regression model and check for overdispersion using the deviance-to-degrees-of-freedom ratio.

**Traps:**
11. "A one-unit increase in X increases the probability of Y=1 by β1." — correct this common misstatement.
12. "My Poisson model's coefficients look fine, so I don't need to check anything else." — what important check is being skipped?
13. Someone claims linear regression, logistic regression, and Poisson regression are "completely different techniques that happen to share some notation." What's the more accurate characterization?

---

*This file covers Kutner Ch. 14 — why OLS fails for binary and count responses, the logistic and Poisson models derived via maximum likelihood, IRLS/Newton-Raphson worked by hand for both models (directly unifying Chapters 11's IRLS and 13's Gauss-Newton), odds ratios and incidence rate ratios as the correct interpretive framework, Wald and likelihood ratio tests, overdispersion, and the full Generalized Linear Model framework tying every model in this course together as one family.*

---

## Course Wrap-Up

This completes the full Kutner *Applied Linear Statistical Models* arc as originally scoped: **Chapters 1–14**, covering simple linear regression, inference, diagnostics, the matrix framework, multiple regression estimation and inference, extra sums of squares, qualitative predictors and interactions, model building and selection, diagnostics for multiple regression, remedial measures (WLS, ridge, robust regression, bootstrapping), autocorrelation, nonlinear regression, and the generalized linear model framework unifying everything into logistic and Poisson regression.

Together, these 14 files form a self-contained reference spanning the statistical foundation expected at the L5 Data Scientist / ML Engineer level at Google-tier companies — from first-principles derivations through the exact machinery (IRLS, MLE, bias-variance tradeoffs) that underlies modern ML tooling.
