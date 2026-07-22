Here is your expanded **Module 10: Assumptions & Diagnostics** — fully updated with deeper interview scenarios, real-world failure cases, visual intuition, diagnostic metrics, mathematical foundations, and end-to-end Python code snippets. Nothing has been deleted.

---

# Module 10 — Assumptions & Diagnostics

## 1. WHY

Every model makes implicit assumptions about the data it's trained on. If those assumptions are badly violated, the model can still *run* and produce numbers — but those numbers become unreliable in ways that are easy to miss unless you know what to check. This module answers: **what does logistic regression silently assume about your data, and how do you catch it if those assumptions are broken?**

**What breaks if you ignore this:** You could ship a model with seemingly reasonable coefficients that are actually unstable, misleading, or wildly overconfident — not because the math is wrong, but because the data violated an assumption the math depends on. This is a favorite area for L5 interviewers because it separates "I can call `.fit()`" from "I understand what I'm fitting."

---

## 2. INTUITION

Think of logistic regression's assumptions like the fine print on a ladder: *"Only use on flat, stable ground."* The ladder still physically exists and you can still climb it on uneven ground — but the manufacturer's safety guarantee no longer applies, and it might wobble or tip in ways you didn't expect.

Assumptions in statistics work the same way: the model still fits and produces output even when assumptions are violated, but the guarantees about that output (reliable coefficients, valid interpretations, stable predictions, reliable p-values) quietly stop holding.

```
       [ Ideal Ground ]                   [ Uneven Ground ]
   (Assumptions Satisfied)             (Assumptions Violated)

         |---|  <- Solid                 |---|  <- Unstable
         |---|  <- Predictable           |/--|  <- Misleading
         |---|  <- Trustworthy          /|---|  <- High Variance
        =======                        =================

```

---

## 3. THE CORE ASSUMPTIONS (EXPANDED)

While standard linear regression requires normality of residuals and homoscedasticity, logistic regression **does not** assume normal residuals or constant variance (since binary outcomes are inherently heteroscedastic: $Var(Y\vert{}X) = p(1-p)$). However, it relies heavily on four distinct structural assumptions:

```
+-------------------------------------------------------------------+
|                  LOGISTIC REGRESSION ASSUMPTIONS                  |
+---------------------------------+---------------------------------+
|   COLUMN-SIDE (Features)        |   ROW-SIDE (Observations)       |
+---------------------------------+---------------------------------+
| 1. Linearity in Log-Odds        | 3. Independence of Errors       |
| 2. No Severe Multicollinearity  | 4. Absence of High-Leverage     |
|                                 |    Outliers / Separation        |
+---------------------------------+---------------------------------+

```

---

### Assumption 1 — Linearity in the Log-Odds

**In words:** Logistic regression assumes that each continuous feature has a **constant, straight-line relationship with the LOG-ODDS** of the outcome.

$$\eta = \ln\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 X_1 + \dots + \beta_k X_k$$

It does **not** assume a linear relationship with raw probabilities (which follow the sigmoidal S-curve) or binary class labels.

**What breaks if violated:** If the true underlying relationship is non-linear (e.g., U-shaped, threshold-based, or logarithmic), a standard linear log-odds model fits a straight line through that curve. This creates systematic region-specific bias — drastically miscalculating probabilities at the extremes or inflection points.

**How to check:**

1. **Empirical Log-Odds Plotting (Binning):** Group a continuous feature into deciles or bins, calculate the sample log-odds $\ln\left(\frac{\hat{p}}{1 - \hat{p}}\right)$ per bin, and plot them against the bin midpoints. Look for non-linear curves.
2. **Box-Tidwell Test:** Add an interaction term $X_i \ln(X_i)$ for each continuous feature into the model. If the coefficient of $X_i \ln(X_i)$ is statistically significant ($p < 0.05$), the linearity-in-log-odds assumption is violated for feature $X_i$.

**Fix if violated:**

* Add non-linear transformations (e.g., $\ln(X)$, $\sqrt{X}$).
* Add higher-order polynomial terms ($X^2, X^3$).
* Use generalized additive models (GAMs) or spline terms (e.g., restricted cubic splines).
* Bin the feature into meaningful categorical buckets (though this loses continuous information).

---

### Assumption 2 — Independence of Observations

**In words:** Each observation (row) must be independent of every other observation. Knowing the outcome of observation $i$ should provide zero information about observation $j$, conditional on the features.

**What breaks if violated:** When data points are clustered or repeated across time/entities (e.g., multiple purchases by the same user, patient visits, spatial neighborhood clusters), standard errors are **underestimated**. The model overestimates its sample size, leading to artificially narrow confidence intervals, inflated $z$-scores, and **false positive significance (Type I errors)**.

**How to check:**

* Audit the data collection process: Are there repeated entity IDs, time-series dependencies, or hierarchical groupings?
* Calculate intra-cluster correlation coefficients (ICC) or examine residual autocorrelation plots across time/groups.

**Fix if violated:**

* **Cluster-Robust Standard Errors (Huber-White / Sandwich estimators):** Adjusts variance estimates without changing point coefficients.
* **Mixed-Effects / Multilevel Logistic Models:** Explicitly models random intercepts/slopes per entity/group.
* **Generalized Estimating Equations (GEE):** Ideal for longitudinal panel data when population-averaged effects are preferred.
* **Aggregation:** Collapse repeated measures into a single summary row per entity.

---

### Assumption 3 — No Severe Multicollinearity

**In words:** Features (columns) should not be highly collinear. No feature should be a linear combination or strong function of other features.

**What breaks if violated:** Multicollinearity inflates the variance of coefficient estimates $\text{Var}(\hat{\beta}_j)$. While global metrics (like overall accuracy, ROC-AUC, or log-loss) often remain stable, individual coefficients become wildly sensitive to tiny perturbations in the training data. Signs can flip unexpectedly (e.g., a known risk factor getting a negative weight), making interpretation impossible.

**How to check:**

1. **Correlation Matrix:** Quick pairwise inspection ($\vert{}r\vert{} > 0.8$ signals high collinearity). *Warning: Misses multi-variable linear combinations.*
2. **Variance Inflation Factor (VIF):** Measures how much the variance of $\hat{\beta}_j$ is inflated due to collinearity with all other features.

$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$

*Where $R_j^2$ is the $R^2$ obtained by regressing feature $X_j$ on all other features.*

| VIF Value | Interpretation | Action Required |
| --- | --- | --- |
| $\text{VIF} = 1$ | Completely un-correlated | None |
| $1 < \text{VIF} < 5$ | Moderate, acceptable correlation | Monitor |
| $5 \le \text{VIF} < 10$ | High correlation | Inspect coefficients closely |
| $\text{VIF} \ge 10$ | Severe multicollinearity | Action required (drop, combine, or regularize) |

**Fix if violated:**

* Drop one of the collinear features based on domain logic.
* Combine correlated features into an index, ratio, or composite score (e.g., `Debt-to-Income`).
* Use **L2 Regularization (Ridge)**: Adds a penalty $\lambda \Vert{}\beta\Vert{}_2^2$ that shrinks collinear coefficients together, stabilizing variance.
* Dimensionality reduction techniques like Principal Component Analysis (PCA).

---

### Assumption 4 — Absence of Extreme Outliers & High-Leverage Points

**In words:** The model should not be heavily dictated by a handful of anomalous rows with extreme feature values or severe misclassifications.

**What breaks if violated:** A few high-leverage points can pull the decision boundary toward them, distorting the fit for the rest of the dataset.

**How to check:**

* **Deviance Residuals:** Measures each row's contribution to overall log-loss.
* **Cook's Distance / DfBeta:** Quantifies how much coefficients change when observation $i$ is removed from the dataset. Observations with Cook's distance $> \frac{4}{N}$ warrant investigation.

---

### Special Case: Quasi-Complete & Complete Separation

While not strictly a traditional linear model assumption, **separation** is a fatal diagnostic failure unique to binary models.

* **Complete Separation:** A feature perfectly splits the binary outcome (e.g., $X > 5 \implies Y=1$ always, $X \le 5 \implies Y=0$ always).
* **Quasi-Complete Separation:** A feature perfectly predicts one class for a range, but has overlap elsewhere.

**What happens:** The Maximum Likelihood Estimator (MLE) attempts to push $\beta \to \infty$ or $-\infty$ to achieve a probability of $1.0$ or $0.0$. The optimizer fails to converge, producing massive standard errors and wildly inflated coefficients.

**Fix:** Apply **Firth's Penalized Likelihood** (adds a Jeffreys prior to the likelihood), use **L2 regularization**, or collapse/re-bin sparse categories.

---

## 4. WORKED NUMERIC EXAMPLE — Checking Linearity in Log-Odds

Let's check whether "months as customer" (tenure) has a linear relationship with log-odds of churn, using binned empirical data:

| Tenure bucket | Customers ($N_i$) | Churned ($y_i$) | Empirical $p$ ($\frac{y_i}{N_i}$) | Empirical Odds ($\frac{p}{1-p}$) | Empirical Log-Odds $\ln\left(\frac{p}{1-p}\right)$ |
| --- | --- | --- | --- | --- | --- |
| 0–6 months | 100 | 40 | 0.40 | 0.40 / 0.60 = 0.667 | $\ln(0.667) = -0.405$ |
| 6–12 months | 100 | 25 | 0.25 | 0.25 / 0.75 = 0.333 | $\ln(0.333) = -1.099$ |
| 12–24 months | 100 | 15 | 0.15 | 0.15 / 0.85 = 0.176 | $\ln(0.176) = -1.735$ |
| 24–36 months | 100 | 10 | 0.10 | 0.10 / 0.90 = 0.111 | $\ln(0.111) = -2.197$ |

### Step 1: Compute Stepwise Differences

Evaluate the change in log-odds ($\Delta \eta$) across successive equal-sized steps:

$$\Delta_1 = -1.099 - (-0.405) = -0.694$$

$$\Delta_2 = -1.735 - (-1.099) = -0.636$$

$$\Delta_3 = -2.197 - (-1.735) = -0.462$$

### Step 2: Diagnostic Evaluation

* **Result:** The differences ($-0.694$, $-0.636$, $-0.462$) show a consistent negative slope with mild attenuation at the high-tenure end.
* **Conclusion:** The relationship is **approximately linear** in log-odds. A standard linear term for tenure is acceptable here.
* **Contrast Scenario (Non-Linearity):** If differences were $+0.80$, $-0.10$, $-3.50$ (changing direction or magnitude drastically), this would signal non-linearity, requiring splines or polynomial terms.

---

## 5. INTERPRETATION & REAL-WORLD IMPLICATIONS

Before trusting your model's coefficients enough to make business claims ("every additional complaint triples churn odds"), it's worth a quick sanity pass on these four assumptions.

> **Key Rule:** A model can have an impressive ROC-AUC on cross-validation while simultaneously possessing broken, uninterpretable, or unstable coefficients due to collinearity or independence violations.

If your goal is **pure prediction** (e.g., ranking leads), mild multicollinearity can be tolerated. But if your goal is **causal inference, policy design, or feature attribution** (e.g., determining which operational lever to pull), assumption diagnostics are strictly non-negotiable.

---

## 6. COMPLETE DIAGNOSTIC SUITE IN PYTHON

This script generates synthetic data, introduces deliberate assumption violations (non-linearity, collinearity, separation), and executes a complete diagnostic pipeline.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# -------------------------------------------------------------------
# 1. Generate Synthetic Dataset with Known Violations
# -------------------------------------------------------------------
np.random.seed(42)
n = 500

# Feature 1: Linear in log-odds
x_linear = np.random.uniform(-3, 3, n)

# Feature 2 & 3: Severe Multicollinearity (r ~ 0.98)
x_collinear1 = np.random.normal(0, 1, n)
x_collinear2 = x_collinear1 * 0.95 + np.random.normal(0, 0.1, n)

# Feature 4: Non-linear in log-odds (U-shaped quadratic relationship)
x_nonlinear = np.random.uniform(-2, 2, n)

# True Log-Odds Equation
log_odds = (
    0.5 
    + 0.8 * x_linear 
    + 0.5 * x_collinear1 
    + 0.0 * x_collinear2  # True weight is 0
    + 1.2 * (x_nonlinear**2) - 1.5  # Quadratic effect
)

# Convert to probabilities via Sigmoid
probs = 1 / (1 + np.exp(-log_odds))
y = np.random.binomial(1, probs)

df = pd.DataFrame({
    'y': y,
    'x_linear': x_linear,
    'x_collinear1': x_collinear1,
    'x_collinear2': x_collinear2,
    'x_nonlinear': x_nonlinear
})

# -------------------------------------------------------------------
# 2. Check Assumption: Multicollinearity (VIF)
# -------------------------------------------------------------------
print("=" * 60)
print("1. MULTICOLLINEARITY DIAGNOSTIC (VIF)")
print("=" * 60)

X_vif = sm.add_constant(df[['x_linear', 'x_collinear1', 'x_collinear2', 'x_nonlinear']])
vif_data = pd.DataFrame({
    "Feature": X_vif.columns,
    "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
})
print(vif_data[vif_data["Feature"] != "const"].to_string(index=False))

# -------------------------------------------------------------------
# 3. Check Assumption: Linearity in Log-Odds (Box-Tidwell Test)
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. LINEARITY IN LOG-ODDS DIAGNOSTIC (Box-Tidwell Terms)")
print("=" * 60)

# Create interaction terms X * ln(X). Shift values to strictly positive first.
df_bt = df.copy()
for col in ['x_linear', 'x_nonlinear']:
    # Shift to positive range for log calculation
    shifted_val = df_bt[col] - df_bt[col].min() + 1
    df_bt[f'{col}_bt'] = shifted_val * np.log(shifted_val)

X_bt = sm.add_constant(df_bt[['x_linear', 'x_nonlinear', 'x_linear_bt', 'x_nonlinear_bt']])
logit_bt = sm.Logit(df_bt['y'], X_bt).fit(disp=False)

print("Box-Tidwell Interaction p-values:")
print(f"  x_linear interaction p-value:    {logit_bt.pvalues['x_linear_bt']:.4f} (Expected > 0.05)")
print(f"  x_nonlinear interaction p-value: {logit_bt.pvalues['x_nonlinear_bt']:.4f} (Expected < 0.05 -> Violation!)")

# -------------------------------------------------------------------
# 4. Check Outliers & Influence (Cook's Distance)
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. OUTLIER & INFLUENCE DIAGNOSTIC (Cook's Distance)")
print("=" * 60)

X_base = sm.add_constant(df[['x_linear', 'x_collinear1', 'x_collinear2', 'x_nonlinear']])
model = sm.Logit(df['y'], X_base).fit(disp=False)

influence = model.get_influence()
cooks_d = influence.cooks_distance[0]
high_influence_threshold = 4 / len(df)
influential_points = np.where(cooks_d > high_influence_threshold)[0]

print(f"Total observations: {len(df)}")
print(f"Threshold (4/N): {high_influence_threshold:.4f}")
print(f"Number of potentially high-leverage points: {len(influential_points)}")

```

---

## 7. FAANG L5 INTERVIEW PREPARATION

### Common Scenario Questions & Expert Responses

#### Scenario A: The Confused Stakeholder

**Interviewer:** *"We ran a churn model. `income` and `wealth_index` both have $p$-values of $0.45$, so our product manager wants to drop both. But when we train a model with ONLY `income`, its $p$-value becomes $0.0001$. What is happening, and how do you advise the team?"*

**L5 Response:**

> "This is a textbook symptom of severe **multicollinearity**. When both features are present, they share high mutual information. The model inflates the standard errors of both coefficients because it cannot attribute variance to one over the other, causing their individual $z$-tests to fail ($p > 0.05$).
> Dropping *both* features would discard genuinely predictive signal. I would check their Variance Inflation Factors (VIF) or pairwise correlation. To fix this, we should either drop the less interpretable feature, combine them into an index, or apply L2 regularization (Ridge) to stabilize the coefficient estimates."

---

#### Scenario B: The Repeated Measures Trap

**Interviewer:** *"We have a dataset of 1,000,000 ad clicks across 10,000 unique users (100 clicks per user). We fit a standard logistic regression model and get extremely small $p$-values ($p < 10^{-12}$) for almost all features. Should we trust these p-values?"*

**L5 Response:**

> "No, those $p$-values are artificially deflated due to a violation of the **Independence of Observations** assumption. Standard logistic regression assumes 1,000,000 independent samples, when in reality we have repeated measurements clustered across 10,000 users.
> The model underestimates coefficient standard errors by treating within-user variance as independent evidence. To fix this without altering point estimates, we must use **cluster-robust standard errors** grouped by `user_id`. Alternatively, we could fit a **Mixed-Effects Logistic Model** with a random intercept per user to account for baseline user propensities."

---

#### Scenario C: Complete Separation

**Interviewer:** *"You train a logistic regression model in `scikit-learn` without regularization (`penalty=None`), and one binary feature `has_visited_pricing_page` gets a coefficient of $+35.2$ with a standard error of $8,400. What happened?"*

**L5 Response:**

> "That is caused by **complete or quasi-complete separation**. Every single user who converted in the training set visited the pricing page, or no one who didn't convert visited it.
> Maximum Likelihood Estimation breaks down here: to make predicted probability as close to 1.0 as possible, the optimizer attempts to drive $\beta \to \infty$. In practice, it stops when reaching solver tolerance limits, leaving an enormous coefficient and an exploded standard error. I would resolve this by adding L2 regularization penalty, applying **Firth’s penalized likelihood**, or combining rare categories."

---

### Comparative Summary Table

| Assumption | Diagnostic Tool | Primary Risk if Violated | Recommended Remediation |
| --- | --- | --- | --- |
| **Linearity in Log-Odds** | Empirical Log-Odds Plots, Box-Tidwell Test | Region-specific systematic prediction bias | Splines, Polynomials, Transformations, Binning |
| **Independence of Errors** | ICC, Residual Autocorrelation, Entity Audits | False precision, artificially low $p$-values | Cluster-Robust SEs, Mixed-Effects Models, GEE |
| **No Multicollinearity** | Correlation Matrix ($\vert{}r\vert{} > 0.8$), VIF ($> 5-10$) | Unstable, erratic, uninterpretable coefficients | Drop redundant feature, L2 Regularization, PCA |
| **No Extreme Outliers** | Deviance Residuals, Cook's Distance ($> \frac{4}{N}$) | Distortion of global decision boundary | Robust scaling, inspection/removal of data errors |
| **No Separation** | Parameter Convergence Checks, Std Error Inspection | Non-convergence, infinite coefficients ($\beta \to \infty$) | L2 Regularization, Firth's Regression, Feature Binning |

---

## 8. CHECK YOUR UNDERSTANDING

### Question 1

You compute a correlation matrix and find two features have a correlation of $0.95$. Your model's overall accuracy and ROC-AUC look completely fine on test data. Should you be concerned? Why or why not, and what specifically would you be worried about?

> **Answer:** It depends strictly on the model's intended use case. High correlation ($0.95$) indicates severe multicollinearity, which inflates parameter variance and renders individual coefficients unstable and uninterpretable. If your objective is **pure prediction**, it is usually not a top-priority blocker because the combined predictive capability remains intact. However, if your objective is **attribution, causal inference, or policy decision-making** ("which feature drives churn?"), you must be deeply concerned—you cannot trust individual coefficient signs or magnitudes.

### Question 2

A colleague states: *"Logistic regression assumes a linear relationship between the input features and the churn outcome."* What is imprecise about this statement, and how would you correct it precisely?

> **Answer:** The statement is incorrect on two fronts:
> 1. Logistic regression does **not** assume linearity with the raw binary outcome ($Y \in \{0, 1\}$).
> 2. It does **not** assume linearity with the probability ($p \in [0, 1]$), which follows a non-linear sigmoidal curve.
> 
> 
> **Correction:** Logistic regression assumes a linear relationship between the continuous input features and the **LOG-ODDS** (logit) of the outcome: $\ln\left(\frac{p}{1-p}\right) = X\beta$.
