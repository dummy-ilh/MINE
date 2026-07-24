# Chapter 8 — Regression Models for Quantitative and Qualitative Predictors
### (Kutner, Nachtsheim, Neter, Li — *Applied Linear Statistical Models*)

Chapters 6–7 built the machinery for multiple regression with continuous predictors. Chapter 8 shows that the *same* machinery — no new math — handles polynomial terms, categorical (qualitative) predictors, and interactions, just by cleverly constructing the columns of $\mathbf{X}$.

---

## 8.1 Polynomial Regression Models

### The Model and Why It's Still "Linear" Regression

$$
Y_i = \beta_0+\beta_1X_i+\beta_2X_i^2+\varepsilon_i
$$

**Why this is still within the linear regression framework:** "linear" refers to linearity in the *parameters* ($\beta_0,\beta_1,\beta_2$ each enter additively, un-transformed), not linearity in $X$. We simply treat $X^2$ as if it were a second, separate predictor column — the entire matrix apparatus from Chapters 5–7 applies completely unchanged.

### The Multicollinearity Problem This Section Warns About — and Its Fix

**Plain English.** $X$ and $X^2$ are almost always strongly correlated over any positive range of X values (bigger X tends to mean bigger X² too) — a built-in, structural multicollinearity problem, not a data-quality issue. This inflates the variance of $b_1, b_2$ exactly as Chapter 7 described, making individual coefficients unstable, even though the fitted curve itself can still be highly accurate.

### Worked Example

Consider $X=1,2,3,4,5$ with $Y = X^2+X$ exactly (a clean quadratic relationship, no noise, chosen to make the arithmetic exact and to isolate the multicollinearity mechanism from estimation noise):

| X | Y |
|---|---|
| 1 | 2 |
| 2 | 6 |
| 3 | 12 |
| 4 | 20 |
| 5 | 30 |

**Correlation between X and X² (uncentered):**
$$
\bar X=3,\ \overline{X^2}=11,\quad S_{XX}=10,\ S_{X^2X^2}=374,\ S_{X,X^2}=60
$$
$$
r(X,X^2) = \frac{60}{\sqrt{10\times374}}=\frac{60}{61.156}=0.9811
$$

**A correlation of 0.981 between your two predictors** — severe built-in multicollinearity, exactly the concern Chapter 7 warned about, here arising purely from the structure of a polynomial term rather than any coincidence in the data.

### The Fix: Centering

Define $X' = X-\bar X = X-3$, giving $X'=-2,-1,0,1,2$.

$$
S_{X'X'} = 10 \ (\text{unchanged — centering never changes spread}), \quad \overline{X'^2}=2,\ S_{X'^2X'^2}=14
$$
$$
S_{X',X'^2}=\sum(X'-0)(X'^2-2) = (-2)(2)+(-1)(-1)+(0)(-2)+(1)(-1)+(2)(2) = -4+1+0-1+4=0
$$
$$
r(X',X'^2) = \frac{0}{\sqrt{10\times14}} = 0
$$

**Centering drove the correlation to exactly zero.** This isn't a coincidence specific to this dataset — for any *symmetric* set of X values around their mean, centered X and centered X² are exactly uncorrelated (an odd function times an even function, summed symmetrically, cancels exactly). For asymmetric X distributions, centering won't achieve *exactly* zero correlation, but it substantially reduces it in virtually all practical cases.

**Re-expressing the model in centered terms:** since $X=X'+3$,
$$
Y = X^2+X = (X'+3)^2+(X'+3) = X'^2+6X'+9+X'+3 = X'^2+7X'+12
$$
So the centered-model coefficients are $\gamma_0=12,\ \gamma_1=7,\ \gamma_2=1$ — and since our toy data was constructed with zero noise, this centered quadratic model fits **exactly** (SSE=0, R²=1).

**Interview question:** *"Why does centering a predictor before adding a squared term help with multicollinearity?"*
**Ideal answer:** X and X² are inherently correlated over most ranges — larger X mechanically produces larger X². Centering X at its mean before squaring removes most (and for a symmetric X distribution, all) of this correlation, because the centered linear term is an odd function around the mean while the centered quadratic term is even — their products cancel in the sum. This reduces the variance inflation on the coefficient estimates without changing the model's fitted values or predictive accuracy at all; it's a pure numerical/interpretive fix, not a different model.

**Practical stakes with real (noisy) data:** in this toy zero-noise example the raw, uncentered model would still fit perfectly too — the multicollinearity problem is invisible without noise. With real noisy data, however, fitting on raw X and X² typically produces wildly unstable, high-variance coefficient estimates (large standard errors, coefficients that swing drastically with small data changes), even though the *fitted curve* itself remains reasonable — centering fixes the instability in the coefficients without altering the shape of the fitted curve.

---

## 8.2 Interaction Regression Models (Quantitative × Quantitative, Briefly)

For two continuous predictors, an interaction term looks like:
$$
Y_i=\beta_0+\beta_1X_{i1}+\beta_2X_{i2}+\beta_3X_{i1}X_{i2}+\varepsilon_i
$$
**Interpretation:** the effect of $X_1$ on Y is no longer a single fixed number — it's $\beta_1+\beta_3X_2$, which *depends on the level of $X_2$*. This is the general mechanism; we'll fully work through the numbers using the qualitative-predictor case below, since interactions between a continuous and a categorical variable are both easier to interpret and far more common in applied interview contexts (e.g., "does the effect of ad spend on conversions differ by customer segment?").

---

## 8.3 Qualitative Predictors: Indicator (Dummy) Variables

### A New Worked Dataset

Predicting exam score (Y) from hours studied (X1, continuous) and whether the student took a prep course (X2 = 1 if yes, 0 if no):

| Group | X1 | Y |
|---|---|---|
| No prep (X2=0) | 2 | 60 |
| No prep (X2=0) | 4 | 68 |
| No prep (X2=0) | 6 | 74 |
| No prep (X2=0) | 8 | 82 |
| Prep (X2=1) | 2 | 68 |
| Prep (X2=1) | 4 | 78 |
| Prep (X2=1) | 6 | 86 |
| Prep (X2=1) | 8 | 96 |

(By design: fitting each group *separately* as simple regressions gives No-prep: $b_0=53, b_1=3.6$; Prep: $b_0=59, b_1=4.6$ — different intercepts **and** different slopes. We built it this way deliberately, to have a clean example for testing both effects.)

### The Additive (No-Interaction) Model: "Parallel Lines"

$$
Y_i = \beta_0+\beta_1X_{i1}+\beta_2X_{i2}+\varepsilon_i
$$

**What this model assumes, structurally:** the *same slope* $\beta_1$ for both groups — prep course only shifts the *intercept* by $\beta_2$, up or down, without changing how much each additional hour of studying is worth. This is a strong, testable assumption (tested formally below).

**Solving the normal equations** (built from $\sum X_1=40,\ \sum X_2=4,\ \sum Y=612,\ \sum X_1^2=240,\ \sum X_2^2=4,\ \sum X_1X_2=20,\ \sum X_1Y=3224,\ \sum X_2Y=328$, $n=8$), via elimination:

$$
b_0=50.5,\qquad b_1=4.1,\qquad b_2=11.0
$$

**Interpretation:** $b_1=4.1$ — holding prep-course status fixed, each additional hour studied predicts 4.1 more points. $b_2=11.0$ — holding hours studied fixed, taking the prep course is associated with 11 additional points, uniformly, regardless of how many hours studied.

**Fitted values and residuals** (worked by hand):

No-prep group ($\hat Y=50.5+4.1X_1$): fitted values 58.7, 66.9, 75.1, 83.3; residuals 1.3, 1.1, -1.1, -1.3.
Prep group ($\hat Y=61.5+4.1X_1$): fitted values 69.7, 77.9, 86.1, 94.3; residuals -1.7, 0.1, -0.1, 1.7.

$$
SSE_{\text{additive}} = 1.69+1.21+1.21+1.69+2.89+0.01+0.01+2.89 = 11.6
$$

**Notice the residual pattern:** in each group, residuals go from positive to negative to positive again in a slight arc — a telltale sign (echoing Chapter 3!) that this model is *systematically* misspecified: it's forcing a single common slope on data that actually has two different true slopes.

---

## 8.4 The Dummy Variable Trap and Reference Category

**Why we used exactly ONE indicator (X2), not two.** If we'd instead created two indicators — $X_2=1$ if prep, $X_3=1$ if no-prep (with $X_2+X_3=1$ always, for every observation) — the design matrix would include the intercept column (all 1's) **and** $X_2+X_3$, which is *also* all 1's. These columns are perfectly linearly dependent, meaning $\mathbf{X}'\mathbf{X}$ becomes **singular** (non-invertible) — directly recalling Chapter 5's rank discussion. This is the infamous **dummy variable trap**: for a categorical variable with $G$ categories, you need exactly $G-1$ indicator variables (plus the intercept), never $G$.

**Why this isn't a loss of information.** The omitted category (here, "no prep") becomes the **reference level** — its effects are absorbed into the intercept $\beta_0$ itself. $\beta_2$ is then interpreted *relative to* that reference category, which is exactly how we read $b_2=11.0$ above: "11 points *more than the no-prep baseline*."

**Interview question:** *"Why do you use G−1 dummy variables for a categorical predictor with G categories, instead of G?"*
**Ideal answer:** Including all G indicator columns alongside the intercept creates exact linear dependence (their sum equals the intercept column), making $\mathbf{X}'\mathbf{X}$ singular and the model unidentifiable — the "dummy variable trap." Using G−1 indicators, with one category designated the reference level absorbed into the intercept, preserves full rank while losing no information: every group's mean is still exactly recoverable (reference group = intercept; any other group = intercept + its own coefficient).

**How this shows up in ML pipelines in practice:** `pd.get_dummies(..., drop_first=True)` or scikit-learn's `OneHotEncoder(drop='first')` — both exist specifically to avoid this trap. Forgetting the `drop_first`/`drop='first'` argument is an extremely common real-world bug that produces a singular or near-singular design matrix (or, with regularized models, silently distorts coefficient interpretation).

---

## 8.5 Modeling Interaction Between a Quantitative and Qualitative Predictor

### The Saturated (Interaction) Model

$$
Y_i = \beta_0+\beta_1X_{i1}+\beta_2X_{i2}+\beta_3(X_{i1}\times X_{i2})+\varepsilon_i
$$

**Why this exactly reproduces two separate group regressions — a shortcut worth understanding rather than grinding through a 4×4 matrix inversion by hand.** This model has exactly 4 parameters for exactly 2 groups × 2 (intercept, slope) — it's **saturated**: fully flexible enough to let each group have its own completely independent intercept and slope. When a model has exactly this much flexibility relative to the group structure, its pooled OLS solution is *mathematically guaranteed* to equal fitting the two groups completely separately (each group's own residuals are minimized independently, since the interaction term "unlocks" a fully separate slope for each group with no shared constraint between them).

For $X_2=0$ (no-prep): the model reduces to $Y=\beta_0+\beta_1X_1$ — matching our separately-fit no-prep regression exactly: $\beta_0=53,\ \beta_1=3.6$.
For $X_2=1$ (prep): the model becomes $Y=(\beta_0+\beta_2)+(\beta_1+\beta_3)X_1$ — matching the prep group's separate fit ($59 = \beta_0+\beta_2 \Rightarrow \beta_2=6$; $4.6=\beta_1+\beta_3\Rightarrow\beta_3=1.0$).

$$
\boxed{b_0=53,\quad b_1=3.6,\quad b_2=6.0,\quad b_3=1.0}
$$

**Interpreting $b_3=1.0$ — the interaction coefficient, the genuinely new object in this section.** It says: the *effect of one more hour of studying* is 1.0 points **larger** for prep-course students than for non-prep students (4.6 vs. 3.6 points per hour). This is fundamentally different information from $b_2=6.0$ (a fixed intercept shift) — $b_3$ tells you the *slope itself* differs by group, i.e., prep-course students get *more* marginal benefit from each additional hour studied, not just a flat head start.

$$
SSE_{\text{interaction}} = (\text{sum of both groups' own SSE}) = 0.8+0.8=1.6
$$

**Compare $SSE_{\text{additive}}=11.6$ vs. $SSE_{\text{interaction}}=1.6$** — a dramatic reduction, telling us visually that forcing parallel slopes cost us real fit quality. Let's formalize that with a test.

---

## 8.7 Comparing Regression Functions Across Groups — the Nested F-Test Hierarchy

This is a direct application of the general linear test (Chapter 2.8 / Chapter 7.3): compare nested models via their SSE and degrees of freedom.

### Test 1: Does the Slope Differ by Group? (Is the Interaction Term Needed?)

Full model = interaction model ($SSE=1.6$, $df=8-4=4$). Reduced model = additive model ($SSE=11.6$, $df=8-3=5$).

$$
F^* = \frac{(SSE_{\text{reduced}}-SSE_{\text{full}})/(df_{\text{reduced}}-df_{\text{full}})}{SSE_{\text{full}}/df_{\text{full}}} = \frac{(11.6-1.6)/1}{1.6/4}=\frac{10.0}{0.4}=25.0
$$

Critical value: $F_{(0.95;1,4)}=7.71$. Since $25.0\gg7.71$, **reject $H_0:\beta_3=0$** — the slopes genuinely differ between groups; the interaction term is real and necessary (exactly as we designed into the data).

### Test 2: Does the Group Matter at All? (Is Even the Intercept Shift Needed?)

Fitting the **fully pooled** model (ignoring group entirely — regress Y on X1 alone, across all 8 observations, both groups combined): using $\bar X_1=5,\ S_{X_1X_1}=40,\ S_{X_1Y}=164$:
$$
b_1=\frac{164}{40}=4.1,\qquad b_0=76.5-4.1(5)=56.0
$$
Computing fitted values and residuals for all 8 points against this single pooled line gives:
$$
SSE_{\text{pooled}} = 253.6, \qquad df=8-2=6
$$

**Full model = additive model** ($SSE=11.6, df=5$). **Reduced model = pooled, single-line model** ($SSE=253.6, df=6$).

$$
F^* = \frac{(253.6-11.6)/1}{11.6/5}=\frac{242.0}{2.32}=104.3
$$

Since $104.3\gg7.71$ (same critical value, $df=1,5$ here technically $F_{(.95;1,5)}\approx6.61$, still far exceeded), **strongly reject** — group membership (prep course) has a real, substantial effect on the intercept.

### The Full Hierarchy, Side by Side

| Model | SSE | df | Interpretation |
|---|---|---|---|
| Pooled (ignore group) | 253.6 | 6 | Wildly misspecified — ignores a huge, real group effect |
| Additive (parallel lines) | 11.6 | 5 | Captures the intercept shift, misses the slope difference |
| Interaction (separate lines) | 1.6 | 4 | Fully captures both the intercept shift and slope difference |

**This nested hierarchy is precisely how Kutner's Section 8.7 frames "comparison of two or more regression functions"** — and it's exactly the same logical structure as the Chow test that shows up in econometrics interviews (testing whether a regression relationship is stable across two distinct regimes/groups/time periods is mathematically identical to this exact procedure).

**Interview question:** *"How would you formally test whether the relationship between a continuous predictor and an outcome differs across two groups?"*
**Ideal answer:** Fit a model with an interaction term between the continuous predictor and a group indicator, and compare it (via a nested F-test / general linear test) against the reduced model without the interaction term. A significant F-test on the interaction term means the slope genuinely differs by group; you can further test whether even the intercept differs by comparing the additive model against a fully pooled model ignoring group altogether. This is exactly the logic behind the classical Chow test for structural stability across regimes or time periods.

---

## Python Implementation — From Scratch (NumPy) and statsmodels

```python
import numpy as np
from scipy import stats

X1 = np.array([2,4,6,8,2,4,6,8], dtype=float)
X2 = np.array([0,0,0,0,1,1,1,1], dtype=float)  # indicator: prep course
Y  = np.array([60,68,74,82,68,78,86,96], dtype=float)
n = len(Y)

def fit_ols(Xcols, Y):
    X = np.column_stack([np.ones(len(Y))] + Xcols)
    b = np.linalg.inv(X.T@X) @ X.T @ Y
    resid = Y - X@b
    SSE = np.sum(resid**2)
    return b, SSE, X.shape[1]

# Pooled model (ignore group)
b_pooled, SSE_pooled, p_pooled = fit_ols([X1], Y)

# Additive model
b_add, SSE_add, p_add = fit_ols([X1, X2], Y)

# Interaction model
b_int, SSE_int, p_int = fit_ols([X1, X2, X1*X2], Y)

print("Pooled:", b_pooled, SSE_pooled)
print("Additive:", b_add, SSE_add)
print("Interaction:", b_int, SSE_int)

# Nested F-tests
def nested_F(SSE_r, df_r, SSE_f, df_f):
    F = ((SSE_r-SSE_f)/(df_r-df_f)) / (SSE_f/df_f)
    p_val = 1 - stats.f.cdf(F, df_r-df_f, df_f)
    return F, p_val

df_pooled, df_add, df_int = n-p_pooled, n-p_add, n-p_int
F1, p1 = nested_F(SSE_add, df_add, SSE_int, df_int)   # interaction test
F2, p2 = nested_F(SSE_pooled, df_pooled, SSE_add, df_add)  # group effect test
print(f"Interaction test: F={F1:.2f}, p={p1:.5f}")
print(f"Group-effect test: F={F2:.2f}, p={p2:.5f}")
```

## statsmodels — with formula interface (recommended for interaction models)

```python
import statsmodels.formula.api as smf
import pandas as pd

df = pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})

model_add = smf.ols('Y ~ X1 + X2', data=df).fit()
model_int = smf.ols('Y ~ X1 * X2', data=df).fit()  # '*' automatically includes X1, X2, and X1:X2

print(model_int.summary())

# Formal nested model comparison
from statsmodels.stats.anova import anova_lm
print(anova_lm(model_add, model_int))
```

---

## Interview Question Bank — Chapter 8

**Conceptual:**
1. Why is "linear regression" still the right term for polynomial regression models like $Y=\beta_0+\beta_1X+\beta_2X^2$?
2. What is the dummy variable trap, and why does it make $\mathbf{X}'\mathbf{X}$ singular?
3. What's the difference in interpretation between an indicator variable's coefficient (e.g., $\beta_2$) and an interaction coefficient (e.g., $\beta_3$) in a model with a continuous and categorical predictor?

**Derivation:**
4. Show why centering X before adding a quadratic term reduces (or, for symmetric designs, eliminates) the correlation between the linear and quadratic terms.
5. Explain why a fully interacted model (continuous × categorical, all terms included) reproduces exactly the same fit as running separate regressions per group.

**ML/Statistics:**
6. How would you test, formally, whether two groups' regression lines have significantly different slopes vs. just different intercepts?
7. In a real feature pipeline, what's the practical bug that recreates the dummy variable trap, and how do standard libraries guard against it?
8. Why might a data scientist deliberately choose an additive (no-interaction) model even after seeing some visual evidence that slopes might differ slightly between groups?

**Coding:**
9. Implement, from scratch, the nested F-test comparing a pooled model, an additive model, and an interaction model for a continuous + categorical predictor setup.
10. Using `statsmodels.formula.api`, fit an interaction model with the `*` operator and interpret each coefficient in the output.

**Traps:**
11. "I one-hot-encoded my categorical variable into all G columns to preserve all information." — what breaks, and how do you fix it?
12. "The additive model's R² was already quite high, so I didn't need to check for an interaction." — why is this reasoning incomplete (connect to the SSE comparison in this chapter)?
13. Someone interprets a large, significant intercept-shift coefficient ($\beta_2$) as evidence the *slope* also differs by group. What's the correct way to test that separately?

---

*This file covers Kutner Ch. 8 — polynomial regression and centering (worked example showing correlation between X and X² dropping from 0.981 to exactly 0 after centering), indicator variables and the dummy variable trap, interaction models between continuous and categorical predictors (fully worked, showing the saturated-model shortcut), and the nested F-test hierarchy for comparing regression functions across groups (pooled vs. additive vs. interaction, F*=104.3 and F*=25.0 respectively). Chapter 9 (Building the Regression Model I) is next — variable selection criteria (adjusted R², Mallows' Cp, AIC/BIC), stepwise procedures, and the model-building philosophy that ties Chapters 6–8 together into a practical workflow.*
