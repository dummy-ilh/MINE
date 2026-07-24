# Chapter 8 — Regression Models for Quantitative and Qualitative Predictors
### (Kutner, Nachtsheim, Neter, Li — *Applied Linear Statistical Models*)

Chapters 6–7 built the machinery of multiple regression assuming all predictors were continuous/quantitative. Chapter 8 extends the *same* machinery to handle **categorical (qualitative)** predictors, and to model situations where the relationship between X and Y genuinely differs across groups.

### A New Worked Dataset

Predicting salary (Y, $k) from years of experience (X1, quantitative) and department (categorical: Engineering vs. Sales), n=8 (4 employees per department):

| i | X1 (years) | Department | D (indicator: 1=Sales, 0=Eng) | Y (salary, $k) |
|---|---|---|---|---|
| 1 | 1 | Engineering | 0 | 50 |
| 2 | 2 | Engineering | 0 | 57 |
| 3 | 3 | Engineering | 0 | 61 |
| 4 | 4 | Engineering | 0 | 69 |
| 5 | 1 | Sales | 1 | 47 |
| 6 | 2 | Sales | 1 | 59 |
| 7 | 3 | Sales | 1 | 67 |
| 8 | 4 | Sales | 1 | 79 |

---

## 8.1 Polynomial Regression Models (Brief Recap)

Already flagged in Chapter 6.1: a model like $Y_i=\beta_0+\beta_1X_i+\beta_2X_i^2+\varepsilon_i$ is still a **linear** model (linear in the $\beta$'s), fit with the exact same $\mathbf{b}=(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{Y}$ machinery — just treat $X_i^2$ as if it were a second, separate predictor column. **One practical caution worth restating here:** X and $X^2$ are often highly correlated over a limited data range, creating exactly the multicollinearity symptoms from Chapter 7 (inflated VIF, unstable individual coefficients) — centering X (subtracting $\bar X$ before squaring) substantially reduces this correlation and is standard practice.

---

## 8.2 Interaction Regression Models (Two Quantitative Predictors)

For two continuous predictors, adding a **product term** $X_1X_2$ lets the effect of one predictor depend on the level of the other:
$$
Y_i = \beta_0+\beta_1X_{i1}+\beta_2X_{i2}+\beta_3X_{i1}X_{i2}+\varepsilon_i
$$
**Interpretation of $\beta_3$:** it's the change in the slope of X1 on Y for each one-unit increase in X2 (or symmetrically, the change in X2's slope per unit of X1) — formally, $\frac{\partial E[Y]}{\partial X_1}=\beta_1+\beta_3X_2$, which is no longer a fixed number but depends on $X_2$'s value. **This is the single biggest conceptual shift interaction terms introduce:** without interaction, "the effect of X1" is one fixed number; with interaction, it's a *function* of the other variable.

We'll develop this same idea far more concretely in Section 8.5 below, using a categorical rather than continuous second variable — much easier to build intuition on, since "different lines for different groups" is more visualizable than "a slope that's a function of another continuous variable."

---

## 8.3 Qualitative Predictors: Indicator (Dummy) Variables

### The Basic Idea

**Plain English.** Department (Engineering/Sales) isn't a number — you can't put "Engineering" into a regression equation. The fix: create an **indicator variable** $D_i$ that equals 1 for one category (Sales) and 0 for the other (Engineering, the **reference category**).

$$
Y_i = \beta_0+\beta_1X_{i1}+\beta_2D_i+\varepsilon_i \qquad \text{(additive, no interaction — "common slope" model)}
$$

**What each piece means:**
- For Engineering ($D_i=0$): $E[Y_i]=\beta_0+\beta_1X_{i1}$.
- For Sales ($D_i=1$): $E[Y_i]=(\beta_0+\beta_2)+\beta_1X_{i1}$.

So $\beta_2$ is literally the **vertical shift in intercept** between the two groups' regression lines — this model forces **both groups to share the exact same slope** ($\beta_1$), differing only in their starting point.

### Worked Example: Fitting the Common-Slope (Additive) Model

Since the model forces one shared slope, the correct "pooled" slope estimate combines both groups' within-group variation:
$$
b_1 = \frac{S_{XY,\text{Eng}}+S_{XY,\text{Sales}}}{S_{XX,\text{Eng}}+S_{XX,\text{Sales}}}
$$

Computing each group's simple-regression building blocks first (same method as Chapter 1):

**Engineering** ($\bar X=2.5,\ \bar Y=59.25$): $S_{XX}=5.0$, $S_{XY}=30.5$ → simple slope would be $6.1$.
**Sales** ($\bar X=2.5,\ \bar Y=63.0$): $S_{XX}=5.0$, $S_{XY}=52.0$ → simple slope would be $10.4$.

**Pooled common slope:**
$$
b_1 = \frac{30.5+52.0}{5.0+5.0}=\frac{82.5}{10}=8.25
$$

**Group-specific intercepts using this shared slope:**
$$
b_{0,\text{Eng}} = 59.25-8.25(2.5)=38.625, \qquad b_{0,\text{Sales}}=63.0-8.25(2.5)=42.375
$$
$$
b_2 = b_{0,\text{Sales}}-b_{0,\text{Eng}} = 42.375-38.625=3.75
$$

**Fitted common-slope model:** $\hat Y = 38.625+8.25X_1+3.75D$

Computing residuals for both groups and summing squares gives $SSE_{\text{common-slope}}=52.125$ (full arithmetic omitted for space, same method as always: fitted value minus actual, squared, summed).

**Interview trap:** People sometimes think a dummy variable coefficient like $b_2=3.75$ means "Sales employees earn $3,750 more than Engineering employees, period." More precisely: it means Sales employees earn about $3,750 more **at any given level of X1 (years of experience)**, under the assumption that the *raise-per-year-of-experience* is identical across departments — an assumption we're about to test, and which will turn out to be false for this data.

---

## 8.4 Considerations in Using Indicator Variables

### The Dummy Variable Trap

**Why you use $c-1$ indicators for $c$ categories, never $c$.** If department had **three** levels (Engineering, Sales, Marketing), you'd create exactly **two** indicators — e.g., $D_1=1$ if Sales, $D_2=1$ if Marketing, both 0 for Engineering (the reference category, absorbed into $\beta_0$). **If you instead created three indicators (one per category, including the reference), the design matrix becomes singular** — because the three indicator columns would always sum to a column of all 1's, which is *already* the intercept column. This is exact linear dependence, meaning $(\mathbf{X}'\mathbf{X})^{-1}$ doesn't exist (directly recalling Chapter 5's "full rank" requirement).

**Interpretation with a reference category:** each indicator's coefficient represents the difference in intercept *relative to the reference category* — e.g., $\beta(D_1)$ = Sales' intercept minus Engineering's intercept; $\beta(D_2)$ = Marketing's intercept minus Engineering's intercept. Choice of reference category is arbitrary and doesn't change the model's fit or predictions — only which comparisons are directly readable off the coefficients (this is exactly the mechanic behind `pd.get_dummies(..., drop_first=True)` or `C(category)` in `statsmodels`/`patsy` formula syntax).

**Interview question:** *"Why do you drop one category when one-hot-encoding a categorical variable for linear regression, but not always for tree-based models?"*
**Ideal answer:** In linear regression, keeping all $c$ indicator columns (plus the intercept) creates exact linear dependence — the indicator columns sum to the intercept column — making $\mathbf{X}'\mathbf{X}$ singular and the coefficients unidentifiable ("the dummy variable trap"). Dropping one category (the reference) resolves this, and its effect gets absorbed into the intercept. Tree-based models don't invert any matrix and don't have collinear coefficients in the same sense, so this isn't a hard requirement for them — though it can still slightly affect split efficiency in practice.

---

## 8.5 Modeling Interactions Between Quantitative and Qualitative Predictors

### Why We Need This: Testing Whether the Common-Slope Assumption Is Actually True

Section 8.3's model forced Engineering and Sales to share one slope. But look at the raw per-group slopes we found: Engineering ≈ 6.1, Sales ≈ 10.4 — visibly different. To let the model capture **different slopes for different groups**, add an interaction term between X1 and D:

$$
Y_i = \beta_0+\beta_1X_{i1}+\beta_2D_i+\beta_3(X_{i1}D_i)+\varepsilon_i
$$

**What each coefficient now means:**
- Engineering ($D=0$): $E[Y]=\beta_0+\beta_1X_1$ — intercept $\beta_0$, slope $\beta_1$.
- Sales ($D=1$): $E[Y]=(\beta_0+\beta_2)+(\beta_1+\beta_3)X_1$ — intercept $\beta_0+\beta_2$, slope $\beta_1+\beta_3$.

$\beta_3$ is the **difference in slopes** between the two groups — precisely what was missing from the additive model. This model is "fully saturated" for two groups: it has 4 parameters, exactly matching what you'd get from fitting two completely separate simple regressions (one per group) — which is exactly how we'll compute it.

### Worked Example: Fitting the Full Interaction Model

Since this model is equivalent to two independent per-group regressions, we already have everything we need:

**Engineering line** (from Section 8.3's building blocks): $b_0=44.0$, $b_1=6.1$ — meaning $\beta_0=44.0,\ \beta_1=6.1$.

*(Full arithmetic: $b_1=30.5/5.0=6.1$; $b_0=59.25-6.1(2.5)=44.0$.)*

**Sales line:** $b_0=37.0$, $b_1=10.4$.

*(Full arithmetic: $b_1=52.0/5.0=10.4$; $b_0=63.0-10.4(2.5)=37.0$.)*

**Converting to the interaction-model parameterization:**
$$
\beta_0=44.0 \ (\text{Eng intercept}), \qquad \beta_1=6.1\ (\text{Eng slope})
$$
$$
\beta_2 = 37.0-44.0=-7.0\ (\text{Sales intercept} - \text{Eng intercept})
$$
$$
\beta_3 = 10.4-6.1=4.3\ (\text{Sales slope} - \text{Eng slope})
$$

**Full fitted model:** $\hat Y = 44.0+6.1X_1-7.0D+4.3(X_1D)$

**Fitted values / residuals** for both groups (computed via each group's own simple regression, since they're mathematically identical to this parameterization):

| Group | X1 | Y | $\hat Y$ | $e$ |
|---|---|---|---|---|
| Eng | 1 | 50 | 50.1 | -0.1 |
| Eng | 2 | 57 | 56.2 | 0.8 |
| Eng | 3 | 61 | 62.3 | -1.3 |
| Eng | 4 | 69 | 68.4 | 0.6 |
| Sales | 1 | 47 | 47.4 | -0.4 |
| Sales | 2 | 59 | 57.8 | 1.2 |
| Sales | 3 | 67 | 68.2 | -1.2 |
| Sales | 4 | 79 | 78.6 | 0.4 |

$$
SSE_{\text{full}} = (0.01+0.64+1.69+0.36)+(0.16+1.44+1.44+0.16)=2.70+3.20=5.90
$$
$$
df_{\text{full}}=n-p=8-4=4, \qquad MSE_{\text{full}}=5.90/4=1.475
$$

### Testing Whether the Slope Difference Is Statistically Real

Since the two groups have identical X1 values (1,2,3,4 each) and share one pooled $MSE$, we can derive standard errors directly:
$$
\text{Var}(b_{1,\text{group}}) = \frac{MSE_{\text{full}}}{S_{XX}} = \frac{1.475}{5.0}=0.295 \ \text{(each group)}
$$
Since $\beta_3 = b_{1,\text{Sales}}-b_{1,\text{Eng}}$ is a difference between two **independent** estimates (different employees in each group):
$$
\text{Var}(\beta_3) = 0.295+0.295=0.59, \qquad s(\beta_3)=\sqrt{0.59}=0.768
$$
$$
t^*_{\beta_3} = \frac{4.3}{0.768}=5.599
$$
Similarly for the intercept difference $\beta_2$: $\text{Var}(b_{0,\text{group}})=MSE(1/4+\bar X^2/S_{XX})=1.475(0.25+1.25)=2.2125$ each, so $\text{Var}(\beta_2)=2(2.2125)=4.425$, $s(\beta_2)=2.104$:
$$
t^*_{\beta_2}=\frac{-7.0}{2.104}=-3.328
$$
Critical value: $t_{(0.975;4)}=2.776$. **Both $|t^*_{\beta_2}|=3.33$ and $|t^*_{\beta_3}|=5.60$ exceed 2.776 — both the intercept difference and the slope difference between departments are statistically significant.** The common-slope model from Section 8.3 was genuinely misspecified for this data; departments truly have both different starting salaries and different raise rates per year of experience.

**Interview question:** *"How do you formally test whether the relationship between X and Y differs across two groups, rather than just eyeballing separate scatter plots?"*
**Ideal answer:** Fit a single regression with an indicator variable for group membership and an interaction term between the indicator and X. The interaction coefficient's t-test (or an equivalent partial F-test) directly tests $H_0:$ "the slope is the same across groups" — a formal, quantified version of "the lines look different," rather than a subjective visual read of separate plots.

---

## 8.7 Comparison of Two (or More) Regression Functions

### The General-Linear-Test Framework, Applied to "Are These Two Lines the Same?"

**The question:** is a single common regression line adequate for both departments, or do we genuinely need separate lines? This is answerable via nested full-vs-reduced model F-tests (directly recalling Chapter 2.8 and Chapter 7.3's machinery).

### Test 1: Do the Two Lines Differ At All? (Joint test on $\beta_2,\beta_3$)

**Reduced model:** ignore department entirely — pool all 8 observations into one simple regression of Y on X1.

Computing this pooled simple regression (X1 values 1,2,3,4,1,2,3,4; $\bar X=2.5$, $\bar Y=61.125$, $S_{XX}=10.0$, $S_{XY}=82.5$):
$$
b_1 = 82.5/10=8.25, \qquad b_0 = 61.125-8.25(2.5)=40.5
$$
Working through residuals for all 8 points against this single line gives:
$$
SSE_{\text{reduced}} = 80.25, \qquad df_{\text{reduced}}=n-2=6
$$

**F-test comparing full (separate lines) vs. reduced (one common line):**
$$
F^* = \frac{[SSE_{\text{reduced}}-SSE_{\text{full}}]/(df_{\text{reduced}}-df_{\text{full}})}{SSE_{\text{full}}/df_{\text{full}}} = \frac{(80.25-5.90)/(6-4)}{5.90/4}=\frac{74.35/2}{1.475}=\frac{37.175}{1.475}=25.20
$$
Critical value: $F_{(0.95;\,2,4)}=6.94$. Since $25.20\gg6.94$, **strongly reject $H_0$: the two departments' salary-experience relationships are not the same** — confirming, via a single joint test, what we found piecemeal with the two individual t-tests above.

### Test 2: Is the Slope Difference Alone Responsible? (Testing $\beta_3=0$ specifically)

**Reduced model here is the common-slope (additive) model from Section 8.3**, which had $SSE=52.125$, $df=5$.

$$
F^* = \frac{(52.125-5.90)/(5-4)}{5.90/4} = \frac{46.225}{1.475}=31.34
$$
Critical value: $F_{(0.95;1,4)}=7.71$. Since $31.34\gg7.71$, reject $H_0:\beta_3=0$ — **confirms the interaction term is genuinely needed** (and note: $31.34\approx(5.599)^2=31.35$ — the familiar $F^*=(t^*)^2$ identity for a single-df test, holding here too).

**Interview question:** *"What's the practical difference between testing 'do these two regression lines differ at all' versus 'do these two lines have different slopes'?"*
**Ideal answer:** The first is a joint test on both the intercept-difference and slope-difference coefficients together (comparing the full separate-lines model against a single pooled line) — it answers "does group membership matter at all." The second isolates just the interaction coefficient (comparing the full model against the common-slope-different-intercept model) — it answers the more specific question of whether the *rate of change* differs by group, separate from whether the *starting point* differs. You'd use the joint test as a first screen, and the more specific test to pin down exactly which aspect (intercept, slope, or both) is driving the difference — precisely how we did it here.

---

## 8.6 More Complex Models (Brief)

Everything here generalizes directly: with a 3-level categorical variable, you'd use 2 indicators, and could add 2 separate interaction terms ($X_1D_1$, $X_1D_2$) to allow all three groups their own slope. With two categorical variables (e.g., department **and** seniority level), you can add indicators and interactions for each, and even interactions *between* the categorical variables themselves — the framework scales the same way, always reducible to "how many genuinely distinct linear combinations of parameters am I estimating," directly connecting back to the rank/identifiability discussion in Chapter 5.

---

## Python Implementation — From Scratch (NumPy) and statsmodels

```python
import numpy as np
from scipy import stats

X1 = np.array([1,2,3,4,1,2,3,4], dtype=float)
D  = np.array([0,0,0,0,1,1,1,1], dtype=float)
Y  = np.array([50,57,61,69,47,59,67,79], dtype=float)
n = len(Y)

# --- Full interaction model ---
X_full = np.column_stack([np.ones(n), X1, D, X1*D])
b_full = np.linalg.inv(X_full.T @ X_full) @ X_full.T @ Y
resid_full = Y - X_full @ b_full
SSE_full = np.sum(resid_full**2)
df_full = n - 4
MSE_full = SSE_full / df_full
print("Full model coefficients [b0, b1, b2(intercept diff), b3(slope diff)]:", b_full)
print(f"SSE_full={SSE_full:.3f}, MSE_full={MSE_full:.4f}")

# --- Reduced model: ignore D entirely ---
X_reduced = np.column_stack([np.ones(n), X1])
b_reduced = np.linalg.inv(X_reduced.T @ X_reduced) @ X_reduced.T @ Y
resid_reduced = Y - X_reduced @ b_reduced
SSE_reduced = np.sum(resid_reduced**2)
df_reduced = n - 2

F_joint = ((SSE_reduced - SSE_full)/(df_reduced - df_full)) / MSE_full
print(f"F* (are the two lines different at all?): {F_joint:.2f}")

# --- Common-slope model: D but no interaction ---
X_common = np.column_stack([np.ones(n), X1, D])
b_common = np.linalg.inv(X_common.T @ X_common) @ X_common.T @ Y
resid_common = Y - X_common @ b_common
SSE_common = np.sum(resid_common**2)
df_common = n - 3

F_slope = ((SSE_common - SSE_full)/(df_common - df_full)) / MSE_full
print(f"F* (is the slope difference alone significant?): {F_slope:.2f}")

# --- t-tests on b2, b3 directly from the full model ---
XtX_inv = np.linalg.inv(X_full.T @ X_full)
se = np.sqrt(MSE_full * np.diag(XtX_inv))
t_stats = b_full / se
print("t-statistics [b0,b1,b2,b3]:", t_stats)
```

```python
# statsmodels with formula interface (handles dummy encoding + interaction automatically)
import statsmodels.formula.api as smf
import pandas as pd

df = pd.DataFrame({'X1': X1, 'Dept': ['Eng']*4 + ['Sales']*4, 'Y': Y})
model = smf.ols('Y ~ X1 * C(Dept, Treatment(reference="Eng"))', data=df).fit()
print(model.summary())
# The X1:C(Dept)[T.Sales] term is exactly our beta_3 (slope difference)
```

---

## Interview Question Bank — Chapter 8

**Conceptual:**
1. Why do you use $c-1$ indicator variables for a $c$-level categorical predictor, never $c$?
2. What does an interaction term between a continuous and a categorical predictor let your model do that an additive model can't?
3. In the additive (no-interaction) model with an indicator, what does the indicator's coefficient represent?

**Derivation:**
4. Show why the fully-saturated interaction model (indicator + interaction) is mathematically equivalent to fitting two completely separate simple regressions, one per group.
5. Derive the variance of the slope-difference estimator $\beta_3 = b_{1,\text{group A}}-b_{1,\text{group B}}$ when the two groups are independent samples.

**ML/Statistics:**
6. How would you test whether two categories in your data have a "different relationship" between a feature and the target, versus just a different average target level?
7. Explain what breaks in a linear model's estimation if you accidentally include all $c$ dummy columns plus an intercept for a $c$-level category.
8. A one-hot-encoded categorical feature in a linear model shows an insignificant coefficient. What are two different explanations (connecting back to Ch. 7's multicollinearity discussion) besides "the category doesn't matter"?

**Coding:**
9. Implement the full interaction model, the common-slope model, and the fully-pooled model from scratch, and compute both F-tests (joint difference, slope-only difference).
10. Using `statsmodels.formula.api`, fit an interaction model between a continuous and categorical variable and interpret each coefficient.

**Traps:**
11. "The dummy variable's coefficient tells me the overall salary difference between departments." — what's the more precise interpretation once an interaction term is present?
12. Someone drops a different reference category and gets different-looking coefficients, and worries the model changed. What's actually different, and what stayed the same?
13. "Since the joint F-test (Test 1) was significant, the intercepts must differ." — is that a valid conclusion on its own, or do you need the more targeted test?

---

*This file covers Kutner Ch. 8 — polynomial and interaction models generally, indicator (dummy) variables and the dummy variable trap, interaction models between quantitative and qualitative predictors (worked in full, showing a genuinely different slope and intercept per department), and the general-linear-test framework for formally comparing two regression functions. Chapter 9 (Building the Regression Model I) is next — model selection criteria (adjusted R², AIC/BIC, Mallow's Cp), stepwise and best-subsets procedures, and the practical tradeoffs of automated model-building, which is where most of the "how do you choose which features to include" interview territory lives.*
