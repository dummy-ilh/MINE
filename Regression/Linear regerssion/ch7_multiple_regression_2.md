# Chapter 7 — Multiple Regression II
### (Kutner, Nachtsheim, Neter, Li — *Applied Linear Statistical Models*)

We continue with the Chapter 6 dataset (hours studied X1, practice tests X2, exam score Y, n=6), where we found:

$$b_0=52.4545,\ b_1=5.5000,\ b_2=0.8409,\quad SSTO=638.833,\ SSE(X_1,X_2)=2.523,\ MSE=0.841$$

Chapter 6 ended on a cliffhanger: X1 and X2 are highly correlated, the overall model is extremely strong ($F^*=378.3$), yet X2's individual coefficient wasn't statistically significant. Chapter 7 gives us the precise machinery to understand *why*, and to formally quantify multicollinearity.

---

## 7.1 Extra Sums of Squares

### The Core Idea

**Plain English.** "Extra sum of squares" answers: *how much additional variation in Y does a predictor explain, given that certain other predictors are already in the model?* Notation: $SSR(X_2|X_1)$ means "the extra SS explained by X2, over and above what X1 alone already explains."

**Definition:**
$$
SSR(X_2|X_1) = SSR(X_1,X_2) - SSR(X_1)
$$
i.e., (regression SS with both predictors) minus (regression SS with only X1). Equivalently, in terms of error: $SSR(X_2|X_1)=SSE(X_1)-SSE(X_1,X_2)$ — how much error X2 additionally soaks up once X1 is already accounted for.

**Why this concept exists.** It's the formal bridge between "does this whole model explain a lot of variance" (the overall F-test) and "does this specific predictor pull its own weight" (the individual t-test) — extra sums of squares are literally what individual t-tests are testing, restated as sums of squares instead of standardized coefficients.

### Worked Example: Computing $SSR(X_2|X_1)$

We need $SSR(X_1)$ — the regression sum of squares from a **simple** regression of Y on X1 alone. Using $X_1$'s summary stats:
$$
\bar X_1=4.5,\ S_{X_1X_1}=17.5,\ S_{X_1Y}=105.5 \quad\Rightarrow\quad b_{1,\text{simple}}=\frac{105.5}{17.5}=6.0286
$$
$$
b_{0,\text{simple}}=79.1667-6.0286(4.5)=52.0381
$$

Fitted values and residuals for this X1-only model (full arithmetic, same method as Chapter 1):

| X1 | Y | $\hat Y$ (X1 only) | $e_i$ |
|---|---|---|---|
| 2 | 65 | 64.095 | 0.905 |
| 3 | 70 | 70.124 | -0.124 |
| 4 | 75 | 76.152 | -1.152 |
| 5 | 82 | 82.181 | -0.181 |
| 6 | 88 | 88.210 | -0.210 |
| 7 | 95 | 94.238 | 0.762 |

$$
SSE(X_1) = 0.819+0.015+1.328+0.033+0.044+0.581 = 2.819
$$
$$
SSR(X_1) = SSTO - SSE(X_1) = 638.833-2.819=636.014
$$

**Now the extra sum of squares from adding X2:**
$$
SSR(X_2|X_1) = SSR(X_1,X_2)-SSR(X_1) = 636.310-636.014 = 0.296
$$

**This is tiny** — X2 adds almost nothing to what X1 already explains. Compare this to the *total* variability X2 alone would explain (computed below) — the contrast is the whole point of this chapter.

---

## 7.2 The ANOVA Table with Extra Sums of Squares (Sequential/"Type I" Decomposition)

When predictors are added to a model one at a time, in a specific order, their sequential contributions sum exactly to the full regression SS:
$$
SSR(X_1,X_2) = SSR(X_1) + SSR(X_2|X_1)
$$
$$
636.310 = 636.014 + 0.296 \quad\checkmark
$$

| Source | SS | df |
|---|---|---|
| $X_1$ | 636.014 | 1 |
| $X_2 \vert X_1$ | 0.296 | 1 |
| Error | 2.523 | 3 |
| Total | 638.833 | 5 |

**Critical subtlety: this decomposition depends on the order you enter variables — this is exactly the "Type I / sequential sums of squares" you may have encountered in R's `anova()` or statistical software.** If we instead entered X2 first, we'd get a *different* decomposition:

### Reversing the Order: $SSR(X_1|X_2)$

We need $SSR(X_2)$ alone first. Using $X_2$'s summary stats ($\bar X_2=2.3333$, $S_{X_2X_2}=7.3333$, $S_{X_2Y}=66.667$):
$$
b_{1,X_2\text{ only}} = \frac{66.667}{7.3333}=9.0909, \qquad b_0 = 79.1667-9.0909(2.3333)=57.9545
$$

Working through fitted values and residuals the same way gives $SSE(X_2)=32.773$, so:
$$
SSR(X_2) = 638.833-32.773=606.061
$$
$$
SSR(X_1|X_2) = SSR(X_1,X_2)-SSR(X_2) = 636.310-606.061=30.249
$$

**Compare the two "extra SS" numbers directly:**
$$
SSR(X_2|X_1)=0.296 \quad \text{(tiny)} \qquad\text{vs.}\qquad SSR(X_1|X_2)=30.249 \quad \text{(substantial)}
$$

**This asymmetry is the whole story of this dataset, quantified precisely:** X1 explains a lot that X2 cannot replace (given X2 is already in the model, X1 still adds 30.249 more SS). But X2 explains almost nothing beyond what X1 already covers (given X1 is already in, X2 adds only 0.296 more SS). **X1 is the "primary" variable; X2 is largely redundant given X1** — exactly consistent with Chapter 6's individual t-test finding.

**Interview trap:** In software with more than 2 predictors, people often misread "Type I / sequential" ANOVA tables as if each row's SS represents that variable's "unique" or "true" importance — but sequential SS for any variable *except the last one entered* depends entirely on the arbitrary order of entry. Only the **last-entered variable's** extra SS in a sequential table equals its true partial contribution (holding all others fixed) — which is exactly why "Type III / partial" sums of squares (each variable's extra SS given *all* others, regardless of entry order) is the more common default for hypothesis testing in modern software, and is what the individual t-tests in Chapter 6 actually correspond to.

---

## 7.3 Uses of Extra Sums of Squares: the Partial F-Test

### The General Logic — a Direct Instance of Chapter 2.8's General Linear Test

$$
F^* = \frac{SSR(X_2|X_1)/1}{MSE(X_1,X_2)}
$$

**Worked example, testing $H_0:\beta_2=0$ given X1 already in the model:**
$$
F^* = \frac{0.296/1}{0.841}=0.352
$$
Critical value: $F_{(0.95;\,1,3)}=10.13$. Since $0.352 < 10.13$, **fail to reject** — consistent with Chapter 6's t-test ($t^*=0.594$; note $t^{*2}=0.594^2=0.353\approx F^*=0.352$, confirming the general fact from Chapter 2.7 that for single-df tests, $F^*=(t^*)^2$, now shown to hold in the multiple-regression setting too).

**Testing $H_0:\beta_1=0$ given X2 already in the model (the reverse order):**
$$
F^* = \frac{30.249/1}{0.841}=35.97
$$
Since $35.97 \gg 10.13$, **strongly reject** $H_0$ — X1 remains highly significant even after X2 is already accounted for. (Check: Chapter 6's $t^*_{b_1}=5.998$, and $5.998^2=35.98\approx35.97$ ✓.)

**Extra sums of squares also let you test multiple coefficients jointly** — e.g., in a model with X1, X2, X3, X4, you could test $H_0:\beta_3=\beta_4=0$ (do the last two variables jointly add anything?) via:
$$
F^*=\frac{[SSR(X_3,X_4|X_1,X_2)]/2}{MSE(\text{full model})}
$$
This generalizes directly to Chapter 9's model-building procedures (stepwise selection is essentially an automated, repeated application of exactly this test).

---

## 7.4 Coefficients of Partial Determination

### Plain English

$R^2$ tells you the proportion of *total* Y variance explained by the whole model. A **coefficient of partial determination** tells you: of the variance in Y **not yet explained** by the predictors already in the model, what proportion does a *new* predictor explain?

$$
R^2_{Y2|1} = \frac{SSR(X_2|X_1)}{SSE(X_1)}
$$

**Worked example:**
$$
R^2_{Y2|1} = \frac{0.296}{2.819}=0.1050
$$
**Interpretation:** of the residual variation left over after X1 alone, X2 explains only about 10.5% — a modest, unimpressive incremental contribution.

**Reverse direction:**
$$
R^2_{Y1|2} = \frac{SSR(X_1|X_2)}{SSE(X_2)}=\frac{30.249}{32.773}=0.9230
$$
**Interpretation:** of the residual variation left over after X2 alone, X1 explains a full 92.3% — an enormous incremental contribution.

**Why this pair of numbers, side by side, is such a clean diagnostic:** it directly quantifies the asymmetric relationship we've suspected since Chapter 6 — X1 is doing essentially all the real work; X2 is largely along for the ride because of its correlation with X1.

**Interview question:** *"What's the difference between R² and a coefficient of partial determination, and when would you use the latter?"*
**Ideal answer:** R² measures the proportion of *total* Y-variance explained by the full model. A coefficient of partial determination measures the proportion of the *remaining, yet-unexplained* variance (after certain predictors are already in the model) that an additional predictor explains. It's the right tool for asking "is it worth adding this specific variable to a model I already have," as opposed to "how good is my model overall" — directly useful in forward-selection style model-building procedures (Chapter 9).

---

## 7.5 The Standardized Multiple Regression Model

### Why Standardize

**Plain English.** Raw coefficients $b_1=5.5$ (points per hour) and $b_2=0.84$ (points per practice test) aren't directly comparable — they're in different units. You can't conclude "X1 matters about 6.5x more than X2" just by comparing 5.5 to 0.84, because "1 hour" and "1 practice test" aren't comparable increments.

**The correlation transformation** rescales every variable (X's and Y) to have mean 0 and a standardized spread, making all coefficients directly comparable in "standard deviations of change in Y per standard deviation of change in X" units:
$$
b_k^* = b_k \times \frac{s_{X_k}}{s_Y}
$$
where $s_{X_k}, s_Y$ are the ordinary sample standard deviations.

### Worked Example

$$
s_{X_1}=\sqrt{S_{X_1X_1}/(n-1)}=\sqrt{17.5/5}=\sqrt{3.5}=1.8708
$$
$$
s_{X_2}=\sqrt{7.3333/5}=\sqrt{1.4667}=1.2111
$$
$$
s_Y=\sqrt{638.833/5}=\sqrt{127.767}=11.3033
$$

$$
b_1^* = 5.5000\times\frac{1.8708}{11.3033}=5.5000(0.16551)=0.9103
$$
$$
b_2^* = 0.8409\times\frac{1.2111}{11.3033}=0.8409(0.10715)=0.0901
$$

**Now directly comparable:** a one-standard-deviation increase in X1 is associated with a 0.91 standard-deviation increase in Y; a one-standard-deviation increase in X2 is associated with only a 0.09 standard-deviation increase in Y. **X1 is roughly 10× more "important" than X2 in these standardized units** — the same conclusion we've been building toward all chapter, now expressed in a single clean, comparable number per predictor.

**Historical note (worth knowing but not overweighting):** Kutner also motivates standardization as a fix for numerical instability in matrix inversion when predictors have wildly different scales (e.g., one predictor in the thousands, another between 0 and 1) — this was a bigger deal with older computing hardware; modern numerical linear algebra libraries handle scale differences far more robustly, so the main modern use of standardized coefficients is **interpretability/comparability**, not numerical necessity.

**Interview trap:** Standardized coefficients are sometimes misused to claim strict causal "importance ranking" of features. They're still just correlational, in-sample, model-dependent quantities — they don't establish causal importance, and they change if you change what else is in the model (exactly like unstandardized coefficients do, per Chapter 6).

---

## 7.6 Multicollinearity and Its Effects

### Formal Definition and the Variance Inflation Factor (VIF)

**Plain English.** Multicollinearity means predictors are substantially correlated with each other. It doesn't bias predictions or hurt overall model fit — but it inflates the *variance* (uncertainty) of individual coefficient estimates, making them unstable and hard to interpret individually, which is precisely the symptom we saw with X2's insignificant t-test.

**The Variance Inflation Factor:**
$$
VIF_k = \frac{1}{1-R_k^2}
$$
where $R_k^2$ is the R² from regressing $X_k$ on **all other predictors** in the model. Rule of thumb: $VIF > 10$ (some practitioners use 5) signals serious multicollinearity concern.

**Why this formula makes sense.** If $X_k$ is highly predictable from the other X's ($R_k^2$ close to 1), then $X_k$ carries very little *independent* information beyond what's already in the other predictors — and $VIF_k=1/(1-R_k^2)$ blows up toward infinity as $R_k^2\to1$. Formally, $VIF_k$ is exactly the factor by which $\text{Var}(b_k)$ is inflated compared to what it would be if $X_k$ were completely uncorrelated with the other predictors.

### Worked Example: Computing VIF for X1 and X2

With only two predictors, $R_k^2$ (X1 regressed on X2, or vice versa) is just the squared correlation $r(X_1,X_2)^2$ (same value for both, since there's only one other variable in each case).

$$
S_{X_1X_2}=\sum(X_1-4.5)(X_2-2.3333) = 3.333+2.0+0.167+0.333+1.0+4.167=11.0
$$
$$
r(X_1,X_2) = \frac{S_{X_1X_2}}{\sqrt{S_{X_1X_1}S_{X_2X_2}}}=\frac{11.0}{\sqrt{17.5\times7.3333}}=\frac{11.0}{\sqrt{128.33}}=\frac{11.0}{11.328}=0.9711
$$
$$
R_k^2 = 0.9711^2 = 0.9430
$$
$$
VIF = \frac{1}{1-0.9430}=\frac{1}{0.0570}=17.54
$$

**$VIF=17.54$, far above the standard concern threshold of 10.** This is the formal confirmation of everything we've observed all chapter: X1 and X2 are so highly correlated ($r=0.971$) that including both inflates each coefficient's standard error substantially — this is *exactly* why $b_2$'s standard error (1.417) was large enough to make its t-test non-significant despite the strong overall model fit.

**What multicollinearity does and doesn't do — commonly confused in interviews:**
- **Does NOT bias coefficient point estimates** (they're still unbiased, on average across repeated sampling).
- **Does NOT hurt overall model fit / prediction accuracy** ($R^2$, $F^*$ stay strong — predictions from the model remain reliable even with severe multicollinearity).
- **DOES inflate variance/standard errors of individual coefficients** — making them unstable (small changes in data can swing $b_1, b_2$ substantially) and their individual significance tests unreliable.
- **DOES make individual coefficient interpretation ("holding X2 constant, X1's effect is...") shaky**, since X1 and X2 rarely vary independently in the actual data — "holding X2 constant while varying X1" may describe a combination of values that barely exists in your sample.

**Interview question:** *"Does multicollinearity bias your regression coefficients?"*
**Ideal answer:** No — multicollinearity doesn't introduce bias; the OLS estimators remain unbiased even under severe multicollinearity. What it does is inflate the *variance* of the coefficient estimates (quantified by the Variance Inflation Factor), making individual coefficients unstable and their standard errors/significance tests unreliable, even though the model's overall fit and predictive accuracy remain unaffected. This is why you can have a highly significant, highly predictive model with individually non-significant, seemingly "unstable" coefficients — it's a symptom of shared information among predictors, not a flaw in the estimation procedure itself.

**Practical remedies (briefly, developed further in later chapters):** drop one of the redundant variables, combine correlated variables into a single index, use ridge regression (which explicitly trades a small amount of bias for a large reduction in coefficient variance — the direct ML-flavored fix for this exact problem), or simply interpret the model's *overall* predictive power while being appropriately cautious about individual coefficient interpretation.

---

## Python Implementation — From Scratch (NumPy) and statsmodels

```python
import numpy as np
from scipy import stats

X1 = np.array([2,3,4,5,6,7], dtype=float)
X2 = np.array([1,1,2,3,3,4], dtype=float)
Y  = np.array([65,70,75,82,88,95], dtype=float)
n = len(Y)

def simple_reg_SSE(X, Y):
    Xbar, Ybar = X.mean(), Y.mean()
    Sxx = np.sum((X-Xbar)**2)
    Sxy = np.sum((X-Xbar)*(Y-Ybar))
    b1 = Sxy/Sxx
    b0 = Ybar - b1*Xbar
    resid = Y - (b0 + b1*X)
    return np.sum(resid**2)

SSTO = np.sum((Y-Y.mean())**2)
SSE_X1 = simple_reg_SSE(X1, Y)
SSE_X2 = simple_reg_SSE(X2, Y)

# full model
Xfull = np.column_stack([np.ones(n), X1, X2])
b_full = np.linalg.inv(Xfull.T@Xfull) @ Xfull.T @ Y
resid_full = Y - Xfull@b_full
SSE_full = np.sum(resid_full**2)
MSE_full = SSE_full/(n-3)

SSR_X1 = SSTO - SSE_X1
SSR_X2 = SSTO - SSE_X2
SSR_full = SSTO - SSE_full

SSR_X2_given_X1 = SSR_full - SSR_X1
SSR_X1_given_X2 = SSR_full - SSR_X2

F_X2_given_X1 = (SSR_X2_given_X1/1) / MSE_full
F_X1_given_X2 = (SSR_X1_given_X2/1) / MSE_full

print(f"SSR(X2|X1)={SSR_X2_given_X1:.3f}, F*={F_X2_given_X1:.3f}")
print(f"SSR(X1|X2)={SSR_X1_given_X2:.3f}, F*={F_X1_given_X2:.3f}")

# Coefficients of partial determination
R2_Y2_given_1 = SSR_X2_given_X1 / SSE_X1
R2_Y1_given_2 = SSR_X1_given_X2 / SSE_X2
print(f"R2(Y,X2|X1)={R2_Y2_given_1:.4f}, R2(Y,X1|X2)={R2_Y1_given_2:.4f}")

# Standardized coefficients
sX1, sX2, sY = X1.std(ddof=1), X2.std(ddof=1), Y.std(ddof=1)
b1_star = b_full[1] * sX1/sY
b2_star = b_full[2] * sX2/sY
print(f"Standardized b1*={b1_star:.4f}, b2*={b2_star:.4f}")

# VIF
r_X1X2 = np.corrcoef(X1, X2)[0,1]
VIF = 1 / (1 - r_X1X2**2)
print(f"r(X1,X2)={r_X1X2:.4f}, VIF={VIF:.2f}")
```

## statsmodels — Type II ANOVA and VIF directly

```python
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.anova import anova_lm
import numpy as np, pandas as pd

df = pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})
X_sm = sm.add_constant(df[['X1','X2']])
model = sm.OLS(df['Y'], X_sm).fit()

# VIF for each predictor
for i, col in enumerate(['const','X1','X2']):
    print(col, variance_inflation_factor(X_sm.values, i))

# Sequential (Type I) ANOVA
model_seq = sm.OLS(df['Y'], sm.add_constant(df[['X1','X2']])).fit()
print(anova_lm(model_seq, typ=1))
```

---

## Interview Question Bank — Chapter 7

**Conceptual:**
1. What does "extra sum of squares" mean, and how does it relate to the individual coefficient t-tests from Chapter 6?
2. Why does sequential (Type I) sum of squares depend on variable entry order, but partial (Type III) sum of squares doesn't?
3. What is a coefficient of partial determination, and how is it different from ordinary R²?

**Derivation:**
4. Show algebraically that $F^*=(t^*)^2$ for a single-variable partial F-test, using the extra sum of squares formula.
5. Derive the standardized coefficient formula $b_k^*=b_k(s_{X_k}/s_Y)$ conceptually from the correlation transformation.

**ML/Statistics:**
6. Explain precisely what multicollinearity does and does not do to a regression model (bias vs. variance).
7. Given VIF values for several features in a model, how would you decide which (if any) to drop?
8. Name a modern ML technique that directly addresses the variance-inflation problem caused by multicollinearity, and explain the mechanism briefly (this foreshadows ridge regression).

**Coding:**
9. Implement extra sums of squares and the corresponding partial F-test from scratch in NumPy, for a 2-predictor model.
10. Compute VIF for each predictor in a multi-predictor dataset using statsmodels, and identify which variables are problematic.

**Traps:**
11. "This variable's Type I sequential sum of squares was small, so it's not an important predictor." — what's wrong with this conclusion if it wasn't the last variable entered?
12. "High VIF means my coefficients are biased and my model's predictions are unreliable." — correct the misconception.
13. Someone claims standardized coefficients directly measure "true causal importance" of predictors. What's the flaw?

---

*This file covers Kutner Ch. 7 — extra sums of squares and their sequential (order-dependent) decomposition, the partial F-test (shown numerically equivalent to Chapter 6's t-tests), coefficients of partial determination (quantifying X1's dominance over X2), standardized regression coefficients, and a full worked derivation of the Variance Inflation Factor exposing severe multicollinearity (VIF≈17.5) in our running example. This closes the two-chapter arc on the mechanics of multiple regression. Chapter 8 (Regression Models for Quantitative and Qualitative Predictors) is next — indicator/dummy variables, interaction terms, and models combining continuous and categorical predictors.*
