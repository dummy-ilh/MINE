# Chapter 10 — Diagnostics and Remedial Measures for Multiple Regression
### (Kutner, Nachtsheim, Neter, Li — *Applied Linear Statistical Models*)

Chapter 3 built diagnostics for simple regression. Chapter 5 introduced the hat matrix and leverage. This chapter completes the picture for the multiple-regression case: **added-variable plots, studentized deleted residuals, leverage, and the influence measures (Cook's Distance, DFFITS, DFBETAS)** that formally separate "this observation is unusual in X" from "this observation is unusual in Y given X" from "this observation actually changes my conclusions."

### A New Worked Dataset — Deliberately Constructed to Contain an Extreme Point

Two predictors X1, X2, response Y, n=7:

| i | X1 | X2 | Y |
|---|---|---|---|
| 1 | 1 | 2 | 18 |
| 2 | 2 | 3 | 24 |
| 3 | 3 | 4 | 27 |
| 4 | 4 | 5 | 34 |
| 5 | 5 | 6 | 37 |
| 6 | 6 | 7 | 44 |
| 7 | 10 | 3 | 65 |

**Notice something structural before we even fit anything:** for observations 1–6, $X_2 = X_1+1$ **exactly** — perfect collinearity. Observation 7 is the *only* point that breaks this exact pattern ($X_1=10$ but $X_2=3$, not 11). Keep this in mind; it will turn out to explain everything unusual that happens in this chapter.

### Fitting the Full Model

Using the same 3×3 matrix method as Chapters 6–7 (full arithmetic omitted here for space, but identical in method): centered sums are $S_{X_1X_1}=53.714$, $S_{X_2X_2}=19.429$, $S_{X_1X_2}=9.143$, $S_{X_1Y}=279.286$, $S_{X_2Y}=43.857$, giving:
$$
b_0=13.272,\quad b_1=5.2345,\quad b_2=-0.2060
$$
$$
\hat Y_i = 13.272+5.2345X_{i1}-0.2060X_{i2}
$$

**Fitted values and residuals:**

| i | $\hat Y_i$ | $e_i$ |
|---|---|---|
| 1 | 18.095 | -0.095 |
| 2 | 23.123 | 0.877 |
| 3 | 28.152 | -1.152 |
| 4 | 33.180 | 0.820 |
| 5 | 38.209 | -1.209 |
| 6 | 43.237 | 0.763 |
| 7 | 64.999 | **0.001** |

$$
SSE=4.820,\qquad df=n-p=7-3=4,\qquad MSE=4.820/4=1.205
$$

**Stop and notice observation 7's residual: essentially zero.** Despite $Y_7=65$ being wildly out of step with the trend the other six points establish (their pattern predicts something far lower for $X_1=10$), the model fits it almost perfectly. **This is not a coincidence — it's the central lesson of this chapter**, and we're about to see exactly why.

---

## 10.3 Identifying Outlying X Observations: Leverage, Revisited

Recall from Chapter 5: $h_{ii}=\mathbf{x}_i'(\mathbf{X}'\mathbf{X})^{-1}\mathbf{x}_i$. Computing all seven (via the same method as Chapter 5's worked example):

| i | $X_1$ | $X_2$ | $h_{ii}$ |
|---|---|---|---|
| 1 | 1 | 2 | 0.524 |
| 2 | 2 | 3 | 0.295 |
| 3 | 3 | 4 | 0.181 |
| 4 | 4 | 5 | 0.181 |
| 5 | 5 | 6 | 0.295 |
| 6 | 6 | 7 | 0.524 |
| 7 | 10 | 3 | **1.000** |

Check: $\sum h_{ii}=0.524+0.295+0.181+0.181+0.295+0.524+1.000=3.000=p$ ✓ (Chapter 5's trace-equals-p property, holding exactly as always.)

**Observation 7 has leverage exactly 1.0 — the mathematical maximum possible.** This is an extreme, illuminating edge case, and it's worth understanding *why* it happens here, not just that it happens.

### Why Leverage = 1 Exactly: the Deeper Reason

**Recall the structural fact we flagged before fitting anything:** for observations 1–6, $X_2=X_1+1$ exactly — a perfect linear relationship. **Observation 7 is the *only* data point that breaks this exact collinearity.** That means observation 7 is the **sole source of information in the entire dataset** that lets the model tell $X_1$'s effect apart from $X_2$'s effect. If you removed observation 7, the remaining six points would make $(\mathbf{X}'\mathbf{X})$ **singular** — you literally could not estimate separate coefficients for $X_1$ and $X_2$ using only observations 1–6, since $X_2$ would be an exact linear function of $X_1$ within that subset (recall Chapter 5's "full rank" requirement — this is precisely a failure of that requirement, restricted to the six-point subsample).

**This is the real meaning of $h_{ii}=1$:** observation $i$ is the unique, load-bearing source of some direction of variability in the predictor space. **A leverage of exactly 1 forces the fitted value to equal the observed value exactly, no matter what that observed value is** — the model has *zero* remaining degrees of freedom left to disagree with that single point along the direction it uniquely defines. This is exactly why observation 7's residual came out to (numerically) zero — it wasn't a good fit in any meaningful sense; it was a **mathematically forced** fit.

**Interview question:** *"What does it mean, mechanically, for an observation to have leverage exactly equal to 1, and why is its own residual always zero in that case?"*
**Ideal answer:** Leverage of 1 means the observation is the unique source of some dimension of variability among the predictors — removing it would make the remaining design matrix rank-deficient (unidentifiable) along that dimension. Since the fitted value is a linear combination of all Y's determined entirely by the geometry of X, and this observation's X-configuration gives it "sole ownership" of a full degree of freedom, the model has no choice but to pass through that point exactly, regardless of its Y value — making its own residual identically zero and completely uninformative about whether it's actually an outlier.

---

## 10.2 Identifying Outlying Y Observations: Studentized (Deleted) Residuals

### Why Raw and Even Semistudentized Residuals Aren't Enough

Chapter 3's semistudentized residuals ($e_i/\sqrt{MSE}$) don't account for the fact that **different observations have different leverage**, and therefore different inherent residual variability. The correct **studentized residual** accounts for this directly:
$$
r_i = \frac{e_i}{\sqrt{MSE(1-h_{ii})}}
$$

**Worked example, contrasting a typical point with the extreme one:**

For observation 3 (moderate leverage 0.181, residual -1.152):
$$
r_3 = \frac{-1.152}{\sqrt{1.205(1-0.181)}}=\frac{-1.152}{\sqrt{0.987}}=\frac{-1.152}{0.9934}=-1.159
$$

**For observation 7:**
$$
r_7 = \frac{0.001}{\sqrt{1.205(1-1.000)}} = \frac{0.001}{\sqrt{0}} = \textbf{undefined (division by zero)}
$$

**The formula itself breaks down completely for observation 7.** This is the formal, unambiguous signal that ordinary residual-based diagnostics are fundamentally inadequate for this point — not just "not very informative," but mathematically undefined.

### Studentized Deleted Residuals: the Real Fix

**Plain English.** Instead of asking "how far is $Y_i$ from what the model (fit *with* $i$ included) predicts," ask the more honest question: **"how far is $Y_i$ from what the model would predict if it had never seen $Y_i$ at all?"** Fit the model on all observations *except* $i$, then measure how far off that model's prediction is for the left-out point. This is precisely the same leave-one-out logic behind Chapter 9's PRESS statistic.

There's a computational shortcut avoiding literally refitting $n$ times:
$$
d_i = e_i\sqrt{\frac{n-p-1}{SSE(1-h_{ii})-e_i^2}}
$$

**For observation 7, try to apply this shortcut:** $h_{77}=1$, so $(1-h_{77})=0$, and the denominator becomes $SSE(0) - e_7^2 = -e_7^2 \approx 0$ (since $e_7\approx0$ too) — an indeterminate $0/0$-flavored breakdown. **The deeper reason the shortcut fails matches exactly what we found for leverage: you literally cannot fit "the model without observation 7"** at all in the ordinary sense, because doing so makes $X_1$ and $X_2$ perfectly collinear among the remaining six points, and $(\mathbf{X}'\mathbf{X})$ becomes singular. **There is no well-defined "prediction from a model that never saw this point,"** because that model can't even be estimated.

**This is a genuinely important, advanced, and often-overlooked practical lesson:** sometimes a "highly influential" or "high-leverage" point isn't just distorting your fit — it may be **structurally necessary for your model to be identifiable at all.** Before deleting or downweighting such a point as "just an outlier," you need to check whether doing so breaks the model's estimability entirely, not merely whether it changes the coefficients.

**Interview question:** *"You compute studentized deleted residuals for your dataset and one observation gives an undefined result (division by zero or near-zero). What's actually going on, and what should you check before deciding what to do with that point?"*
**Ideal answer:** This almost always traces back to that observation having leverage at or extremely close to 1 — meaning it's the sole source of some dimension of variation among the predictors. Before treating it as a removable outlier, check whether the remaining data (with that point excluded) can even estimate all the model's parameters — if the remaining predictors become collinear or rank-deficient without it, the point isn't just an outlier, it's structurally load-bearing for the model's identifiability, and removing it isn't really an option without also removing one of the correlated predictors it was disambiguating.

---

## 10.4 Identifying Influential Cases: Cook's Distance, DFFITS, DFBETAS

### Cook's Distance — Combining Leverage and Residual Size into One Number

**Plain English.** Cook's Distance asks: "if I deleted observation $i$ and refit, how much would the **entire vector of fitted values** shift, on average, across all $n$ observations?" It combines both ingredients that make a point dangerous — a large residual (surprising Y) **and** high leverage (extreme X) — into a single influence score:

$$
D_i = \frac{e_i^2}{p\cdot MSE}\cdot\frac{h_{ii}}{(1-h_{ii})^2}
$$

**Why it needs both ingredients multiplicatively, not just one.** A large residual with *low* leverage is just an ordinary outlier — it doesn't have the geometric position to swing the whole line. High leverage with a *small* residual (a point that happens to sit right on the trend, despite an extreme X) also doesn't distort anything — it's extreme in X but doesn't disagree with the model. **Only the combination — extreme X position *and* a value that disagrees with what the rest of the data implies — produces real influence.**

**Worked example, three points for contrast:**

**Observation 3** (moderate leverage 0.181, residual -1.152):
$$
D_3 = \frac{(-1.152)^2}{3(1.205)}\times\frac{0.181}{(1-0.181)^2} = \frac{1.326}{3.615}\times\frac{0.181}{0.671}=0.3668\times0.2698=0.099
$$

**Observation 5** (leverage 0.295, residual -1.209):
$$
D_5 = \frac{1.462}{3.615}\times\frac{0.295}{(0.705)^2}=0.4044\times0.5943=0.240
$$

**Observation 7** (leverage 1.000): the denominator $(1-h_{77})^2=0$, so
$$
D_7 = \frac{(0.001)^2}{3.615}\times\frac{1.0}{0} \rightarrow \boldsymbol{\infty} \ (\text{formally undefined/unbounded})
$$

**Rule of thumb** (one common convention): flag $D_i > \frac{4}{n}$ (here $4/7=0.571$) for further scrutiny, or the more conservative $D_i>1$. Observations 3 and 5 are both comfortably below either threshold — unremarkable. **Observation 7's Cook's Distance is not merely "large" — it's mathematically unbounded**, the most extreme possible signal of influence this diagnostic can produce, entirely consistent with everything found above.

### DFFITS and DFBETAS — Briefly

**DFFITS** measures the *standardized* change in the fitted value for observation $i$ specifically, if $i$ is deleted:
$$
(DFFITS)_i = d_i\sqrt{\frac{h_{ii}}{1-h_{ii}}}
$$
(where $d_i$ is the studentized deleted residual) — again undefined for observation 7 for the same $1-h_{ii}=0$ reason.

**DFBETAS** measures the standardized change in **each individual coefficient** ($b_0, b_1, b_2$ separately) when observation $i$ is deleted — useful for pinpointing *which* coefficient a suspicious point is distorting, rather than just the overall fitted values. For observation 7, since deleting it makes the model inestimable entirely, DFBETAS for $b_1$ and $b_2$ specifically are not just large but literally undefined — you can't compute "the change in $b_1$ when the model without observation 7 can't estimate $b_1$ or $b_2$ separately in the first place."

**Interview question:** *"What's the difference between what Cook's Distance tells you and what DFBETAS tells you?"*
**Ideal answer:** Cook's Distance gives one aggregate number summarizing how much an observation shifts the *overall* fitted values across the whole dataset. DFBETAS breaks that influence down coefficient-by-coefficient, showing specifically which parameter estimate(s) a given observation is distorting — useful when you want to know not just "is this point influential" but "is it specifically corrupting my estimate of this particular variable's effect," which matters a lot if that's the coefficient your business decision hinges on.

---

## 10.1 Added-Variable Plots (Partial Regression Plots)

**Plain English.** An added-variable plot for $X_1$ (given $X_2$ already in the model) visualizes exactly the "partial slope" idea from Chapter 6: (1) regress Y on $X_2$ alone, keep the residuals; (2) regress $X_1$ on $X_2$ alone, keep those residuals too; (3) plot the first set of residuals against the second. **The slope of a simple regression line through this residual-vs-residual plot exactly equals $b_1$ from the full multiple regression** — a beautiful, direct visualization of "the part of $X_1$'s relationship with Y that isn't already explained by $X_2$."

**Why this is a genuinely useful diagnostic beyond just illustrating partial slopes:** points that look unremarkable in an ordinary $Y$-vs-$X_1$ scatterplot can reveal themselves as highly leveraged or influential *specifically with respect to X1's partial effect* once you look at this residual-vs-residual view — it isolates exactly the piece of variation relevant to that one coefficient, filtering out everything already explained by the other predictors.

**In our example:** since observation 7 is *the* point breaking collinearity between X1 and X2, it would appear as an extreme outlying point along the horizontal axis of the added-variable plot for either predictor — visually confirming its outsized leverage on that specific partial relationship, echoing everything found numerically above.

---

## 10.5 Multicollinearity Diagnostics: VIF, Revisited

Chapter 7 introduced $VIF_k=1/(1-R_k^2)$. **Worth restating here as a diagnostic, not just an explanatory concept:** in our dataset, if we computed $VIF$ using **only observations 1–6**, we'd find $R_k^2\to1$ (since $X_2=X_1+1$ exactly) and $VIF\to\infty$ — a formal echo of the exact-collinearity problem this whole chapter has been circling. Including observation 7 breaks that perfect collinearity and makes $VIF$ finite again — but at the cost of concentrating enormous leverage and influence onto that single point. **This is the essential, often underappreciated tradeoff**: breaking multicollinearity sometimes requires data whose very rarity (being the "different" point) makes it disproportionately powerful over your conclusions.

---

## Python Implementation — From Scratch (NumPy) and statsmodels

```python
import numpy as np

X1 = np.array([1,2,3,4,5,6,10], dtype=float)
X2 = np.array([2,3,4,5,6,7,3], dtype=float)
Y  = np.array([18,24,27,34,37,44,65], dtype=float)
n = len(Y)

X = np.column_stack([np.ones(n), X1, X2])
XtX_inv = np.linalg.inv(X.T @ X)
b = XtX_inv @ X.T @ Y
Y_hat = X @ b
resid = Y - Y_hat
p = X.shape[1]
SSE = np.sum(resid**2)
MSE = SSE / (n - p)

H = X @ XtX_inv @ X.T
leverage = np.diag(H)
print("Leverages:", np.round(leverage, 4))
print("Sum of leverages (should = p):", leverage.sum())

# Studentized residuals (will show inf/nan for h_ii=1)
with np.errstate(divide='ignore', invalid='ignore'):
    studentized = resid / np.sqrt(MSE * (1 - leverage))
print("Studentized residuals:", studentized)

# Studentized DELETED residuals (shortcut formula)
with np.errstate(divide='ignore', invalid='ignore'):
    denom = SSE*(1-leverage) - resid**2
    deleted_resid = resid * np.sqrt((n-p-1) / denom)
print("Studentized deleted residuals:", deleted_resid)

# Cook's Distance
with np.errstate(divide='ignore', invalid='ignore'):
    cooks_d = (resid**2 / (p*MSE)) * (leverage / (1-leverage)**2)
print("Cook's Distance:", cooks_d)
```

```python
# statsmodels: get everything via the built-in influence object
import statsmodels.api as sm

X_sm = sm.add_constant(np.column_stack([X1, X2]))
model = sm.OLS(Y, X_sm).fit()
infl = model.get_influence()

print("Leverage:", infl.hat_matrix_diag)
print("Studentized residuals:", infl.resid_studentized_internal)
print("Studentized DELETED residuals:", infl.resid_studentized_external)
print("Cook's D:", infl.cooks_distance[0])
print("DFBETAS:\n", infl.dfbetas)

# Check: is (X'X) close to singular without observation 7?
X_no7 = X_sm[:-1]
print("Rank without obs 7:", np.linalg.matrix_rank(X_no7), "vs. needed:", X_sm.shape[1])
print("Condition number without obs 7:", np.linalg.cond(X_no7.T @ X_no7))
```

---

## Interview Question Bank — Chapter 10

**Conceptual:**
1. What's the difference between an observation being "unusual in X" (leverage), "unusual in Y given X" (large residual), and "influential" (Cook's D)?
2. Why can a point with very high leverage have a deceptively small ordinary residual?
3. What does an added-variable plot visualize that an ordinary scatterplot of Y vs. X1 doesn't?

**Derivation:**
4. Show why $h_{ii}=1$ forces the fitted value to equal the observed value exactly for that point.
5. Explain, using the structure of Cook's Distance, why it requires *both* a large residual and high leverage to produce a large value — neither alone is sufficient.

**ML/Statistics:**
6. You find one observation with leverage extremely close to 1. What should you check before deciding whether to remove it as an "outlier"?
7. Why might a highly influential point actually be structurally necessary for your model, rather than a data error to discard?
8. In modern ML pipelines with many features, what's the practical analog of "checking leverage" before trusting a model's coefficients?

**Coding:**
9. Implement the leverage, studentized residual, and Cook's Distance calculations from scratch in NumPy for an arbitrary design matrix.
10. Write code that checks whether removing any single observation from a dataset would make the remaining design matrix rank-deficient.

**Traps:**
11. "This point's residual is nearly zero, so it must be a good, unremarkable observation." — what should you check before accepting this?
12. "Cook's Distance for this point is infinite/undefined, so I should just delete it." — what's the more careful conclusion, given what we found about identifiability?
13. Someone claims high VIF and high leverage are the same kind of problem. What's the actual distinction between multicollinearity (a property of the predictor set as a whole) and leverage (a property of an individual observation)?

---

*This file covers Kutner Ch. 10 — added-variable plots, leverage (with a rare, fully-worked example of an observation achieving the mathematical maximum leverage of exactly 1, and a deep explanation of what that means for identifiability), studentized and studentized deleted residuals (both shown to be literally undefined for the extreme point), and Cook's Distance/DFFITS/DFBETAS for influence. This closes out the two-chapter diagnostic arc for multiple regression (Ch. 3 for simple regression, Ch. 10 for multiple regression), and completes the entire multiple-regression core of this course: Chapters 6 through 10 have now covered estimation, inference, extra sums of squares, qualitative predictors, model building, and diagnostics — the full applied toolkit expected for an L5 Data Scientist / ML Engineer interview at Google-tier companies.*
