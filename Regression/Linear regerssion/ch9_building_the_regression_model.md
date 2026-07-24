# Chapter 9 — Building the Regression Model I
### (Kutner, Nachtsheim, Neter, Li — *Applied Linear Statistical Models*)

Chapters 6–8 assumed you already knew which predictors belonged in the model. Chapter 9 asks the question every real project actually starts with: **given a pool of candidate predictors, which subset should you use?**

### A New Worked Dataset (3 candidate predictors)

Predicting monthly sales (Y, units) from advertising spend (X1, $1000s), price (X2, $), and store size (X3, 1000 sq ft), n=8 stores:

| i | X1 | X2 | X3 | Y |
|---|---|---|---|---|
| 1 | 1 | 10 | 2 | 35 |
| 2 | 2 | 9  | 2 | 35 |
| 3 | 3 | 10 | 3 | 45 |
| 4 | 4 | 8  | 3 | 45 |
| 5 | 5 | 9  | 4 | 56 |
| 6 | 6 | 7  | 4 | 60 |
| 7 | 7 | 8  | 5 | 65 |
| 8 | 8 | 6  | 5 | 67 |

(Behind the scenes, Y was generated mainly from X1 and X3, with some added noise; X2 happens to be correlated with X1 — negatively, since discount pricing often accompanies bigger ad pushes — but has little genuine independent relationship with Y. Keep this "ground truth" in your back pocket; part of this chapter's lesson is watching how well — or poorly — the formal criteria recover it.)

$$
n=8,\quad SSTO=1142,\quad \bar Y=51.0
$$

---

## 9.1 Overview of the Model-Building Process

Kutner frames model-building as an iterative cycle, not a one-shot calculation:

1. **Data collection** — including deciding which candidate predictors are even worth measuring.
2. **Model refinement** — fitting candidate subsets, checking diagnostics (Ch. 3), refining functional form.
3. **Model validation** — checking the selected model holds up on **new data**, not just the data used to build it.

**Why Step 3 is not optional, and is the step most people skip in practice.** Any selection procedure that chooses a subset by optimizing a criterion *on the same data used to fit the models* will be optimistic — the selected model's apparent fit overstates how well it will perform on genuinely new data. This is the classical-statistics ancestor of the **train/validation/test split** discipline now standard in ML — Kutner is teaching the exact same lesson decades earlier, under the name "model validation," typically via a holdout sample or cross-validation.

---

## 9.3 Criteria for Model Selection

We first fit **all subsets** of {X1, X2, X3} to build a comparison table. (Full arithmetic for each subset follows exactly the same matrix/normal-equation methods from Chapters 1, 6, and 7 — shown in full for one representative 2-predictor case below, with the rest computed identically.)

### Worked Example: Fitting $\{X_1, X_3\}$ by Hand

Using centered sums $S_{X_1X_1}=42$, $S_{X_3X_3}=10$, $S_{X_1X_3}=20$, $S_{X_1Y}=215$, $S_{X_3Y}=106$:
$$
42b_1+20b_3=215 \qquad 20b_1+10b_3=106
$$
Solving: from the second equation $b_3=10.6-2b_1$; substituting into the first gives $b_1=1.5$, $b_3=7.6$, and $b_0=51-1.5(4.5)-7.6(3.5)=17.65$.
$$
SSR(X_1,X_3)=b_1S_{X_1Y}+b_3S_{X_3Y}=1.5(215)+7.6(106)=322.5+805.6=1128.1
$$
$$
SSE(X_1,X_3)=SSTO-SSR=1142-1128.1=13.9
$$

Every other subset in the table below was obtained the same way (single predictors via Chapter 1's simple-regression formulas; the remaining pairs and the full 3-predictor model via the same normal-equation/matrix-inversion method from Chapters 6–7).

### The Full Comparison Table

| Model | p | SSE | $R^2_p$ |
|---|---|---|---|
| $\{X_1\}$ | 2 | 41.40 | 0.9637 |
| $\{X_2\}$ | 2 | 477.68 | 0.5817 |
| $\{X_3\}$ | 2 | 18.40 | 0.9839 |
| $\{X_1,X_2\}$ | 3 | 20.76 | 0.9818 |
| $\{X_1,X_3\}$ | 3 | 13.90 | 0.9878 |
| $\{X_2,X_3\}$ | 3 | 13.17 | 0.9885 |
| $\{X_1,X_2,X_3\}$ | 4 | 13.08 | 0.9885 |

($p$ = number of parameters including the intercept.)

### $R^2_p$ — Why It's Insufficient on Its Own

**Plain English.** $R^2_p$ always **weakly increases** (never decreases) as you add more predictors, even useless ones — because least squares can never do worse by having more flexibility to exploit, even if that "exploitation" is just fitting noise. Notice in the table: $R^2$ crawls from 0.9885 ($\{X_2,X_3\}$) to 0.9885 (all three) — a razor-thin, meaningless improvement for the cost of an entire extra parameter. **This mechanical property makes raw $R^2$ useless for comparing models with different numbers of predictors** — it will always favor the biggest model, regardless of whether the added predictors are meaningful.

### Adjusted $R^2_p$ — Penalizing for Model Size

$$
R^2_{a,p} = 1-\left(\frac{n-1}{n-p}\right)\frac{SSE_p}{SSTO}
$$

**Why this fixes the problem.** Unlike $R^2$, adjusted $R^2$ can **decrease** when you add a predictor that doesn't pull its weight — because the penalty term $\frac{n-1}{n-p}$ grows as $p$ grows, and if $SSE_p$ doesn't shrink enough to offset that growing penalty, the whole expression gets worse.

**Worked example — computing adjusted $R^2$ for all 3 predictors:**
$$
R^2_{a,\{1,2,3\}} = 1-\frac{7}{4}\times\frac{13.08}{1142}=1-1.75(0.01145)=1-0.02005=0.9799
$$
**Compare to $\{X_2,X_3\}$ alone:**
$$
R^2_{a,\{2,3\}} = 1-\frac{7}{5}\times\frac{13.17}{1142}=1-1.4(0.01153)=1-0.01614=0.9839
$$

**Adjusted $R^2$ actually goes DOWN when you add X1 to the $\{X_2,X_3\}$ model** (0.9839 → 0.9799), even though raw $R^2$ barely moved (and technically still rose, from 0.9885 to 0.9885 — the tiny residual improvement wasn't enough to offset the added parameter's cost). This is exactly the diagnostic adjusted $R^2$ is designed to catch.

### Mallow's $C_p$ — A Bias/Variance Tradeoff Criterion

**Plain English.** $C_p$ formalizes a very ML-flavored idea: a model with **too few** predictors is *biased* (missing real signal); a model with **too many** predictors has needlessly inflated *variance* (overfitting noise). $C_p$ is built to be minimized around the sweet spot.

$$
C_p = \frac{SSE_p}{MSE(\text{full model})} - (n-2p)
$$

where "full model" means the model containing **all** candidate predictors under consideration (here, all of X1, X2, X3), and its MSE serves as your best available estimate of the true $\sigma^2$.

**The rule of thumb:** look for models where $C_p \approx p$ (a small $C_p$ close to the number of parameters signals low bias *and* reasonable variance); models with $C_p \gg p$ have too few predictors (bias); interestingly, **the full model's $C_p$ always equals $p$ exactly** — a mechanical identity, not a sign the full model is actually "good" — so you're really looking for the *smallest* model whose $C_p$ is still close to its own $p$.

**Worked example.** First, $MSE(\text{full}) = SSE_{\{1,2,3\}}/(n-4)=13.08/4=3.27$.

$$
C_{p,\{X_3\}} = \frac{18.4}{3.27}-(8-4) = 5.627-4=1.627
$$
$$
C_{p,\{X_1,X_3\}} = \frac{13.9}{3.27}-(8-6)=4.251-2=2.251
$$
$$
C_{p,\{X_2,X_3\}} = \frac{13.17}{3.27}-2=4.028-2=2.028
$$
$$
C_{p,\{X_1,X_2,X_3\}} = \frac{13.08}{3.27}-(8-8)=4.0-0=4.0 \quad (\text{= } p \text{, exactly, as expected})
$$

$\{X_3\}$ alone has $C_p=1.627$, quite close to its own $p=2$ — a strong, parsimonious candidate. $\{X_2,X_3\}$'s $C_p=2.028$ is essentially exactly its $p=3$ — also an excellent candidate. Both are far better than the full 3-predictor model's inflated bias-adjusted picture.

### $AIC_p$ and $BIC_p$ — Information-Theoretic Criteria

$$
AIC_p = n\ln\left(\frac{SSE_p}{n}\right)+2p, \qquad BIC_p = n\ln\left(\frac{SSE_p}{n}\right)+p\ln(n)
$$

**Why these have the shape they do.** The first term rewards fit (smaller SSE → more negative/smaller first term); the second term penalizes complexity (more parameters → bigger penalty). **BIC's penalty ($p\ln n$) grows faster than AIC's ($2p$) whenever $n>7$ or so** ($\ln 8 \approx 2.08>2$ here) — meaning BIC pushes toward *more parsimonious* models than AIC does, especially as sample size grows. This is a direct echo of the same idea underlying $L_1$/$L_2$ regularization penalty strength in ML: how hard you penalize complexity is itself a tunable design choice, not a universal constant.

**Worked example (using $\ln 8 = 2.0794$):**

| Model | p | SSE | $SSE/n$ | AIC | BIC |
|---|---|---|---|---|---|
| $\{X_3\}$ | 2 | 18.40 | 2.300 | 10.663 | 10.822 |
| $\{X_1,X_3\}$ | 3 | 13.90 | 1.738 | 10.422 | 10.660 |
| $\{X_2,X_3\}$ | 3 | 13.17 | 1.646 | 9.986 | 10.224 |
| $\{X_1,X_2,X_3\}$ | 4 | 13.08 | 1.635 | 11.934 | 12.252 |
| $\{X_1,X_2\}$ | 3 | 20.76 | 2.595 | 13.630 | 13.868 |
| $\{X_1\}$ | 2 | 41.40 | 5.175 | 17.150 | 17.309 |

**Both AIC and BIC favor $\{X_2,X_3\}$ here** — the lowest value of both criteria (smaller is better for both AIC and BIC). Notice the full 3-predictor model is actually one of the *worst* by these criteria despite having the (tied) lowest raw SSE — a clean, concrete demonstration of why raw fit statistics without a complexity penalty mislead you toward overfit models.

### PRESS$_p$ (Prediction Sum of Squares) — Briefly

**Plain English.** PRESS asks a more honest question than any of the criteria above: *for each observation, if you refit the model with that single observation excluded, how well does the resulting model predict the excluded point?*
$$
PRESS_p = \sum_{i=1}^n (Y_i - \hat Y_{i(i)})^2
$$
where $\hat Y_{i(i)}$ is the fitted value for observation $i$ **using a model fit without observation $i$** — this is exactly **leave-one-out cross-validation**, dating to this book, decades before "cross-validation" became standard ML vocabulary. **Why it's more trustworthy than in-sample SSE-based criteria:** it directly measures out-of-sample-style predictive accuracy (each prediction is made "as if" for new data), rather than relying on a fixed mathematical penalty term to approximate that idea.

**Interview connection, explicit and important:** PRESS *is* leave-one-out cross-validation, expressed in 1980s regression-textbook notation. If you're asked "what's the difference between AIC and cross-validation for model selection," a strong answer notes: AIC approximates the same overfitting-penalty idea analytically (assuming certain asymptotic conditions), while PRESS/LOO-CV estimates it directly by actually refitting — more computationally expensive, but doesn't rely on those asymptotic approximations holding.

---

## 9.4 Automatic Search Procedures

### Forward Selection — Worked Through Step by Step

**The algorithm:** (1) start with no predictors; (2) at each step, add whichever candidate predictor produces the largest partial F-statistic (Chapter 7 machinery) among those not yet in the model; (3) stop when no remaining candidate's partial F exceeds a chosen threshold (a common convention: an F-to-enter around 4, roughly corresponding to $\alpha\approx0.05$–$0.10$ for reasonably sized samples).

**Step 1 — best single predictor.** From the table, $\{X_3\}$ has by far the smallest SSE (18.40) and highest $R^2$ (0.9839) among single predictors — it enters first.

**Step 2 — test adding X1 to $\{X_3\}$:**
$$
SSR(X_1|X_3) = SSE(X_3)-SSE(X_1,X_3)=18.40-13.90=4.50
$$
$$
F^*=\frac{SSR(X_1|X_3)/1}{MSE(X_1,X_3)}=\frac{4.50}{13.90/5}=\frac{4.50}{2.78}=1.618
$$

**Test adding X2 to $\{X_3\}$:**
$$
SSR(X_2|X_3)=18.40-13.17=5.23, \qquad F^*=\frac{5.23}{13.17/5}=\frac{5.23}{2.634}=1.985
$$

**Neither exceeds the F-to-enter threshold of ≈4.** Forward selection **stops at $\{X_3\}$ alone** — the algorithm concludes no other single addition earns its keep, at conventional significance levels.

**Why this matters, stated plainly:** even though $\{X_2,X_3\}$ had the best AIC/BIC/adjusted-$R^2$ numbers above, the formal stepwise **entry test** doesn't clear the bar for adding X2. **Different, individually reasonable selection procedures can disagree** — this is a genuine, honest feature of model selection, not a contradiction to be papered over.

### Backward Elimination — Worked Through Step by Step

**The algorithm:** start with **all** candidates in the model; at each step, remove whichever predictor has the *smallest* partial F (least significant, given everything else currently in the model); stop when every remaining predictor's partial F exceeds a chosen threshold (F-to-stay).

**Step 1 — compute each variable's partial F in the full 3-predictor model**, using extra sums of squares against the two-predictor models:
$$
SSR(X_1|X_2,X_3)=SSE(X_2,X_3)-SSE(X_1,X_2,X_3)=13.17-13.08=0.09 \Rightarrow F^*=\frac{0.09}{3.27}=0.028
$$
$$
SSR(X_2|X_1,X_3)=SSE(X_1,X_3)-SSE(\text{full})=13.90-13.08=0.82 \Rightarrow F^*=\frac{0.82}{3.27}=0.251
$$
$$
SSR(X_3|X_1,X_2)=SSE(X_1,X_2)-SSE(\text{full})=20.76-13.08=7.68 \Rightarrow F^*=\frac{7.68}{3.27}=2.348
$$

**All three are below any conventional F-to-stay threshold (≈4)** — remarkable, since the overall model fits extremely well ($R^2=0.9885$). **X1 has the smallest partial F (0.028) — remove it first**, leaving $\{X_2,X_3\}$.

**Step 2 — re-check the remaining two variables in $\{X_2,X_3\}$:** we already computed $SSR(X_2|X_3)$'s F-value above as 1.985, still below threshold — **remove X2 next**, leaving $\{X_3\}$ alone.

**Step 3 — only X3 remains; nothing left to test against.** Backward elimination **also converges to $\{X_3\}$ alone** — the same answer as forward selection, reassuring agreement between the two procedures despite starting from opposite ends.

**Why the full model's individual partial F-tests were all so weak despite excellent overall fit:** this is multicollinearity again (Chapter 7) — X1, X2, and X3 are all substantially correlated with each other in this data (recall X1 and X2 were built to move together), so once *any two* are in the model, the third has very little left to contribute. This is the same phenomenon from Chapters 6–7's worked examples, now playing out at model-selection scale.

**Interview question:** *"Forward selection and backward elimination can, in principle, land on different final models. Why might that happen, and what does it tell you?"*
**Ideal answer:** Each procedure is a greedy, locally-optimal search — forward selection can miss a predictor that only becomes valuable in combination with others not yet added, while backward elimination can be misled early by multicollinearity making an ultimately-important variable look weak in the full model. When they agree (as in this example), that's reassuring convergent evidence; when they disagree, it's a signal that the data has enough correlation structure or noise that the "best subset" isn't clearly defined, and you should lean more on domain knowledge, cross-validation performance, and stability across resamples rather than trusting either single greedy path.

---

## 9.5 Final Comments on Automatic Selection Procedures

Kutner closes the chapter with explicit cautions, all still directly relevant to modern ML feature selection:

1. **No single criterion is "correct."** $C_p$, AIC, BIC, and adjusted $R^2$ can and do disagree (as seen above — $C_p$ mildly favored $\{X_3\}$'s extreme parsimony; AIC/BIC favored $\{X_2,X_3\}$; the formal stepwise F-tests favored $\{X_3\}$ again). None is objectively "the" right answer — they encode different implicit tradeoffs between fit and complexity (this is the classical-statistics version of choosing a regularization strength in ML).

2. **Automatic procedures don't guarantee recovering the "true" generating model**, especially under multicollinearity — our data was actually built mainly from X1 and X3, yet several reasonable procedures pointed toward $\{X_2,X_3\}$ or $\{X_3\}$ alone instead of the "true" $\{X_1,X_3\}$, precisely because X1 and X2's correlation makes them substitutable in-sample.

3. **Stepwise procedures' p-values/F-tests are invalid if taken at face value after the fact** — since you searched over many candidate models and reported only the best one, the reported significance levels don't account for that search (the multiple-comparisons problem from Chapter 4, now at the scale of an entire model-search procedure). This is exactly why **held-out validation performance**, not in-sample test statistics, should be the final arbiter of a selected model's quality.

4. **Domain knowledge should override purely mechanical criteria** when they conflict — if theory strongly suggests a predictor belongs (e.g., known causal mechanism), that's a legitimate reason to keep it even if a stepwise procedure would drop it.

**Interview question:** *"Why shouldn't you trust the p-values reported for coefficients in a model that was chosen via stepwise selection on the same data?"*
**Ideal answer:** Because the selection process itself searched over many possible models and kept whichever looked best by chance as well as by real signal — the reported p-values were computed as if the model had been specified in advance, not chosen after peeking at the data, so they understate the true uncertainty (this is a form of the multiple-comparisons/data-dredging problem). The valid way to assess a stepwise-selected model's real quality is to evaluate it on a genuinely held-out sample it had no part in selecting, echoing this book's train/validate discipline and directly paralleling the modern ML practice of never trusting training-set metrics alone.

---

## Python Implementation — From Scratch (NumPy) and Automated (statsmodels / sklearn)

```python
import numpy as np
from itertools import combinations

X1 = np.array([1,2,3,4,5,6,7,8], dtype=float)
X2 = np.array([10,9,10,8,9,7,8,6], dtype=float)
X3 = np.array([2,2,3,3,4,4,5,5], dtype=float)
Y  = np.array([35,35,45,45,56,60,65,67], dtype=float)
n = len(Y)
predictors = {'X1': X1, 'X2': X2, 'X3': X3}
SSTO = np.sum((Y - Y.mean())**2)

def fit_subset(cols):
    X = np.column_stack([np.ones(n)] + [predictors[c] for c in cols])
    b = np.linalg.inv(X.T@X) @ X.T @ Y
    resid = Y - X@b
    return np.sum(resid**2), X.shape[1]

# Fit the full model first (needed for Cp)
SSE_full, p_full = fit_subset(['X1','X2','X3'])
MSE_full = SSE_full / (n - p_full)

results = []
all_cols = ['X1','X2','X3']
for k in range(1, 4):
    for combo in combinations(all_cols, k):
        SSE, p = fit_subset(combo)
        R2 = 1 - SSE/SSTO
        adjR2 = 1 - (n-1)/(n-p) * SSE/SSTO
        Cp = SSE/MSE_full - (n - 2*p)
        AIC = n*np.log(SSE/n) + 2*p
        BIC = n*np.log(SSE/n) + p*np.log(n)
        results.append((combo, p, SSE, R2, adjR2, Cp, AIC, BIC))

for r in sorted(results, key=lambda x: x[6]):  # sort by AIC
    print(f"{r[0]}: p={r[1]}, SSE={r[2]:.2f}, R2={r[3]:.4f}, adjR2={r[4]:.4f}, "
          f"Cp={r[5]:.3f}, AIC={r[6]:.3f}, BIC={r[7]:.3f}")
```

```python
# statsmodels: forward selection via partial F, using formula interface
import statsmodels.formula.api as smf
import pandas as pd

df = pd.DataFrame({'X1':X1,'X2':X2,'X3':X3,'Y':Y})

def partial_F(reduced_formula, full_formula, data):
    m_r = smf.ols(reduced_formula, data=data).fit()
    m_f = smf.ols(full_formula, data=data).fit()
    df_r, df_f = m_r.df_resid, m_f.df_resid
    F = ((m_r.ssr - m_f.ssr)/(df_r - df_f)) / (m_f.ssr/df_f)
    return F

print("F(X1|X3):", partial_F('Y ~ X3', 'Y ~ X3 + X1', df))
print("F(X2|X3):", partial_F('Y ~ X3', 'Y ~ X3 + X2', df))
```

```python
# sklearn: leave-one-out cross-validation (the modern equivalent of PRESS)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score
import numpy as np

X = np.column_stack([X2, X3])  # e.g., testing the {X2,X3} subset
loo = LeaveOneOut()
scores = cross_val_score(LinearRegression(), X, Y, cv=loo, scoring='neg_mean_squared_error')
PRESS_equivalent = -scores.sum() * len(Y)  # sum of squared LOO errors (PRESS statistic)
print("PRESS-equivalent (sum of squared LOO errors):", PRESS_equivalent)
```

---

## Interview Question Bank — Chapter 9

**Conceptual:**
1. Why does raw $R^2$ never decrease when you add predictors, and why does that make it unsuitable alone for model comparison?
2. What tradeoff is Mallow's $C_p$ specifically designed to balance?
3. Why does the full model's $C_p$ always equal exactly $p$, and why doesn't that make the full model automatically "good"?

**Derivation:**
4. Derive why BIC penalizes model complexity more heavily than AIC once $n$ is moderately large.
5. Show, using extra sums of squares, how backward elimination's per-step test at any given stage is just a partial F-test from Chapter 7.

**ML/Statistics:**
6. Explain the exact correspondence between PRESS and leave-one-out cross-validation.
7. Why can forward selection and backward elimination converge on different final models, in general (even though they agreed in our worked example)?
8. Why are p-values reported for a stepwise-selected model's final coefficients untrustworthy if interpreted at face value?

**Coding:**
9. Implement a best-subsets search from scratch in NumPy that computes SSE, adjusted R², $C_p$, AIC, and BIC for every possible predictor subset.
10. Implement forward selection using partial F-tests as the entry criterion, from scratch.

**Traps:**
11. "This model has the highest R² among all subsets I tried, so it's the best model." — what's the flaw?
12. "My stepwise procedure selected these 3 features and they were all significant at p<0.05, so I'm confident they're the right features." — what's wrong with trusting this p-value directly?
13. Two model-selection criteria disagree on the best subset. Which one should you trust, and how would you actually decide?

---

*This file covers Kutner Ch. 9 — the model-building process and the importance of validation, the full family of selection criteria ($R^2_p$, adjusted $R^2_p$, Mallow's $C_p$, AIC, BIC, and PRESS/leave-one-out), and forward selection and backward elimination worked step-by-step via partial F-tests, converging on the same final model from opposite directions. Chapter 10 (Diagnostics and Remedial Measures for Multiple Regression) is next — extending Chapter 3's single-predictor diagnostics to the multivariate case: added-variable plots, studentized deleted residuals, leverage/Cook's distance for influence, and a full formal treatment of the Variance Inflation Factor's role in diagnosing the multicollinearity this chapter kept running into.*
