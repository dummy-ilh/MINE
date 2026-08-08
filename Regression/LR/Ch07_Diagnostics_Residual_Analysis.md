# Chapter 7 — Diagnostics I: Residual Analysis

*Synthesized from Kutner, Montgomery, Sheather, and ESL/ISL. Continuing Chapter 5's noisy dataset ($\hat{\beta}_0=38.2,\hat{\beta}_1=4.6,\hat{\beta}_2=7$; residuals $e = 0.2,\ 0.6,\ -1,\ -0.6,\ 0.8$; $MSE=1.2$) and the hat matrix machinery from Chapter 3.*

---

## 7.1 The Motivating Question

Chapter 1 introduced the residual plot informally — "look for random scatter." This chapter makes that instinct rigorous. Two problems with using **raw** residuals $e_i$ directly:

1. **Raw residuals don't have equal variance across observations**, even under perfect homoscedasticity of the true errors — because $\text{Var}(e_i)=\sigma^2(1-h_{ii})$, where $h_{ii}$ is that observation's leverage (Chapter 3, §3.6). High-leverage points mechanically have *smaller* residual variance, so comparing raw residuals side by side is comparing apples whose sizes differ for reasons that have nothing to do with model fit.
2. **Raw residuals have no natural scale** — is $e_i=2$ large or small? Depends entirely on the units and spread of $y$. You need a standardized yardstick.

This chapter fixes both problems, then uses the fixed version to build the formal diagnostic toolkit.

---

## 7.2 Standardized (Semi-Studentized) Residuals

The simplest fix — divide every residual by the overall residual standard error:

$$ d_i = \frac{e_i}{\sqrt{MSE}} = \frac{e_i}{s} $$

This puts every residual on a common, unit-free scale, but it **still ignores the leverage problem** above — it uses the same denominator $s$ for every point regardless of that point's own leverage. It's a quick sanity check, not the full diagnostic tool.

---

## 7.3 Internally Studentized Residuals — Fixing the Leverage Problem

The correct fix accounts for each point's own leverage $h_{ii}$:

$$ r_i = \frac{e_i}{s\sqrt{1-h_{ii}}} $$

**Plain-English reading:** this rescales each residual by *how much variance that specific point's residual is even allowed to have* — a high-leverage point ($h_{ii}$ close to 1) has its residual pulled toward zero mechanically (the fitted line is forced close to it), so dividing by $\sqrt{1-h_{ii}}$ un-shrinks it back to a fair, comparable scale. Under the model's assumptions, $r_i$ approximately follows a t-distribution.

**Worked numbers.** Recall the leverage values (computed from $\mathbf{H}=\mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$ using Chapter 5's $(\mathbf{X}^T\mathbf{X})^{-1}$, which only depends on the predictors, not $y$):

| Student | $e_i$ | $h_{ii}$ | $1-h_{ii}$ | $r_i$ (studentized) |
|---|---|---|---|---|
| 1 | 0.2 | 0.733 | 0.267 | 0.354 |
| 2 | 0.6 | 0.600 | 0.400 | 0.866 |
| 3 | -1.0 | 0.333 | 0.667 | -1.118 |
| 4 | -0.6 | 0.600 | 0.400 | -0.866 |
| 5 | 0.8 | 0.733 | 0.267 | 1.414 |

(Check: $\sum h_{ii} = 0.733+0.6+0.333+0.6+0.733 = 3.0$, exactly matching $\text{trace}(\mathbf{H})=p+1=3$ from Chapter 3, §3.6 — always true, useful as an arithmetic sanity check.)

**Notice** student 3 has the *smallest* raw residual magnitude relative to student 5 (|-1| vs |0.8|) but a *larger* studentized residual (-1.118 vs 1.414 is actually smaller in magnitude — let's read this carefully: student 5's raw residual of 0.8 becomes the largest studentized residual, 1.414, precisely because student 5 also carries high leverage, which mechanically shrinks how large a raw residual "should" be able to get — a small raw residual at a high-leverage point is actually more anomalous once properly rescaled).

None of these exceed common flagging thresholds (±2 or ±3), so nothing here looks like an outlier — expected, given how small and clean this dataset is.

---

## 7.4 Externally Studentized (Deleted) Residuals

Internally studentized residuals have a subtle flaw: $s$ (used in the denominator) was computed **using** every point, including the one you're checking — if point $i$ is a genuine outlier, it inflates $s$ itself, making its own studentized residual look artificially smaller than it should ("the outlier hides its own effect on the ruler used to measure it").

The fix: refit the model **excluding** point $i$, get a residual standard error $s_{(i)}$ that doesn't include point $i$'s influence, and use that instead:

$$ t_i = \frac{e_i}{s_{(i)}\sqrt{1-h_{ii}}} $$

Refitting $n$ separate models by hand would be tedious, but there's a shortcut formula that avoids literally refitting:

$$ s_{(i)}^2 = \frac{(n-p-1)MSE - \dfrac{e_i^2}{1-h_{ii}}}{n-p-2} $$

**Worked example for student 3** (the point with the largest internal studentized residual):

$$ \frac{e_3^2}{1-h_{33}} = \frac{(-1)^2}{0.667} = 1.5 $$

$$ s_{(3)}^2 = \frac{(2)(1.2) - 1.5}{1} = \frac{2.4-1.5}{1} = 0.9 \quad\Rightarrow\quad s_{(3)} \approx 0.949 $$

$$ t_3 = \frac{-1}{0.949\times\sqrt{0.667}} = \frac{-1}{0.775} \approx -1.29 $$

Compare to $r_3=-1.118$ (internal): the externally studentized version is slightly larger in magnitude, because excluding student 3 actually *reduced* the estimated residual variance — a hint (though far from conclusive with only 1 residual degree of freedom remaining after deletion, $n-p-2=1$ here) that student 3 might be worth a second look in a larger dataset. **Caution flagged honestly:** with $n-p-2=1$ degree of freedom, this specific numeric comparison is more a demonstration of the mechanics than a trustworthy outlier test — you'd want considerably more data before taking any single externally studentized residual seriously.

**Interview-critical distinction:** internal studentized residuals use $s$ from the full model; external (deleted) studentized residuals use $s_{(i)}$ with point $i$ excluded — the latter is the statistically correct tool for formal outlier testing (it follows an exact t-distribution with $n-p-2$ df under the null of no outlier), while the former is a faster approximation.

---

## 7.5 The Four-Panel Diagnostic Plot — Each Panel Tied to a Specific Assumption

Formalizing Chapter 1's informal residual-plot intuition, standard regression software (R's `plot.lm()`, or the equivalent in Python) produces four panels, **each one targeted at a specific Gauss-Markov / LINE assumption:**

| Panel | What's plotted | Assumption being checked | Healthy pattern | Warning pattern |
|---|---|---|---|---|
| 1. Residuals vs. Fitted | $e_i$ (or $r_i$) vs. $\hat{y}_i$ | **Linearity** | Random scatter around zero | Curved/U-shaped pattern → missing nonlinear term |
| 2. Normal Q-Q plot | Sorted studentized residuals vs. theoretical normal quantiles | **Normality** | Points close to the 45° diagonal line | S-curve or heavy tails → non-normal errors |
| 3. Scale-Location | $\sqrt{|r_i|}$ vs. $\hat{y}_i$ | **Equal variance (homoscedasticity)** | Flat, horizontal band | Upward/downward trend → variance changes with fitted value |
| 4. Residuals vs. Leverage | $r_i$ vs. $h_{ii}$, with Cook's distance contours overlaid | **Influential points** | All points inside Cook's distance contours | Points outside contours → high-influence outliers (full treatment in Chapter 8) |

*(Diagram to visualize: a 2×2 grid of the four panels described above — top-left a random cloud around a horizontal zero line; top-right a straight diagonal line of points; bottom-left a flat horizontal band of points; bottom-right a scatter with two curved dashed contour lines near the plot edges marking Cook's distance thresholds.)*

**Why this matters as a system, not four separate checks:** each panel maps to one distinct assumption. A model can pass three panels and fail one — e.g., perfectly linear and homoscedastic residuals that are clearly non-normal in the Q-Q plot. Diagnosing *which* assumption is broken (not just "something looks off") is what determines the correct fix — a curved Panel 1 calls for a transformation (Chapter 12), a fanning Panel 3 calls for WLS (Chapter 19), while Panel 2 issues are often tolerable in large samples due to the Central Limit Theorem protecting your inference even when errors aren't exactly normal.

---

## 7.6 Where the Textbooks Differ

- **Kutner** introduces standardized and studentized residuals with the most complete algebraic derivation, precisely deriving $\text{Var}(e_i)=\sigma^2(1-h_{ii})$ from the hat matrix properties in Chapter 3.
- **Montgomery** is the strongest source specifically on the four-panel plot as an integrated diagnostic *system*, with extensive real engineering-data examples of each violation pattern.
- **Sheather** emphasizes computing and interpreting these diagnostics directly from R's `rstandard()` and `rstudent()` functions, treating the formulas as background for interpreting software output rather than something to compute by hand.
- **ESL/ISL** barely discuss residual diagnostics at all — their attention is on predictive performance (train/test error, cross-validation) rather than classical assumption-checking, reflecting the more ML-oriented, less inference-oriented philosophy of that text.

---

## 7.7 Interview Q&A

**Q: Why isn't a raw residual $e_i$ enough to judge whether a point is unusual?**
A: Raw residuals don't have equal variance across observations — $\text{Var}(e_i)=\sigma^2(1-h_{ii})$ — so a high-leverage point's residual is mechanically shrunk regardless of model fit. Comparing raw residuals across points with different leverage is comparing on an inconsistent scale.

**Q: What's the difference between internally and externally studentized residuals?**
A: Internal studentized residuals use the residual standard error $s$ computed from the full model (including the point itself); external (deleted) studentized residuals refit excluding that point, giving a statistically valid t-distributed test for whether that point is an outlier — internal residuals are a faster approximation that can understate an outlier's own effect on the fit.

**Q: Which diagnostic plot checks which assumption?**
A: Residuals vs. fitted → linearity; Normal Q-Q → normality; scale-location → homoscedasticity; residuals vs. leverage → influential points/outliers. Each panel targets one specific assumption, not a generic "does this look okay" check.

**Q: If your Q-Q plot shows non-normal residuals but the other three panels look fine, should you panic?**
A: Not necessarily — OLS point estimates don't require normality (Gauss-Markov, Chapter 6), and with a reasonably large sample, the Central Limit Theorem means t-tests and F-tests remain approximately valid even with non-normal errors. It's a bigger concern in small samples, where exact-distribution inference relies on normality more heavily.

**Q: How would you formally test whether a specific point is a statistical outlier?**
A: Compute its externally studentized (deleted) residual $t_i$ and compare it to a t-distribution with $n-p-2$ degrees of freedom — typically with a Bonferroni correction if you're testing multiple points simultaneously, since checking every point for outliers is itself multiple testing.

---

*End of Chapter 7. Next: Chapter 8 — Diagnostics II: Leverage & Influence (formalizing Cook's distance and DFBETAS, and precisely distinguishing a high-leverage point from a high-influence point — they are not the same thing).*
