# Chapter 7 — Diagnostics I: Residual Analysis — Interview-Boosted Edition

*Synthesized from Kutner, Montgomery, Sheather, and ESL/ISL. Continuing Chapter 5's noisy dataset ($\hat\beta_0=38.2,\hat\beta_1=4.6,\hat\beta_2=7$; residuals $e=0.2,0.6,-1,-0.6,0.8$; $MSE=1.2$) and the hat matrix machinery from Chapter 3.*

**Why this chapter carries real interview weight:** almost every regression interview question that sounds like "how do you know your model is any good?" is secretly asking about this chapter. It's also the direct prerequisite for Chapter 8 (leverage/influence) and the diagnostic trigger for half the remedies in this syllabus — a curved Panel 1 sends you to Chapter 12, a fanning Panel 3 sends you to Chapter 19, and so on. Getting this chapter fluent makes the rest of the syllabus click into place as "which panel told me to come here."

---

## 7.0 The 60-Second Answer

> "Raw residuals aren't directly comparable across points, because high-leverage points mechanically have smaller residual variance — $\text{Var}(e_i)=\sigma^2(1-h_{ii})$ — regardless of fit quality. So you standardize: divide by $s$ for a quick look (standardized), or by $s\sqrt{1-h_{ii}}$ to correct for each point's own leverage (internally studentized). The most rigorous version, externally studentized, refits the model *excluding* that point so an outlier can't inflate its own yardstick, and follows an exact t-distribution with $n-p-2$ df. Then the four-panel plot checks four *different* assumptions — fitted-vs-residual for linearity, Q-Q for normality, scale-location for equal variance, leverage-vs-residual for influential points — so 'the residuals look fine' actually means four separate claims, and you should be able to say which panel supports which one."

Everything below unpacks and proves this paragraph.

---

## 7.1 The Motivating Question

Chapter 1 said "eyeball the residuals, they should look like random noise." Two hidden traps in that advice:

1. **Not all residuals are allowed the same amount of variance**, even under perfect homoscedasticity — $\text{Var}(e_i)=\sigma^2(1-h_{ii})$, where $h_{ii}$ is that point's leverage (Chapter 3, §3.6). A high-leverage point is *structurally* forced toward a smaller residual, independent of how good the model actually is. Comparing raw residuals across points is comparing apples of different sizes for reasons that have nothing to do with fit.
2. **Raw residuals have no natural scale.** Is $e_i=2$ large or small? Depends entirely on the units and spread of $y$. You need a universal yardstick before judging.

This chapter builds that yardstick, then uses it for a formal diagnostic system.

---

## 7.2 Standardized Residuals — the Quick, Incomplete Fix

$$ d_i = \frac{e_i}{\sqrt{MSE}} = \frac{e_i}{s} $$

**What it fixes / what it doesn't:** puts every residual on a unit-free scale (fixes trap 2), but uses the *same* denominator for every point regardless of leverage (still misses trap 1). Treat this as a first-glance sanity check, not a final outlier call.

---

## 7.3 Internally Studentized Residuals — Fixing the Leverage Problem

$$ r_i = \frac{e_i}{s\sqrt{1-h_{ii}}} $$

**The tug-of-war analogy:** a high-leverage point ($h_{ii}$ near 1) pulls the fitted line toward itself almost regardless of its $y$-value — so its residual is *artificially* small, not because the model fit it well, but because the line was forced close to it. Dividing by $\sqrt{1-h_{ii}}$ un-shrinks the residual back onto a fair, comparable scale — "how surprising is this miss, given how much wiggle room this point actually had?" Under the model assumptions, $r_i$ approximately follows a t-distribution.

**Worked numbers**, using the leverage values (from $\mathbf{H}=\mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$, built from Chapter 5's $(\mathbf{X}^T\mathbf{X})^{-1}$ — leverage depends only on the predictors, never on $y$):

| Student | $e_i$ | $h_{ii}$ | $1-h_{ii}$ | $r_i$ |
|---|---|---|---|---|
| 1 | 0.2 | 0.733 | 0.267 | 0.354 |
| 2 | 0.6 | 0.600 | 0.400 | 0.866 |
| 3 | -1.0 | 0.333 | 0.667 | -1.118 |
| 4 | -0.6 | 0.600 | 0.400 | -0.866 |
| 5 | 0.8 | 0.733 | 0.267 | 1.414 |

**Sanity check:** $\sum h_{ii}=0.733+0.6+0.333+0.6+0.733=3.0$, exactly $\text{trace}(\mathbf{H})=p+1=3$ — this identity should hold for *any* dataset and is worth running as a check whenever you compute leverages by hand.

**The one result worth internalizing:** student 5's raw residual (0.8) is smaller than student 3's (−1.0 in magnitude), yet student 5 ends up with the *largest* studentized residual (1.414). Why: student 5 has high leverage ($h_{55}=0.733$), so the line was pulled hard toward it — you'd expect its residual to shrink almost automatically. That it still missed by 0.8 despite that pull is more surprising than student 3's larger raw miss at lower leverage ($h_{33}=0.333$, where the line had no special reason to be close). Studentizing formalizes exactly that intuition.

None of these exceed the usual flagging thresholds (±2 or ±3) — expected, given how small and clean this dataset is.

---

## 7.4 Externally Studentized (Deleted) Residuals — the Rigorous Version

**The flaw in the internal version:** $s$ is computed *using* every point, including the one being checked. If point $i$ is a genuine outlier, it inflates $s$ itself, making its own studentized residual look smaller than it should — an outlier partially hides its own effect on the ruler used to measure it (like calculating "average height" using a group that includes the very tall person you're trying to flag).

**The fix:** refit excluding point $i$, get $s_{(i)}$ without that point's influence:

$$ t_i = \frac{e_i}{s_{(i)}\sqrt{1-h_{ii}}} $$

**Shortcut formula (no literal refitting needed):**

$$ s_{(i)}^2 = \frac{(n-p-1)MSE - \dfrac{e_i^2}{1-h_{ii}}}{n-p-2} $$

**Worked for student 3** (largest internal studentized residual):

$$ \frac{e_3^2}{1-h_{33}}=\frac{1}{0.667}=1.5 \qquad s_{(3)}^2=\frac{(2)(1.2)-1.5}{1}=\frac{0.9}{1}=0.9 \quad\Rightarrow\quad s_{(3)}\approx0.949 $$

$$ t_3=\frac{-1}{0.949\times\sqrt{0.667}}=\frac{-1}{0.775}\approx-1.29 $$

Compare to $r_3=-1.118$: the external version is larger in magnitude, because removing student 3 actually *lowered* the estimated noise level ($\sqrt{1.2}\approx1.095\to0.949$) — a mild signal student 3 was contributing more than its share of noise. **Honesty check:** with $n-p-2=1$ residual df remaining after deletion, this is a mechanics demonstration, not a trustworthy test — you'd want far more data before treating any single toy-dataset result as a real outlier finding.

**The one-line distinction that gets tested:** internal studentized residuals use $s$ from the *full* model (fast, approximate); external studentized residuals use $s_{(i)}$ with the point excluded (statistically exact, follows a genuine t-distribution with $n-p-2$ df) — external is the correct tool for a *formal* outlier test.

---

## 7.5 The Four-Panel Diagnostic Plot — One Panel, One Assumption

| Panel | Plotted | Assumption checked | Healthy | Warning |
|---|---|---|---|---|
| 1. Residuals vs. Fitted | $e_i$ (or $r_i$) vs. $\hat y_i$ | **Linearity** | Random scatter around zero | Curve/U-shape → missing nonlinear term |
| 2. Normal Q-Q | Sorted studentized residuals vs. normal quantiles | **Normality** | Points hug the 45° line | S-curve / heavy tails → non-normal errors |
| 3. Scale-Location | $\sqrt{|r_i|}$ vs. $\hat y_i$ | **Homoscedasticity** | Flat horizontal band | Trend/funnel → variance changes with fitted value |
| 4. Residuals vs. Leverage | $r_i$ vs. $h_{ii}$, Cook's distance contours overlaid | **Influential points** | All points inside contours | Points outside → high-influence (Chapter 8) |

**One line per panel, why it works:**

- **Panel 1** — a random cloud means the model's *shape* is right; a curve means the true relationship bends and a straight line can't capture it → the fix is a transformation (Chapter 12), not more data.
- **Panel 2** — compares sorted residuals to what a bell curve would produce; deviation, especially at the tails, means more extreme errors than normality predicts → matters for how much to trust p-values/CIs, less so for the point estimate itself.
- **Panel 3** — checks whether the *size* of typical mistakes is constant across the range of predictions; a funnel shape means the model is noisier in some regions than others → the fix is Weighted Least Squares (Chapter 19).
- **Panel 4** — flags points that are *both* unusual in their predictor values (high leverage) *and* poorly fit (large residual) at once. High leverage alone is harmless if the point is well fit; it's the combination that matters, because that combination means removing the point could meaningfully change the whole fitted line. Cook's distance (Chapter 8) formalizes exactly this combination into one number.

**Why this matters as a system, not four separate checks:** a model can pass three panels and fail one — e.g., linear and homoscedastic residuals that are visibly non-normal in the Q-Q plot. The point of naming *which* panel is broken is that it determines the correct fix directly: curved Panel 1 → transform; fanning Panel 3 → WLS; Panel 2 issues alone are often tolerable in large samples (Central Limit Theorem protects your inference even with non-normal errors — same estimation-vs-inference split as Chapter 6). Saying "I checked the residual plots" is a weak interview answer; saying "Panel 3 fanned out, so I diagnosed heteroscedasticity and moved to WLS" is a strong one.

---

## 7.6 Common Interview Traps

- **"A large residual always means an outlier."** → No — check leverage first. A large raw residual at *low* leverage is more suspicious than the same raw residual at *high* leverage (§7.3's student-5-vs-3 comparison is the canonical example).
- **"Internal and external studentized residuals give the same answer."** → They agree closely when no point is a real outlier, but diverge exactly when it matters most — a genuine outlier inflates its own internal yardstick, understating its own studentized residual. That's the entire reason the external version exists.
- **"A non-normal Q-Q plot means my model is broken."** → No — it means your *p-values and confidence intervals* might be untrustworthy, especially in small samples. Your point estimates (the fitted line itself) don't require normality at all (Gauss-Markov, Chapter 6).
- **"High leverage means a bad data point."** → No — leverage is about an unusual *predictor* value, not a bad fit. A high-leverage point that's well-fit is harmless. Only leverage *combined with* a large residual is dangerous (Panel 4, formalized in Chapter 8).
- **"Checking multiple points for outliers is just repeating the same t-test many times."** → Testing many points at once is a multiple-testing problem; a rigorous approach applies a Bonferroni correction to the threshold rather than using the naive per-point significance level.

---

## 7.7 Rapid-Fire Flashcards

| Q | A |
|---|---|
| Formula for $\text{Var}(e_i)$? | $\sigma^2(1-h_{ii})$ |
| Standardized residual formula? | $e_i/s$ |
| Internally studentized formula? | $e_i/(s\sqrt{1-h_{ii}})$ |
| Externally studentized formula? | $e_i/(s_{(i)}\sqrt{1-h_{ii}})$ |
| What distribution does the external version follow? | t-distribution with $n-p-2$ df |
| Why does the internal version understate real outliers? | The outlier's own residual inflates the $s$ used to judge it |
| Panel 1 checks? | Linearity |
| Panel 2 checks? | Normality |
| Panel 3 checks? | Homoscedasticity |
| Panel 4 checks? | Influential points (leverage + large residual together) |
| Sanity-check identity for leverages? | $\sum h_{ii}=\text{trace}(\mathbf{H})=p+1$ |
| Fix for a curved Panel 1? | Transformation (Chapter 12) |
| Fix for a fanning Panel 3? | Weighted Least Squares (Chapter 19) |
| Does non-normality break the point estimate? | No — only exact-distribution inference (p-values, CIs) |

---

## 7.8 Where the Textbooks Differ

- **Kutner** gives the most complete algebraic derivation, precisely deriving $\text{Var}(e_i)=\sigma^2(1-h_{ii})$ from hat-matrix properties (Chapter 3).
- **Montgomery** is the strongest source on the four-panel plot as an integrated diagnostic *system*, with extensive real engineering-data examples of each violation pattern.
- **Sheather** emphasizes computing these diagnostics via R's `rstandard()`/`rstudent()`, treating the formulas as background for reading software output.
- **ESL/ISL** barely cover residual diagnostics at all — their focus is predictive performance (train/test error, cross-validation) over classical assumption-checking.

---

## 7.9 Interview Q&A

**Q: Why isn't a raw residual enough to judge whether a point is unusual?**
A: Raw residuals don't have equal variance across observations — $\text{Var}(e_i)=\sigma^2(1-h_{ii})$ — so a high-leverage point's residual is mechanically shrunk regardless of fit quality. You need to correct for that before comparing points.

**Q: What's the difference between internally and externally studentized residuals?**
A: Internal uses $s$ from the full model (fast, approximate, can understate an outlier's own effect); external refits excluding that point, giving an exact t-distributed test — the correct tool for formal outlier testing.

**Q: Which diagnostic plot checks which assumption?**
A: Residuals-vs-fitted → linearity; Normal Q-Q → normality; scale-location → homoscedasticity; residuals-vs-leverage → influential points. Each panel targets exactly one assumption.

**Q: Non-normal Q-Q plot but the other three panels look fine — should you panic?**
A: Not necessarily. OLS point estimates don't require normality (Gauss-Markov). With reasonable sample size, the CLT keeps t-/F-tests approximately valid even with non-normal errors — a bigger concern only in small samples.

**Q: How would you formally test whether a specific point is a statistical outlier?**
A: Compute its externally studentized residual and compare to a t-distribution with $n-p-2$ df, applying a Bonferroni correction if testing multiple points simultaneously (checking every point is itself multiple testing, which inflates the false-positive rate).

---

*End of Chapter 7 (interview-boosted). Next: Chapter 8 — Diagnostics II: Leverage & Influence (formalizing Cook's distance and DFBETAS, and precisely distinguishing a high-leverage point from a high-influence point — they are not the same thing).*
