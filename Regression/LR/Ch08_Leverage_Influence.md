# Chapter 8 — Diagnostics II: Leverage & Influence

*Synthesized from Kutner, Montgomery, Sheather, and ESL/ISL. Continuing Chapter 7's dataset and leverage/residual table in full.*

Recall from Chapter 7:

| Student | $e_i$ | $h_{ii}$ | $r_i$ (studentized) |
|---|---|---|---|
| 1 | 0.2 | 0.733 | 0.354 |
| 2 | 0.6 | 0.600 | 0.866 |
| 3 | -1.0 | 0.333 | -1.118 |
| 4 | -0.6 | 0.600 | -0.866 |
| 5 | 0.8 | 0.733 | 1.414 |

---

## 8.1 The Motivating Question — Leverage Is Not Influence

This is the single most important distinction in the chapter, and one interviewers specifically probe because the two concepts are so easy to conflate:

- **Leverage** ($h_{ii}$) measures how *unusual* a point's predictor values ($x$) are — how far it sits from the center of the predictor space. It's computed purely from $\mathbf{X}$, **without ever looking at $y$**.
- **Influence** measures how much a point actually **changes the fitted model** if you remove it — it depends on *both* the point's leverage **and** how far its $y$-value sits from what the model would have predicted for it (its residual).

**The key insight:** a point can have high leverage but low influence (if its $y$-value happens to agree with what the rest of the data already predicted), and a point can have low leverage but still noticeably shift the fit if its residual is large enough. **High leverage is necessary but not sufficient for high influence** — you need an unusual $x$ *and* a surprising $y$ together.

---

## 8.2 Cook's Distance — Combining Leverage and Residual Into One Number

Cook's distance for observation $i$ is:

$$ D_i = \frac{r_i^2}{p} \cdot \frac{h_{ii}}{1-h_{ii}} $$

where $p$ is the number of estimated parameters (here $p=3$: intercept, $\hat{\beta}_1$, $\hat{\beta}_2$) and $r_i$ is the internally studentized residual from Chapter 7.

**Plain-English reading:** the first factor ($r_i^2/p$) captures "how surprising is this point's $y$-value" — the second factor ($h_{ii}/(1-h_{ii})$) captures "how much leverage does this point have to drag the fit toward itself." $D_i$ is large only when **both** factors are large together — exactly formalizing the intuition from §8.1.

**Worked numbers for all 5 students** (using $p=3$):

| Student | $r_i^2$ | $h_{ii}/(1-h_{ii})$ | $D_i = \frac{r_i^2}{3}\times\frac{h_{ii}}{1-h_{ii}}$ |
|---|---|---|---|
| 1 | 0.125 | 2.750 | 0.1146 |
| 2 | 0.750 | 1.500 | 0.375 |
| 3 | 1.250 | 0.500 | 0.2083 |
| 4 | 0.750 | 1.500 | 0.375 |
| 5 | **2.000** | **2.750** | **1.833** |

**Student 5's Cook's distance (1.833) dwarfs every other point.** A common rule of thumb flags $D_i > 4/n$ (here $4/5=0.8$) or the more conservative classical threshold $D_i > 1$ — student 5 clears **both** thresholds decisively, while every other student sits comfortably below 0.4. This is the formal confirmation of what Chapter 7 hinted at informally: student 5 combines both high leverage ($h_{55}=0.733$, tied for highest) **and** a large residual ($e_5=0.8$, tied for second-largest) — exactly the combination that produces outsized influence.

**Contrast with student 3:** despite having the single *largest raw residual* ($e_3=-1$, larger in magnitude than student 5's 0.8), student 3's Cook's distance (0.208) is far smaller than student 5's — because student 3's leverage (0.333) is comparatively low. This is the cleanest possible illustration of §8.1's core point: **a big residual alone doesn't guarantee big influence; it needs leverage too.**

---

## 8.3 DFBETAS — Influence on Individual Coefficients

Cook's distance summarizes a point's overall influence on the *entire* fitted model in one number. Sometimes you need to know something more specific: **how much does removing this one point shift a particular coefficient?** That's what DFBETAS measures:

$$ DFBETAS_{j,i} = \frac{\hat{\beta}_j - \hat{\beta}_{j(i)}}{s_{(i)}\sqrt{[(\mathbf{X}^T\mathbf{X})^{-1}]_{jj}}} $$

where $\hat{\beta}_{j(i)}$ is coefficient $j$ refit **without** observation $i$, and $s_{(i)}$ is the residual standard error also computed without observation $i$ (the same deleted quantity from Chapter 7, §7.4).

**Worked illustration — removing student 5.** The remaining 4 students (1–4) have $x_1=1,2,3,4$, $x_2=1,1,2,2$, $y=50,55,65,70$ — and it turns out these 4 points satisfy $y=40+5x_1+5x_2$ **exactly** (this is the original perfect-fit relationship from Chapter 4, before student 5's score was nudged to 83 in Chapter 5). So:

$$ \hat{\beta}_{0(5)}=40,\quad \hat{\beta}_{1(5)}=5,\quad \hat{\beta}_{2(5)}=5 \qquad\text{with } SSE_{(5)}=0 $$

**Raw coefficient shifts from removing student 5:**

$$ \hat{\beta}_1-\hat{\beta}_{1(5)} = 4.6-5 = -0.4 \qquad \hat{\beta}_2-\hat{\beta}_{2(5)} = 7-5 = 2.0 $$

**An instructive edge case:** because $SSE_{(5)}=0$ exactly, $s_{(5)}=0$, which makes the *standardized* DFBETAS formula divide by zero — technically undefined (infinite). Rather than a flaw in the example, **this is the sharpest possible illustration of the concept**: every bit of "unexplained noise" in the full 5-point model was contributed by student 5 alone. Remove that single point, and the remaining four fit a perfect plane with zero residual variance left over — meaning student 5 wasn't just influential, it was **solely** responsible for the entire model's residual variation. Common software (R, statsmodels) would report this as `NaN` or `Inf` in practice, and a `NaN` DFBETAS/Cook's-distance value is itself a diagnostic signal worth investigating, not something to silently discard.

**General interpretation rule (for well-behaved, non-degenerate cases):** $|DFBETAS_{j,i}| > 2/\sqrt{n}$ is a common flagging threshold for "this point meaningfully shifts coefficient $j$."

---

## 8.4 Other Influence Measures (Briefly, for Completeness)

- **DFFITS** — analogous to DFBETAS, but measures the shift in the *fitted value* $\hat{y}_i$ itself (rather than a specific coefficient) when point $i$ is removed. Threshold: $|DFFITS_i| > 2\sqrt{p/n}$.
- **COVRATIO** — measures how removing point $i$ changes the *precision* (variance-covariance matrix determinant) of the coefficient estimates as a whole, rather than their point values.

These are used less often by hand in interviews than Cook's distance, but recognizing their names and general purpose ("influence on fitted values" / "influence on estimation precision," respectively) is worth having ready.

---

## 8.5 What to Actually Do With a Flagged Point

Finding a high-Cook's-distance point is the **start** of an investigation, not an automatic deletion order. In order of preference:

1. **Check for a data-entry error first** — often the simplest and most defensible fix.
2. **Consider whether the point reflects a genuinely different regime** — e.g., an unusual but legitimate observation that suggests the linear model is misspecified for part of the data (ties back to Chapter 7's linearity diagnostic).
3. **Report results with and without the point** — transparency about sensitivity to a single observation is often more valuable than picking one version and hiding the other.
4. **Only as a last resort, and with clear justification, exclude it** — blind deletion of "inconvenient" points is a well-known way to manufacture false confidence in a model; both Kutner and Montgomery explicitly warn against this.

---

## 8.6 Where the Textbooks Differ

- **Kutner** derives Cook's distance and DFBETAS with the most complete algebraic connection back to the hat matrix and studentized residuals — the most proof-complete treatment.
- **Montgomery** is the strongest source on the *practitioner workflow* in §8.5 — extensive case studies of what to actually do once an influential point is found, particularly in engineering/manufacturing contexts.
- **Sheather** emphasizes reading these diagnostics directly from software (`cooks.distance()`, `dfbetas()` in R) and interpreting the standard flagged-point plots, rather than hand-deriving the formulas.
- **ESL/ISL** essentially skip this topic — influence diagnostics are a classical-inference concern, and ESL/ISL's predictive, cross-validation-driven philosophy handles problematic points implicitly (a single influential point matters less when you're evaluating out-of-sample predictive accuracy across many resamples) rather than explicitly flagging them one at a time.

---

## 8.7 Interview Q&A

**Q: What's the difference between leverage and influence?**
A: Leverage ($h_{ii}$) depends only on a point's predictor ($x$) values — how far it sits from the center of the predictor space. Influence depends on leverage **combined with** the point's residual — how much removing that point would actually change the fitted model. High leverage alone doesn't guarantee high influence.

**Q: A point has the largest raw residual in the dataset but a low Cook's distance. How is that possible?**
A: Its leverage must be low. Cook's distance requires both a large residual **and** meaningful leverage to be large — a big residual at a low-leverage point (near the center of the predictor space) has limited ability to pull the fitted line/plane toward itself.

**Q: What does DFBETAS measure that Cook's distance doesn't?**
A: Cook's distance summarizes overall influence on the whole fitted model in one number; DFBETAS isolates the influence on one **specific** coefficient, which matters when you care about interpreting that particular predictor's effect rather than overall model fit.

**Q: What should you do if you find a point with a very high Cook's distance?**
A: Investigate before acting — check for a data-entry error, consider whether it reflects real but unusual data (possibly signaling model misspecification), and report sensitivity with and without the point rather than silently deleting it.

**Q: Can a point have zero residual but still be influential?**
A: It can still have high leverage, but Cook's distance specifically would be low if its residual is exactly (or near) zero — because Cook's distance requires both factors. However, such a point could still substantially affect *other* diagnostics, like standard errors, simply by its extreme predictor value.

---

*End of Chapter 8. Next: Chapter 9 — Multicollinearity (formal VIF and condition-number diagnostics, building directly on the coefficient instability first observed in Chapter 5, §5.4).*
