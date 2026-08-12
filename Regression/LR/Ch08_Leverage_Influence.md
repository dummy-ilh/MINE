# Chapter 8 — Diagnostics II: Leverage & Influence

*Synthesized from Kutner, Montgomery, Sheather, and ESL/ISL — expanded with plain-language explanations, a simplified formula cheat-sheet, and fully worked numerics. Continues Chapter 7's dataset and leverage/residual table.*

**Dataset recap (from Chapter 7):**

| Student | $e_i$ (raw residual) | $h_{ii}$ (leverage) | $r_i$ (internally studentized residual) |
|---|---|---|---|
| 1 | 0.2 | 0.733 | 0.354 |
| 2 | 0.6 | 0.600 | 0.866 |
| 3 | -1.0 | 0.333 | -1.118 |
| 4 | -0.6 | 0.600 | -0.866 |
| 5 | 0.8 | 0.733 | 1.414 |

$p = 3$ parameters are estimated (intercept, $\hat\beta_1$, $\hat\beta_2$); $n = 5$ observations.

---

## 8.1 The Motivating Question — Leverage Is Not Influence

This is the single most important distinction in the chapter, and interviewers probe it specifically because the two concepts are so easy to conflate.

| | Depends on | Ignores |
|---|---|---|
| **Leverage** ($h_{ii}$) | Only the predictor values $x$ — how far a point sits from the center of predictor space | The $y$-value entirely |
| **Influence** | Leverage **and** the residual — how surprising $y$ was, combined with how much power the point has to pull the fit | Nothing — it's the full picture |

**The key insight:** a point can have high leverage but low influence (its $y$ agrees with what was already predicted), and a point can have low leverage but still shift the fit noticeably if its residual is large enough. **High leverage is necessary but not sufficient for high influence** — you need an unusual $x$ *and* a surprising $y$ together.

**See-saw analogy:**
- **Leverage** = how far out on the see-saw you're sitting (position only — nothing about $y$ yet).
- **Influence** = did you actually tip it (position **combined with** whether your actual weight, i.e. your $y$-value, was a surprise).

Someone sitting at the very end has more *potential* to tip things, but only tips it if their weight is unexpected too.

---

## 8.2 Cook's Distance — Combining Leverage and Residual Into One Number

### The formula, and a simplified way to read it

$$ D_i = \frac{r_i^2}{p} \cdot \frac{h_{ii}}{1-h_{ii}} $$

Strip away the notation and this is just:

$$ D_i \;=\; \underbrace{\left(\frac{r_i^2}{p}\right)}_{\text{"how surprising was }y_i\text{?"}} \;\times\; \underbrace{\left(\frac{h_{ii}}{1-h_{ii}}\right)}_{\text{"how much power does this point have to drag the fit?"}} $$

**One-sentence version:** Cook's distance multiplies two separate "danger scores" together — a *surprise score* and a *leverage score*. If either score is near zero, the product is near zero too. A point needs to score high on **both** to end up with a large $D_i$.

### Fully worked numbers, all 5 students

| Student | $r_i^2$ | $\dfrac{h_{ii}}{1-h_{ii}}$ | $D_i=\dfrac{r_i^2}{3}\times\dfrac{h_{ii}}{1-h_{ii}}$ |
|---|---|---|---|
| 1 | $0.354^2=0.125$ | $\frac{0.733}{0.267}=2.745$ | $\frac{0.125}{3}\times2.745=0.1146$ |
| 2 | $0.866^2=0.750$ | $\frac{0.600}{0.400}=1.500$ | $\frac{0.750}{3}\times1.500=0.3750$ |
| 3 | $(-1.118)^2=1.250$ | $\frac{0.333}{0.667}=0.4995$ | $\frac{1.250}{3}\times0.4995=0.2083$ |
| 4 | $(-0.866)^2=0.750$ | $\frac{0.600}{0.400}=1.500$ | $\frac{0.750}{3}\times1.500=0.3750$ |
| 5 | $1.414^2=2.000$ | $\frac{0.733}{0.267}=2.745$ | $\frac{2.000}{3}\times2.745=\mathbf{1.833}$ |

**Flagging thresholds (two common conventions):**

| Rule of thumb | Value here | Student 5 |
|---|---|---|
| $D_i > 4/n$ | $4/5 = 0.800$ | $1.833 > 0.800$ ✅ flagged |
| $D_i > 1$ (classical/conservative) | $1.000$ | $1.833 > 1.000$ ✅ flagged |

**Student 5 clears both thresholds decisively**, while every other student sits below 0.4. Student 5 combines the joint-highest leverage ($h_{55}=0.733$) **and** a large residual ($e_5=0.8$, second-largest in magnitude) — exactly the combination that produces outsized influence.

### The cleanest contrast in the chapter: Student 3 vs. Student 5

| | Raw residual $\lvert e_i\rvert$ | Leverage $h_{ii}$ | Cook's $D_i$ |
|---|---|---|---|
| Student 3 | **1.0** (largest in the dataset) | 0.333 (low) | 0.208 |
| Student 5 | 0.8 (second-largest) | **0.733** (tied-highest) | **1.833** |

Student 3 has the single biggest raw miss, yet Student 5 — with a *smaller* residual — has a Cook's distance nearly **9× larger**. The only thing that differs is leverage. This is the sharpest possible illustration of §8.1: **a big residual alone doesn't guarantee big influence — it needs leverage too.**

**Plain words:** Student 3 "yelled the loudest" but sat near the middle of the see-saw, so the yell didn't move the fitted plane much. Student 5 yelled almost as loud while sitting way out at the far end — the same-ish size of miss translated into a massively bigger effect. You can never judge influence from the residual alone; you always have to ask *where the point was sitting too*.

---

## 8.3 DFBETAS — Influence on One Specific Coefficient

Cook's distance summarizes influence on the *whole* model in one number. DFBETAS narrows the question to: **did removing this point change this one particular coefficient?**

### Simplified formula reading

$$ DFBETAS_{j,i} = \frac{\hat\beta_j - \hat\beta_{j(i)}}{s_{(i)}\sqrt{[(\mathbf X^T\mathbf X)^{-1}]_{jj}}} \;=\; \frac{\text{(raw shift in coefficient }j\text{ from removing point }i)}{\text{(typical noise level, with point }i\text{ removed)}} $$

- $\hat\beta_{j(i)}$: coefficient $j$ refit **without** observation $i$.
- $s_{(i)}$: residual standard error, also computed **without** observation $i$ (the "deleted" quantity from §7.4).
- The denominator just rescales the raw shift into "number of standard errors," so DFBETAS values are comparable across coefficients and datasets.

**One-sentence version:** DFBETAS = (how far this coefficient moved) ÷ (how much noise is normally expected) — it tells you whether a shift is big *relative to* the model's usual wobble, not just big in absolute terms.

### Worked illustration — removing Student 5

Students 1–4 alone have $x_1=1,2,3,4$, $x_2=1,1,2,2$, $y=50,55,65,70$, and these four points satisfy $y = 40 + 5x_1 + 5x_2$ **exactly** (the original perfect-fit relationship from Chapter 4, before Student 5's score was nudged to 83 in Chapter 5). So:

$$ \hat\beta_{0(5)} = 40, \quad \hat\beta_{1(5)} = 5, \quad \hat\beta_{2(5)} = 5, \quad SSE_{(5)} = 0 $$

**Raw coefficient shifts caused by Student 5:**

| Coefficient | With Student 5 | Without Student 5 | Raw shift |
|---|---|---|---|
| $\hat\beta_1$ | 4.6 | 5.0 | $4.6-5.0=-0.4$ |
| $\hat\beta_2$ | 7.0 | 5.0 | $7.0-5.0=+2.0$ |

**Plain words:** with Student 5 in the data, the slope on $x_1$ came out to 4.6 and the slope on $x_2$ came out to 7.0. Pull Student 5 out and both numbers *snap* to a clean 5 and 5, with zero leftover error. Student 5 alone was responsible for dragging $\hat\beta_1$ below 5 and $\hat\beta_2$ above 5 — not "a bit noisy," but single-handedly bending both coefficients away from an otherwise-perfect relationship.

### The edge case: dividing by zero

Because $SSE_{(5)}=0$ exactly, $s_{(5)}=0$ — so the *standardized* DFBETAS formula divides by zero and is technically undefined (infinite).

**Why this is a feature, not a flaw:** DFBETAS normally measures "how big was the shift, *relative to* how noisy the model still is without this point." Here, without Student 5, the model has **zero** noise — it's a perfect plane. Dividing a real, nonzero shift (like $-0.4$) by "zero leftover noise" is asking "how big is this shift as a multiple of nothing" — the answer blows up to infinity because *any* nonzero shift looks infinitely large next to zero noise. The formula is correctly screaming: **this single point explains 100% of the imperfection in the entire model** — about as influential as a single data point can possibly be.

Common software (R, statsmodels) reports this as `NaN` or `Inf`. Treat a `NaN`/`Inf` diagnostic value as a signal worth investigating, never something to silently discard.

### General flagging rule (well-behaved, non-degenerate cases)

$$ |DFBETAS_{j,i}| > \frac{2}{\sqrt n} $$

With $n=5$: threshold $= 2/\sqrt5 \approx 0.894$. (Student 5's case exceeds this trivially, since it's infinite — the degenerate case above is the extreme end of this same rule.)

---

## 8.4 DFFITS and COVRATIO — Two More Influence Measures

### DFFITS — "how much does this point's own prediction change?"

$$ DFFITS_i = t_i\sqrt{\frac{h_{ii}}{1-h_{ii}}} \qquad\text{(}t_i\text{ = externally studentized residual, deleting point }i\text{)} $$

**Plain-language reading:** same two-factor structure as Cook's distance (surprise × leverage), but measuring the shift in the point's own *fitted value* $\hat y_i$ rather than the whole model. It's a more localized cousin of Cook's distance.

**Approximate numbers here.** We only computed *internally* studentized residuals $r_i$ (not the externally studentized $t_i$ that DFFITS technically requires), but there's a useful identity connecting the two:

$$ DFFITS_i \approx \sqrt{p\cdot D_i} $$

Using this as a teaching approximation:

| Student | $D_i$ | $DFFITS_i\approx\sqrt{3 D_i}$ |
|---|---|---|
| 1 | 0.1146 | $\sqrt{0.344}=0.586$ |
| 2 | 0.3750 | $\sqrt{1.125}=1.061$ |
| 3 | 0.2083 | $\sqrt{0.625}=0.790$ |
| 4 | 0.3750 | $\sqrt{1.125}=1.061$ |
| 5 | 1.8330 | $\sqrt{5.499}=\mathbf{2.345}$ |

**Threshold:** $|DFFITS_i| > 2\sqrt{p/n} = 2\sqrt{3/5} = 1.549$.

Only **Student 5** clears this threshold ($2.345 > 1.549$) — consistent with everything Cook's distance already told us. Students 2 and 4 come reasonably close (1.061) but stay under the line.

### COVRATIO — "does this point make my estimates more or less precise?"

$$ COVRATIO_i = \frac{\det\!\big[s_{(i)}^2(\mathbf X_{(i)}^T\mathbf X_{(i)})^{-1}\big]}{\det\!\big[s^2(\mathbf X^T\mathbf X)^{-1}\big]} $$

**Plain-language reading:** this is a subtler question than the others — not "did the coefficient *values* move?" but "did my *confidence* in those coefficients change?" $COVRATIO_i < 1$ means removing point $i$ would tighten (improve) precision — i.e., point $i$ was adding noise to the estimates. $COVRATIO_i > 1$ means point $i$ was actually *helping* precision, often because it fills in a sparse region of predictor space even if it doesn't shift the coefficient estimates much.

**Flagging rule of thumb:** $|COVRATIO_i - 1| > 3p/n$. Here $3p/n = 9/5 = 1.8$, i.e. flag if $COVRATIO_i$ falls outside roughly $(-0.8,\ 2.8)$ — a wide band for such a small dataset, which is itself a reminder that these thresholds are asymptotic guidelines, not hard cutoffs, and are most meaningful in larger samples.

---

## 8.5 One-Table Summary — Every Diagnostic, Side by Side

| Student | $e_i$ | $h_{ii}$ | $r_i$ | $D_i$ (Cook's) | $DFFITS_i$ (approx.) | Flagged? |
|---|---|---|---|---|---|---|
| 1 | 0.2 | 0.733 | 0.354 | 0.115 | 0.586 | No |
| 2 | 0.6 | 0.600 | 0.866 | 0.375 | 1.061 | No |
| 3 | -1.0 | 0.333 | -1.118 | 0.208 | 0.790 | No |
| 4 | -0.6 | 0.600 | -0.866 | 0.375 | 1.061 | No |
| 5 | 0.8 | **0.733** | 1.414 | **1.833** | **2.345** | **Yes — by every measure** |

Every diagnostic in the chapter — Cook's distance, DFFITS, and (with an undefined/infinite value) DFBETAS — converges on the same conclusion: **Student 5 is the influential point**, and it's influential for the textbook reason: high leverage *combined with* a large residual, not either one alone.

---

## 8.6 What to Actually Do With a Flagged Point

Finding a high-Cook's-distance point is the **start** of an investigation, not an automatic deletion order.

1. **Check for a data-entry error first** — often the simplest, most defensible fix.
2. **Consider whether the point reflects a genuinely different regime** — an unusual but legitimate observation suggesting the linear model is misspecified for part of the data (ties back to Chapter 7's linearity diagnostic).
3. **Report results with and without the point** — transparency about sensitivity to one observation is often more valuable than silently picking a version.
4. **Only as a last resort, with clear justification, exclude it** — blind deletion of "inconvenient" points is a well-known way to manufacture false confidence; both Kutner and Montgomery explicitly warn against this.

**Smoke-alarm analogy:** a flagged point is like a smoke alarm going off — it tells you to go check, not to immediately start throwing things out. Look for a boring explanation first (a mistyped number). If that's not it, ask whether the point is revealing something true that a straight line can't capture. If still unsure, show both versions of your results and let the reader see how much it matters. Deletion is the last resort, and only with a clear, stated justification — otherwise you risk quietly cherry-picking your way to a nicer-looking but less honest result.

---

## 8.7 Where the Textbooks Differ

| Source | Distinctive contribution |
|---|---|
| **Kutner** | Most algebraically complete — derives Cook's distance and DFBETAS with full ties back to the hat matrix and studentized residuals. |
| **Montgomery** | Strongest on §8.6's *practitioner workflow* — extensive case studies from engineering/manufacturing on what to actually do once a point is flagged. |
| **Sheather** | Emphasizes reading diagnostics directly from software (`cooks.distance()`, `dfbetas()` in R) and interpreting standard flagged-point plots over hand-derivation. |
| **ESL/ISL** | Essentially skip this topic — influence diagnostics are a classical-inference concern; ESL/ISL's cross-validation-driven philosophy handles problematic points implicitly (one influential point matters less when evaluating out-of-sample accuracy across many resamples). |

---

## 8.8 Formula Cheat-Sheet (Everything in One Place)

| Quantity | Formula | Plain-English question it answers |
|---|---|---|
| Leverage | $h_{ii}$ (diagonal of hat matrix) | "How unusual is this point's $x$?" |
| Studentized residual | $r_i = e_i / \big(s\sqrt{1-h_{ii}}\big)$ | "How surprising is this point's $y$, in standardized units?" |
| Cook's distance | $D_i = \dfrac{r_i^2}{p}\cdot\dfrac{h_{ii}}{1-h_{ii}}$ | "Overall, did this point shake up the whole model?" |
| DFBETAS | $DFBETAS_{j,i} = \dfrac{\hat\beta_j-\hat\beta_{j(i)}}{s_{(i)}\sqrt{[(\mathbf X^T\mathbf X)^{-1}]_{jj}}}$ | "Did this point distort *this one* coefficient specifically?" |
| DFFITS | $DFFITS_i = t_i\sqrt{\dfrac{h_{ii}}{1-h_{ii}}}$ | "Did this point's *own* prediction change a lot?" |
| COVRATIO | ratio of generalized variances, with vs. without point $i$ | "Did this point make my estimates more or less *precise*?" |

**Common flagging thresholds:**

| Diagnostic | Threshold | Value here ($n=5,\ p=3$) |
|---|---|---|
| Cook's $D_i$ | $>4/n$ or $>1$ | $0.8$ or $1.0$ |
| $\lvert DFBETAS_{j,i}\rvert$ | $>2/\sqrt n$ | $0.894$ |
| $\lvert DFFITS_i\rvert$ | $>2\sqrt{p/n}$ | $1.549$ |
| $\lvert COVRATIO_i-1\rvert$ | $>3p/n$ | $1.8$ |

---

## 8.9 Interview Q&A

**Q: What's the difference between leverage and influence?**
A: Leverage ($h_{ii}$) depends only on a point's predictor ($x$) values — how far it sits from the center of predictor space. Influence depends on leverage **combined with** the residual — how much removing the point would actually change the fitted model. High leverage alone doesn't guarantee high influence.
*(Simple version: leverage = how far out on the see-saw you're sitting; influence = whether your actual weight was surprising enough to tip it.)*

**Q: A point has the largest raw residual in the dataset but a low Cook's distance. How is that possible?**
A: Its leverage must be low. Cook's distance requires both a large residual **and** meaningful leverage — a big residual at a low-leverage point has limited power to pull the line toward itself. (Student 3 in this dataset is exactly this case: largest residual, $D_3=0.208$, well below Student 5's $1.833$.)
*(Simple version: it "yelled loudly" but sat too close to the middle of the see-saw to move it.)*

**Q: What does DFBETAS measure that Cook's distance doesn't?**
A: Cook's distance summarizes overall influence on the whole model in one number; DFBETAS isolates influence on **one specific coefficient** — useful when that predictor has real interpretive meaning and you want to know if a single point is the whole reason it looks the way it does.
*(Simple version: Cook's distance = "did this point shake up the model as a whole?" DFBETAS = "did this point specifically distort this one number I care about?")*

**Q: What's the difference between DFFITS and Cook's distance, if they're built from the same two ingredients?**
A: DFFITS measures the shift in the point's *own* fitted value $\hat y_i$; Cook's distance measures a broader, aggregated shift across *all* fitted values (equivalently, all coefficients at once). They're highly correlated in practice — as seen here, both flag Student 5 — but DFFITS is the more localized diagnostic.

**Q: What should you do if you find a point with a very high Cook's distance?**
A: Investigate before acting — check for a data-entry error, consider whether it reflects real but unusual data (possibly signaling model misspecification), and report sensitivity with and without the point rather than silently deleting it.
*(Simple version: treat it like a smoke alarm — go check what's happening, don't just yank the batteries out.)*

**Q: Can a point have zero residual but still be influential?**
A: Cook's distance specifically would be low if the residual is near zero, since it requires both factors. But such a point could still substantially affect *other* things, like the standard errors of the coefficients (reflected in COVRATIO), purely from its extreme predictor value.
*(Simple version: sitting far out on the see-saw with exactly the expected weight won't trip Cook's distance's alarm — but it can still quietly affect how "wobbly" your overall estimates are.)*

**Q: Why does DFBETAS become infinite/undefined in the Student-5 example, and is that a bug?**
A: Not a bug — it's the clearest possible signal. Removing Student 5 leaves a model with exactly zero residual variance ($s_{(5)}=0$), so any nonzero coefficient shift divided by "zero leftover noise" is undefined/infinite. This correctly indicates Student 5 was responsible for 100% of the model's imperfection — software reporting `NaN`/`Inf` here should be treated as a diagnostic finding, not an error to suppress.

---

*End of Chapter 8. Next: Chapter 9 — Multicollinearity (formal VIF and condition-number diagnostics, building directly on the coefficient instability first observed in Chapter 5, §5.4).*
