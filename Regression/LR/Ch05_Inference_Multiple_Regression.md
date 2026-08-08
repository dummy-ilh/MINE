# Chapter 5 — Inference in Multiple Regression

*Synthesized from Kutner, Montgomery, Sheather, and ESL/ISL. Chapter 4's dataset fit perfectly (SSE=0), which is useless for inference — you can't compute a standard error from zero residual variance. Here we add small, realistic noise to one observation so every test below is fully worked and meaningful.*

**Dataset for this chapter** (same $x_1$, $x_2$ as Chapter 4; only student 5's score is nudged from 80 to 83 — a slightly lucky day):

| Student | $x_1$ (hours) | $x_2$ (practice tests) | $y$ (score) |
|---|---|---|---|
| 1 | 1 | 1 | 50 |
| 2 | 2 | 1 | 55 |
| 3 | 3 | 2 | 65 |
| 4 | 4 | 2 | 70 |
| 5 | 5 | 3 | **83** |

Fitting this (same method as Chapter 4) gives $\hat{\beta}_0=38.2,\ \hat{\beta}_1=4.6,\ \hat{\beta}_2=7$, with residuals $0.2,\ 0.6,\ -1,\ -0.6,\ 0.8$ (they sum to zero, as always). $SSE = 2.4$.

---

## 5.1 The Motivating Question — Three Different Questions, Not One

With multiple predictors, "is this model any good?" splits into genuinely distinct questions, and this is the single biggest conceptual shift from Chapters 1–2:

1. **"Do the predictors, taken together, explain a significant amount of variation in y?"** → the **overall F-test**
2. **"Does this one specific predictor matter, holding the others fixed?"** → an **individual t-test**
3. **"Does adding this specific group of predictors improve the model over a simpler version?"** → a **partial F-test**

These can — and often do — disagree with each other, which is exactly why interviewers ask about all three separately rather than treating "is my model significant" as one question.

---

## 5.2 The Overall F-Test

**Hypotheses:** $H_0: \beta_1=\beta_2=...=\beta_p=0$ (none of the predictors matter at all) vs. $H_a:$ at least one $\beta_j \neq 0$.

**Test statistic**, reusing the ANOVA decomposition from Chapter 2 (SST = SSR + SSE, same identity, now with $p$ predictors instead of 1):

$$ F = \frac{SSR/p}{SSE/(n-p-1)} = \frac{MSR}{MSE} $$

**Plain-English reading:** the numerator is "average variance explained, per predictor used"; the denominator is "average leftover noise, per remaining degree of freedom." A big F means the predictors are explaining far more than you'd expect from noise alone.

**Worked numbers:** first, $SST = \sum(y_i-\bar{y})^2$ with $\bar{y}=64.6$:

$$ SST = (-14.6)^2+(-9.6)^2+(0.4)^2+(5.4)^2+(18.4)^2 = 213.16+92.16+0.16+29.16+338.56 = 673.2 $$

$$ SSR = SST-SSE = 673.2-2.4 = 670.8 $$

With $p=2$ predictors and $n-p-1 = 5-2-1=2$ residual degrees of freedom:

$$ F = \frac{670.8/2}{2.4/2} = \frac{335.4}{1.2} \approx 279.5 $$

Comparing to $F_{critical}(2,2)$ at $\alpha=0.05 \approx 19.0$: since $279.5 \gg 19.0$, we reject $H_0$ decisively — the two predictors, together, explain far more than chance. (As in earlier chapters, this toy dataset's near-perfect fit produces dramatic test statistics; real data is rarely this clean.)

---

## 5.3 Individual t-Tests on Each Coefficient

Each coefficient gets its own test, exactly as in Chapter 2, but now the standard error must come from the full variance-covariance matrix (Chapter 3, §3.8) since coefficients are no longer independent of each other.

$$ \text{Var}(\hat{\boldsymbol{\beta}}) = MSE \cdot (\mathbf{X}^T\mathbf{X})^{-1} $$

Since $\mathbf{X}^T\mathbf{X}$ only depends on the predictors (not $y$), it's identical to Chapter 4's, and its inverse works out to:

$$ (\mathbf{X}^T\mathbf{X})^{-1} = \frac{1}{15}\begin{bmatrix} 21 & 3 & -15 \\ 3 & 14 & -25 \\ -15 & -25 & 50 \end{bmatrix} $$

With $MSE = SSE/(n-p-1) = 2.4/2 = 1.2$:

$$ SE(\hat{\beta}_1) = \sqrt{1.2 \times \tfrac{14}{15}} = \sqrt{1.12} \approx 1.058 $$

$$ SE(\hat{\beta}_2) = \sqrt{1.2 \times \tfrac{50}{15}} = \sqrt{4} = 2.0 $$

**t-statistics** (each with $n-p-1=2$ degrees of freedom):

$$ t_{\beta_1} = \frac{4.6}{1.058} \approx 4.35 \qquad t_{\beta_2} = \frac{7}{2.0} = 3.5 $$

Critical value $t^*_{(0.025,\ 2)} = 4.303$.

- $\hat{\beta}_1$: $t=4.35$ **barely exceeds** 4.303 — significant, but only just.
- $\hat{\beta}_2$: $t=3.5$ **does not exceed** 4.303 — **not significant** at $\alpha=0.05$, individually.

**This is the crucial, non-obvious result of the chapter:** the overall F-test screamed significance ($F=279.5$), yet the individual test for $\hat{\beta}_2$ fails to reject $H_0:\beta_2=0$. How can the whole model be so clearly significant while one coefficient looks statistically weak on its own?

---

## 5.4 Why This Happens — A First Real Look at Multicollinearity

Look at the off-diagonal entry linking $\hat{\beta}_1$ and $\hat{\beta}_2$ in the variance-covariance matrix: $\text{Cov}(\hat{\beta}_1,\hat{\beta}_2) = MSE\times(-25/15) = 1.2\times(-1.667)=-2.0$. The implied **correlation between the two coefficient estimates** is:

$$ \text{Corr}(\hat{\beta}_1,\hat{\beta}_2) = \frac{-2.0}{1.058\times2.0} \approx -0.945 $$

A correlation this strongly negative between coefficient estimates is a direct symptom of **multicollinearity**: $x_1$ (hours studied) and $x_2$ (practice tests) move together in this data, so the model has trouble deciding how much credit to assign to each individually — it can trade credit back and forth between them almost freely while barely changing the overall fit. The *combined* effect of the two predictors is estimated precisely (hence the huge overall F), but each one *individually* is estimated with much more uncertainty. We'll build the formal diagnostic for this (VIF) in Chapter 9 — this is the intuition that motivates it.

---

## 5.5 The Partial F-Test — Comparing Nested Models

**The question:** does adding $x_2$ (practice tests) meaningfully improve the model over having $x_1$ (hours studied) alone? This is a different question from "is $\hat{\beta}_2$'s t-test significant" — though, as you'll see, for a *single* added predictor they turn out to be mathematically the same test.

**Step 1 — fit the reduced model** (simple regression of $y$ on $x_1$ only, same data): this gives $\hat{\beta}_0=40.3,\ \hat{\beta}_1=8.1$ (recomputed fresh on this chapter's $y$-values, using the same method as Chapter 1), with $SSE_{reduced}=17.1$.

**Step 2 — apply the partial F-test formula:**

$$ F = \frac{(SSE_{reduced}-SSE_{full})/(df_{reduced}-df_{full})}{SSE_{full}/df_{full}} $$

Here $df_{reduced}=n-2=3$ (2 parameters), $df_{full}=n-3=2$ (3 parameters), so 1 parameter was added:

$$ F = \frac{(17.1-2.4)/1}{2.4/2} = \frac{14.7}{1.2} = 12.25 $$

Comparing to $F_{critical}(1,2)$ at $\alpha=0.05 \approx 18.51$: since $12.25 < 18.51$, adding $x_2$ does **not** significantly improve the model at the 5% level — consistent with §5.3's individual t-test result for $\hat{\beta}_2$.

**The consistency check every textbook emphasizes:** for a partial F-test adding exactly **one** predictor, $F = t^2$ for that predictor's individual t-test. Check: $3.5^2 = 12.25$ — **exact match.** This is not a coincidence; it's a mathematical identity, and it only holds when exactly one predictor is being tested at a time. For groups of two or more added predictors, the partial F-test and individual t-tests genuinely diverge and answer different questions.

---

## 5.6 When to Use Which Test — A Decision Guide

| Question | Test | Use when |
|---|---|---|
| "Is the model as a whole useful?" | Overall F-test | First check before interpreting any individual coefficient |
| "Does this one predictor matter, controlling for the others already in the model?" | Individual t-test | Deciding whether to keep/interpret a specific existing predictor |
| "Does adding this group of predictors improve the model significantly?" | Partial F-test | Comparing two nested models — e.g., justifying a whole new feature category, not just one variable |

**Interview-critical practical rule:** always check the overall F-test *first*. If it's not significant, individual t-tests become nearly meaningless to interpret (you're picking through effects in a model that, as a whole, may not be distinguishable from noise).

---

## 5.7 Where the Textbooks Differ

- **Kutner** builds this chapter as a direct generalization of the ANOVA table, formally deriving the general linear test approach (which the partial F-test is a special case of) — the most proof-complete treatment of the three.
- **Montgomery** spends the most time on the practical interpretation of "significant overall model, insignificant individual predictor" scenarios like §5.4 — this is a hallmark Montgomery-style worked example.
- **Sheather** emphasizes reading these three tests directly off a single piece of `lm()`/`statsmodels` software output — the F-statistic at the bottom of a summary table, the individual t-values in the coefficient table, and `anova()` model comparisons for partial F-tests.
- **ESL/ISL** de-emphasize classical hypothesis testing in this chapter entirely, preferring cross-validation-based model comparison — a preview of the philosophical shift you'll see fully in Chapter 14 (Model Selection) and Chapter 15 (Overfitting).

---

## 5.8 Interview Q&A

**Q: Can the overall F-test be significant while every individual t-test is not?**
A: Yes — this is a classic multicollinearity signature. The predictors jointly explain substantial variation, but strong correlation among them makes it statistically difficult to attribute that explained variation to any one predictor individually.

**Q: What's the difference between an individual t-test and a partial F-test for a single added predictor?**
A: For exactly one added predictor, they're mathematically identical: $F=t^2$. They diverge only when testing *multiple* predictors jointly — the partial F-test can test a whole group at once, while an individual t-test only ever isolates one coefficient.

**Q: Why should you check the overall F-test before looking at individual coefficients?**
A: If the whole model isn't distinguishable from noise, interpreting individual "significant-looking" coefficients risks capitalizing on chance (data dredging) — the overall test is the gatekeeper.

**Q: If two predictors are highly correlated, does that bias their coefficient estimates?**
A: No — the estimates remain unbiased. What increases is their *variance* (Chapter 3, §3.8), making individual estimates unstable and individual significance tests underpowered, even while overall model fit remains strong.

**Q: How would you test whether adding three new predictors as a group significantly improves a model?**
A: A partial F-test comparing the reduced model (without the three) to the full model (with them), using $F=\frac{(SSE_{reduced}-SSE_{full})/3}{SSE_{full}/df_{full}}$ — individual t-tests on each of the three wouldn't answer the joint question, since they don't account for correlation among the three added predictors.

---

*End of Chapter 5. Next: Chapter 6 — The Gauss-Markov Theorem (why OLS is BLUE — Best Linear Unbiased Estimator — and the full proof of why no other linear unbiased estimator can have lower variance).*
