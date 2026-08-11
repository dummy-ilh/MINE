# Chapter 5 — Inference in Multiple Regression (Revised: Why + How added throughout)

*Synthesized from Kutner, Montgomery, Sheather, and ESL/ISL. Chapter 4's dataset fit perfectly (SSE=0), which is useless for inference. Here we add small, realistic noise to one observation so every test is fully worked and meaningful.*

**Dataset for this chapter** (same $x_1$, $x_2$ as Chapter 4; student 5's score nudged from 80 to 83):

| Student | $x_1$ (hours) | $x_2$ (practice tests) | $y$ (score) |
|---|---|---|---|
| 1 | 1 | 1 | 50 |
| 2 | 2 | 1 | 55 |
| 3 | 3 | 2 | 65 |
| 4 | 4 | 2 | 70 |
| 5 | 5 | 3 | **83** |

Fitting this gives $\hat{\beta}_0=38.2,\ \hat{\beta}_1=4.6,\ \hat{\beta}_2=7$, residuals $0.2,\ 0.6,\ -1,\ -0.6,\ 0.8$ (sum to zero, as always). $SSE=2.4$.

---

## 5.1 The Motivating Question — Three Different Questions, Not One

With multiple predictors, "is this model any good?" splits into genuinely distinct questions:

1. **"Do the predictors, together, explain a significant amount of variation in y?"** → the **overall F-test**
2. **"Does this one specific predictor matter, holding the others fixed?"** → an **individual t-test**
3. **"Does adding this specific group of predictors improve the model over a simpler version?"** → a **partial F-test**

**Why three separate tests instead of one?** Because a model can be jointly informative while individual predictors within it are hard to pin down (§5.4 shows exactly this happening), and because sometimes you care about a *block* of predictors (e.g., a whole categorical variable's worth of dummy columns) rather than one at a time. One number can't answer three different questions — that's why interviewers probe all three separately.

---

## 5.2 The Overall F-Test

**Hypotheses:** $H_0: \beta_1=\beta_2=...=\beta_p=0$ vs. $H_a:$ at least one $\beta_j\neq0$.

**Test statistic** (ANOVA decomposition from Chapter 2, SST = SSR + SSE, now with $p$ predictors):

$$ F = \frac{SSR/p}{SSE/(n-p-1)} = \frac{MSR}{MSE} $$

**Why is it built this way, specifically?** Both numerator and denominator are variances (sums of squares divided by their degrees of freedom), so their ratio is a pure signal-to-noise comparison, unaffected by the units of $y$. $MSR$ answers "on average, how much variance did each predictor buy you?" $MSE$ answers "how much variance is left over that *no* predictor explains — i.e., what does pure noise look like in this dataset?" Dividing gives you a single number that's large exactly when the explained-per-predictor variance dwarfs the noise floor, and hovers near 1 when the predictors are indistinguishable from noise. This is also *why* it's an F-distribution and not, say, a t-distribution: F is specifically the distribution of a ratio of two independent scaled chi-squared quantities, which is exactly what $MSR/MSE$ is under $H_0$.

**How to actually compute it, step by step:**

**Step 1 — compute $SST=\sum(y_i-\bar{y})^2$.** With $\bar{y}=64.6$:

$$ SST = (-14.6)^2+(-9.6)^2+(0.4)^2+(5.4)^2+(18.4)^2 = 213.16+92.16+0.16+29.16+338.56=673.2 $$

**Step 2 — get $SSR$ by subtraction:** $SSR=SST-SSE=673.2-2.4=670.8$.

**Step 3 — divide each by its degrees of freedom.** $p=2$ predictors used $\Rightarrow SSR$ has $p=2$ df. $n-p-1=5-2-1=2$ observations' worth of information remain unused $\Rightarrow SSE$ has 2 df.

$$ F=\frac{670.8/2}{2.4/2}=\frac{335.4}{1.2}\approx279.5 $$

**Step 4 — compare to the critical value.** Look up $F_{critical}(2,2)$ at $\alpha=0.05\approx19.0$ in an F-table (indexed by numerator df, denominator df). Since $279.5\gg19.0$, reject $H_0$ decisively.

*(As in earlier chapters, this toy dataset's near-perfect fit produces dramatic statistics; real data is rarely this clean.)*

---

## 5.3 Individual t-Tests on Each Coefficient

**Why can't you reuse Chapter 2's simple-regression $SE$ formula here?** Because in multiple regression, coefficients are no longer estimated independently — they're all solved simultaneously from the same normal equations, so their sampling variability is entangled. You need the *full* variance-covariance matrix (Chapter 3, §3.8), not a single-number variance:

$$ \text{Var}(\hat{\boldsymbol{\beta}}) = MSE\cdot(\mathbf{X}^T\mathbf{X})^{-1} $$

**How $(\mathbf{X}^T\mathbf{X})^{-1}$ is actually obtained here.** Since $\mathbf{X}^T\mathbf{X}$ depends only on the predictor columns (not on $y$), it's identical to Chapter 4's: $\begin{bmatrix}5&15&9\\15&55&32\\9&32&19\end{bmatrix}$. Inverting a $3\times3$ by hand uses the determinant + cofactor (adjugate) method: the determinant works out to $5(55{\cdot}19-32{\cdot}32)-15(15{\cdot}19-32{\cdot}9)+9(15{\cdot}32-55{\cdot}9)=5(21)-15(-3)+9(-15)=105+45-135=15$, and each cofactor (e.g., the $(1,1)$ entry is $55{\cdot}19-32{\cdot}32=21$) gets divided by that determinant, giving:

$$ (\mathbf{X}^T\mathbf{X})^{-1} = \frac{1}{15}\begin{bmatrix} 21 & 3 & -15 \\ 3 & 14 & -25 \\ -15 & -25 & 50 \end{bmatrix} $$

You won't be asked to invert $3\times3$ matrices by hand in most interviews, but knowing *this is a determinant-and-cofactors computation, not magic* is what lets you reason about what makes it blow up (a near-zero determinant — precisely what happens under severe multicollinearity, previewed in §5.4).

**How to get each $SE$ from there.** $MSE=SSE/(n-p-1)=2.4/2=1.2$. Pull the diagonal entries corresponding to $\beta_1$ and $\beta_2$ — the $(2,2)$ and $(3,3)$ entries of $(\mathbf{X}^T\mathbf{X})^{-1}$, i.e. $14/15$ and $50/15$:

$$ SE(\hat{\beta}_1)=\sqrt{1.2\times\tfrac{14}{15}}=\sqrt{1.12}\approx1.058 \qquad SE(\hat{\beta}_2)=\sqrt{1.2\times\tfrac{50}{15}}=\sqrt{4}=2.0 $$

**t-statistics**, each with $n-p-1=2$ degrees of freedom:

$$ t_{\beta_1}=\frac{4.6}{1.058}\approx4.35 \qquad t_{\beta_2}=\frac{7}{2.0}=3.5 $$

Critical value $t^*_{(0.025,2)}=4.303$ (from a t-table, 2 df, two-tailed $\alpha=0.05$).

- $\hat{\beta}_1$: $t=4.35$ **barely exceeds** 4.303 — significant, but only just.
- $\hat{\beta}_2$: $t=3.5$ **does not exceed** 4.303 — **not significant** at $\alpha=0.05$, individually.

**The crucial, non-obvious result:** the overall F-test screamed significance ($F=279.5$), yet the individual test for $\hat{\beta}_2$ fails to reject $H_0:\beta_2=0$. How can the whole model be so clearly significant while one coefficient looks statistically weak on its own? §5.4 explains why.

---

## 5.4 Why This Happens — A First Real Look at Multicollinearity

**Why does the variance-covariance matrix have off-diagonal entries at all, and why do they matter?** Off-diagonal entry $(i,j)$ of $\text{Var}(\hat{\boldsymbol{\beta}})$ is the *covariance* between $\hat{\beta}_i$ and $\hat{\beta}_j$ — literally, "when this coefficient's estimate is unusually high due to sampling noise, does the other one tend to move too?" A near-zero covariance means the two estimates fluctuate independently across hypothetical resamples; a large-magnitude covariance means the model is trading credit back and forth between the two predictors, which is exactly the multicollinearity signature.

**How to compute the correlation between two coefficient estimates:**

**Step 1 — pull the relevant covariance.** The $(2,3)$ entry of $(\mathbf{X}^T\mathbf{X})^{-1}$ is $-25/15$, so $\text{Cov}(\hat{\beta}_1,\hat{\beta}_2)=MSE\times(-25/15)=1.2\times(-1.667)=-2.0$.

**Step 2 — normalize by both standard errors** (same logic as any correlation = covariance / (SD × SD)):

$$ \text{Corr}(\hat{\beta}_1,\hat{\beta}_2)=\frac{-2.0}{1.058\times2.0}\approx-0.945 $$

**Why $-0.945$ specifically explains §5.3's puzzle:** a correlation this strongly negative means that in repeated samples, whenever $\hat{\beta}_1$ happens to come out high, $\hat{\beta}_2$ tends to come out low, and vice versa — the two estimates can trade credit almost freely between them while barely disturbing the overall fitted line (since $x_1$ and $x_2$ move together in this data, $5x_1+5x_2$-style combinations can be re-split many ways and still nearly match the data). That trade-off inflates each *individual* coefficient's standard error even though the combined predictive power ($SSR$, and hence the overall F) stays rock-solid — because $SSR$ only cares about the fitted values as a whole, not how credit is split between $x_1$ and $x_2$. This is the intuition behind the formal VIF diagnostic built in Chapter 9.

---

## 5.5 The Partial F-Test — Comparing Nested Models

**Why does this need to be a *separate* test from the individual t-test at all?** Because it generalizes to testing a *group* of predictors jointly (§5.8 below), which no single t-test can do — and because it's built directly from a model-comparison logic (how much does SSE drop when you add these specific predictors) rather than a single coefficient's sampling distribution, which turns out to matter once you leave the one-predictor-at-a-time case.

**The question here:** does adding $x_2$ (practice tests) meaningfully improve the model over having $x_1$ (hours studied) alone?

**How to fit the reduced model — shown in full, not skipped this time.** Simple regression of $y$ on $x_1$ alone, using this chapter's $y$-values ($50,55,65,70,83$) and Chapter 1's formulas:

$$ \hat{\beta}_1=\frac{n\sum x_1y-\sum x_1\sum y}{n\sum x_1^2-(\sum x_1)^2} = \frac{5(1050)-15(323)}{5(55)-15^2}=\frac{5250-4845}{275-225}=\frac{405}{50}=8.1 $$

$$ \hat{\beta}_0=\bar{y}-\hat{\beta}_1\bar{x}_1 = \frac{323-8.1(15)}{5}=\frac{201.5}{5}=40.3 $$

Fitted values from $\hat{y}=40.3+8.1x_1$, and residuals against the actual scores:

| Student | $x_1$ | $\hat y=40.3+8.1x_1$ | Actual $y$ | Residual | Residual$^2$ |
|---|---|---|---|---|---|
| 1 | 1 | 48.4 | 50 | 1.6 | 2.56 |
| 2 | 2 | 56.5 | 55 | -1.5 | 2.25 |
| 3 | 3 | 64.6 | 65 | 0.4 | 0.16 |
| 4 | 4 | 72.7 | 70 | -2.7 | 7.29 |
| 5 | 5 | 80.8 | 83 | 2.2 | 4.84 |

$$ SSE_{reduced}=2.56+2.25+0.16+7.29+4.84=17.1 $$

**How to apply the partial F-test formula:**

$$ F=\frac{(SSE_{reduced}-SSE_{full})/(df_{reduced}-df_{full})}{SSE_{full}/df_{full}} $$

$df_{reduced}=n-2=3$ (2 parameters), $df_{full}=n-3=2$ (3 parameters), so 1 parameter was added:

$$ F=\frac{(17.1-2.4)/1}{2.4/2}=\frac{14.7}{1.2}=12.25 $$

Comparing to $F_{critical}(1,2)$ at $\alpha=0.05\approx18.51$: since $12.25<18.51$, adding $x_2$ does **not** significantly improve the model at the 5% level — consistent with §5.3's t-test on $\hat{\beta}_2$.

**Why this matches the t-test exactly:** for a partial F-test adding exactly **one** predictor, $F=t^2$ for that predictor's individual t-test. Check: $3.5^2=12.25$ — exact match. This isn't a coincidence — it's a mathematical identity that only holds when testing one predictor at a time; for two-or-more-predictor groups, the two tests genuinely diverge (§5.8).

**Why must the two models be nested for any of this to work?** "Nested" means the reduced model's predictors are a strict subset of the full model's. This guarantees $SSE_{reduced}\ge SSE_{full}$ always (more predictors can only reduce or match SSE, never increase it — Chapter 4, §4.8), which is what makes "$SSE_{reduced}-SSE_{full}$" interpretable as *error removed by the added predictors specifically*, rather than some uninterpretable mix of two unrelated models' errors. Comparing two non-nested models (different, non-overlapping predictor sets) needs different tools entirely — AIC, BIC, or cross-validation (Chapter 14).

---

## 5.6 When to Use Which Test — A Decision Guide

| Question | Test | Use when |
|---|---|---|
| "Is the model as a whole useful?" | Overall F-test | First check before interpreting any individual coefficient |
| "Does this one predictor matter, controlling for the others already in the model?" | Individual t-test | Deciding whether to keep/interpret a specific existing predictor |
| "Does adding this group of predictors improve the model significantly?" | Partial F-test | Comparing two nested models — e.g., justifying a whole new feature category, not just one variable |

**Why check the overall F-test first, as a hard rule?** If it's not significant, the model as a whole may be indistinguishable from noise — interpreting individual "significant-looking" coefficients at that point risks *data dredging*: with enough coefficients, some will look significant by chance alone, and the overall F-test is the gatekeeper that protects against over-trusting those.

---

## 5.7 Where the Textbooks Differ

- **Kutner** builds this chapter as a direct generalization of the ANOVA table, formally deriving the general linear test approach (which the partial F-test is a special case of).
- **Montgomery** spends the most time on the practical interpretation of "significant overall model, insignificant individual predictor" scenarios like §5.4.
- **Sheather** emphasizes reading these three tests directly off software output — the F-statistic at the bottom of a `lm()`/`statsmodels` summary, the individual t-values in the coefficient table, `anova()` model comparisons for partial F-tests.
- **ESL/ISL** de-emphasize classical hypothesis testing here entirely, preferring cross-validation-based model comparison — a preview of Chapter 14 (Model Selection) and Chapter 15 (Overfitting).

---

## 5.8 Extending to Groups of Predictors — Why a Single t-Test Isn't Enough

**Why you can't just run individual t-tests on a block of related predictors:** suppose "school region" is a categorical variable with 4 categories, encoded as 3 dummy columns (one category dropped as baseline). If those 3 dummies are correlated with each other (regions often correlate with other things in the dataset), each dummy's *individual* t-test can look non-significant — exactly the §5.4 phenomenon, now spread across 3 coefficients instead of 2 — even while the region variable *as a whole* is jointly significant. Conversely, one dummy could look spuriously significant on its own and stop mattering once you account for the group. Testing them one at a time misses the real question, which is joint.

**How to test a group of $q$ predictors at once — the general partial F-test:**

$$ F=\frac{(SSE_{reduced}-SSE_{full})/q}{SSE_{full}/(n-k-q-1)} $$

where the reduced model drops all $q$ predictors in the group simultaneously and the full model includes all of them. This is exactly §5.5's formula with $q$ generalized beyond 1.

$$ H_0:\ \beta_{region1}=\beta_{region2}=\beta_{region3}=0 $$

One F-test with $q=3$ answers "does region matter at all," which is the actually-useful question — not three separate, individually underpowered t-tests.

---

## 5.9 How This Powers Stepwise Selection — Why It's the Natural Engine

**Why the F-test (or its $q=1$ shortcut, the t-test) is the natural tool for stepwise procedures:** at every step of adding or removing one feature, you're comparing exactly two nested models — with vs. without that one feature — which is precisely the setup §5.5 was built for. It gives a principled, consistent stopping rule instead of an arbitrary "does $R^2$ look better" judgment call (recall Chapter 4, §4.8: $R^2$ mechanically never decreases, so it can't be trusted alone).

**How forward selection actually runs, step by step:**

1. Start with the **null model** (intercept only, or your current feature set).
2. For every candidate feature *not yet in the model*, temporarily add it and compute its partial F-test (equivalently, its t-test p-value) against the current model.
3. Pick the candidate with the **most significant** result (largest F / smallest p-value).
4. If that best candidate's p-value clears your threshold (commonly p < 0.05, the "F-to-enter"), **permanently add it**.
5. If no remaining candidate clears the threshold, **stop**.
6. Repeat from step 2 with the updated model.

**How backward elimination runs (the mirror image):**

1. Start with **all candidate features** in the model.
2. Find the feature with the **least significant** coefficient (smallest F / largest p-value).
3. If its p-value exceeds your threshold ("F-to-remove"), **remove it**.
4. Refit and repeat until every remaining feature is significant.

**Why re-test against the *current* model at every step, rather than testing all candidates against the null model once?** Multicollinearity again: a feature's significance depends on what's already in the model (§5.4 showed this directly — $\hat\beta_2$'s significance changed once $x_1$ was accounted for). Testing everything against the null simultaneously would miss features that only become significant after others are already included, or falsely flag features that stop mattering once a correlated feature enters.

**Why this method has a well-known weakness:** running many sequential significance tests inflates the overall false-positive rate (a multiple-testing problem) — stepwise-selected models often look more significant than they really are on held-out data. This is why cross-validation or regularization (Lasso) are often preferred for feature selection in practice, previewed here and covered properly in Chapter 14.

---

## 5.10 Interview Q&A

**Q: Can the overall F-test be significant while every individual t-test is not?**
A: Yes — a classic multicollinearity signature. The predictors jointly explain substantial variation, but strong correlation among them makes it statistically difficult to attribute that explained variation to any one predictor individually. §5.4 works this out with actual numbers: a $-0.945$ correlation between $\hat\beta_1$ and $\hat\beta_2$.

**Q: What's the difference between an individual t-test and a partial F-test for a single added predictor?**
A: For exactly one added predictor, they're mathematically identical: $F=t^2$. They diverge only when testing multiple predictors jointly — the partial F-test can test a whole group at once (§5.8), while a t-test only ever isolates one coefficient.

**Q: Why should you check the overall F-test before looking at individual coefficients?**
A: If the whole model isn't distinguishable from noise, interpreting individual "significant-looking" coefficients risks data dredging — the overall test is the gatekeeper.

**Q: If two predictors are highly correlated, does that bias their coefficient estimates?**
A: No — the estimates remain unbiased. What increases is their *variance*, making individual estimates unstable and individual significance tests underpowered, even while overall model fit stays strong. (§5.4 shows why: the off-diagonal covariance entry lets the model trade credit between correlated predictors almost freely.)

**Q: How would you test whether adding three new predictors as a group significantly improves a model?**
A: A partial F-test comparing the reduced model (without the three) to the full model (with them): $F=\frac{(SSE_{reduced}-SSE_{full})/3}{SSE_{full}/df_{full}}$. Individual t-tests on each of the three wouldn't answer the joint question, since they don't account for correlation among the added predictors.

**Q: Why can't you just compare $R^2$ between two models to decide whether to add a feature?**
A: $R^2$ is mathematically guaranteed to never decrease when you add any predictor, even pure noise, because OLS will always exploit whatever tiny correlation exists by chance. The F-test corrects for this by weighing the SSE improvement against the degrees of freedom spent buying it.

**Q: Why must the two models being compared with a partial F-test be nested?**
A: Nesting guarantees $SSE_{reduced}\ge SSE_{full}$, which is what makes "SSE removed" interpretable as the effect of the added predictors specifically, rather than an uninterpretable difference between two unrelated models. Non-nested comparisons need AIC, BIC, or cross-validation instead.

---

## 5.11 One-Paragraph Summary

Multiple regression inference splits "is this model good?" into three genuinely different, sometimes-disagreeing questions: the overall F-test (are the predictors jointly useful?), individual t-tests (does this one predictor matter, holding others fixed?), and the partial F-test (does adding this group of predictors help?). The overall F-test can be wildly significant while an individual t-test isn't — a direct symptom of multicollinearity, traceable to the off-diagonal covariance entries of $\text{Var}(\hat{\boldsymbol\beta})=MSE(\mathbf{X}^T\mathbf{X})^{-1}$. For a single added predictor, the partial F-test collapses exactly to the t-test ($F=t^2$), which is why stepwise selection procedures can operate one feature at a time using ordinary coefficient p-values — but the general F-test (with $q>1$) is what you need whenever adding or removing a *group* of predictors together, such as a categorical variable's full set of dummy columns.

---

*End of Chapter 5 (revised). Next: Chapter 6.*
