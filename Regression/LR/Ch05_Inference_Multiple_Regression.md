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

# The F-Test for Adding Features to Regression

## 1. The question this test answers

You have a regression model. You're wondering: *"If I add this new feature (or these new features), does the model actually get meaningfully better — or does it only look better by chance, because more predictors always inflate R² a little?"*

The **F-test** (specifically the **partial F-test**, also called the **nested F-test**) gives you a formal, statistical answer to that question — instead of eyeballing "R² went up a bit, I guess that's good."

This is the core tool behind **forward selection**, **backward elimination**, and general **model comparison** in regression.

---

## 2. Why you can't just compare R² directly

Here's the trap: **R² can never decrease when you add a feature** — even a completely useless, random-noise feature — because OLS will always find *some* tiny amount of correlation with the residuals by chance, and use it to reduce SSE slightly.

So "R² went up" is *not* evidence the new feature is useful. You need a test that asks: **"Is the improvement in R² bigger than what we'd expect from random noise alone, given how many degrees of freedom I spent to get it?"**

That's exactly what the F-test does — it penalizes you for the number of new parameters you added, the same spirit as adjusted R², but as a proper hypothesis test with a p-value.

---

## 3. Setting up the two competing models

You compare a **reduced model** (without the new feature(s)) against a **full model** (with them). The two models must be **nested** — meaning the reduced model's predictors are a strict subset of the full model's predictors.

**Reduced model** (k predictors):
```
y = β₀ + β₁x₁ + ... + βₖxₖ + ε
```

**Full model** (k + q predictors — you added q new features):
```
y = β₀ + β₁x₁ + ... + βₖxₖ + βₖ₊₁xₖ₊₁ + ... + βₖ₊qxₖ₊q + ε
```

### The hypotheses

```
H₀:  βₖ₊₁ = βₖ₊₂ = ... = βₖ₊q = 0    (the new features add nothing — reduced model is "true")
H₁:  at least one of βₖ₊₁, ..., βₖ₊q ≠ 0    (at least one new feature genuinely helps)
```

Notice: this is a **joint test**. Even if you're adding just *one* feature (q=1), it's still phrased as "is this one coefficient zero" — which, as you'll see in §6, connects directly back to the t-test you already know.

---

## 4. The F-statistic — built from SSE, piece by piece

Recall from your regression fundamentals: **SSE (sum of squared errors)** measures how much error is left after fitting a model. Adding predictors can only decrease (or leave unchanged) SSE.

```
SSE_reduced = SSE of the model WITHOUT the new features
SSE_full    = SSE of the model WITH the new features
```

Since the full model has more flexibility, `SSE_full ≤ SSE_reduced` always. The question is: **how much smaller, relative to how many parameters you spent buying that improvement?**

```
        (SSE_reduced - SSE_full) / q
F  =  ────────────────────────────────
              SSE_full / (n - k - q - 1)
```

Let's name every piece:

| Symbol | Meaning |
|---|---|
| `SSE_reduced - SSE_full` | Extra error *removed* by adding the new features (bigger = better) |
| `q` | Number of new parameters added (the "cost" you paid) |
| `n` | Number of observations |
| `k` | Number of predictors in the reduced model |
| `n - k - q - 1` | Residual degrees of freedom of the *full* model (subtract 1 for the intercept) |

**Read it as a ratio of two things:**
- **Numerator**: average error-reduction *per new parameter* — "how much did each new predictor buy you, on average"
- **Denominator**: average leftover error *per remaining degree of freedom* in the full model — this is basically the full model's residual variance, `σ̂²`

So the F-statistic is really asking: **"Is the improvement per new parameter large relative to the model's typical unexplained noise?"** If the new features are pure noise, you'd expect the numerator to be roughly the same *size* as random noise itself, giving F ≈ 1. If the features are genuinely useful, the numerator will be much bigger than the noise floor, giving F >> 1.

This F-statistic follows an **F-distribution** with `(q, n-k-q-1)` degrees of freedom under H₀, which is how you get your p-value.

---

## 5. Worked numerical example

Let's extend your existing 5-student regression setup conceptually with a slightly bigger illustrative dataset (n = 10 students) so the F-test has enough degrees of freedom to be meaningful.

Suppose you're predicting **exam score** from **hours studied** (reduced model), and you're considering adding **hours slept** as a second feature (full model).

| Model | Predictors | SSE | # params (incl. intercept) |
|---|---|---|---|
| Reduced | hours studied | 180 | 2 (β₀, β₁) |
| Full | hours studied + hours slept | 120 | 3 (β₀, β₁, β₂) |

- `n = 10`
- `k = 1` (reduced model has 1 predictor)
- `q = 1` (we added 1 new feature: hours slept)
- Residual df of full model: `n - k - q - 1 = 10 - 1 - 1 - 1 = 7`

**Step 1 — numerator:**
```
(SSE_reduced - SSE_full) / q = (180 - 120) / 1 = 60
```

**Step 2 — denominator:**
```
SSE_full / (n - k - q - 1) = 120 / 7 = 17.14
```

**Step 3 — F-statistic:**
```
F = 60 / 17.14 = 3.50
```

**Step 4 — compare to critical value.** Look up `F(1, 7)` at α = 0.05 → critical value ≈ **5.59**.

Since `3.50 < 5.59`, we **fail to reject H₀** — at the 5% significance level, we don't have enough evidence that "hours slept" meaningfully improves the model, despite SSE dropping from 180 to 120. That drop could plausibly be due to chance given only 7 residual degrees of freedom.

*(If instead SSE_full had dropped to, say, 60, you'd get F = (120/1)/(60/7) = 120/8.57 = 14.0, which would clear the 5.59 threshold — a case where you'd reject H₀ and keep the feature.)*

---

## 6. Special case: adding exactly ONE feature (q = 1)

This is the most common case in forward selection, and it connects to something you already know: **the t-test on a single coefficient.**

When you add just one feature, there's a clean identity:

```
F = t²
```

where `t` is the usual t-statistic for testing `H₀: βₖ₊₁ = 0` (the coefficient of the new feature alone) from the full model's regression output.

**This means:** for a single added feature, the F-test and the t-test on that feature's coefficient give you *exactly the same conclusion* — they're mathematically the same test viewed two ways. You don't need to run both. This is worth stating explicitly in an interview: it shows you understand *why* forward selection can just check "is the new coefficient's p-value < threshold?" instead of manually computing an F-statistic every time.

---

## 7. How this powers stepwise selection procedures

### Forward selection algorithm

1. Start with the **null model** (intercept only, or your current feature set).
2. For every candidate feature *not yet in the model*, temporarily add it and compute its partial F-test (equivalently, its t-test p-value) against the current model.
3. Pick the candidate with the **most significant** result (largest F / smallest p-value).
4. If that best candidate's p-value is below your threshold (commonly p < 0.05, sometimes called `F-to-enter`), **permanently add it** to the model.
5. If no remaining candidate clears the threshold, **stop**.
6. Repeat from step 2 with the updated model.

### Backward elimination (the mirror image)

1. Start with **all candidate features** in the model.
2. Find the feature with the **least significant** coefficient (smallest F / largest p-value).
3. If that p-value exceeds your threshold (`F-to-remove`), **remove it**.
4. Refit and repeat until every remaining feature is significant.

### Why the F-test is the natural engine for both

At every step, you're comparing two *nested* models (with vs. without one feature) — exactly the setup the partial F-test was built for. The F-test gives you a principled, consistent stopping rule instead of an arbitrary "does R² look better" judgment call.

---

## 8. Testing a GROUP of features at once (q > 1)

Sometimes you don't want to add features one at a time — e.g., you have a categorical variable that expands into 4 dummy variables, and it only makes sense to add or remove them together as a block.

Here the general F-test (§4, with `q > 1`) is essential — you *can't* just look at individual t-tests for each dummy, because:
- Individual coefficients might each look non-significant on their own (especially if the dummies are correlated with each other),
- but the **block as a whole** could still be jointly significant.

**Example:** testing whether "school region" (4 categories → 3 dummy variables after dropping a baseline) belongs in a model predicting exam scores.

```
H₀: β_region1 = β_region2 = β_region3 = 0
```

You'd run one F-test with `q = 3`, not three separate t-tests. This is a common interview trap — testing correlated dummy variables one-by-one with t-tests can miss a jointly significant effect (or find spuriously significant individual dummies that don't hold up jointly).

---

## 9. Why this helps in practice — the core intuition, restated

| Without F-test | With F-test |
|---|---|
| "R² went from 0.71 to 0.73, seems better, let's keep it" | "That 0.02 R² gain isn't statistically distinguishable from noise, given how many parameters it cost — drop it" |
| Risk: overfitting by greedily adding any feature that nudges R² up | Protection: only add features whose improvement clears a noise-adjusted bar |
| No principled stopping rule for stepwise selection | F-to-enter / F-to-remove thresholds give an explicit, defensible stopping criterion |
| Can't test a *group* of correlated features (e.g., dummy variables) coherently | Naturally extends to joint tests via `q > 1` |

---

## 10. Relationship to other tools you already know

| Concept | Relationship to the partial F-test |
|---|---|
| **Overall F-test** (from Ch2: "is the regression significant at all?") | Special case where the reduced model is intercept-only (k=0) and the full model is your entire feature set — same formula, just q = all your predictors |
| **t-test on a coefficient** | Special case of partial F-test with q=1; `F = t²` exactly |
| **Adjusted R²** | Also penalizes for added parameters, but gives you a descriptive number, not a hypothesis test with a p-value — F-test is the formal significance-testing counterpart |
| **AIC / BIC** | Alternative model-comparison criteria that also penalize complexity, but don't require nested models the way the F-test does — useful when comparing non-nested models |

---

## 11. Interview Q&A

**Q: Why can't you just compare R² between two models to decide whether to add a feature?**
A: R² is mathematically guaranteed to never decrease when you add any predictor, even pure noise, because OLS will always exploit whatever tiny correlation exists by chance. The F-test corrects for this by weighing the SSE improvement against the degrees of freedom spent.

**Q: What's the relationship between the F-test and the t-test when adding a single feature?**
A: They're identical tests: `F = t²`, and the p-values match exactly. The F-test framework generalizes to adding multiple features at once, which a single t-test cannot do.

**Q: Why must the two models be "nested" for this F-test?**
A: The test relies on the reduced model's predictor set being a strict subset of the full model's, so that `SSE_reduced - SSE_full` is guaranteed to be ≥ 0 and interpretable as "error removed by the added predictors." Comparing non-nested models (different, non-overlapping predictor sets) needs different tools (AIC, BIC, cross-validation).

**Q: In forward selection, why not just add every feature with p < 0.05 in one shot instead of one at a time?**
A: Multicollinearity — a feature's significance can depend on what's already in the model. Adding features one at a time (always re-testing remaining candidates against the *current* model) accounts for this; testing all candidates against the null model simultaneously would miss features that only become significant after others are already included (or falsely flag features that stop mattering once a correlated feature is added).

**Q: What's a key weakness of stepwise selection via repeated F-tests?**
A: It doesn't correct for multiple testing — running many sequential significance tests inflates the overall false-positive rate, so stepwise-selected models often look more significant than they really are on held-out data. This is a well-known criticism; cross-validation or regularization (Lasso) are often preferred for feature selection in practice.

---

## 12. One-paragraph summary

The partial F-test compares two nested regression models — with and without a candidate feature (or block of features) — by measuring how much SSE drops per new parameter added, relative to the leftover noise in the fuller model. A large F-statistic means the improvement is bigger than you'd expect from chance; a small one means you likely just fit noise. For a single added feature, this collapses exactly to the familiar t-test (`F = t²`), which is why forward selection and backward elimination can operate one-feature-at-a-time using ordinary coefficient p-values — but the general F-test (with q > 1) is what you need whenever you're adding or removing a *group* of features together, such as a categorical variable's dummy encoding.
