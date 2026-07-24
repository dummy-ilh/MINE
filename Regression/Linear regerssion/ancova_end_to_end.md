# Analysis of Covariance (ANCOVA) — End to End
### A self-contained reference (in the style of Kutner, Nachtsheim, Neter, Li — *Applied Linear Statistical Models*)

This file is written to stand alone. Anywhere it leans on an idea from regression or ANOVA, the idea is re-derived or re-explained on the spot rather than just cited.

---

## 1. What Problem ANCOVA Solves

**Plain English.** You want to compare several treatment groups (e.g., three webpage designs) on some outcome Y (e.g., revenue per visitor). Ordinary one-way ANOVA (comparing raw group means) works perfectly well **if the groups are otherwise identical except for treatment**. But in practice, groups often differ — by chance or by design — on some other variable that also affects Y. If you ignore that variable, part of what looks like a "treatment effect" is actually just a **confound**: the groups were never on equal footing to begin with.

**The fix:** measure that other variable (call it X, the **covariate**) and **statistically adjust** each group's mean for the fact that its members happened to have higher or lower X on average. ANCOVA is exactly this: **ANOVA (comparing group means) + regression (adjusting for a continuous covariate), combined into a single model.**

**Why this matters even in a randomized experiment.** Randomization guarantees groups are *equal in expectation* on every covariate, known or unknown (this is Chapter-15-style logic: random assignment breaks the link between treatment and any other variable). But **in any specific finite sample, random chance can still leave groups imbalanced** on a covariate that happens to matter a lot for Y. ANCOVA corrects for this leftover, chance imbalance, and — even when groups are perfectly balanced — it also **reduces unexplained error variance**, which tightens confidence intervals and increases the power to detect a real treatment effect. This dual benefit (bias reduction under imbalance, variance reduction under balance) is why ANCOVA-style adjustment (under the modern name **CUPED**, "Controlled-experiment Using Pre-Experiment Data") is now a standard variance-reduction technique in industry A/B testing platforms.

---

## 2. The Model

### Building Blocks You Need (self-contained recap)

**Indicator (dummy) variables.** To include a categorical factor with $g$ groups in a regression, you use $g-1$ indicator variables, each equal to 1 for one group and 0 otherwise, with one group left out as the **reference category** (its effect gets absorbed into the intercept). For 3 groups A, B, C with A as reference: $D_B=1$ if the observation is in group B (else 0), $D_C=1$ if in group C (else 0).

**Least squares with multiple predictors.** Given predictors $X_1,\ldots,X_p$, the fitted model $\hat Y=b_0+b_1X_1+\cdots+b_pX_p$ is found by minimizing $\sum(Y_i-\hat Y_i)^2$; each $b_k$ represents that predictor's effect **holding the other predictors constant** — a "partial" effect, not the predictor's raw, unadjusted association with Y.

### The ANCOVA Model Itself

$$
Y_{ij} = \beta_0 + \tau_1 D_{1,ij}+\tau_2D_{2,ij}+\cdots+\tau_{g-1}D_{g-1,ij} + \gamma X_{ij} + \varepsilon_{ij}
$$

**What each piece means:**
- $D_{1},\ldots,D_{g-1}$: group indicators, exactly as above.
- $\tau_k$: the **adjusted treatment effect** of group $k+1$ relative to the reference group — the difference in intercepts, i.e., the group difference in Y **after** accounting for the covariate's effect.
- $X_{ij}$: the covariate (e.g., prior engagement level of visitor $j$ in group $i$).
- $\gamma$: the **common slope** — how much Y changes per unit of X, assumed **the same across all groups**. This shared-slope assumption is the single most important assumption in this whole framework, examined closely in Section 4.

**Why this is exactly Chapter 8's "common slope, different intercept" indicator-variable regression, just under a new name.** Each group's implied regression line is $\hat Y=(\beta_0+\tau_k)+\gamma X$ — same slope $\gamma$ for every group, different starting height ($\beta_0+\tau_k$) per group. Geometrically: **parallel lines**, one per group, offset vertically by each group's treatment effect.

---

## 3. Estimating the Model by Hand

### The Pooled (Common) Slope

**Why you can't just fit one simple regression ignoring groups, and why you can't just fit each group separately either.** Ignoring groups entirely would let genuine treatment differences masquerade as scatter around one wrong, blended line. Fitting each group with its own separate slope (ignoring the shared-slope assumption) throws away information — if the slope really is the same across groups, pooling that information across all groups gives a more precise, lower-variance estimate of it than estimating it separately, group by group, from smaller samples each.

**The fix: pool the *within-group* covariance and variance information across groups.** For each group $i$, compute the usual building blocks (deviations from that group's own means):
$$
S_{XX,i}=\sum_j(X_{ij}-\bar X_i)^2, \qquad S_{XY,i}=\sum_j(X_{ij}-\bar X_i)(Y_{ij}-\bar Y_i)
$$
Then the **pooled common slope** is:
$$
\hat\gamma = \frac{\sum_i S_{XY,i}}{\sum_i S_{XX,i}}
$$
**Why summing across groups before dividing (rather than averaging each group's own slope) is correct:** this weights each group's contribution by how much genuine X-variability it contains ($S_{XX,i}$), giving groups with more spread-out covariate values more say in pinning down the shared slope — exactly the same "spread your X out to reduce slope variance" logic that governs simple regression precision.

### Adjusted Intercepts and the Treatment Effects

Once $\hat\gamma$ is known, each group's own intercept (at $X=0$) is:
$$
b_{0,i} = \bar Y_i - \hat\gamma\bar X_i
$$
And the **adjusted treatment effect** of group $i$ relative to the reference group is simply the difference in these intercepts.

### Adjusted Means — the Standard ANCOVA Output

**Plain English.** Rather than reporting intercepts (the predicted Y at the arguably meaningless value $X=0$), ANCOVA conventionally reports each group's **adjusted mean**: what that group's average Y would have been if every group had had the *same* average covariate value (the overall grand mean $\bar X_{\text{grand}}$), given the fitted common slope.

$$
\text{Adjusted mean}_i = \bar Y_i - \hat\gamma(\bar X_i-\bar X_{\text{grand}})
$$

**Why this formula makes sense.** If group $i$'s covariate mean $\bar X_i$ was *above* the grand mean, some of its raw Y advantage is "unfairly" attributable to the covariate, not the treatment — so we subtract $\hat\gamma$ times that excess. If group $i$'s covariate mean was *below* the grand mean, the correction adds back the estimated shortfall the covariate cost it. **The adjusted means answer: "what would the group means have looked like if all groups had been equally positioned on the covariate?"** — exactly the comparison you actually want when judging the treatment's own effect.

---

## 4. The Crucial Assumption: Homogeneity of Regression Slopes

**Plain English.** The whole "single common slope $\gamma$" setup only makes sense if the covariate genuinely affects Y **the same way in every group**. If the covariate's effect is actually stronger in one group than another, forcing a single shared slope onto the data is a **misspecified model** — exactly Chapter 8's "does the slope differ across groups" problem, applied here to the covariate instead of a general predictor.

### How to Test It

Fit a **richer** model that allows each group its own slope (adding interaction terms between each indicator and X):
$$
Y_{ij} = \beta_0+\tau_1D_1+\cdots+\tau_{g-1}D_{g-1}+\gamma X_{ij}+\delta_1(D_1X_{ij})+\cdots+\delta_{g-1}(D_{g-1}X_{ij})+\varepsilon_{ij}
$$
If the $\delta$'s (the interaction terms) are jointly **not** significantly different from zero (tested via an F-test comparing this richer model's error sum of squares to the simpler common-slope model's — the same "extra sum of squares" logic used throughout regression: fit both models, compare how much the extra terms reduce the error sum of squares, relative to how many extra parameters they cost), you have support for the homogeneous-slopes assumption, and the simpler common-slope ANCOVA model above is appropriate.

**If homogeneity fails:** there is no single "the treatment effect" — the treatment's advantage genuinely depends on the covariate's value. The honest move is to report simple effects at specific, meaningful covariate values, or use the **Johnson-Neyman technique**, which finds the exact range of covariate values within which the groups differ significantly and the range within which they don't — rather than forcing a single, misleading "adjusted mean difference" onto a relationship that isn't actually parallel.

---

## 5. Testing the Treatment Effect (Adjusted for the Covariate)

**The logic:** fit two models and compare.
- **Full model:** intercept + group indicators + covariate (the ANCOVA model above).
- **Reduced model:** intercept + covariate **only** (no group indicators at all) — this says "groups don't matter, once you've accounted for the covariate."

$$
F^* = \frac{[SSE(\text{reduced})-SSE(\text{full})]/(g-1)}{SSE(\text{full})/df_{\text{full}}}
$$

**Why this specific comparison isolates the treatment effect correctly.** The reduced model already gets full credit for whatever the covariate alone explains. Whatever *additional* reduction in error the group indicators provide, on top of that, is the treatment effect's own, covariate-independent contribution — exactly the "extra sum of squares" idea used throughout regression model-building: never credit a variable with explaining something another variable already accounts for.

---

## 6. Full Worked Numerical Example

### The Data

Three webpage designs (A, B, C), $n=5$ visitors each. Covariate $X$ = each visitor's prior page-view count (a proxy for engagement, measured *before* seeing the design). Response $Y$ = revenue.

| Group | X | Y |
|---|---|---|
| A | 2 | 8 |
| A | 3 | 12 |
| A | 4 | 11 |
| A | 5 | 15 |
| A | 6 | 19 |
| B | 4 | 15 |
| B | 5 | 19 |
| B | 6 | 18 |
| B | 7 | 22 |
| B | 8 | 26 |
| C | 6 | 22 |
| C | 7 | 26 |
| C | 8 | 25 |
| C | 9 | 29 |
| C | 10 | 33 |

**Notice something important before doing any analysis: the groups are *not* balanced on the covariate.** Group A's visitors averaged fewer prior page views than Group C's ($\bar X_A=4$ vs. $\bar X_C=8$) — exactly the kind of chance (or self-selection) imbalance ANCOVA is built to handle.

### Step 1 — Raw (Unadjusted) Group Means

$$
\bar Y_A = 13, \qquad \bar Y_B=20, \qquad \bar Y_C=27
$$
Raw differences: $\bar Y_B-\bar Y_A=7$, $\bar Y_C-\bar Y_A=14$. **At face value, this looks like a big, escalating design effect.** Keep this number in mind — we're about to see how much of it survives adjustment.

### Step 2 — Within-Group Covariate/Response Building Blocks

**Group A** ($\bar X_A=4,\bar Y_A=13$): $S_{XX,A}=\sum(X-4)^2=4+1+0+1+4=10$.
$$
S_{XY,A}=(-2)(-5)+(-1)(-1)+(0)(-2)+(1)(2)+(2)(6)=10+1+0+2+12=25
$$

**Group B** ($\bar X_B=6,\bar Y_B=20$): $S_{XX,B}=10$ (identical structure), $S_{XY,B}=25$ (identical arithmetic pattern).

**Group C** ($\bar X_C=8,\bar Y_C=27$): $S_{XX,C}=10$, $S_{XY,C}=25$.

**All three groups show exactly the same within-group relationship between X and Y** ($S_{XY}/S_{XX}=25/10=2.5$ in every group) — a clean, idealized case where the homogeneity-of-slopes assumption holds *exactly*, letting us walk through the whole procedure without ambiguity.

### Step 3 — The Pooled Common Slope

$$
\hat\gamma = \frac{S_{XY,A}+S_{XY,B}+S_{XY,C}}{S_{XX,A}+S_{XX,B}+S_{XX,C}} = \frac{25+25+25}{10+10+10}=\frac{75}{30}=2.5
$$

### Step 4 — Group Intercepts and Adjusted Treatment Effects

$$
b_{0,A}=13-2.5(4)=3, \qquad b_{0,B}=20-2.5(6)=5, \qquad b_{0,C}=27-2.5(8)=7
$$
**Adjusted treatment effects (relative to A):** $\tau_B=5-3=2$, $\tau_C=7-3=4$.

**Compare to the raw differences:** raw was $+7$ (B vs A) and $+14$ (C vs A); **adjusted is only $+2$ and $+4$.** Most of the raw gap has evaporated — it was never really about the design; it was mostly about Group C's visitors already being more engaged before they ever saw the page.

### Step 5 — Adjusted Means

Grand covariate mean: $\bar X_{\text{grand}}=6$ (average of 4, 6, 8).
$$
\text{Adj. mean}_A = 13-2.5(4-6)=13+5=18
$$
$$
\text{Adj. mean}_B = 20-2.5(6-6)=20
$$
$$
\text{Adj. mean}_C = 27-2.5(8-6)=27-5=22
$$

**Adjusted means: 18, 20, 22 — evenly spaced by exactly 2**, matching the $\tau$'s above perfectly ($20-18=2=\tau_B$; $22-18=4=\tau_C$) as it must, since these are just two equivalent ways of expressing the same adjusted comparison.

### Step 6 — Residuals and SSE for the Full ANCOVA Model

Using each group's fitted line ($\hat Y = b_{0,i}+2.5X$):

**Group A** ($\hat Y=3+2.5X$): fitted values $8, 10.5, 13, 15.5, 18$; residuals $0, 1.5, -2, -0.5, 1$.
**Group B** ($\hat Y=5+2.5X$): fitted values $15, 17.5, 20, 22.5, 25$; residuals $0, 1.5, -2, -0.5, 1$.
**Group C** ($\hat Y=7+2.5X$): fitted values $22, 24.5, 27, 29.5, 32$; residuals $0, 1.5, -2, -0.5, 1$.

(Identical residual pattern in every group — a direct consequence of the identical within-group $S_{XX}, S_{XY}$ found above.)

$$
SSE_{\text{group}} = 0^2+1.5^2+(-2)^2+(-0.5)^2+1^2 = 0+2.25+4+0.25+1=7.5
$$
$$
SSE_{\text{full}} = 3\times7.5=22.5, \qquad df_{\text{full}}=n-p=15-4=11, \qquad MSE_{\text{full}}=22.5/11=2.045
$$

### Step 7 — The Reduced Model (Covariate Only, No Groups)

Pool all 15 points and fit a single regression of Y on X, ignoring group membership entirely.

$$
\bar X_{\text{grand}}=6, \qquad \bar Y_{\text{grand}}=20
$$

**Total $S_{XX}$** decomposes into within-group ($10+10+10=30$) plus between-group variability:
$$
S_{XX,\text{between}} = 5\left[(4-6)^2+(6-6)^2+(8-6)^2\right]=5(8)=40 \quad\Rightarrow\quad S_{XX,\text{total}}=30+40=70
$$

**Total $S_{XY}$** similarly decomposes into within-group ($25+25+25=75$) plus between-group:
$$
S_{XY,\text{between}} = 5\left[(-2)(-7)+(0)(0)+(2)(7)\right]=5(28)=140 \quad\Rightarrow\quad S_{XY,\text{total}}=75+140=215
$$
(Here the between-group deviations used are each group's $(\bar X_i-\bar X_{\text{grand}})$ paired with $(\bar Y_i-\bar Y_{\text{grand}})$: for A, $(-2,-7)$; for B, $(0,0)$; for C, $(2,7)$.)

$$
b_{1,\text{reduced}} = \frac{215}{70}=3.071, \qquad SSR_{\text{reduced}} = \frac{S_{XY,\text{total}}^2}{S_{XX,\text{total}}}=\frac{215^2}{70}=660.36
$$

**Total sum of squares of Y** (needed to get $SSE_{\text{reduced}}$), via the same within/between decomposition applied to Y alone: within-group $SS_Y$ is $70$ per group (verify for Group A: deviations from mean 13 are $-5,-1,-2,2,6$, squares $25,1,4,4,36$, sum $70$ — identical for B and C by the same constructed symmetry), so within-group total $=210$; between-group $SS_Y=5[(13-20)^2+(20-20)^2+(27-20)^2]=5(98)=490$.
$$
SSTO = 210+490=700
$$
$$
SSE_{\text{reduced}} = SSTO - SSR_{\text{reduced}} = 700-660.36=39.64, \qquad df_{\text{reduced}}=n-2=13
$$

### Step 8 — The F-Test for the Treatment Effect, Adjusted for the Covariate

$$
F^* = \frac{[SSE_{\text{reduced}}-SSE_{\text{full}}]/(df_{\text{reduced}}-df_{\text{full}})}{MSE_{\text{full}}} = \frac{(39.64-22.5)/(13-11)}{2.045}=\frac{17.14/2}{2.045}=\frac{8.57}{2.045}=4.19
$$

Critical value: $F_{(0.95;2,11)}=3.98$. Since $4.19>3.98$, **the treatment effect remains statistically significant after adjusting for the covariate — but notice how much closer this is to the threshold** than the raw picture suggested.

### Step 9 — The Payoff: Comparing to a Naive One-Way ANOVA That Ignores the Covariate

If you'd ignored the covariate entirely and just run a one-way ANOVA on the raw group means:
$$
SSTR = 490 \ (\text{between-group } SS_Y \text{ from above}), \qquad df=2, \qquad MSTR=245
$$
$$
SSE_{\text{one-way}} = 210 \ (\text{within-group } SS_Y), \qquad df=12, \qquad MSE_{\text{one-way}}=17.5
$$
$$
F^*_{\text{one-way}} = \frac{245}{17.5}=14.0
$$

**This is dramatically larger than ANCOVA's $F^*=4.19$.** Compare $F_{(0.95;2,12)}=3.89$: the naive one-way test would have you walking away thinking the design effect is overwhelming ($14.0\gg3.89$), when properly adjusting for the pre-existing engagement covariate shows a real but much more modest, only-barely-significant effect.

**Why this happened — the important, slightly subtle lesson.** In the naive one-way ANOVA, *all* of the between-group variation (490) gets credited to "treatment," including the part that was really just because Group C started out with more-engaged visitors. In ANCOVA, that confounded portion gets correctly reassigned to the covariate instead — leaving a much smaller genuine treatment signal (the extra-sum-of-squares numerator shrank from 490 all the way down to 17.14). **At the same time**, ANCOVA's error variance is also much smaller (MSE dropped from 17.5 to 2.045), since the covariate explains away a lot of the previously "unexplained" within-group-looking noise. **Both the signal and the noise shrank — and in this particular dataset, the noise shrank by less than the signal did, so the adjusted F-statistic ended up smaller than the naive one**, even though ANCOVA's test is the *honest* one. This is the crucial, non-obvious point: ANCOVA's variance reduction doesn't automatically mean a *bigger* F-statistic — it means a *correct* one. When the covariate is confounded with treatment (as it is here), part of what looked like signal was actually noise-in-disguise, and ANCOVA correctly removes it from the numerator, not just the denominator.

### Step 10 — Checking Homogeneity of Slopes (Confirming the Assumption Holds)

Recall Section 4: this requires comparing the common-slope model's SSE to a richer model that lets each group have its own slope. Because our data was built so that each group's own slope is *exactly* $S_{XY,i}/S_{XX,i}=25/10=2.5$ — identical across all three groups — fitting the richer, separate-slopes model would produce **zero** improvement in fit over the common-slope model (the extra interaction terms would all come out to exactly zero, and the F-test comparing the two models would give $F=0$). **This confirms, as cleanly as a real dataset ever will, that the homogeneity-of-slopes assumption holds** — licensing the entire common-slope ANCOVA analysis above.

---

## 7. Diagnostics

Nothing new beyond what applies to any regression model: check residual plots for non-linearity, non-constant variance, and non-normality (the same tools built for simple and multiple regression), and — specific to ANCOVA — always check homogeneity of slopes *first*, since if it fails, the entire "single adjusted treatment effect" framing above is not a valid description of the data, and you need the separate-slopes model (Section 4) instead.

---

## 8. The Modern Connection: CUPED in A/B Testing

**CUPED** ("Controlled-experiment Using Pre-Experiment Data"), widely used at Microsoft, Google, and other tech companies for variance reduction in online experiments, is — mechanically — exactly this chapter. A pre-experiment metric (e.g., a user's historical engagement, revenue, or usage level, measured *before* the experiment starts) is used as the covariate $X$; the treatment/control assignment is the group factor; and the "CUPED-adjusted" treatment effect is precisely this chapter's adjusted-mean-difference. **Why this is such a valuable technique in practice:** because randomization already guarantees $X$ is balanced across arms *in expectation*, the primary practical benefit in this setting is the **variance reduction** side of ANCOVA (a smaller MSE, exactly as computed in Step 6 above) rather than the bias-correction side — a genuinely tighter confidence interval and higher power to detect a real treatment effect, at zero cost to the validity of the causal comparison, using data you likely already have sitting around from before the experiment even began.

**Interview question:** *"How does CUPED relate to classical statistical methods you might have learned in a regression or experimental design course?"*
**Ideal answer:** CUPED is a direct, modern application of ANCOVA — using a pre-experiment covariate correlated with the outcome to adjust each arm's observed mean and reduce residual variance, exactly the mechanism developed for combining regression covariates with group comparisons. In a properly randomized experiment, the covariate is balanced across arms in expectation, so CUPED's main benefit is the variance-reduction (power) side of ANCOVA rather than correcting for confounding — though the same technique also corrects for any leftover chance imbalance in the covariate across arms, exactly as ANCOVA would in a non-randomized comparison.

---

## Python Implementation

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

df = pd.DataFrame({
    'Group': ['A']*5 + ['B']*5 + ['C']*5,
    'X': [2,3,4,5,6, 4,5,6,7,8, 6,7,8,9,10],
    'Y': [8,12,11,15,19, 15,19,18,22,26, 22,26,25,29,33]
})

# --- Step 1: check homogeneity of slopes first ---
interaction_model = smf.ols('Y ~ C(Group) * X', data=df).fit()
print(anova_lm(interaction_model, typ=2))
# If the Group:X interaction term is not significant, proceed with the common-slope ANCOVA model below

# --- Step 2: fit the ANCOVA model (common slope) ---
ancova_model = smf.ols('Y ~ C(Group) + X', data=df).fit()
print(ancova_model.summary())

# --- Step 3: F-test for the treatment effect, adjusted for covariate ---
reduced_model = smf.ols('Y ~ X', data=df).fit()
f_test = anova_lm(reduced_model, ancova_model)
print(f_test)

# --- Step 4: compute adjusted means directly ---
grand_X = df['X'].mean()
adjusted_means = {}
for g in df['Group'].unique():
    sub = df[df['Group']==g]
    y_bar, x_bar = sub['Y'].mean(), sub['X'].mean()
    gamma_hat = ancova_model.params['X']
    adjusted_means[g] = y_bar - gamma_hat*(x_bar - grand_X)
print("Adjusted means:", adjusted_means)

# --- Step 5: compare to a naive one-way ANOVA (no covariate) ---
oneway_model = smf.ols('Y ~ C(Group)', data=df).fit()
print(anova_lm(oneway_model, typ=2))
```

---

## Interview Question Bank

**Conceptual:**
1. What problem does ANCOVA solve that a plain one-way ANOVA cannot?
2. Why does ANCOVA still provide value even in a properly randomized experiment where the covariate is balanced in expectation?
3. What does the "homogeneity of regression slopes" assumption mean, and what would it look like for it to fail?

**Derivation:**
4. Derive the pooled common-slope formula from the within-group $S_{XX}$ and $S_{XY}$ values, and explain why you sum before dividing rather than averaging each group's own slope.
5. Derive the adjusted-mean formula and explain, in words, what it's correcting for.
6. Show how the F-test for the treatment effect, adjusted for the covariate, follows the general "compare a full and reduced model's error sums of squares" logic.

**ML/Statistics:**
7. Explain CUPED as a direct application of ANCOVA, including exactly which classical benefit (bias correction vs. variance reduction) it typically provides in a randomized A/B test.
8. In our worked example, the naive one-way ANOVA's F-statistic (14.0) was much larger than the ANCOVA-adjusted F-statistic (4.19). Explain why, precisely — what happened to both the numerator and denominator of the F-ratio?
9. If the homogeneity-of-slopes assumption fails, what alternative approach should you use instead of reporting a single adjusted mean difference?

**Coding:**
10. Implement the pooled common-slope ANCOVA estimation from scratch in NumPy/pandas, including adjusted means, and verify against `statsmodels`.
11. Implement the homogeneity-of-slopes test (interaction F-test) from scratch.

**Traps:**
12. "Since randomization guarantees balance, there's no point running ANCOVA in an A/B test — the raw group means are already unbiased." — what benefit of ANCOVA does this overlook?
13. "The adjusted means differ by less than the raw means, so ANCOVA must have found a smaller, less real effect." — what's the more precise way to state what changed?
14. Someone fits an ANCOVA model without first checking the interaction term for homogeneity of slopes. What could go wrong with their reported "adjusted treatment effect" if that assumption actually fails?

---

*This file is self-contained: it re-derives the indicator-variable regression setup, the pooled-slope estimation, the extra-sum-of-squares F-test logic, and the homogeneity-of-slopes check needed to understand and apply ANCOVA without reference to any other file, while noting (for context only) that it is the direct synthesis of the regression machinery (multiple regression, indicator variables, extra sums of squares) and the ANOVA/experimental-design machinery covered elsewhere in this course.*
