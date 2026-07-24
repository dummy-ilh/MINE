# Chapter 16 — Single-Factor Studies
### (Kutner, Nachtsheim, Neter, Li — *Applied Linear Statistical Models*)

Chapter 15 showed that one-way ANOVA is mathematically identical to Chapter 8's indicator-variable regression. This chapter formalizes the ANOVA model in its own right, covers diagnostics specific to this setting, introduces the nonparametric alternative (Kruskal-Wallis), and draws the fixed-vs-random-effects distinction that directly foreshadows modern hierarchical/mixed-effects models.

We continue the Chapter 15 dataset: three webpage designs (A, B, C), $n_j=5$ each, response = revenue per visitor.

---

## 16.1–16.2 Two Equivalent Parameterizations of the ANOVA Model

### The Cell Means Model

$$
Y_{ij} = \mu_i + \varepsilon_{ij}
$$
**Plain English.** Each group $i$ has its own true mean $\mu_i$; $Y_{ij}$ (the $j$-th observation in group $i$) deviates from that group's mean by random error $\varepsilon_{ij}$. This is the most direct, least-abstracted way to write the model — literally "each group has a mean, plus noise."

### The Factor Effects Model

$$
Y_{ij} = \mu+\tau_i+\varepsilon_{ij}, \qquad \text{subject to } \sum_i\tau_i=0
$$
**What's different, and why introduce a second parameterization at all.** Here $\mu$ is the **overall (grand) mean**, and $\tau_i$ is the **effect of being in group $i$** — how much group $i$'s mean deviates from the grand mean. The constraint $\sum\tau_i=0$ ensures $\mu$ and the $\tau_i$'s are uniquely identifiable (otherwise you could shift $\mu$ up and shift every $\tau_i$ down by the same amount and get an identical fit — an identifiability issue directly analogous to Chapter 8's dummy-variable-trap discussion). **This parameterization is exactly Chapter 8's reference-category regression, just with a different, symmetric identifying constraint** instead of "one category's indicator is dropped, absorbed into the intercept." Both parameterizations produce the exact same fitted values and F-test; they differ only in which linear combination of the raw group means is called "the intercept."

**Worked correspondence, using Chapter 15's data** ($\bar Y_A=13,\bar Y_B=17,\bar Y_C=21,\bar Y_{grand}=17$):

Cell means model: $\hat\mu_A=13,\ \hat\mu_B=17,\ \hat\mu_C=21$ (each group's own sample mean — nothing more).

Factor effects model: $\hat\mu=17$ (grand mean), $\hat\tau_A=13-17=-4,\ \hat\tau_B=17-17=0,\ \hat\tau_C=21-17=4$. Check the constraint: $-4+0+4=0$ ✓.

**Interview question:** *"What's the practical difference between the cell means and factor effects parameterizations of ANOVA?"*
**Ideal answer:** They're mathematically equivalent reparameterizations of the identical model, producing identical fitted values, residuals, and F-tests — differing only in which particular linear combination of the group means is labeled the "intercept" and how the remaining information is encoded. The factor effects model is often preferred when you want to talk about deviations from an overall average (e.g., "Design C adds $4 above the grand mean"), while the cell means model is more direct when you just want each group's own estimated mean without reference to any baseline.

---

## 16.4 Evaluating the Appropriateness of the ANOVA Model: Diagnostics

Exactly Chapter 1's four assumptions, now checked group-by-group rather than point-by-point.

### Checking Constant Variance Across Groups: Brown-Forsythe, Generalized to $k>2$ Groups

**Recall Chapter 3's Brown-Forsythe test** compared two groups' absolute deviations from their medians via a t-test. **With more than two groups, the natural generalization is to run a one-way ANOVA F-test *on those absolute deviations themselves*** — directly reusing this chapter's own machinery on a transformed version of the data.

**Worked example.** Each group's median: $\tilde Y_A=13,\ \tilde Y_B=17,\ \tilde Y_C=21$ (same as the means here, since each group's 5 values are symmetric around its center).

**Absolute deviations from each group's own median:**

| Group A | Group B | Group C |
|---|---|---|
| \|12-13\|=1 | \|16-17\|=1 | \|20-21\|=1 |
| \|14-13\|=1 | \|18-17\|=1 | \|22-21\|=1 |
| \|11-13\|=2 | \|15-17\|=2 | \|19-21\|=2 |
| \|13-13\|=0 | \|17-17\|=0 | \|21-21\|=0 |
| \|15-13\|=2 | \|19-17\|=2 | \|23-21\|=2 |

**Notice: each group produces the identical set of deviations $\{1,1,2,0,2\}$** (a direct consequence of how this illustrative dataset was constructed — every group has the same internal spread, just shifted). Each group's mean deviation is $\frac{1+1+2+0+2}{5}=1.2$ — **identical across all three groups**, so the between-group sum of squares on these deviations is **exactly zero**, giving $F=0$ on this test.

**Conclusion: this is about as clean a "pass" on the constant-variance assumption as you'll ever see** — real data essentially never looks this tidy, but it cleanly demonstrates the mechanism (compare this to Chapter 3's original 2-group example, which showed a case where a real, if non-significant, difference in spread appeared between groups).

### Checking Normality and Independence

Same tools as Chapter 3: a Q-Q plot of the pooled residuals (each observation's deviation from its own group mean) should fall roughly along a straight line; a plot of residuals against the order of data collection checks for hidden sequential dependence. Nothing conceptually new here beyond applying Chapter 3's exact machinery to across-group residuals rather than a single regression line's residuals.

---

## 16.5 Remedial Measures: the Kruskal-Wallis Nonparametric Test

### When Parametric ANOVA Assumptions Are in Serious Doubt

**Plain English.** If normality is badly violated or outliers are severe (and a transformation doesn't fix it), the **Kruskal-Wallis test** provides a distribution-free alternative: instead of comparing group **means**, it compares group **rank sums** — testing whether the groups' distributions are shifted relative to each other, without assuming any particular shape.

### The Procedure

1. Pool **all** $N$ observations across all groups and **rank them 1 to $N$** (averaging ranks for ties).
2. Compute each group's **rank sum** $R_i$.
3. Test statistic:
$$
H = \frac{12}{N(N+1)}\sum_i\frac{R_i^2}{n_i} - 3(N+1)
$$
Under $H_0$ (all groups drawn from the same distribution), $H$ is approximately $\chi^2$ with $k-1$ degrees of freedom.

### Worked Example (same data — note groups A and B share the tied value 15; groups B and C share the tied value 19)

**All 15 values, ranked** (ties averaged):

| Value | 11 | 12 | 13 | 14 | 15(A) | 15(B) | 16 | 17 | 18 | 19(B) | 19(C) | 20 | 21 | 22 | 23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Group | A | A | A | A | A | B | B | B | B | B | C | C | C | C | C |
| Rank | 1 | 2 | 3 | 4 | 5.5 | 5.5 | 7 | 8 | 9 | 10.5 | 10.5 | 12 | 13 | 14 | 15 |

**Rank sums:**
$$
R_A = 1+2+3+4+5.5=15.5, \qquad R_B=5.5+7+8+9+10.5=40, \qquad R_C=10.5+12+13+14+15=64.5
$$
Check: $15.5+40+64.5=120=\frac{N(N+1)}{2}=\frac{15(16)}{2}$ ✓

**Computing $H$:**
$$
\frac{R_A^2}{5}=\frac{240.25}{5}=48.05, \quad \frac{R_B^2}{5}=\frac{1600}{5}=320, \quad \frac{R_C^2}{5}=\frac{4160.25}{5}=832.05
$$
$$
H = \frac{12}{15(16)}(48.05+320+832.05)-3(16) = \frac{12}{240}(1200.1)-48=60.005-48=12.005
$$

**(A minor tie-correction factor** $C=1-\frac{\sum(t^3-t)}{N^3-N}$ **is conventionally applied by dividing $H$ by $C$** when ties are present, slightly increasing $H$; we omit the small correction here for clarity, but a careful analysis would include it.)

Compared to $\chi^2_{(0.95,2)}=5.991$: since $12.005>5.991$, **reject $H_0$ — the groups' distributions differ significantly**, matching the parametric F-test's conclusion (F=32) from Chapter 15 exactly in direction, giving reassuring cross-validation between the parametric and nonparametric approaches on this dataset.

**Interview question:** *"When would you reach for Kruskal-Wallis instead of a standard one-way ANOVA F-test?"*
**Ideal answer:** When the normality assumption is seriously doubtful (strong skew, heavy outliers) or the sample sizes are too small to lean on the Central Limit Theorem, and a variance-stabilizing transformation doesn't adequately fix the issue. Kruskal-Wallis sacrifices some statistical power relative to a correctly-specified parametric ANOVA (since it only uses rank information, discarding the magnitude of differences), but it's far more robust to violations of the parametric assumptions, since it makes no distributional assumption about the errors beyond exchangeability under the null.

---

## 16.6 Fixed vs. Random Factor Levels — Foreshadowing Mixed-Effects Models

### The Distinction

**Fixed effects:** the specific factor levels in your study **are the only levels you care about**, and conclusions apply only to exactly those levels. "Design A vs. B vs. C" in an A/B test is a fixed effect — you specifically care about comparing *these three particular designs*, not some broader population of possible designs.

**Random effects:** the factor levels in your study are treated as a **random sample from a larger population of possible levels**, and you want conclusions that **generalize to that whole population**, not just the specific levels observed.

**Worked contrast, directly relevant to your recommendation-systems background.** Suppose instead of testing 3 specific webpage designs, you're measuring average order value across 3 *randomly selected* retail stores, planning to generalize to "stores in general" (the entire chain), not just those 3 specific locations. Here, "store" should be modeled as a **random effect** — you don't care about *these particular* 3 stores' idiosyncratic means specifically; you care about the *variance* store-to-store variability contributes overall, and you want your conclusions about any other factor (e.g., a promotion's effect) to generalize across stores broadly, not be tied to these three specific ones.

**Why this distinction matters mechanically, not just philosophically.** In more complex multi-factor designs (developed in Kutner's later chapters), whether a factor is fixed or random changes **which mean square serves as the correct denominator** for an F-test on other factors in the model — using the wrong error term (treating a random factor as fixed, or vice versa) can produce a badly incorrect F-test, even though the point estimates themselves might look identical.

**Direct connection to modern ML — this is worth stating explicitly, since it's exactly the theoretical root of a widely-used family of models in your recommendation-systems work.** The random-effects idea generalizes directly into **mixed-effects / hierarchical models** — e.g., a recommendation system might treat individual users or items as random effects (each has their own latent "intercept" drawn from a shared population distribution, rather than being individually, separately estimated as fixed, unrelated parameters) — this is precisely the statistical justification behind techniques like user/item bias terms with regularization (shrinkage) in collaborative filtering, and directly the same idea underlying `lme4`/`statsmodels MixedLM`-style hierarchical models. The "shrinkage toward the grand mean" you get from a random-effects model is mathematically related to the ridge-regression shrinkage from Chapter 11 — both trade a controlled amount of bias for reduced variance, just applied to *group-level* means rather than regression coefficients.

**Interview question:** *"If you're building a model that includes a 'store' or 'user' effect, how would you decide whether to treat it as fixed or random?"*
**Ideal answer:** Ask whether you care about the specific entities in your data (fixed — e.g., comparing 3 named, deliberately chosen products) or whether they represent a sample from a broader population you want to generalize over (random — e.g., a sample of users or stores standing in for all users or stores). Practically, if there are many levels (thousands of users) and you mainly care about the overall variance they contribute plus generalizable conclusions about other factors, a random-effects/hierarchical treatment is usually appropriate — it also has the practical benefit of automatically shrinking noisy, low-data entities' estimated effects toward the population average, which a fixed-effects treatment (giving each entity an entirely separate, unregularized parameter) does not do.

---

## Python Implementation

```python
import numpy as np
from scipy import stats

A = np.array([12,14,11,13,15], dtype=float)
B = np.array([16,18,15,17,19], dtype=float)
C = np.array([20,22,19,21,23], dtype=float)

# --- Brown-Forsythe test generalized to k groups ---
def brown_forsythe(*groups):
    devs = [np.abs(g - np.median(g)) for g in groups]
    F, p = stats.f_oneway(*devs)
    return F, p

F_bf, p_bf = brown_forsythe(A, B, C)
print(f"Brown-Forsythe F={F_bf:.4f}, p={p_bf:.4f}")

# --- Kruskal-Wallis test ---
H, p_kw = stats.kruskal(A, B, C)
print(f"Kruskal-Wallis H={H:.4f}, p={p_kw:.6f}")

# --- Standard one-way ANOVA, for comparison ---
F_anova, p_anova = stats.f_oneway(A, B, C)
print(f"One-way ANOVA F={F_anova:.4f}, p={p_anova:.6f}")
```

```python
# --- Random effects / mixed model example (conceptual illustration) ---
import statsmodels.formula.api as smf
import pandas as pd

# Simulated: revenue observations nested within randomly-selected stores
df = pd.DataFrame({
    'store': ['S1']*3+['S2']*3+['S3']*3,
    'revenue': [100,105,98, 120,118,125, 90,95,88]
})
# 'store' as a random effect (random intercept model)
mixed_model = smf.mixedlm('revenue ~ 1', df, groups=df['store']).fit()
print(mixed_model.summary())
# Compare to treating store as fixed (ordinary one-way ANOVA / regression with dummies)
fixed_model = smf.ols('revenue ~ C(store)', data=df).fit()
print(fixed_model.summary())
```

---

## Interview Question Bank — Chapter 16

**Conceptual:**
1. What's the difference between the cell means and factor effects parameterizations of one-way ANOVA, and why do both give the same F-test?
2. How does the Brown-Forsythe test generalize from two groups (Chapter 3) to more than two groups?
3. What's the core tradeoff between Kruskal-Wallis and a parametric one-way ANOVA F-test?

**Derivation:**
4. Show why the factor effects model's constraint $\sum\tau_i=0$ is needed for identifiability, drawing the parallel to Chapter 8's dummy variable trap.
5. Derive the Kruskal-Wallis test statistic's logic: why does comparing rank sums (rather than raw means) provide a distribution-free test?

**ML/Statistics:**
6. Explain, with an example, the practical difference between treating a grouping variable as a fixed effect versus a random effect.
7. How does a random-effects model's "shrinkage toward the grand mean" relate to ridge regression's shrinkage (Chapter 11)?
8. In a recommendation system, why might user-level and item-level effects be modeled as random rather than fixed?

**Coding:**
9. Implement the Kruskal-Wallis test from scratch (ranking, rank sums, H statistic) and verify against `scipy.stats.kruskal`.
10. Fit both a fixed-effects (dummy-variable) and random-effects (mixed) model to grouped data and compare the estimated group-level effects.

**Traps:**
11. "Kruskal-Wallis and one-way ANOVA test exactly the same hypothesis, so it doesn't matter which you use." — what subtle difference in what's being tested (means vs. distributions/ranks) should you flag?
12. "Since this factor has many levels, I should always treat it as random." — what's the more precise criterion for the fixed-vs-random decision?
13. A student says the "factor effects" and "cell means" models must give different p-values since they have different parameters. What's wrong with this reasoning?

---

*This file covers Kutner Ch. 16 — the cell means and factor effects parameterizations of one-way ANOVA (shown equivalent to Chapter 8's regression), the Brown-Forsythe test generalized to multiple groups, the Kruskal-Wallis nonparametric alternative (worked by hand with tied ranks), and the fixed-vs-random effects distinction connected directly to modern mixed-effects/hierarchical models and ridge-regression-style shrinkage. Chapter 17 (Analysis of Factor Level Means) is next — covering planned contrasts and the multiple-comparison procedures (Tukey's HSD, Scheffé, Bonferroni-adjusted pairwise tests) needed once an omnibus F-test says "the groups differ" but you need to know *which* pairs differ.*
