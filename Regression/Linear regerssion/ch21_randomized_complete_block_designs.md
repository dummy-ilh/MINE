# Chapter 21 — Randomized Complete Block Designs
### (Kutner, Nachtsheim, Neter, Li — *Applied Linear Statistical Models*)

Chapter 15 previewed blocking; Chapter 19 showed that a two-factor design with one observation per cell has exactly the right mathematical structure to serve as a blocking design. This chapter makes that connection explicit and formal, and introduces the one genuinely new practical tool: **quantifying how much blocking actually helped** — the relative efficiency calculation.

### The Worked Example

Testing **4 ad creatives** (A, B, C, D — the treatment of real interest) across **5 cities** (the blocking factor — a nuisance source of variability we want to control for, not something we're trying to compare in its own right, since cities obviously differ in baseline conversion rate for reasons unrelated to the ad creative). One observation per city×creative combination (20 total). Response: conversion rate (%).

| | A | B | C | D | Row mean |
|---|---|---|---|---|---|
| **City 1** | 14 | 14 | 18 | 18 | 16 |
| **City 2** | 14 | 18 | 18 | 22 | 18 |
| **City 3** | 18 | 18 | 22 | 22 | 20 |
| **City 4** | 18 | 22 | 22 | 26 | 22 |
| **City 5** | 22 | 22 | 26 | 26 | 24 |
| **Col mean** | 17.2 | 18.8 | 21.2 | 22.8 | **Grand: 20** |

**Notice the cities differ substantially** (row means range from 16 to 24 — an 8-point spread) — exactly the kind of nuisance variability blocking is meant to absorb before it ever reaches the treatment comparison.

---

## 21.1 Why Block: the Core Idea, Restated Precisely

**Plain English.** If you ignored city entirely and just compared the four creatives' raw averages pooled across all cities, the substantial city-to-city variability would sit entirely inside your "error" term — inflating it and making real treatment differences harder to detect. **Blocking works by literally subtracting out each block's own average level before comparing treatments within it** — since every treatment appears exactly once in every block, the comparison between treatments *within* a block is completely unaffected by that block's own baseline level. This is mathematically the exact same maneuver as Chapter 6's "controlling for a covariate" and Chapter 20's ANCOVA — just for a **categorical** nuisance variable (city) instead of a continuous one (a pre-experiment covariate).

---

## 21.2 The Model — Mathematically Identical to Chapter 19

$$
Y_{ij} = \mu+\rho_i+\tau_j+\varepsilon_{ij}
$$

where $\rho_i$ is the block (city) effect and $\tau_j$ is the treatment (creative) effect. **This is precisely Chapter 19's additive two-factor, one-observation-per-cell model** — the only thing that's changed is *intent*: here, we don't care about comparing cities to each other; we're using them purely to control variance while we focus entirely on the $\tau_j$'s. That difference in intent, not any difference in the mathematics, is what separates "a two-factor study with one case per treatment" from "a randomized complete block design."

**And exactly as in Chapter 19: with one observation per cell, $SS(\text{Blocks}\times\text{Treatment interaction})$ and $SSE$ are confounded** — we again pool them into a single "remainder," using it as our error term, and (exactly as before) can apply Tukey's one-degree-of-freedom test for (multiplicative) non-additivity to check whether that pooling is defensible.

---

## 21.3 The ANOVA

**Sums of squares** (identical mechanics to Chapters 18–20; $t=4$ treatments, $r=5$ blocks):
$$
SS(\text{Blocks}) = t\sum_i(\bar Y_{i.}-\bar Y_{..})^2 = 4\left[(-4)^2+(-2)^2+0^2+2^2+4^2\right]=4(40)=160
$$
$$
SS(\text{Treatments}) = r\sum_j(\bar Y_{.j}-\bar Y_{..})^2 = 5\left[(-2.8)^2+(-1.2)^2+(1.2)^2+(2.8)^2\right]=5(18.56)=92.8
$$
$$
SSTO = \sum_{ij}(Y_{ij}-20)^2 = 272 \quad(\text{direct sum of all 20 squared deviations})
$$
$$
SS(\text{remainder}) = SSTO-SS(\text{Blocks})-SS(\text{Treatments}) = 272-160-92.8=19.2
$$

| Source | SS | df | MS | F* |
|---|---|---|---|---|
| Blocks (City) | 160 | $r-1=4$ | 40 | 25.0 |
| Treatments (Creative) | 92.8 | $t-1=3$ | 30.93 | 19.33 |
| Remainder | 19.2 | $(r-1)(t-1)=12$ | 1.6 | |
| Total | 272 | $rt-1=19$ | | |

$$
F^*_{\text{Treatment}} = \frac{30.93}{1.6}=19.33 \quad(\text{vs. } F_{(0.95;3,12)}=3.49 \Rightarrow \text{highly significant})
$$

**The four ad creatives genuinely differ in conversion rate, even after controlling for city.** (The block F-test, 25.0, is often reported too — mainly as a sanity check that blocking was worthwhile — though if cities are treated as a *random* rather than fixed factor, as discussed in Section 21.6, this particular F-test isn't really the right question to be asking of them.)

---

## 21.4 Relative Efficiency of Blocking — the Key New Practical Tool

### The Question This Answers

**Plain English.** You've now seen that blocking dramatically shrank the error term (the "remainder," 1.6) relative to how big the city-to-city variation was (160). **How much would a Completely Randomized Design — one that ignored city entirely — have suffered, in concrete, actionable terms?** Specifically: *how many times more observations would a CRD need, to achieve the same precision this RCBD achieved with just 20?*

### Deriving It From First Principles

**If you'd ignored blocking entirely**, the city-to-city variation wouldn't disappear — it would simply become part of the unexplained error, pooled in with the remainder:
$$
SSE_{\text{if CRD}} = SS(\text{Blocks})+SS(\text{remainder}) = 160+19.2=179.2
$$
$$
df_{\text{if CRD}} = df(\text{Blocks})+df(\text{remainder}) = 4+12=16
$$
$$
MSE_{\text{if CRD}} = 179.2/16=11.2
$$

**Compare this to the RCBD's actual error variance** ($MSE_{\text{RCBD}}=1.6$, the remainder alone). The simple **relative efficiency**:
$$
RE = \frac{MSE_{\text{if CRD}}}{MSE_{\text{RCBD}}} = \frac{11.2}{1.6}=7.0
$$

**Interpretation, made concrete and actionable:** blocking by city reduced the error variance by a factor of **7**. Equivalently: **a Completely Randomized Design would need approximately 7 times as many total observations as this RCBD to achieve the same precision** for comparing the four creatives — a dramatic, quantified illustration of exactly why blocking (or its modern covariate-based cousin, CUPED) is worth the design effort whenever a strong nuisance variable is available and identifiable in advance.

### The Small-Sample Refinement

Because the RCBD's error estimate and the hypothetical CRD's error estimate are based on **different degrees of freedom** (12 vs. 16 here), and an estimated variance based on fewer degrees of freedom is itself a noisier estimate of the true variance, the raw ratio above slightly **overstates** the true benefit. The standard correction discounts the raw ratio by a factor based on both designs' degrees of freedom:
$$
RE_{\text{adjusted}} = RE\times\frac{(df_{\text{RCBD}}+1)(df_{\text{CRD}}+3)}{(df_{\text{RCBD}}+3)(df_{\text{CRD}}+1)}
$$
$$
= 7.0\times\frac{(12+1)(16+3)}{(12+3)(16+1)}=7.0\times\frac{13\times19}{15\times17}=7.0\times\frac{247}{255}=7.0(0.9686)=6.78
$$

**With reasonably large degrees of freedom in both designs (as here), this correction is small** (7.0 → 6.78) — it matters more when either design's error df is quite small, since the correction factor approaches 1 as both degrees of freedom grow large (the asymptotic case where finite-sample noise in the variance estimates themselves stops mattering).

**Interview question:** *"How would you explain 'relative efficiency of blocking' to a product manager deciding whether a stratified experiment design is worth the added complexity?"*
**Ideal answer:** It directly answers "how much bigger would our experiment need to be if we didn't stratify/block on this variable?" — computed by comparing the error variance actually achieved with blocking to what the error variance would have been if the blocking variable's variation had simply been left inside the noise term. A relative efficiency of, say, 7 means an unblocked design would need roughly 7 times the sample size to detect the same effect with the same precision — a very concrete way to justify the extra design complexity of blocking or stratified randomization when a strong, identifiable nuisance factor (like city, device type, or a pre-experiment engagement metric) is available.

---

## 21.5 Checking Appropriateness: Tukey's Test for Additivity, Revisited

**Exactly the same tool from Chapter 19 applies here without modification** — a significant "block × treatment interaction" would mean the treatment effect genuinely differs across blocks (e.g., Creative D works great in some cities but not others), which would undermine the assumption that a single "adjusted treatment effect" fairly describes all blocks. The mechanics (compute row effects and column effects, form $\sum a_ib_jY_{ij}$, test the resulting one-degree-of-freedom statistic against $F_{(1,\,df_{\text{remainder}}-1)}$) are identical to Chapter 19's worked example — we won't re-derive them here, but the same important limitation applies: **it only detects a specific multiplicative form of block-treatment interaction**, not any possible departure from additivity.

---

## 21.6 Fixed vs. Random Blocks — Direct Callback to Chapter 16

**If the specific 5 cities are the only ones you care about** (a fixed effect, Chapter 16's terminology): the block F-test (25.0) is a legitimate test of whether *these particular* cities differ. **If these 5 cities are meant to represent a broader population of cities** you want to generalize to (a random effect): you don't really care about testing whether *these specific* cities differ from each other — you care about how much city-to-city variability exists *in general*, and whether the treatment effect generalizes across that variability. In the random-blocks case, the treatment F-test computed above remains valid exactly as computed, but the "block F-test" itself isn't really answering a meaningful question — you'd instead report an estimated **variance component** for blocks (how much of the total variance is attributable to between-city differences) rather than a formal significance test on the specific cities observed.

---

## Python Implementation

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

data = []
Y = [[14,14,18,18],[14,18,18,22],[18,18,22,22],[18,22,22,26],[22,22,26,26]]
for i, city in enumerate(['City1','City2','City3','City4','City5']):
    for j, creative in enumerate(['A','B','C','D']):
        data.append({'City': city, 'Creative': creative, 'Conversion': Y[i][j]})
df = pd.DataFrame(data)

# --- RCBD ANOVA ---
model = smf.ols('Conversion ~ C(City) + C(Creative)', data=df).fit()
print(anova_lm(model, typ=2))

# --- Relative efficiency of blocking ---
anova_table = anova_lm(model, typ=2)
SS_block = anova_table.loc['C(City)', 'sum_sq']
SS_remainder = anova_table.loc['Residual', 'sum_sq']
df_block = anova_table.loc['C(City)', 'df']
df_remainder = anova_table.loc['Residual', 'df']

SSE_if_CRD = SS_block + SS_remainder
df_CRD = df_block + df_remainder
MSE_if_CRD = SSE_if_CRD / df_CRD
MSE_RCBD = SS_remainder / df_remainder

RE = MSE_if_CRD / MSE_RCBD
RE_adjusted = RE * ((df_remainder+1)*(df_CRD+3)) / ((df_remainder+3)*(df_CRD+1))
print(f"Relative Efficiency (simple): {RE:.2f}")
print(f"Relative Efficiency (adjusted): {RE_adjusted:.2f}")
```

---

## Interview Question Bank — Chapter 21

**Conceptual:**
1. In what precise sense is a Randomized Complete Block Design mathematically identical to Chapter 19's two-factor, one-case-per-treatment design?
2. What question does "relative efficiency of blocking" answer, in plain, actionable terms?
3. Why does testing whether specific block levels differ make sense for fixed blocks but not really for random blocks?

**Derivation:**
4. Derive the relative efficiency formula from the idea of "what would the error variance have been if block variation were left inside the noise term."
5. Explain why the small-sample correction factor for relative efficiency approaches 1 as both designs' degrees of freedom grow large.

**ML/Statistics:**
6. Connect blocking's variance-reduction mechanism to ANCOVA and to CUPED in modern A/B testing — what's the same, and what's different (categorical nuisance factor vs. continuous covariate)?
7. If you computed a relative efficiency of 1.05 for a proposed blocking variable, would you recommend using it? Why or why not?
8. Why would you use a variance component (rather than a significance test on specific block levels) when blocks represent a random sample from a larger population?

**Coding:**
9. Implement the RCBD ANOVA and relative efficiency calculation from scratch in NumPy/pandas.
10. Simulate data with a strong blocking variable and verify that the computed relative efficiency correctly reflects how much the blocking variable's variance dominates the total variance.

**Traps:**
11. "Blocking always improves your experiment, so you should block on every variable you can measure." — what's the flaw, especially considering variables with low relative efficiency?
12. "Relative efficiency of 7 means we'd get exactly 7x more statistical power." — what's the more precise statement about what this quantifies?
13. Someone reports a significant F-test on their random blocking factor and concludes something about those specific block levels. What's the more appropriate interpretation given the factor's random rather than fixed role?

---

*This file covers Kutner Ch. 21 — the Randomized Complete Block Design as an application of Chapter 19's math with a change in intent, the full ANOVA worked by hand, and the relative efficiency of blocking (derived from first principles and computed, including the small-sample correction) — the single most practically important new tool in this chapter, directly generalizing the variance-reduction logic from ANCOVA and CUPED into a concrete "how much bigger would my unblocked experiment need to be" answer. This is a reasonable place to consider the design-of-experiments arc complete for interview purposes — remaining Kutner chapters (nested designs, repeated measures, Latin squares, factorial/fractional-factorial designs, response surface methodology) move further into specialized industrial DOE territory.*
