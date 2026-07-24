# Chapter 18 — Two-Factor Studies with Equal Sample Sizes
### (Kutner, Nachtsheim, Neter, Li — *Applied Linear Statistical Models*)

Chapters 15–17 handled **one** factor (webpage design). This chapter adds a **second** factor, letting us ask a richer set of questions: does each factor matter on its own (**main effects**), and does the effect of one factor **depend on the level of the other** (**interaction**) — a direct generalization of Chapter 8's interaction-model idea into the ANOVA table itself.

### Extending the Worked Example

We add a second factor to the webpage-design study: **traffic source** (Organic vs. Paid), giving a $3\times2$ factorial design — 3 designs × 2 traffic sources = 6 "cells," with $n=3$ replicates each (revenue per visitor, 18 observations total):

| | Organic | Paid |
|---|---|---|
| **Design A** | 9, 10, 11 | 13, 14, 15 |
| **Design B** | 13, 14, 15 | 17, 18, 19 |
| **Design C** | 17, 18, 19 | 21, 22, 23 |

**Cell means:** $\bar Y_{A,Org}=10,\ \bar Y_{A,Paid}=14,\ \bar Y_{B,Org}=14,\ \bar Y_{B,Paid}=18,\ \bar Y_{C,Org}=18,\ \bar Y_{C,Paid}=22$.

---

## 18.1 The Two-Factor Model

$$
Y_{ijk} = \mu+\alpha_i+\beta_j+(\alpha\beta)_{ij}+\varepsilon_{ijk}
$$

**What each new piece means, in plain English:**
- $\mu$: overall grand mean.
- $\alpha_i$: **main effect of Factor A** (Design) — how much Design level $i$'s average shifts the mean, averaged across both traffic sources.
- $\beta_j$: **main effect of Factor B** (Traffic) — how much traffic source $j$'s average shifts the mean, averaged across all designs.
- $(\alpha\beta)_{ij}$: the **interaction effect** — the part of cell $(i,j)$'s mean that **isn't** explained by simply adding the two main effects together. This is the genuinely new object this chapter introduces.

**Why interaction is the single most important new concept here.** If $(\alpha\beta)_{ij}=0$ for every cell, the two factors act **independently and additively** — Design's effect is the same regardless of traffic source, and vice versa. If interaction is present, this breaks down: **the effect of Design genuinely depends on which traffic source you're looking at** — exactly Chapter 8's "the slope depends on group membership" idea, now expressed as "the row effect depends on the column."

---

## 18.2–18.3 Partitioning the Sum of Squares

**Marginal means** (averaging across the other factor):

$$
\bar Y_{A..}=12,\ \bar Y_{B..}=16,\ \bar Y_{C..}=20 \qquad(\text{Design marginal means})
$$
$$
\bar Y_{.Org.}=14,\ \bar Y_{.Paid.}=18 \qquad(\text{Traffic marginal means})
$$
$$
\bar Y_{...}=16 \qquad(\text{grand mean})
$$

### $SSA$ — Main Effect of Design

$$
SSA = bn\sum_i(\bar Y_{i..}-\bar Y_{...})^2, \qquad b=2\text{ (traffic levels)},\ n=3
$$
$$
= 6\left[(12-16)^2+(16-16)^2+(20-16)^2\right]=6[16+0+16]=6(32)=192
$$

### $SSB$ — Main Effect of Traffic

$$
SSB = an\sum_j(\bar Y_{.j.}-\bar Y_{...})^2, \qquad a=3\text{ (design levels)}
$$
$$
=9\left[(14-16)^2+(18-16)^2\right]=9[4+4]=9(8)=72
$$

### $SSAB$ — Interaction

**Plain English formula intuition.** For each cell, subtract off what the main effects alone would predict ($\bar Y_{i..}+\bar Y_{.j.}-\bar Y_{...}$), leaving only the "leftover," genuinely cell-specific deviation:
$$
SSAB = n\sum_{i,j}\left[\bar Y_{ij.}-\bar Y_{i..}-\bar Y_{.j.}+\bar Y_{...}\right]^2
$$

**Worked out, cell by cell** (the bracketed term for each cell):
$$
(A,Org): 10-12-14+16=0 \qquad (A,Paid): 14-12-18+16=0
$$
$$
(B,Org): 14-16-14+16=0 \qquad (B,Paid): 18-16-18+16=0
$$
$$
(C,Org): 18-20-14+16=0 \qquad (C,Paid): 22-20-18+16=0
$$

**Every single interaction term is exactly zero.** $SSAB=3\times(0^2\times6)=0$.

**Why this happened, and what it means.** Notice the "Paid boost" is exactly $+4$ for every single design (A: $14-10=4$; B: $18-14=4$; C: $22-18=4$) — traffic source's effect is **perfectly consistent** regardless of which design it's paired with. This is precisely what "no interaction" looks like numerically: **the two factors' effects are perfectly additive.**

### $SSE$ (within-cell error)

Each cell's 3 replicates deviate from their own cell mean by exactly $\{-1,0,+1\}$ (by construction), contributing $SS=(-1)^2+0^2+1^2=2$ per cell:
$$
SSE = 6\text{ cells}\times2 = 12
$$

### The Full ANOVA Table

| Source | SS | df | MS | F* |
|---|---|---|---|---|
| Design (A) | 192 | $a-1=2$ | 96 | 96.0 |
| Traffic (B) | 72 | $b-1=1$ | 72 | 72.0 |
| Interaction (AB) | 0 | $(a-1)(b-1)=2$ | 0 | 0.0 |
| Error | 12 | $ab(n-1)=12$ | 1.0 | |
| Total | 276 | $abn-1=17$ | | |

(Check: $2+1+2+12=17$ ✓, and $192+72+0+12=276$ ✓ matches $SSTO$ computed directly from all 18 observations around the grand mean.)

**F-tests:**
$$
F^*_A=\frac{96}{1.0}=96 \quad(\text{vs. } F_{(0.95;2,12)}=3.89 \Rightarrow \text{highly significant})
$$
$$
F^*_B=\frac{72}{1.0}=72 \quad(\text{vs. } F_{(0.95;1,12)}=4.75 \Rightarrow \text{highly significant})
$$
$$
F^*_{AB}=\frac{0}{1.0}=0 \quad(\text{clearly not significant — no evidence of interaction})
$$

**Conclusion: both Design and Traffic Source significantly affect revenue on their own, and they do so *independently* — there's no evidence the size of one factor's effect depends on the other.**

---

## 18.4 Interpreting Interaction (or Its Absence) Visually

**Plain English.** Plot each design's mean revenue, with one line for Organic and one line for Paid, connected across A→B→C. **When there's no interaction (our case), these two lines are exactly parallel** — both climb by 4 units per design step, just offset from each other by a constant 4. **When interaction is present, the lines are not parallel — they might converge, diverge, or even cross.**

**Why this matters enormously for practical interpretation.** If the lines were *not* parallel — say, the Paid boost was $+2$ for Design A but $+8$ for Design C — then it would be misleading to report a single, blanket "main effect of Traffic Source" (Kutner's classical caution, directly echoing Chapter 8's warning about interpreting a coefficient "holding other variables constant" when an interaction term is present). **You would instead need to examine "simple effects"** — the effect of Traffic *within* each specific Design level separately — since the overall averaged main effect would obscure real, meaningfully different behavior across designs.

**Interview question:** *"You run a two-way ANOVA and find a significant interaction effect. Can you still meaningfully interpret the main effects on their own?"*
**Ideal answer:** Not cleanly — a significant interaction means each factor's effect genuinely depends on the level of the other factor, so a single "average" main effect can mask substantial, meaningfully different behavior across subgroups (e.g., a factor could help in one condition and hurt in another, averaging out to a deceptively small or even null-looking main effect). The right approach is to examine **simple effects** — the effect of one factor at each specific level of the other — rather than reporting the marginal main effect as if it applied uniformly.

---

## 18.5 The Direct Regression Equivalent — Extending Chapter 8

**Exactly as one-way ANOVA is Chapter 8's single-indicator regression (Chapter 15), two-way ANOVA is Chapter 8's *interaction* regression model, generalized to more than two levels per factor:**
$$
Y = \beta_0+\beta_1D_B+\beta_2D_C+\beta_3D_{Paid}+\beta_4(D_B\times D_{Paid})+\beta_5(D_C\times D_{Paid})+\varepsilon
$$
where $D_B, D_C$ are Design indicators (Design A as reference) and $D_{Paid}$ is the Traffic indicator (Organic as reference). **$SSA$ is exactly the extra sum of squares from $D_B,D_C$; $SSB$ is exactly the extra sum of squares from $D_{Paid}$; $SSAB$ is exactly the extra sum of squares from the two product terms** — precisely Chapter 7's extra-sums-of-squares framework, now applied to a richer, two-factor design. **Every F-test in the ANOVA table above could be reproduced, number-for-number, by fitting this regression and running the corresponding partial F-tests from Chapter 7** — there is, once again, no new mathematics here, only new vocabulary and a more convenient tabular organization for the specific case of fully categorical predictors.

---

## Python Implementation

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

data = {
    'Design': ['A']*6 + ['B']*6 + ['C']*6,
    'Traffic': (['Organic']*3 + ['Paid']*3)*3,
    'Revenue': [9,10,11,13,14,15,  13,14,15,17,18,19,  17,18,19,21,22,23]
}
df = pd.DataFrame(data)

# --- Two-way ANOVA via regression with interaction (Chapter 8 machinery, extended) ---
model = smf.ols('Revenue ~ C(Design) * C(Traffic)', data=df).fit()
anova_table = anova_lm(model, typ=2)
print(anova_table)

# --- Interaction plot ---
import matplotlib.pyplot as plt
means = df.groupby(['Design','Traffic'])['Revenue'].mean().unstack()
means.plot(marker='o')
plt.ylabel('Mean Revenue')
plt.title('Interaction Plot: parallel lines = no interaction')
plt.show()
```

---

## Interview Question Bank — Chapter 18

**Conceptual:**
1. What does "interaction" mean in a two-factor ANOVA, in plain terms?
2. What does it look like, visually, when two factors have no interaction versus a strong interaction?
3. Why is it misleading to interpret main effects in isolation when a significant interaction is present?

**Derivation:**
4. Derive the interaction sum of squares formula and explain intuitively why it's zero when effects are perfectly additive.
5. Show how the two-way ANOVA F-tests correspond exactly to partial F-tests in an equivalent regression with indicator variables and product terms.

**ML/Statistics:**
6. In an A/B test with two simultaneous treatments (e.g., a UI change and a pricing change), why would you want to test for interaction rather than just running two separate single-factor tests?
7. If a significant interaction is found, what follow-up analysis would you do instead of just reporting the two main effects?
8. Connect this chapter's two-way ANOVA table to the general linear test / extra sums of squares framework from Chapter 7.

**Coding:**
9. Implement the two-way ANOVA sum-of-squares decomposition (SSA, SSB, SSAB, SSE) from scratch in NumPy/pandas for a balanced factorial dataset.
10. Fit the equivalent interaction regression model and verify its ANOVA table matches a classical two-way ANOVA computation.

**Traps:**
11. "Since both main effects are significant and the interaction isn't, the two factors don't relate to each other at all." — what's the more precise statement?
12. "A large main effect always means that factor matters more than a factor with a smaller main effect." — what could a hidden interaction do to this comparison?
13. Someone reports only the marginal means from a two-way design with a strong, significant interaction. What important information are they hiding?

---

*This file covers Kutner Ch. 18 — the two-factor ANOVA model, the SSA/SSB/SSAB/SSE decomposition worked in full on a balanced 3×2 factorial design (deliberately constructed with zero interaction to make the concept concrete), interaction plots and their interpretation, and the direct regression-equivalence extending Chapter 8's interaction models. This is likely a natural stopping point for the ML-interview-focused portion of this course — subsequent Kutner chapters (randomized block designs, unequal-sample-size factorial designs, and specialized study designs) become increasingly specialized to formal experimental design work rather than core ML/DS interview content.*
