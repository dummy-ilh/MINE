# Chapter 20 — Multi-Factor Studies
### (Kutner, Nachtsheim, Neter, Li — *Applied Linear Statistical Models*)

Chapters 18–19 covered two factors. This chapter extends the same machinery to **three or more factors simultaneously** — the natural setting for a "multivariate test" (testing several changes at once) — and confronts the real practical challenge that emerges: **interactions get harder to interpret, and harder to detect, as their order grows.**

### The Worked Example: A $2\times2\times2$ Factorial Design

Three factors, each at 2 levels, replicated $n=2$ times per combination (8 cells × 2 = 16 observations): **Design** (Old/New), **Traffic** (Organic/Paid), **Device** (Desktop/Mobile). Response: revenue per visitor.

| Design | Traffic | Device | Observations | Cell mean |
|---|---|---|---|---|
| Old | Organic | Desktop | 13, 15 | 14 |
| Old | Organic | Mobile | 15, 17 | 16 |
| Old | Paid | Desktop | 19, 21 | 20 |
| Old | Paid | Mobile | 21, 23 | 22 |
| New | Organic | Desktop | 17, 19 | 18 |
| New | Organic | Mobile | 19, 21 | 20 |
| New | Paid | Desktop | 23, 25 | 24 |
| New | Paid | Mobile | 25, 27 | 26 |

(Built with genuine main effects for all three factors and **zero interaction of any order** — deliberately, to make the mechanics of the decomposition unambiguous. We'll discuss what nonzero higher-order interaction would look like immediately after.)

---

## 20.1 The Three-Factor Model

$$
Y_{ijkl} = \mu+\alpha_i+\beta_j+\gamma_k+(\alpha\beta)_{ij}+(\alpha\gamma)_{ik}+(\beta\gamma)_{jk}+(\alpha\beta\gamma)_{ijk}+\varepsilon_{ijkl}
$$

**What's genuinely new here, beyond just "one more factor":** the **three-way interaction term** $(\alpha\beta\gamma)_{ijk}$. **What this means in plain English:** it asks whether the **two-way interaction between two factors itself changes depending on the level of the third**. For example: suppose New Design helps more with Paid traffic than Organic (a Design×Traffic interaction) — a three-way interaction would mean *that specific pattern itself is different on Desktop versus Mobile* (e.g., the "New Design helps Paid traffic more" effect is strong on Desktop but weak or reversed on Mobile).

**Why three-way interactions are notoriously hard to communicate and act on.** A significant main effect is one sentence ("New Design increases revenue"). A significant two-way interaction is already a more complex sentence ("New Design helps more with Paid traffic than Organic"). A three-way interaction requires a sentence with *three* moving parts ("New Design helps more with Paid traffic than Organic, but only on Desktop — on Mobile this pattern is reversed") — genuinely difficult to act on practically, and (as we'll see) also genuinely difficult to detect statistically with typical sample sizes.

---

## 20.2 Computing the Full Sum-of-Squares Decomposition

### Marginal Means

**Design:** $\bar Y_{Old}=18,\ \bar Y_{New}=22$ (difference: 4). **Traffic:** $\bar Y_{Org}=17,\ \bar Y_{Paid}=23$ (difference: 6). **Device:** $\bar Y_{Desk}=19,\ \bar Y_{Mob}=21$ (difference: 2). **Grand mean:** $\bar Y_{....}=20$.

### Main Effect Sums of Squares

With $a=b=c=2$ levels each and $n=2$ replicates, each main effect's multiplier is (product of the *other two* factors' levels) × $n$:
$$
SSA = (bcn)\sum_i(\bar Y_{i...}-\bar Y_{....})^2 = 8\left[(18-20)^2+(22-20)^2\right]=8(8)=64
$$
$$
SSB = (acn)\sum_j(\bar Y_{.j..}-\bar Y_{....})^2 = 8\left[(17-20)^2+(23-20)^2\right]=8(18)=144
$$
$$
SSC = (abn)\sum_k(\bar Y_{..k.}-\bar Y_{....})^2 = 8\left[(19-20)^2+(21-20)^2\right]=8(2)=16
$$

### Two-Way Interaction Sums of Squares

**Design×Traffic** two-way means (averaging over Device): Old-Org=15, Old-Paid=21, New-Org=19, New-Paid=25.
$$
\text{Bracket}_{ij} = \bar Y_{ij..}-\bar Y_{i...}-\bar Y_{.j..}+\bar Y_{....}
$$
$$
(Old,Org): 15-18-17+20=0 \quad (Old,Paid): 21-18-23+20=0
$$
$$
(New,Org): 19-22-17+20=0 \quad (New,Paid): 25-22-23+20=0
$$
**All exactly zero** — $SSAB=0$. The same computation for Design×Device and Traffic×Device (omitted here for space, identical mechanics) also gives **exactly zero** for both $SSAC$ and $SSBC$ — confirming there is no two-way interaction of any kind in this data, as intended.

### The Three-Way Interaction Sum of Squares

$$
\text{Bracket}_{ijk} = \bar Y_{ijk.} - \bar Y_{ij..}-\bar Y_{i.k.}-\bar Y_{.jk.}+\bar Y_{i...}+\bar Y_{.j..}+\bar Y_{..k.}-\bar Y_{....}
$$
**Worked for the (Old, Organic, Desktop) cell** (mean 14; needed two-way means: Old-Org=15, Old-Desk=17, Org-Desk=16; needed main-effect means: Old=18, Org=17, Desk=19; grand=20):
$$
14-15-17-16+18+17+19-20 = 0
$$
**Every cell gives exactly zero this way** (a direct consequence of the fully additive construction) — $SSABC=0$.

### Error and Total

Each cell's two replicates deviate from their own cell mean by $\pm1$, contributing $1^2+1^2=2$ per cell:
$$
SSE = 8\text{ cells}\times2=16, \qquad df_{SSE}=abc(n-1)=8(1)=8, \qquad MSE=16/8=2.0
$$
$$
SSTO = SSA+SSB+SSC+SSAB+SSAC+SSBC+SSABC+SSE = 64+144+16+0+0+0+0+16=240
$$
(Directly verified by summing all 16 squared deviations from the grand mean 20 — matches exactly.)

### The Full ANOVA Table

| Source | SS | df | MS | F* |
|---|---|---|---|---|
| Design (A) | 64 | 1 | 64 | 32.0 |
| Traffic (B) | 144 | 1 | 144 | 72.0 |
| Device (C) | 16 | 1 | 16 | 8.0 |
| A×B | 0 | 1 | 0 | 0.0 |
| A×C | 0 | 1 | 0 | 0.0 |
| B×C | 0 | 1 | 0 | 0.0 |
| A×B×C | 0 | 1 | 0 | 0.0 |
| Error | 16 | 8 | 2.0 | |
| Total | 240 | 15 | | |

(Check df: $1+1+1+1+1+1+1+8=15$ ✓)

**F-tests:** all three main effects clear $F_{(0.95;1,8)}=5.32$ easily (Design: 32.0, Traffic: 72.0, Device: 8.0 — all significant), while every interaction term is exactly zero — **a clean result: all three factors matter on their own, and none of their effects depend on the others.**

---

## 20.3 Why Higher-Order Interactions Are Hard to Detect (Not Just Hard to Interpret)

**This is worth stating explicitly, since it's a genuine statistical fact, not just a communication problem.** Notice the three-way interaction has only **1 degree of freedom** here (same as each main effect, in this particular $2\times2\times2$ design) — but as the number of levels per factor grows, higher-order interaction terms consume **rapidly growing** degrees of freedom (e.g., a 3-level × 3-level × 4-level design's three-way interaction has $(3-1)(3-1)(4-1)=12$ df), spreading a fixed total sample size across more and more parameters to estimate. **This means higher-order interactions are estimated with progressively less precision per parameter**, for a fixed total sample size — you need substantially more data to detect a real three-way interaction with the same reliability as a main effect, even when the underlying true effect sizes are comparable. This is a direct, practical reason experimenters are often reluctant to chase higher-order interactions without a strong prior reason to expect one and a large enough sample to have real power to detect it.

**Interview question:** *"Why are three-way (or higher) interactions in a factorial experiment generally harder to detect than main effects, holding the true effect size constant?"*
**Ideal answer:** Higher-order interaction terms consume more degrees of freedom as the number of factor levels grows, and for a fixed total sample size, more parameters means less precision per parameter — the same total information has to be spread across more distinct effects to estimate. Combined with the fact that true higher-order interaction effects are often smaller in magnitude than main effects in many real systems, this compounds into substantially lower statistical power for detecting genuine higher-order interactions compared to main effects, even when the experiment is otherwise well-powered for the main effects themselves.

---

## 20.4 The Hierarchical Principle and Pooling Strategy

### The Hierarchical Principle

**Plain English.** If a higher-order interaction (say, $A\times B$) is retained in a model, convention (and good interpretive practice) says you should also retain the corresponding lower-order terms ($A$ and $B$ main effects) **regardless of whether they're individually statistically significant**. **Why:** an interaction term without its "parent" main effects present is not meaningfully interpretable on its own — the coefficient on an interaction term represents a *deviation from* an additive baseline that the main effects define; removing the main effects while keeping the interaction effectively redefines what "baseline" means in a way that makes the interaction's coefficient no longer comparable across analyses or easily interpretable. This is exactly the same principle behind never removing $X$ from a model that still contains $X^2$ (Chapter 8/13's polynomial regression context) — the lower-order term anchors the higher-order term's interpretation.

### Pooling Non-Significant Higher-Order Terms

**A common, pragmatic strategy** (directly connecting to Chapter 9's model-building themes): if higher-order interactions (especially three-way and beyond) come back clearly non-significant, some analysts **pool their sums of squares back into the error term**, gaining additional error degrees of freedom and thereby increasing power for testing the remaining (main effect and lower-order interaction) terms that are actually of interest. **The tradeoff, stated honestly:** this is only appropriate when there's good reason (theoretical or empirical) to believe the true higher-order interaction is genuinely negligible — pooling a term that's actually real, just underpowered to detect in this particular sample, would bias the resulting error term and invalidate the remaining tests, exactly the same caution that applies to any model-simplification decision (Chapter 9's warnings about stepwise selection apply here too).

**Interview question:** *"When might you pool a non-significant three-way interaction into the error term, and what's the risk of doing so?"*
**Ideal answer:** Pooling is reasonable when there's a solid theoretical or prior-empirical reason to expect the true three-way interaction is negligible, and doing so frees up degrees of freedom that increase power for the main effects and lower-order interactions you actually care about testing. The risk is that if the three-way interaction is real but simply underpowered to detect in this particular sample (a real possibility given how much less power higher-order terms tend to have, as discussed above), pooling it into the error term contaminates that error term with real, systematic variation — potentially invalidating the very tests you were trying to make more powerful, and risking a false main-effect conclusion built on a mis-specified model.

---

## 20.5 Connection to Multivariate (MVT) Testing in Industry

**This chapter's framework is exactly the statistical backbone of multivariate testing (MVT)** — running several simultaneous experiment factors at once (e.g., testing a new headline, a new button color, and a new layout all together) rather than one factor at a time. **The practical lesson from this chapter directly applies:** MVT designs let you detect interactions between simultaneous changes that separate, sequential A/B tests would completely miss — but they also require substantially larger sample sizes to have real power for anything beyond the main effects, and results involving three-way-or-higher interactions among the tested factors are both statistically underpowered and practically difficult to act on. This is exactly why most industry experimentation guidance recommends MVT primarily when you have a specific, prior reason to suspect an interaction between particular factors, rather than as a default way to test many things simultaneously "for efficiency."

---

## Python Implementation

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

data = {
    'Design': ['Old']*4 + ['New']*4,
    'Traffic': (['Organic']*2 + ['Paid']*2)*2,
    'Device': ['Desktop','Mobile']*4,
}
# expand to include both replicates per cell
rows = []
values = [[13,15],[15,17],[19,21],[21,23],[17,19],[19,21],[23,25],[25,27]]
i = 0
for design in ['Old','New']:
    for traffic in ['Organic','Paid']:
        for device in ['Desktop','Mobile']:
            for v in values[i]:
                rows.append({'Design':design,'Traffic':traffic,'Device':device,'Revenue':v})
            i += 1
df = pd.DataFrame(rows)

model = smf.ols('Revenue ~ C(Design)*C(Traffic)*C(Device)', data=df).fit()
print(anova_lm(model, typ=2))
```

---

## Interview Question Bank — Chapter 20

**Conceptual:**
1. In plain English, what does a significant three-way interaction actually mean?
2. Why does the hierarchical principle say you should keep lower-order terms even if they're not individually significant, once a higher-order term involving them is retained?
3. Why do higher-order interactions generally require larger sample sizes to detect reliably than main effects?

**Derivation:**
4. Write out the bracket formula for the three-way interaction term and explain, term by term, what each piece is removing.
5. Show how the degrees of freedom for a three-way interaction grow as the number of levels per factor increases, and connect this to statistical power.

**ML/Statistics:**
6. What's the practical risk of pooling a non-significant higher-order interaction into the error term to gain power for other tests?
7. How does this chapter's logic apply directly to multivariate (MVT) experimentation in industry, and what's the tradeoff versus running sequential single-factor A/B tests?
8. Connect the hierarchical principle to an analogous convention in polynomial regression (Chapters 8/13).

**Coding:**
9. Implement the full three-way ANOVA sum-of-squares decomposition from scratch in NumPy for a balanced factorial design.
10. Simulate a dataset with a genuine three-way interaction and verify your from-scratch computation detects it, then simulate one without and verify it correctly comes back null.

**Traps:**
11. "None of the interaction terms were significant, so I'll just report the three main effects as if they act completely independently." — under what circumstance is this actually the fully correct conclusion, versus potentially masking an underpowered but real interaction?
12. "A 2x2x2 factorial with 16 total observations gives us just as much power to detect the three-way interaction as it does the main effects." — what's wrong with this claim?
13. Someone runs an MVT test with 4 simultaneous factors and reports a marginally significant four-way interaction. How much skepticism is warranted, and why?

---

*This file covers Kutner Ch. 20 — the three-factor ANOVA model and its full sum-of-squares decomposition (worked entirely by hand on a clean 2×2×2 factorial with zero true interaction at any order), the practical and statistical challenges of higher-order interactions (interpretability and detection power both degrading with order), the hierarchical principle, the pooling strategy for non-significant higher-order terms, and the direct connection to multivariate (MVT) testing in industry. Chapter 21 (Randomized Complete Block Designs) is next if you'd like to continue — formalizing the blocking preview from Chapter 15 and the block/factor duality already flagged at the end of Chapter 19.*
