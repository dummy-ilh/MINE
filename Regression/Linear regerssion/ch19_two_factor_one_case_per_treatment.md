# Chapter 19 — Two-Factor Studies: One Case per Treatment
### (Kutner, Nachtsheim, Neter, Li — *Applied Linear Statistical Models*)

Chapter 18 assumed **replication** ($n>1$ per cell), which let us cleanly separate interaction ($SSAB$) from pure error ($SSE$). This chapter covers the common practical situation where **replication isn't feasible** — each factor combination is observed exactly **once** ($n=1$). This single change forces a real statistical compromise, and introduces the one specialized tool built to partially work around it: **Tukey's test for additivity**.

### The Worked Example

Two factors: **Machine** (4 levels, M1–M4) and **Operator** (3 levels, O1–O3), each combination run **exactly once** (a common real-world constraint — e.g., each machine/operator pairing is expensive or time-consuming to test, so replicating all 12 combinations isn't practical). Response: output quality score.

| | O1 | O2 | O3 | Row mean |
|---|---|---|---|---|
| **M1** | 15 | 17 | 19 | 17 |
| **M2** | 17 | 22 | 21 | 20 |
| **M3** | 19 | 21 | 23 | 21 |
| **M4** | 21 | 23 | 25 | 23 |
| **Col mean** | 18 | 20.75 | 22 | **Grand: 20.25** |

(Notice cell M2,O2 = 22 stands out somewhat from the otherwise smooth pattern the rest of the table follows — keep this in mind; it's the one place genuine non-additivity was deliberately built into this example.)

---

## 19.1 Why $n=1$ Per Cell Forces a Real Compromise

**Recall Chapter 18's decomposition:** $SSTO=SSA+SSB+SSAB+SSE$, with $df_{SSE}=ab(n-1)$. **With $n=1$, $df_{SSE}=ab(1-1)=0$ exactly.** There is **no possible way to separately estimate pure error** — every observation is the *only* observation in its cell, so there's nothing to compare it against to isolate "noise" from "this cell's true interaction effect." $SSAB$ and $SSE$ become **completely confounded**: any leftover variation, after removing the two main effects, could be genuine interaction, pure noise, or (almost always) some blend of both — and the data alone cannot tell you which.

**The practical resolution:** **assume no interaction** (the additive model), and treat the leftover variation — call it $SS_{\text{remainder}}$ — as if it were pure error, so you at least have *some* error term against which to test the two main effects. This is a real compromise, not a free lunch: **if genuine interaction is actually present, this remainder is contaminated, and the main-effect F-tests below are no longer strictly valid.** This is exactly why this chapter introduces a specific diagnostic (Section 19.3) to check the additivity assumption before trusting the result.

---

## 19.2 The Additive Model and Main-Effect F-Tests

$$
Y_{ij} = \mu+\alpha_i+\beta_j+\varepsilon_{ij} \qquad(\text{no interaction term — it's structurally unavailable})
$$

### Computing the Sums of Squares

$$
SSA = b\sum_i(\bar Y_{i.}-\bar Y_{..})^2 = 3\left[(-3.25)^2+(-0.25)^2+(0.75)^2+(2.75)^2\right]=3(18.75)=56.25
$$
$$
SSB = a\sum_j(\bar Y_{.j}-\bar Y_{..})^2 = 4\left[(-2.25)^2+(0.5)^2+(1.75)^2\right]=4(8.375)=33.5
$$
$$
SSTO = \sum_{ij}(Y_{ij}-\bar Y_{..})^2 = 94.25 \quad(\text{direct sum of all 12 squared deviations from } 20.25)
$$
$$
SS_{\text{remainder}} = SSTO-SSA-SSB = 94.25-56.25-33.5=4.5
$$
$$
df_{\text{remainder}} = (a-1)(b-1) = 3\times2=6 \quad(\text{this df would have been split between interaction and error, had } n>1)
$$
$$
MS_{\text{remainder}} = 4.5/6 = 0.75
$$

### The Main-Effect F-Tests

$$
F^*_{\text{Machine}} = \frac{SSA/(a-1)}{MS_{\text{remainder}}} = \frac{56.25/3}{0.75}=\frac{18.75}{0.75}=25.0
$$
Compared to $F_{(0.95;3,6)}=4.76$: since $25.0\gg4.76$, **Machine has a significant effect.**

$$
F^*_{\text{Operator}} = \frac{SSB/(b-1)}{MS_{\text{remainder}}}=\frac{33.5/2}{0.75}=\frac{16.75}{0.75}=22.33
$$
Compared to $F_{(0.95;2,6)}=5.14$: since $22.33\gg5.14$, **Operator has a significant effect.**

**Both conclusions are only as trustworthy as the additivity assumption underlying $MS_{\text{remainder}}$.** That's exactly what Section 19.3 checks.

---

## 19.3 Tukey's One-Degree-of-Freedom Test for Non-Additivity

### The Idea

**Plain English.** Since we can't test *general* interaction with only 1 df available in the remainder to spend, Tukey devised a test for **one specific, restrictive form** of non-additivity: interaction that is **proportional to the product of the row effect and column effect** — i.e., cells where both the row effect and column effect are large (in the same direction) show an extra boost (or an extra penalty) proportional to $\hat\alpha_i\times\hat\beta_j$. This is a real, common real-world pattern (sometimes called "multiplicative" non-additivity — e.g., a machine that's already fast combined with an operator who's already fast might get a disproportionate *extra* boost together, beyond what adding their separate effects would predict) — but it is **not the only possible kind of interaction**, a crucial limitation to keep in mind (Section 19.4).

### The Formula

Let $a_i=\bar Y_{i.}-\bar Y_{..}$ (row effects) and $b_j=\bar Y_{.j}-\bar Y_{..}$ (column effects). The test statistic:
$$
SS_N = \frac{\left[\sum_{ij}a_ib_jY_{ij}\right]^2}{\left(\sum_i a_i^2\right)\left(\sum_j b_j^2\right)}, \qquad df_N=1
$$

**Why raw $Y_{ij}$ can be used directly (a nice simplification worth understanding, not just accepting)**: because $\sum_ia_i=0$ and $\sum_jb_j=0$ exactly (both main effects are deviations from a mean, so they must sum to zero), every part of $Y_{ij}$'s decomposition that comes from the row mean, column mean, or grand mean alone contributes **exactly zero** to $\sum a_ib_jY_{ij}$ — leaving only the genuine residual (interaction-or-error) part to actually drive the statistic. This means you can plug in the raw data directly without first computing residuals — a computational convenience, not a different calculation.

### Worked Example

**Row effects:** $a_1=-3.25,\ a_2=-0.25,\ a_3=0.75,\ a_4=2.75$ (sum: $-3.25-0.25+0.75+2.75=0$ ✓).
**Column effects:** $b_1=-2.25,\ b_2=0.5,\ b_3=1.75$ (sum: $-2.25+0.5+1.75=0$ ✓).

$$
\sum_i a_i^2 = 18.75, \qquad \sum_j b_j^2 = 8.375
$$

**Computing $\sum_{ij}a_ib_jY_{ij}$** (each term is $a_i\times b_j\times Y_{ij}$, summed across all 12 cells — full arithmetic, row by row):

Row 1 ($a_1=-3.25$): $(7.3125)(15)+(-1.625)(17)+(-5.6875)(19) = 109.6875-27.625-108.0625=-26.0$
Row 2 ($a_2=-0.25$): $(0.5625)(17)+(-0.125)(22)+(-0.4375)(21)=9.5625-2.75-9.1875=-2.375$
Row 3 ($a_3=0.75$): $(-1.6875)(19)+(0.375)(21)+(1.3125)(23)=-32.0625+7.875+30.1875=6.0$
Row 4 ($a_4=2.75$): $(-6.1875)(21)+(1.375)(23)+(4.8125)(25)=-129.9375+31.625+120.3125=22.0$

$$
\sum_{ij}a_ib_jY_{ij} = -26.0-2.375+6.0+22.0=-0.375
$$

$$
SS_N = \frac{(-0.375)^2}{18.75\times8.375}=\frac{0.140625}{157.031}=0.000896
$$

**Splitting the remainder:**
$$
SS_{\text{pure error}} = SS_{\text{remainder}}-SS_N = 4.5-0.000896\approx4.499, \qquad df=6-1=5
$$
$$
MS_{\text{pure error}} = 4.499/5=0.8998
$$
$$
F^*_{\text{Tukey}} = \frac{SS_N/1}{MS_{\text{pure error}}}=\frac{0.000896}{0.8998}\approx0.001
$$

Compared to $F_{(0.95;1,5)}=6.61$: since $0.001\ll6.61$, **fail to reject $H_0$ — no significant evidence of this specific (multiplicative) form of non-additivity.**

### The Honest, Important Limitation This Result Reveals

**We deliberately built a genuine departure from additivity into this data** (the +3 bump at cell M2,O2, which is exactly why $SS_{\text{remainder}}=4.5$ is nonzero in the first place, rather than exactly 0 as it would be under perfect additivity). **Yet Tukey's test came back essentially null.** This is not a mistake — it's the test correctly doing exactly what it's built to do: **Tukey's test only detects non-additivity that specifically follows the multiplicative $a_i\times b_j$ pattern.** A single, idiosyncratic bump at one cell — which is a very different, more general kind of departure from additivity — doesn't happen to align with that specific structured pattern, so the test has no particular power to catch it.

**Interview question:** *"Tukey's test for additivity comes back non-significant. Does this mean you can be confident the additive model is appropriate?"*
**Ideal answer:** Only with an important caveat: Tukey's test is a single, one-degree-of-freedom test specifically calibrated to detect non-additivity that takes the form of an interaction proportional to the product of the row and column effects — not a fully general test for *any* possible departure from additivity. A dataset can fail to be genuinely additive in other ways (e.g., an isolated outlier cell, or a more complex nonlinear pattern) and still pass Tukey's test cleanly, since those departures don't match the specific multiplicative structure the test targets. A non-significant Tukey's test is reassuring evidence, not proof, and should be combined with residual plots and substantive judgment about whether interaction is plausible.

---

## 19.4 When You Suspect Other Forms of Interaction

If you have genuine reason to suspect non-multiplicative interaction (e.g., from domain knowledge, or a residual plot showing a suspicious localized pattern like our single-cell bump), **the only real fix is replication** — going back to Chapter 18's design with $n>1$ per cell, which lets $SSAB$ and $SSE$ be estimated separately and tested with a fully general interaction F-test, rather than relying on the restrictive one-df proxy this chapter provides as a partial substitute when replication isn't an option.

---

## 19.5 The Direct Connection to Randomized Block Designs

**This exact design — one factor of primary interest, one additional factor with exactly one observation per combination — is mathematically identical to a Randomized Complete Block Design (RCBD)**, just under different framing. If "Operator" here were reframed not as a factor you care about comparing in its own right, but as a **nuisance variable** you're controlling for (a "block") to reduce unexplained variability while studying Machine's effect, this table and every computation above would be **exactly** an RCBD analysis — same sums of squares, same F-test for the factor of interest, using the blocking factor's variation to soak up what would otherwise be unexplained error (directly the same blocking logic previewed in Chapter 15 and connected to ANCOVA's variance-reduction motivation). The distinction between "two-factor study, one case per treatment" and "randomized complete block design" is really about **intent** (are you interested in comparing both factors, or is one just a nuisance you're controlling for?), not about any difference in the underlying mathematics.

---

## Python Implementation

```python
import numpy as np
from scipy import stats

Y = np.array([
    [15, 17, 19],
    [17, 22, 21],
    [19, 21, 23],
    [21, 23, 25]
], dtype=float)
a, b = Y.shape  # a=4 machines, b=3 operators

row_means = Y.mean(axis=1)
col_means = Y.mean(axis=0)
grand_mean = Y.mean()

SSA = b * np.sum((row_means - grand_mean)**2)
SSB = a * np.sum((col_means - grand_mean)**2)
SSTO = np.sum((Y - grand_mean)**2)
SS_remainder = SSTO - SSA - SSB
df_remainder = (a-1)*(b-1)
MS_remainder = SS_remainder / df_remainder

F_A = (SSA/(a-1)) / MS_remainder
F_B = (SSB/(b-1)) / MS_remainder
print(f"SSA={SSA}, SSB={SSB}, SS_remainder={SS_remainder}")
print(f"F_Machine={F_A:.2f}, F_Operator={F_B:.2f}")

# --- Tukey's test for non-additivity ---
a_i = row_means - grand_mean
b_j = col_means - grand_mean
numerator = sum(a_i[i]*b_j[j]*Y[i,j] for i in range(a) for j in range(b))
SS_N = numerator**2 / (np.sum(a_i**2) * np.sum(b_j**2))
SS_pure_error = SS_remainder - SS_N
df_pure_error = df_remainder - 1
MS_pure_error = SS_pure_error / df_pure_error
F_tukey = SS_N / MS_pure_error

print(f"SS_N={SS_N:.5f}, F_Tukey={F_tukey:.5f}")
print(f"Critical F(0.95,1,{df_pure_error}):", stats.f.ppf(0.95, 1, df_pure_error))
```

---

## Interview Question Bank — Chapter 19

**Conceptual:**
1. Why does having exactly one observation per cell make it impossible to separately estimate interaction and pure error?
2. What specific, restrictive form of non-additivity does Tukey's test detect, and what forms can it miss?
3. In what sense is this chapter's design mathematically identical to a randomized complete block design?

**Derivation:**
4. Explain why $\sum_{ij}a_ib_jY_{ij}$ can be computed directly from raw data rather than residuals, using the fact that row and column effects each sum to zero.
5. Derive why $SS_N$ has exactly 1 degree of freedom.

**ML/Statistics:**
6. If you suspect a more general (non-multiplicative) form of interaction, what design change is required to test for it properly?
7. Why might a "nuisance factor" in an experiment be better thought of as a block rather than a factor of direct interest, and how does that framing (not the math) change how you'd report results?
8. Connect the compromise made in this chapter (assuming additivity due to lack of replication) to the bias-variance tradeoff ideas from ridge regression or random effects.

**Coding:**
9. Implement the full one-case-per-treatment ANOVA decomposition and Tukey's test for non-additivity from scratch in NumPy.
10. Simulate data with genuine multiplicative interaction and verify Tukey's test correctly detects it, then simulate data with a single-cell outlier and verify the test misses it.

**Traps:**
11. "Tukey's test for additivity wasn't significant, so I can be sure the additive model is correct." — what's the precise, more careful statement?
12. "With one observation per cell, you simply can't do a two-factor ANOVA at all." — what's the correct, more nuanced statement about what you give up versus what remains possible?
13. Someone treats their blocking factor as if it were a factor of direct scientific interest and reports pairwise comparisons among block levels. What's likely wrong with this approach, given the factor's actual role in the design?

---

*This file covers Kutner Ch. 19 — the two-factor, one-observation-per-cell design and the compromise it forces (assuming additivity to obtain any usable error term), the main-effect F-tests using the pooled remainder, Tukey's one-degree-of-freedom test for (multiplicative) non-additivity worked fully by hand, its important and often-overlooked limitation, and the direct mathematical identity between this design and Randomized Complete Block Designs — the next natural topic if you want to continue building out the design-of-experiments arc.*
