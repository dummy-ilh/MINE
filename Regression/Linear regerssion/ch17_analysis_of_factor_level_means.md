# Chapter 17 — Analysis of Factor Level Means
### (Kutner, Nachtsheim, Neter, Li — *Applied Linear Statistical Models*)

Chapters 15–16 established that the three webpage designs differ significantly overall ($F^*=32$). That omnibus result answers only "do the groups differ at all?" — it says nothing about **which specific pairs differ**, or whether some more complex comparison (e.g., "A vs. the average of B and C") is meaningful. This chapter is entirely about answering *that* more specific, more useful question — and it's a direct, explicit continuation of Chapter 4's simultaneous inference machinery, now applied to factor-level means.

We continue the same dataset: $\bar Y_A=13,\ \bar Y_B=17,\ \bar Y_C=21$, $MSE=2.5$, $n_j=5$ each, $df_{error}=12$.

---

## 17.1–17.2 Contrasts: Generalizing "Pairwise Difference" to Any Linear Comparison

### Definition

A **contrast** is any linear combination of factor level means whose coefficients sum to zero:
$$
L = \sum_i c_i\mu_i, \qquad \sum_i c_i = 0
$$

**Why the zero-sum constraint specifically.** It guarantees $L$ measures a genuine *comparison* between groups — a pure "difference" concept, not contaminated by the overall average level. A simple pairwise difference $\mu_A-\mu_B$ is the contrast $c=(1,-1,0)$; but contrasts are far more general — e.g., "is Design A different from the *average* of the two redesigns B and C?" is the contrast $c=(1,-0.5,-0.5)$.

### Worked Example: A Non-Pairwise Contrast

$$
L = \mu_A - \frac{\mu_B+\mu_C}{2}, \qquad c=(1,-0.5,-0.5)
$$
$$
\hat L = 13-\frac{17+21}{2}=13-19=-6
$$

**Variance of the estimated contrast:**
$$
\text{Var}(\hat L) = MSE\sum_i\frac{c_i^2}{n_i} = 2.5\left(\frac{1}{5}+\frac{0.25}{5}+\frac{0.25}{5}\right)=2.5(0.3)=0.75
$$
$$
s(\hat L)=\sqrt{0.75}=0.866
$$

**If this comparison was planned *before* looking at the data** (a single, pre-registered hypothesis), an ordinary t-test suffices, exactly as in Chapter 2:
$$
t^* = \frac{\hat L}{s(\hat L)}=\frac{-6}{0.866}=-6.93, \qquad df=12
$$
Since $|-6.93|\gg t_{(0.975,12)}=2.179$, **overwhelming evidence that Design A underperforms the average of the two redesigns.**

**But if this specific contrast was chosen *after* looking at the data** (e.g., you noticed A looked worst and *then* decided to compare it to the average of the others), a plain t-test is no longer valid on its own — you've implicitly searched over many possible contrasts and reported the most extreme-looking one, exactly Chapter 9's stepwise-selection p-value warning, now in the ANOVA context. This is precisely why the multiple-comparison procedures below exist.

---

## 17.3 Tukey's Honestly Significant Difference (HSD) — Built Specifically for All Pairwise Comparisons

### The Idea

**Plain English.** If you want to compare **every pair** of $k$ group means simultaneously, with a guaranteed overall (family-wise) confidence level, Tukey's HSD uses the **studentized range distribution** — specifically calibrated for exactly this "compare all pairs" scenario, making it the *tightest* (most powerful) valid procedure for that specific job.

$$
\text{Critical difference} = q_{(1-\alpha;\,k,\,df_{error})}\times\sqrt{\frac{MSE}{n}} \quad \text{(equal } n_j\text{ case)}
$$

where $q$ is the studentized range critical value (from its own table, a generalization of the t-distribution designed specifically for the maximum of several pairwise comparisons).

### Worked Example: All Three Pairwise Comparisons

$$
q_{(0.95;\,3,\,12)}\approx3.77 \quad(\text{standard published table value})
$$
$$
SE=\sqrt{\frac{MSE}{n}}=\sqrt{\frac{2.5}{5}}=\sqrt{0.5}=0.7071
$$
$$
\text{Critical difference} = 3.77(0.7071)=2.666
$$

**Compare to the observed differences:**
$$
|\bar Y_A-\bar Y_B|=4,\quad |\bar Y_A-\bar Y_C|=8,\quad |\bar Y_B-\bar Y_C|=4
$$
**All three exceed $2.666$ — every pairwise comparison is significant** at family-wise $\alpha=0.05$: Design C > Design B > Design A, each pair distinguishable.

---

## 17.4 The Scheffé Method — Valid for ANY Contrast, Planned or Not

### Why a More Conservative Procedure Is Sometimes Necessary

**Plain English.** Tukey's HSD is calibrated *specifically* for simple pairwise comparisons. If you want protection for **any conceivable contrast** — including complex ones like "A vs. the average of B and C" that you might dream up *after* seeing the data — you need a procedure valid across the entire (infinite) space of possible contrasts. Scheffé's method provides exactly this, at the cost of being more conservative (wider intervals) than Tukey's for the specific case of simple pairwise comparisons.

$$
S = \sqrt{(k-1)F_{(1-\alpha;\,k-1,\,df_{error})}}, \qquad \text{Critical margin for contrast } L = S\times s(\hat L)
$$

### Worked Example: Scheffé Applied to a Pairwise Comparison

$$
S = \sqrt{2\times F_{(0.95;2,12)}}=\sqrt{2(3.89)}=\sqrt{7.78}=2.789
$$
For a simple pairwise contrast, $s(\hat L)=\sqrt{MSE(1/n_i+1/n_j)}=\sqrt{2.5(0.4)}=\sqrt{1.0}=1.0$:
$$
\text{Critical margin} = 2.789(1.0)=2.789
$$

**Direct comparison to Tukey's margin (2.666) for the same pairwise comparisons: Scheffé's margin is wider.** This is the expected, textbook relationship — **for pure pairwise comparisons specifically, Tukey HSD is always at least as tight as Scheffé** (since Scheffé is "paying" for validity across an unlimited range of possible contrasts, a generality you don't need if you only ever intend to look at pairs).

### Worked Example: Scheffé Applied to the Non-Pairwise Contrast

Recall $\hat L=-6$, $s(\hat L)=0.866$ for the "A vs. average of B,C" contrast:
$$
\text{Critical margin} = 2.789(0.866)=2.415
$$
Since $|\hat L|=6\gg2.415$, **still overwhelmingly significant, even under Scheffé's full protection against having searched over any possible contrast.** This is the case where Scheffé's extra conservatism is actually *necessary* — Tukey's HSD is not built to cover this kind of comparison at all, so it wouldn't even apply here.

**Interview question:** *"When would you use Scheffé's method instead of Tukey's HSD, given that Scheffé gives wider, less powerful intervals?"*
**Ideal answer:** Tukey's HSD is specifically calibrated for, and optimal for, the scenario where you want simultaneous confidence across *all pairwise* comparisons among the groups. If you instead want to test a more complex contrast — especially one you didn't specify in advance, like a comparison discovered by eyeballing the data — Tukey's method doesn't provide valid protection for that; Scheffé's method remains valid for literally *any* linear contrast you might construct, planned or not, which is exactly the protection you need when the comparison of interest wasn't decided before looking at the results.

---

## 17.5 The Bonferroni Approach, Revisited from Chapter 4

**Directly Chapter 4's machinery**, applied to a fixed, pre-specified list of $g$ comparisons: use $t_{(1-\alpha/(2g);\,df_{error})}$ instead of the ordinary $t_{(1-\alpha/2)}$.

### Worked Example: Bonferroni for the Same 3 Pairwise Comparisons

$$
g=3, \qquad t_{(1-0.05/6;\,12)} = t_{(0.99167;\,12)}\approx2.806 \ (\text{interpolated from standard t-tables})
$$
$$
\text{Critical margin} = 2.806(1.0)=2.806
$$

### Comparing All Three Procedures Side by Side

| Method | Critical margin (for a pairwise comparison) |
|---|---|
| Tukey HSD | 2.666 |
| Scheffé | 2.789 |
| Bonferroni | 2.806 |

**The ordering here is the standard, expected pattern**: Tukey HSD is tightest specifically because it's purpose-built for exactly this job (all pairwise comparisons among $k$ groups); Scheffé is a bit wider since it protects against *any* contrast; Bonferroni, applied to exactly these 3 pre-specified pairwise tests, comes out slightly wider still in this particular case (the relative ordering of Scheffé vs. Bonferroni isn't universal — it depends on the specific number of comparisons $g$ relative to $k$ — but Tukey being tightest for pure pairwise comparisons *is* a reliable, general pattern worth remembering).

**Interview question:** *"You need to compare all pairs among 5 treatment groups. Which multiple-comparison procedure would you reach for by default, and why?"*
**Ideal answer:** Tukey's HSD, since it's specifically calibrated for the all-pairwise-comparisons scenario and is provably the tightest (most powerful) valid procedure for exactly that job — using Scheffé or Bonferroni instead would give up power unnecessarily, since neither is optimized for this specific, common comparison pattern. Scheffé should be reserved for when you need validity for arbitrary, possibly post-hoc contrasts beyond simple pairs; Bonferroni is simplest and most transparent when you have a small, fixed, pre-planned set of specific comparisons (not necessarily all pairs) and don't need Tukey's specialized calibration.

---

## Practical Guidance: Choosing a Procedure

1. **A small number of comparisons decided *before* seeing the data** → plain t-tests (or Bonferroni if there are a few of them and you want formal joint protection) — Chapter 4's logic directly.
2. **All pairwise comparisons among $k$ groups, decided in advance that you want every pair** → Tukey's HSD, since it's purpose-built and most powerful for exactly this.
3. **Any complex or data-driven/post-hoc contrast, including "I noticed this pattern after looking at the results"** → Scheffé, since it's the only procedure among these three that remains valid no matter which (or how many) contrasts you end up examining.

**A final, important honesty point for interviews:** none of these procedures are "better" in some absolute sense — each is calibrated for a specific *scope* of comparisons, and using a narrower-scope procedure (Tukey) for a broader-scope problem (arbitrary post-hoc contrasts) would silently invalidate your stated confidence level, exactly the multiple-comparisons trap from Chapter 4 and Chapter 9 recurring in yet another form.

---

## Python Implementation

```python
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

A = np.array([12,14,11,13,15], dtype=float)
B = np.array([16,18,15,17,19], dtype=float)
C = np.array([20,22,19,21,23], dtype=float)

data = np.concatenate([A,B,C])
labels = ['A']*5+['B']*5+['C']*5

# --- Tukey's HSD (built-in) ---
tukey_result = pairwise_tukeyhsd(data, labels, alpha=0.05)
print(tukey_result)

# --- Manual Scheffé method for a general contrast ---
MSE = 2.5
n = 5
df_error = 12
k = 3

def scheffe_test(means, coeffs, n_per_group, MSE, k, df_error, alpha=0.05):
    L_hat = sum(c*m for c, m in zip(coeffs, means))
    var_L = MSE * sum(c**2/n_per_group for c in coeffs)
    se_L = np.sqrt(var_L)
    S = np.sqrt((k-1) * stats.f.ppf(1-alpha, k-1, df_error))
    margin = S * se_L
    return L_hat, se_L, margin, (L_hat-margin, L_hat+margin)

means = {'A':13, 'B':17, 'C':21}
# Contrast: A vs average(B,C)
L, se, margin, ci = scheffe_test([13,17,21], [1,-0.5,-0.5], n, MSE, k, df_error)
print(f"Contrast L={L}, SE={se:.4f}, Scheffe margin={margin:.4f}, CI={ci}")

# --- Bonferroni-adjusted pairwise t-tests ---
from itertools import combinations
groups = {'A':A, 'B':B, 'C':C}
g = 3  # number of pairwise comparisons
alpha = 0.05
t_crit = stats.t.ppf(1 - alpha/(2*g), df_error)
for (name1, g1), (name2, g2) in combinations(groups.items(), 2):
    diff = g1.mean() - g2.mean()
    se = np.sqrt(MSE*(1/n + 1/n))
    margin = t_crit * se
    print(f"{name1} vs {name2}: diff={diff}, Bonferroni margin={margin:.4f}, significant={abs(diff)>margin}")
```

---

## Interview Question Bank — Chapter 17

**Conceptual:**
1. What is a contrast, and why must its coefficients sum to zero?
2. Why is Tukey's HSD the "tightest" valid procedure specifically for all-pairwise comparisons, but not for arbitrary contrasts?
3. What's the practical risk of using Tukey's HSD for a contrast that was chosen after looking at the data?

**Derivation:**
4. Derive the variance of a general contrast estimator $\hat L=\sum c_i\bar Y_i$ from the assumption that group means are independent.
5. Explain why Scheffé's critical value uses the $F_{(k-1,df_{error})}$ distribution rather than a simple t-distribution.

**ML/Statistics:**
6. Compare Tukey, Scheffé, and Bonferroni in terms of what scope of comparisons each protects, and how you'd choose among them for a specific analysis.
7. How does the logic of these multiple-comparison procedures relate to feature-selection multiple-testing concerns from Chapter 9?
8. In an A/B test with 5 variants, how would you correctly compare all pairs after a significant omnibus test, and why not just run 10 separate unadjusted t-tests?

**Coding:**
9. Implement Tukey's HSD, Scheffé, and Bonferroni-adjusted pairwise comparisons from scratch, and verify against `statsmodels`' `pairwise_tukeyhsd`.
10. Given a significant omnibus ANOVA F-test, write code that reports which specific pairs of groups differ significantly using an appropriately chosen procedure.

**Traps:**
11. "I found this interesting comparison by exploring the data, then confirmed it with a t-test at α=0.05, so it's valid." — what's wrong, and what should have been used instead?
12. "Bonferroni is always the most conservative (widest) procedure." — is this always true relative to Scheffé and Tukey? What does our worked example show?
13. Someone runs Tukey's HSD for all pairwise comparisons, and separately wants to test whether the average of two groups differs from a third. Can they reuse the same Tukey critical value for that new comparison?

---

*This file covers Kutner Ch. 17 — contrasts as the general framework for comparing factor level means, and the three major multiple-comparison procedures (Tukey's HSD, Scheffé, and Bonferroni) worked side-by-side on the same data with their critical values directly compared, plus practical guidance on choosing among them. This completes Kutner's Part III (Design of Experimental and Observational Studies, Chapters 15-17) and, combined with Chapters 1-14, gives a comprehensive foundation spanning simple and multiple regression, GLMs, and single-factor experimental design — the great majority of applied statistics tested in FAANG-level L5 ML/DS interviews. Kutner's book continues into Part IV (Multi-Factor Studies: two-way ANOVA, randomized block designs, and beyond) if you'd like to keep going, though the remaining material is progressively more specialized to experimental design and less central to typical ML interview content.*
