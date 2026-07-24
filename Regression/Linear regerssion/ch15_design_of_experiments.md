# Chapter 15 — Introduction to the Design of Experimental and Observational Studies
### (Kutner, Nachtsheim, Neter, Li — *Applied Linear Statistical Models*)

Chapters 1–14 covered the *modeling* side of statistics — given data, how to fit and interpret a relationship. This chapter shifts to the *design* side: **how the data got collected in the first place determines what conclusions you're entitled to draw** — most importantly, whether you can say anything **causal**, or only associational. This is Part III of Kutner's book, the bridge into ANOVA and experimental design, and it's directly the statistical backbone of A/B testing.

---

## 15.1 Experimental vs. Observational Studies, and the Causation Question

### The Core Distinction

**Plain English.** In an **experimental study**, the researcher actively **assigns** units to different treatment conditions (ideally at random). In an **observational study**, units arrive with their treatment/exposure already determined by nature, choice, or circumstance — the researcher only *observes* what happened.

**Why this distinction, more than sample size or model sophistication, determines whether you can claim causation.** Chapter 6 already flagged this: a regression coefficient captures association, not automatically causation, because any observed X might be correlated with unmeasured confounders that are the *real* drivers of Y. **Random assignment breaks this link by construction** — if treatment is assigned by a coin flip, it cannot be correlated (in expectation) with anything else about the unit, known or unknown, measured or unmeasured. This is the single most powerful design tool in all of statistics for licensing causal claims, and it's *why* experiments are prized above even very large, very carefully-modeled observational datasets when causal conclusions matter.

**Interview question:** *"Why can a small, well-randomized experiment sometimes give more trustworthy causal conclusions than a massive observational dataset with sophisticated controls?"*
**Ideal answer:** Regression with controls (Chapter 6-7) can only adjust for confounders you've actually measured and included — any unmeasured confounder correlated with both the "treatment" and the outcome will bias the estimated effect, no matter how large the dataset or how many covariates you control for. Randomization breaks the treatment's correlation with *every* other variable, measured or not, by construction — so even a modest-sized randomized experiment can support a causal claim that no amount of clever adjustment in observational data can fully guarantee.

---

## 15.2 Key Vocabulary for Experimental Design

- **Experimental unit**: the entity treatment is applied to and measured on (a user, a store, a patient).
- **Factor**: the variable being manipulated (e.g., "webpage design").
- **Factor levels / treatments**: the specific values or versions of the factor being compared (e.g., "Design A," "Design B," "Design C").
- **Response variable**: the outcome being measured (e.g., revenue per visitor).
- **Replication**: having multiple experimental units per treatment level (essential for estimating variability and computing standard errors — a single unit per treatment gives you a point estimate with no way to gauge its precision).
- **Randomization**: the random assignment of units to treatment levels — the mechanism that licenses causal interpretation, as established above.
- **Blocking**: grouping experimental units into similar subsets (blocks) *before* randomizing treatment within each block — a technique for soaking up known sources of variability, previewed at the end of this chapter and developed fully in later chapters (Randomized Complete Block Designs).

---

## 15.3 The Completely Randomized Design (CRD) — and Its Direct Identity with Chapter 8's Regression

### The Simplest Experimental Design

**Plain English.** In a CRD, experimental units are assigned to treatment levels **completely at random**, with no blocking or stratification. It's the experimental-design analog of "no additional structure" — the baseline design against which more sophisticated designs (blocking, factorial designs) are compared in later chapters.

### The Crucial Connection: One-Way ANOVA *Is* Chapter 8's Indicator-Variable Regression

**This is the single most important conceptual bridge in this chapter.** A completely randomized design with one factor at several levels is analyzed via **one-way ANOVA** — which is not a separate technique from regression, but **exactly** Chapter 8's qualitative-predictor regression model, generalized from 2 groups to $g$ groups using $g-1$ indicator variables.

### Worked Example: Three Webpage Designs, A Completely Randomized Design

15 visitors are **randomly assigned**, 5 each, to one of three webpage designs (A, B, C). Response: revenue per visitor ($).

| Design A | Design B | Design C |
|---|---|---|
| 12 | 16 | 20 |
| 14 | 18 | 22 |
| 11 | 15 | 19 |
| 13 | 17 | 21 |
| 15 | 19 | 23 |

$\bar Y_A=13,\ \bar Y_B=17,\ \bar Y_C=21,\ \bar Y_{\text{grand}}=17$ (balanced design, so the grand mean is just the average of the group means: $\frac{13+17+21}{3}=17$).

**Set up exactly as Chapter 8's indicator-variable regression**, with Design A as the reference category:
$$
Y_i = \beta_0+\beta_1D_{B,i}+\beta_2D_{C,i}+\varepsilon_i
$$

Since this design is **balanced** (equal group sizes) and each group's fitted value under this model is simply its own group mean:
$$
b_0=\bar Y_A=13,\quad b_1=\bar Y_B-\bar Y_A=17-13=4,\quad b_2=\bar Y_C-\bar Y_A=21-13=8
$$

**Residuals** (each observation minus its own group's mean) — by construction identical in pattern across all three groups here: $-1, 1, -2, 0, 2$ within each group.
$$
SSE = 3\times[(-1)^2+1^2+(-2)^2+0^2+2^2] = 3\times10=30, \qquad df_{error}=n-p=15-3=12, \qquad MSE=2.5
$$

**The "between-groups" sum of squares** — in ANOVA vocabulary, $SSTR$ (Treatment SS) — is **exactly Chapter 6-8's $SSR$** (regression sum of squares explained by the group indicators):
$$
SSTR = \sum_j n_j(\bar Y_j-\bar Y_{\text{grand}})^2 = 5(13-17)^2+5(17-17)^2+5(21-17)^2=80+0+80=160
$$
$$
SSTO = SSTR+SSE = 160+30=190
$$

**The overall F-test** — in ANOVA vocabulary, testing $H_0:$ all treatment means are equal — is **exactly Chapter 6's overall F-test** for whether the group-indicator predictors jointly explain significant variance:
$$
df_{\text{treatment}}=g-1=2, \qquad MSTR=\frac{160}{2}=80
$$
$$
F^* = \frac{MSTR}{MSE}=\frac{80}{2.5}=32
$$
Critical value: $F_{(0.95;2,12)}=3.89$. Since $32\gg3.89$, **strongly reject $H_0$: the three designs produce significantly different mean revenue** — with the causal license to say so specifically *because* assignment was randomized.

**Why this identity matters enormously for an interview.** If asked "what's the difference between ANOVA and regression," the strongest possible answer is: **there isn't one, mathematically** — one-way ANOVA is regression with categorical-indicator predictors, using the exact same least-squares machinery, the exact same F-test logic (Chapter 6-7's extra sums of squares), and the exact same underlying assumptions (Chapter 1's four assumptions on the errors). ANOVA is simply the traditional vocabulary/notation used when all predictors are categorical, largely for historical reasons predating widespread matrix-based regression computation.

**Interview question:** *"Is ANOVA a fundamentally different statistical technique from linear regression?"*
**Ideal answer:** No — one-way ANOVA is mathematically identical to a regression model with $g-1$ indicator variables representing $g$ treatment groups (Chapter 8's framework, generalized beyond two groups). The "treatment sum of squares" is exactly the regression sum of squares from the group indicators, and the ANOVA F-test is exactly the overall F-test for whether those indicators jointly explain significant variance. The distinct vocabulary and table format persist for historical and pedagogical reasons (ANOVA predates widespread matrix-based regression computing and is often taught to audiences without a regression background), but the underlying mathematics, assumptions, and estimation procedure are the same.

---

## 15.4 Why Replication Matters: A Direct Echo of Chapter 2

**Plain English.** Just as Chapter 2 showed $\text{Var}(b_1)=\sigma^2/S_{XX}$ shrinks as X-values spread out or sample size grows, the precision of an estimated **treatment mean** in a CRD is:
$$
\text{Var}(\bar Y_j) = \frac{\sigma^2}{n_j}
$$
**More replicates per treatment group directly reduces the uncertainty in that group's estimated mean** — the exact same logic underlying **statistical power** in A/B testing: to reliably detect a real difference between treatment means, you need enough replicates (users) per arm that the *signal* (the true difference between means) is large relative to the *noise* (each mean's own standard error, $\sigma/\sqrt{n_j}$). This is precisely why every A/B test requires a **power analysis** before launch — using an assumed effect size and variance to compute the minimum sample size needed per arm to detect that effect reliably, exactly generalizing Chapter 2's variance formula into a sample-size-planning tool.

---

## Preview: Blocking (Developed Fully in Later Chapters)

**Plain English.** If there's a known **nuisance factor** — something that affects the response but isn't the factor you actually care about testing (e.g., day-of-week effects on revenue, or traffic-source differences) — **blocking** groups experimental units by that nuisance factor first, then randomizes treatment **within** each block.

**Why this helps, connecting directly to Chapter 6's partial-regression logic.** Blocking is mechanically similar to adding a covariate to a regression model (Chapter 6-7): it soaks up variability that would otherwise land in $SSE$, shrinking $MSE$ and thereby increasing the power of the treatment-effect test — exactly the same mechanism by which adding a genuinely explanatory covariate reduces $SSE$ and tightens standard errors in ordinary regression.

**A/B testing connection:** this is precisely the logic behind **stratified randomization** (e.g., randomizing separately within each country, or each user-tenure bucket) and **CUPED**-style variance reduction techniques widely used in industry experimentation platforms — both are direct applications of the blocking principle, reducing unexplained variance to increase the test's sensitivity without needing a larger sample.

---

## Practical A/B Testing Pitfalls (Connecting Design Theory to Real Practice)

Since this chapter's framework is the direct statistical foundation of industry A/B testing, it's worth naming the standard real-world violations and complications an interviewer will expect you to recognize:

1. **SUTVA violations (Stable Unit Treatment Value Assumption).** The entire CRD framework implicitly assumes one unit's treatment assignment doesn't affect another unit's outcome. In social networks, marketplaces, or any system with **interference/network effects** (e.g., a referral feature, a two-sided marketplace where showing more of product A to some users reduces its availability for others), this assumption is directly violated — the observed "treatment effect" can be badly biased, and specialized designs (cluster randomization, switchback experiments) are needed instead of a naive CRD.

2. **Multiple testing across many metrics** — directly Chapter 4's Bonferroni problem, now at the scale of a typical A/B test dashboard tracking dozens of metrics simultaneously. Reporting "significant" on whichever metric happened to cross $p<0.05$ without correction inflates the true false-positive rate substantially.

3. **The "peeking" problem** — repeatedly checking a test's significance as data accumulates and stopping as soon as it crosses significance inflates the false-positive rate far beyond the nominal $\alpha$, since you're implicitly running many sequential hypothesis tests without correction. Proper practice uses either a fixed, pre-registered sample size (classical approach, directly following this chapter's power-analysis logic) or formal **sequential testing** methods (e.g., always-valid p-values, sequential probability ratio tests) designed to allow legitimate early stopping.

4. **Novelty and primacy effects** — an initial spike (or dip) in a treatment's effect purely because it's *new*, which fades over time — a threat to the assumption that the measured effect during the experiment window reflects the long-run steady-state effect.

**Interview question:** *"What does SUTVA mean, and why might it fail in a marketplace or social-network product?"*
**Ideal answer:** SUTVA (Stable Unit Treatment Value Assumption) requires that one unit's assigned treatment doesn't affect another unit's potential outcomes — implicitly assumed by the standard completely-randomized-design framework. In a marketplace, treating some users with a feature that changes their search/booking behavior can change the availability or pricing seen by *other* users (treated or not) — meaning the control group's outcomes are contaminated by the treatment group's behavior. This violates SUTVA and can badly bias a naively-computed treatment effect; standard remedies include cluster-level randomization (randomizing entire markets or regions rather than individual users) or specially designed switchback experiments.

---

## Python Implementation — From Scratch (Showing the ANOVA/Regression Identity Directly)

```python
import numpy as np
from scipy import stats

# Data: three treatment groups (webpage designs)
A = np.array([12,14,11,13,15], dtype=float)
B = np.array([16,18,15,17,19], dtype=float)
C = np.array([20,22,19,21,23], dtype=float)

Y = np.concatenate([A,B,C])
n = len(Y)
groups = ['A']*5 + ['B']*5 + ['C']*5

# --- Classical one-way ANOVA computation ---
grand_mean = Y.mean()
SSTR = 5*(A.mean()-grand_mean)**2 + 5*(B.mean()-grand_mean)**2 + 5*(C.mean()-grand_mean)**2
SSE = np.sum((A-A.mean())**2) + np.sum((B-B.mean())**2) + np.sum((C-C.mean())**2)
SSTO = SSTR + SSE
df_tr, df_err = 2, n-3
MSTR, MSE = SSTR/df_tr, SSE/df_err
F_stat = MSTR/MSE
p_value = 1 - stats.f.cdf(F_stat, df_tr, df_err)
print(f"SSTR={SSTR}, SSE={SSE}, F={F_stat}, p={p_value:.6f}")

# --- Same result via indicator-variable regression (Chapter 8 machinery) ---
D_B = np.array([0]*5 + [1]*5 + [0]*5, dtype=float)
D_C = np.array([0]*5 + [0]*5 + [1]*5, dtype=float)
X = np.column_stack([np.ones(n), D_B, D_C])
b = np.linalg.inv(X.T@X) @ X.T @ Y
resid = Y - X@b
SSE_reg = np.sum(resid**2)
print("Regression coefficients [b0, b1(B-A), b2(C-A)]:", b)
print("SSE via regression (should match ANOVA SSE):", SSE_reg)

# --- scipy/statsmodels built-ins for cross-check ---
from scipy.stats import f_oneway
f_stat_scipy, p_scipy = f_oneway(A, B, C)
print(f"scipy f_oneway: F={f_stat_scipy}, p={p_scipy:.6f}")
```

```python
# --- A/B test sample size / power calculation (direct application of Var(Ybar_j)=sigma^2/n_j) ---
from scipy.stats import norm

def required_sample_size(sigma, min_detectable_effect, alpha=0.05, power=0.8):
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)
    n_per_arm = 2 * ((z_alpha+z_beta)*sigma/min_detectable_effect)**2
    return int(np.ceil(n_per_arm))

n_needed = required_sample_size(sigma=5.0, min_detectable_effect=2.0)
print(f"Required sample size per arm: {n_needed}")
```

---

## Interview Question Bank — Chapter 15

**Conceptual:**
1. Why does random assignment license causal claims in a way that no amount of regression adjustment on observational data can fully replicate?
2. In what precise sense is one-way ANOVA "the same thing" as Chapter 8's indicator-variable regression?
3. What is blocking, and how is it mechanically similar to adding a covariate in regression?

**Derivation:**
4. Show that the ANOVA "treatment sum of squares" ($SSTR$) is exactly the regression sum of squares ($SSR$) from a model with group-indicator predictors.
5. Derive why the variance of a treatment group's sample mean shrinks as replication ($n_j$) increases, and connect this to A/B test sample size planning.

**ML/Statistics:**
6. What is SUTVA, and describe a realistic product scenario where it's violated.
7. Why does "peeking" at an A/B test's significance repeatedly, without correction, inflate the false-positive rate? Name a remedy.
8. How would you explain to a product manager why a large observational dataset showing a correlation isn't sufficient to justify shipping a feature change, when a proper A/B test would be?

**Coding:**
9. Implement one-way ANOVA from scratch via indicator-variable regression, and verify the F-statistic and SSE match a classical ANOVA calculation.
10. Implement a basic sample-size/power calculator for a two-arm A/B test given an assumed effect size and variance.

**Traps:**
11. "Our observational data shows users who use Feature X convert at twice the rate of those who don't, so shipping Feature X to everyone will double conversion." — what's missing to justify this causal claim?
12. "ANOVA and regression give different p-values for the same data, so they must be testing different things." — what's the likely explanation if this were actually observed (hint: check the model specifications match exactly)?
13. A team ships a new feature, monitors 15 different metrics, and highlights the one that hit p<0.05 as "the win." What statistical principle from earlier chapters does this violate, and what's the fix?

---

*This file covers Kutner Ch. 15 — the causal/observational distinction and why randomization uniquely licenses causal claims, core experimental design vocabulary, the completely randomized design worked in full and shown to be mathematically identical to Chapter 8's indicator-variable regression (a genuinely unifying result), the replication/power connection back to Chapter 2's variance formulas, a preview of blocking, and the standard real-world A/B testing pitfalls (SUTVA violations, multiple testing, peeking, novelty effects) that connect this classical design theory directly to modern industry experimentation practice. This begins Part III of Kutner's book (Design of Experimental and Observational Studies) — Chapter 16 (Single-Factor Studies, covering ANOVA model diagnostics and remedial measures) and Chapter 17 (Analysis of Factor Level Means, covering multiple comparison procedures like Tukey's HSD) continue this arc if you'd like to proceed.*
