# Chapter 23 — Repeated Measures Designs
### (Kutner, Nachtsheim, Neter, Li — *Applied Linear Statistical Models*)

Chapter 21 blocked on a nuisance *factor* (city). This chapter blocks on something even more powerful: **the same experimental unit measuring itself** — each subject experiences **every** treatment condition, rather than being randomly assigned to just one. This is a **within-subjects** design, as opposed to every prior chapter's **between-subjects** design (Chapters 16–21, where each unit received exactly one treatment).

### The Worked Example

Six users each try **all three** webpage designs (A, B, C) — the same person testing every design, rather than being randomly split into three separate groups. Response: a conversion-related score.

| Subject | A | B | C | Row mean |
|---|---|---|---|---|
| S1 | 13 | 15 | 17 | 15 |
| S2 | 14 | 16 | 21 | 17 |
| S3 | 15 | 20 | 22 | 19 |
| S4 | 19 | 21 | 23 | 21 |
| S5 | 20 | 22 | 27 | 23 |
| S6 | 21 | 26 | 28 | 25 |
| **Col mean** | 17 | 20 | 23 | **Grand: 20** |

---

## 23.1 Within-Subjects vs. Between-Subjects: Why Repeat Measures At All

**Plain English.** In a between-subjects design (a standard CRD, Chapters 16-18), each person contributes data to exactly one treatment group — meaning any comparison between groups is contaminated by ordinary person-to-person variability (some people are just naturally faster, more engaged, etc., regardless of treatment). In a **repeated measures** design, **each person serves as their own control** — since the same person tries every design, their own personal baseline level cancels out of any within-person comparison between designs.

**Why this dramatically increases statistical power, and why the mechanism is identical to blocking.** This is **mathematically the exact same maneuver as Chapter 21's Randomized Complete Block Design** — just relabel "Block" as "Subject" and "Treatment" as "Design." Person-to-person variability, which would otherwise sit entirely inside the error term, gets explicitly subtracted out before comparing treatments, exactly as city-to-city variability was subtracted out in Chapter 21.

---

## 23.2 The Model and the Direct Identity to Chapter 21's RCBD

$$
Y_{ij} = \mu+\pi_i+\tau_j+\varepsilon_{ij}
$$

where $\pi_i$ is the subject effect and $\tau_j$ is the treatment effect — **structurally identical to Chapter 21's $Y_{ij}=\mu+\rho_i+\tau_j+\varepsilon_{ij}$**, with Subject playing the role Block played there.

### The Full ANOVA, Computed Exactly as in Chapter 21

$$
SSTO = \sum_{ij}(Y_{ij}-20)^2 = 330 \quad(\text{direct sum over all 18 observations})
$$
$$
SS(\text{Subjects}) = t\sum_i(\bar Y_{i.}-\bar Y_{..})^2 = 3\left[(-5)^2+(-3)^2+(-1)^2+1^2+3^2+5^2\right]=3(70)=210
$$
$$
SS(\text{Treatments}) = n\sum_j(\bar Y_{.j}-\bar Y_{..})^2 = 6\left[(-3)^2+0^2+3^2\right]=6(18)=108
$$
$$
SS(\text{remainder}) = 330-210-108=12
$$

| Source | SS | df | MS | F* |
|---|---|---|---|---|
| Subjects | 210 | $n-1=5$ | 42 | 35.0 |
| Treatment (Design) | 108 | $t-1=2$ | 54 | 45.0 |
| Remainder | 12 | $(n-1)(t-1)=10$ | 1.2 | |
| Total | 330 | $nt-1=17$ | | |

$$
F^*_{\text{Treatment}} = \frac{54}{1.2}=45.0 \quad(\text{vs. } F_{(0.95;2,10)}=4.10 \Rightarrow \text{highly significant})
$$

**The three designs genuinely differ, and this test is dramatically more powerful than a between-subjects comparison would have been** — exactly Chapter 21's relative-efficiency lesson, here delivered by having each person cancel out their own baseline rather than by grouping similar external units into blocks.

---

## 23.3 A Genuinely New Practical Concern: Order Effects and Counterbalancing

**Plain English.** Since each subject experiences *all* treatments in some sequence, a new problem appears that never arose in Chapter 21's blocking (where "blocks" were separate physical entities like cities, not a single subject moving through a sequence of conditions): **order effects**. If every subject tried Design A first, then B, then C, any improvement from A to C might just reflect the subject getting more practiced, more fatigued, or more familiar with the task — not a real difference between the designs at all.

**The standard remedy: counterbalancing** — systematically varying the order across subjects (e.g., some try A→B→C, others B→C→A, others C→A→B — a Latin-square-style rotation) so that any genuine order/practice effect gets spread evenly across all three designs and washes out of the treatment comparison, rather than systematically favoring whichever design happens to come later (or earlier) in the sequence for everyone.

**Interview question:** *"What's a risk specific to within-subject (repeated measures) designs that doesn't really arise in a standard between-subjects blocked design, and how do you address it?"*
**Ideal answer:** Order effects — since the same subject experiences every condition in some sequence, practice, fatigue, or carryover effects from earlier conditions can contaminate later ones, confounding the true treatment effect with a systematic order effect. The standard fix is counterbalancing: varying the order of conditions across subjects (e.g., via a Latin square) so any genuine order effect is distributed evenly across all treatments rather than systematically favoring one.

---

## 23.4 The Crucial New Assumption: Sphericity

### Why This Matters More Here Than in Chapter 21's Blocking

**Plain English.** The single pooled error term ($MS(\text{remainder})=1.2$ above) implicitly assumes that the variance of the *difference* between any two treatments is the same, no matter which pair you pick — e.g., the variability in (A−B) differences across subjects should be about the same as the variability in (A−C) or (B−C) differences. This is called **sphericity** (a specific form of it, **compound symmetry**, is the simplest sufficient case). **Why this deserves special attention specifically in repeated measures, more than in Chapter 21's block design:** when the *same person* provides multiple correlated measurements, there's a real, concrete mechanism for this assumption to fail — e.g., adjacent conditions in a sequence might be more similar to each other (carryover) than distant ones, or some pairs of conditions might be inherently easier to tell apart consistently than others. In Chapter 21's blocking, blocks were typically separate physical/spatial units without this specific same-person-correlation mechanism, so the analogous concern, while technically present, is usually far less pressing in practice.

### Checking It Directly: Compute the Variance of Every Pairwise Difference

**A−B differences** (per subject): $-2,-2,-5,-2,-2,-5$ → mean $-3$, sample variance $=12/5=2.4$.
**A−C differences:** $-4,-7,-7,-4,-7,-7$ → mean $-6$, sample variance $=12/5=2.4$.
**B−C differences:** $-2,-5,-2,-2,-5,-2$ → mean $-3$, sample variance $=12/5=2.4$.

**All three pairwise difference variances are exactly equal (2.4) in this data** — sphericity holds cleanly here (by deliberate construction, to demonstrate the concept unambiguously; real data is essentially never this tidy). Because sphericity holds, the single pooled F-test computed in Section 23.2 is fully valid as-is, with no correction needed.

### If Sphericity Had Failed

**Mauchly's test** formally tests $H_0:$ sphericity holds, using the full covariance matrix of the repeated measures. **If sphericity is violated, the standard F-test's Type I error rate becomes inflated** — you'll see "significant" results more often than your stated $\alpha$ actually promises, since the single pooled error term is no longer a fair, uniform yardstick across all the pairwise comparisons contributing to the omnibus test.

**Standard remedies:**
1. **Greenhouse-Geisser correction:** multiply both the numerator and denominator degrees of freedom by an estimated correction factor $\hat\varepsilon$ (between $1/(t-1)$ and 1), shrinking the effective degrees of freedom to compensate — a conservative fix.
2. **Huynh-Feldt correction:** a similar but slightly less conservative adjustment, often preferred when the true sphericity violation is believed to be mild.
3. **Multivariate approach (repeated-measures MANOVA):** sidesteps the sphericity assumption entirely by treating the repeated measures as a multivariate outcome vector (using Wilks' Lambda or Hotelling's $T^2$) — more robust, at the cost of requiring a larger sample for adequate power.

**Interview question:** *"What happens to your repeated-measures ANOVA's Type I error rate if sphericity is violated and you don't correct for it?"*
**Ideal answer:** It becomes inflated — you'll reject the null hypothesis more often than your nominal significance level suggests, because the single pooled error term used across all pairwise comparisons is no longer a uniformly fair estimate of variability for every comparison. The standard fixes are the Greenhouse-Geisser or Huynh-Feldt degrees-of-freedom corrections (which shrink the effective df to compensate) or switching to a multivariate (MANOVA-style) test that doesn't require the sphericity assumption at all.

---

## 23.5 The Modern Alternative: Mixed-Effects Models with Explicit Covariance Structures

**The most robust modern practice sidesteps the sphericity issue entirely** by fitting a mixed-effects model with an *explicitly specified* (or flexibly estimated) covariance structure for the repeated measurements — rather than assuming a specific restrictive structure (sphericity/compound symmetry) upfront and correcting for violations after the fact. Common choices include an **unstructured** covariance (estimate every pairwise correlation freely, most flexible but requires more data), an **autoregressive** structure (correlation decays with how far apart in sequence two measurements are — natural for genuine time-ordered repeated measures), or **compound symmetric** (the classical sphericity assumption, as a special, most-restrictive case). **This is exactly the same philosophical shift already seen in Chapter 22** — where a correctly specified mixed model automatically gets the error structure right, rather than requiring you to manually diagnose and correct for a violated classical assumption.

**Interview question:** *"Why might a modern analyst prefer a mixed-effects model with an explicit covariance structure over a classical repeated-measures ANOVA with a sphericity correction?"*
**Ideal answer:** A mixed-effects model lets you directly specify (or flexibly estimate) the actual correlation structure among repeated measurements — e.g., an autoregressive structure if measurements closer together in time are more correlated — rather than assuming the restrictive compound-symmetry/sphericity structure and applying a blunt degrees-of-freedom correction after detecting a violation. This tends to be both more accurate (modeling the true dependence structure rather than just discounting for its absence) and more flexible (naturally handling missing data or unbalanced designs, which the classical repeated-measures ANOVA framework struggles with).

---

## Python Implementation

```python
import numpy as np
from scipy import stats

Y = np.array([
    [13,15,17],
    [14,16,21],
    [15,20,22],
    [19,21,23],
    [20,22,27],
    [21,26,28]
], dtype=float)
n, t = Y.shape  # n=6 subjects, t=3 treatments

subject_means = Y.mean(axis=1)
treatment_means = Y.mean(axis=0)
grand_mean = Y.mean()

SS_subjects = t * np.sum((subject_means - grand_mean)**2)
SS_treatment = n * np.sum((treatment_means - grand_mean)**2)
SSTO = np.sum((Y - grand_mean)**2)
SS_remainder = SSTO - SS_subjects - SS_treatment

df_treatment, df_remainder = t-1, (n-1)*(t-1)
F_stat = (SS_treatment/df_treatment) / (SS_remainder/df_remainder)
print(f"F={F_stat:.2f}, critical={stats.f.ppf(0.95, df_treatment, df_remainder):.2f}")

# --- Check sphericity: variance of each pairwise difference ---
pairs = [(0,1), (0,2), (1,2)]
labels = ['A-B','A-C','B-C']
for (i,j), label in zip(pairs, labels):
    diff = Y[:,i] - Y[:,j]
    print(f"{label}: mean={diff.mean():.2f}, variance={diff.var(ddof=1):.3f}")
```

```python
# --- Full repeated-measures ANOVA with sphericity correction, via pingouin ---
import pandas as pd
import pingouin as pg

rows = []
for i in range(n):
    for j, design in enumerate(['A','B','C']):
        rows.append({'Subject': f"S{i+1}", 'Design': design, 'Score': Y[i,j]})
df = pd.DataFrame(rows)

aov = pg.rm_anova(data=df, dv='Score', within='Design', subject='Subject', detailed=True)
print(aov)  # includes Mauchly's test and Greenhouse-Geisser correction automatically

# --- Modern alternative: mixed model with explicit covariance structure ---
import statsmodels.formula.api as smf
mixed = smf.mixedlm("Score ~ C(Design)", df, groups=df["Subject"]).fit()
print(mixed.summary())
```

---

## Interview Question Bank — Chapter 23

**Conceptual:**
1. In what precise sense is a repeated-measures design mathematically identical to Chapter 21's RCBD?
2. Why does a repeated-measures design typically have more statistical power than an equivalent between-subjects design?
3. What is sphericity, and why is it a more pressing concern in repeated measures than in ordinary blocking?

**Derivation:**
4. Derive the sum-of-squares decomposition for a repeated-measures ANOVA from first principles, and show it matches Chapter 21's RCBD formulas exactly.
5. Explain, using the variance-of-differences framing, exactly what sphericity requires and how you'd check it directly from data.

**ML/Statistics:**
6. What's the practical consequence of violating sphericity and not correcting for it?
7. Compare the Greenhouse-Geisser correction approach to using a mixed-effects model with an explicit covariance structure — what's the key philosophical difference?
8. Why is counterbalancing necessary in repeated measures designs but not typically a concern in Chapter 21's block designs?

**Coding:**
9. Implement the repeated-measures ANOVA sum-of-squares decomposition from scratch in NumPy, and verify against a library like `pingouin`.
10. Compute the variance of every pairwise treatment difference from a repeated-measures dataset to directly check sphericity.

**Traps:**
11. "Since each subject tries every treatment, we don't need to worry about the order they experience them in." — what's the flaw, and what's the standard fix?
12. "The repeated-measures F-test is always more powerful than a between-subjects test, so it should always be preferred." — under what circumstance might repeated measures actually be inappropriate or infeasible (hint: think about carryover effects that can't be counterbalanced away, like a permanent behavior change)?
13. Someone runs a repeated-measures ANOVA, ignores a Mauchly's test warning about violated sphericity, and reports the uncorrected p-value. What's the risk?

---

*This file covers Kutner Ch. 23 — the within-subjects repeated measures design and its direct mathematical identity to Chapter 21's RCBD, order effects and counterbalancing as a genuinely new practical concern, the sphericity assumption (worked and confirmed to hold via pairwise difference variances), Mauchly's test and the Greenhouse-Geisser/Huynh-Feldt corrections, and the modern mixed-effects-model alternative that sidesteps the sphericity issue entirely. This substantially completes the core design-of-experiments arc most relevant to ML/DS interview and applied-analytics work — remaining Kutner chapters (Latin squares, fractional factorial designs, response surface methodology) move into specialized industrial DOE territory with comparatively little interview relevance.*
