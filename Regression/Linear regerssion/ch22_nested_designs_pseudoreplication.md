# Chapter 22 — Nested Designs, Subsampling, and the Pseudo-Replication Trap
### (Kutner, Nachtsheim, Neter, Li — *Applied Linear Statistical Models*)

Chapters 18–21 all assumed **crossed** factors — every level of Factor A appears alongside every level of Factor B. This chapter covers **nested** factors — where that's structurally impossible — and, in doing so, exposes one of the single most common and consequential errors in real-world applied data analysis: **pseudo-replication**, using the wrong error term and dramatically overstating your true sample size.

---

## 22.1 Nested vs. Crossed Factors: the Core Distinction

**Plain English.** In a **crossed** design (Chapters 18–20), "Design A, Traffic Organic" and "Design A, Traffic Paid" are both meaningful, comparable combinations — every Design level pairs with every Traffic level. In a **nested** design, this pairing doesn't make sense. Consider: 3 raw-material **Suppliers**, and from each supplier, 2 **Batches** of material are received. "Batch 1" from Supplier 1 and "Batch 1" from Supplier 2 are not the same thing, not comparable, and not "crossed" in any meaningful sense — they're just arbitrarily numbered, entirely distinct physical batches that happen to share a label. **Batch is nested within Supplier.**

**Why this distinction is not just semantic — it changes the entire error structure of the analysis**, as the rest of this chapter shows.

---

## 22.2 The Nested Model

$$
Y_{ijk} = \mu+\alpha_i+\beta_{j(i)}+\varepsilon_{k(ij)}
$$

**Notation, explained precisely:**
- $\alpha_i$: effect of Supplier $i$ (the factor of primary interest).
- $\beta_{j(i)}$: effect of Batch $j$, **nested within** Supplier $i$ — read "$j(i)$" as "batch $j$, which only exists inside supplier $i$." There is no single, shared "Batch 1 effect" across suppliers; each supplier has its own private set of batch effects.
- $\varepsilon_{k(ij)}$: the **subsampling error** — repeated measurements $k$ taken from the *same physical batch* $j(i)$. These are not independent replicates of the *treatment* (Supplier) — they're just repeated looks at one single batch.

### The Worked Example

Three Suppliers, 2 batches each (nested — Supplier 1's two batches are physically distinct material lots from Supplier 2's two batches), and 2 measurements (subsamples) taken from each batch — 12 total observations. Response: a quality measurement.

| Supplier | Batch | Measurements | Batch mean |
|---|---|---|---|
| S1 | B1 | 47, 49 | 48 |
| S1 | B2 | 45, 47 | 46 |
| S2 | B1 | 51, 53 | 52 |
| S2 | B2 | 47, 49 | 48 |
| S3 | B1 | 51, 53 | 52 |
| S3 | B2 | 53, 55 | 54 |

**Supplier means:** $\bar Y_{S1}=47,\ \bar Y_{S2}=50,\ \bar Y_{S3}=53$. **Grand mean:** $50$.

---

## 22.3 The Sum-of-Squares Decomposition

$$
SSTO = SSA + SSB(A) + SSE
$$

### $SSA$ — Between Suppliers

$$
SSA = (bn)\sum_i(\bar Y_{i..}-\bar Y_{...})^2, \qquad b=2\text{ batches/supplier},\ n=2\text{ subsamples/batch}
$$
$$
= 4\left[(-3)^2+0^2+3^2\right]=4(18)=72
$$

### $SSB(A)$ — Batches Within Suppliers ("Nested" Sum of Squares)

**Plain English.** This measures how much batches vary *within* each supplier — batch-to-batch variability that has nothing to do with which supplier it's from, but everything to do with natural lot-to-lot variation.
$$
SSB(A) = n\sum_i\sum_j(\bar Y_{ij.}-\bar Y_{i..})^2
$$
Computing each supplier's batch deviations from its own mean: S1 ($48,46$ vs. mean $47$: deviations $\pm1$, sum of squares $2$); S2 ($52,48$ vs. mean $50$: deviations $\pm2$, sum of squares $8$); S3 ($52,54$ vs. mean $53$: deviations $\mp1$, sum of squares $2$).
$$
SSB(A) = 2\times(2+8+2) = 2(12)=24
$$

### $SSE$ — Subsampling Error (True Pure Error)

Each batch's two measurements deviate from their own batch mean by $\pm1$, contributing $1^2+1^2=2$ per batch:
$$
SSE = 6\text{ batches}\times2=12
$$

### The Full Table

| Source | SS | df | MS |
|---|---|---|---|
| Supplier (A) | 72 | $a-1=2$ | 36 |
| Batch within Supplier, B(A) | 24 | $a(b-1)=3$ | 8 |
| Error (subsampling), E | 12 | $ab(n-1)=6$ | 2 |
| Total | 108 | $abn-1=11$ | |

(Check: $2+3+6=11$ ✓; $72+24+12=108$, verified directly against $\sum(Y_{ijk}-50)^2$ summed over all 12 observations.)

---

## 22.4 The Critical Lesson: Which Mean Square Is the *Correct* Denominator?

### The Mistake Almost Everyone Makes the First Time

**It's extremely tempting to test the Supplier effect using $MSE$ (the subsampling error) as the denominator**, since that's the "smallest," most granular error term available, and it's what a naive one-way-ANOVA-style analysis (treating all 12 measurements as if they were 12 independent observations) would default to:
$$
F_{\text{(incorrect)}} = \frac{MSA}{MSE} = \frac{36}{2}=18.0
$$
Compared to $F_{(0.95;2,6)}=5.14$: this looks **overwhelmingly significant** ($18.0\gg5.14$).

### Why This Is Wrong

**The two measurements taken from the same batch are not independent replicates of the Supplier effect — they're just two looks at the *same* physical batch.** The *true* unit of replication for comparing Suppliers is the **batch**, not the individual measurement. Each supplier really only contributes **2 independent pieces of information** (its 2 batches) about what that supplier is "truly" like — not 4. **Treating the 4 measurements per supplier as if they were 4 independent replicates inflates the apparent sample size and understates the true uncertainty in the Supplier comparison** — this is **pseudo-replication**, and it is one of the single most common, consequential errors in applied statistics and data science.

### The Correct Test

$$
F_{\text{(correct)}} = \frac{MSA}{MSB(A)} = \frac{36}{8}=4.5
$$
Compared to $F_{(0.95;2,3)}=9.55$ (note: $df=3$, from the *nested batch* term, not 6): since $4.5<9.55$, **we fail to reject $H_0$ — there is NOT significant evidence that suppliers differ**, once the analysis correctly accounts for the fact that batches, not individual measurements, are the true experimental unit.

**The contrast is stark and worth sitting with: $F=18.0$ (using the wrong error term) says "overwhelming evidence of a real supplier difference." $F=4.5$ (using the correct error term) says "not enough evidence to conclude suppliers really differ."** Same data, two different conclusions, purely because of which mean square was used as the denominator.

**Why this makes intuitive sense, beyond the formal derivation:** with only 2 batches per supplier, you genuinely don't have much information about whether a supplier's *true, long-run* quality level differs from another's — a lot of what you're seeing could just be batch-to-batch luck (recall $MSB(A)=8$, a fairly substantial batch-to-batch variance relative to $MSA=36$). Measuring each of those 2 batches twice (the subsamples) tells you more precisely what *those two particular batches* measured — but it tells you *nothing new* about whether the supplier, in general, differs from another supplier. More subsamples of the same batches would keep making $MSE$ smaller and smaller, but would never legitimately increase your confidence about the *supplier*-level comparison — only more **batches** would do that.

**Interview question:** *"Your team ran an experiment on 3 servers, with 100 requests logged per server, and found a highly significant difference between servers using a standard ANOVA on all 300 requests. What should you check before trusting that result?"*
**Ideal answer:** Check whether the 100 requests per server are truly independent replicates, or whether they're more like subsamples/repeated measurements from the same underlying unit (the server itself, and whatever state or configuration it was in during the test). If servers themselves are the real unit of comparison and you only have 1 server per condition (or a small number), then the 100 requests per server are analogous to this chapter's subsamples — pseudo-replicates that inflate the apparent sample size and understate true uncertainty. The correct analysis would use the between-server (or between-unit) variability as the error term for comparing conditions, not the pooled within-server variability across all 300 individual requests.

---

## 22.5 The Direct Connection to Hierarchical / Mixed-Effects Models

**This entire chapter is the classical-ANOVA ancestor of hierarchical (multilevel) models**, directly continuing Chapter 16's fixed-vs-random-effects discussion. **Batch-within-Supplier is naturally modeled as a random effect**: batches are a sample of the many possible batches a supplier could produce, and you want your conclusions about the *supplier* to generalize beyond just these 2 specific observed batches. A modern mixed-effects model for this exact data would specify a random intercept for batch, nested within a fixed effect for supplier — and critically, **fitting it correctly automatically produces the right error structure for testing the supplier effect**, without requiring you to manually figure out which mean square is the "correct" denominator, as we had to do by hand above. This is a major practical reason mixed-effects software (e.g., `lme4` in R, `statsmodels.MixedLM` in Python) is so valuable for real hierarchical data: **it encodes the correct nested error structure automatically**, precisely avoiding the pseudo-replication trap that a naive flat analysis (or a poorly-specified fixed-effects ANOVA) would fall into.

**The broader, extremely common real-world pattern this generalizes to:** any time your data has a natural hierarchy — users nested within stores, sessions nested within users, pageviews nested within sessions, repeated sensor readings nested within a device — **the innermost, most granular level of data is essentially always "subsamples,"** and naively treating every single row as an independent observation when computing a standard error or p-value is the pseudo-replication trap in its most common modern guise. This shows up constantly in A/B testing (multiple events per user, incorrectly treated as independent when computing test significance), IoT/sensor analysis (multiple readings per device), and any clustered or panel dataset.

**Interview question:** *"In an A/B test, you have 10,000 users but log 500,000 total page-view events across them. Why is it a mistake to compute your significance test's standard error as if you had 500,000 independent observations?"*
**Ideal answer:** Page-view events from the same user are correlated with each other (a given user's behavior tends to be consistent within themselves across their own events) — they are not independent replicates in the sense a standard significance test assumes. This is exactly this chapter's pseudo-replication problem: the true unit of independent replication is the *user* (10,000), not the *event* (500,000), and computing a standard error as if you had 500,000 independent data points will produce a dramatically understated standard error and a falsely significant result. The correct approach aggregates to the user level first (or uses a clustered/hierarchical standard error that properly accounts for within-user correlation) before computing significance.

---

## Python Implementation

```python
import numpy as np
from scipy import stats

# Data: 3 suppliers, 2 batches each, 2 subsamples per batch
data = {
    ('S1','B1'): [47,49], ('S1','B2'): [45,47],
    ('S2','B1'): [51,53], ('S2','B2'): [47,49],
    ('S3','B1'): [51,53], ('S3','B2'): [53,55],
}

suppliers = ['S1','S2','S3']
batch_means = {k: np.mean(v) for k,v in data.items()}
supplier_means = {s: np.mean([batch_means[(s,b)] for b in ['B1','B2']]) for s in suppliers}
grand_mean = np.mean(list(supplier_means.values()))

b, n = 2, 2  # batches per supplier, subsamples per batch
SSA = b*n*sum((supplier_means[s]-grand_mean)**2 for s in suppliers)
SSB_A = n*sum((batch_means[(s,bt)]-supplier_means[s])**2 for s in suppliers for bt in ['B1','B2'])
SSE = sum((v[0]-np.mean(v))**2 + (v[1]-np.mean(v))**2 for v in data.values())

dfA, dfB_A, dfE = len(suppliers)-1, len(suppliers)*(b-1), len(suppliers)*b*(n-1)
MSA, MSB_A, MSE = SSA/dfA, SSB_A/dfB_A, SSE/dfE

F_incorrect = MSA/MSE
F_correct = MSA/MSB_A

print(f"SSA={SSA}, SSB(A)={SSB_A}, SSE={SSE}")
print(f"F using MSE (INCORRECT, pseudo-replication): {F_incorrect:.2f}, "
      f"critical={stats.f.ppf(0.95, dfA, dfE):.2f}")
print(f"F using MSB(A) (CORRECT): {F_correct:.2f}, "
      f"critical={stats.f.ppf(0.95, dfA, dfB_A):.2f}")
```

```python
# --- Mixed-effects model automatically gets the error structure right ---
import pandas as pd
import statsmodels.formula.api as smf

rows = []
for (s, bt), vals in data.items():
    for v in vals:
        rows.append({'Supplier': s, 'Batch': f"{s}_{bt}", 'Y': v})
df = pd.DataFrame(rows)

# Batch nested within Supplier as a random effect
mixed = smf.mixedlm("Y ~ C(Supplier)", df, groups=df["Batch"]).fit()
print(mixed.summary())
```

---

## Interview Question Bank — Chapter 22

**Conceptual:**
1. What makes a factor "nested" rather than "crossed," using the Supplier/Batch example?
2. Why is $MSE$ (subsampling error) the *wrong* denominator for testing the Supplier effect?
3. What is pseudo-replication, in one sentence, and why does it typically make results look more significant than they really are?

**Derivation:**
4. Derive $SSB(A)$ and explain, term by term, what it measures that $SSA$ does not.
5. Explain precisely why more subsamples per batch would never legitimately increase confidence in the Supplier-level comparison.

**ML/Statistics:**
6. Connect this chapter's nested design directly to a mixed-effects/hierarchical model, and explain why fitting the mixed model correctly sidesteps the "which mean square do I use" problem entirely.
7. Give a concrete modern data-science example (outside manufacturing) where pseudo-replication commonly occurs, and explain the correct fix.
8. Why does this chapter's lesson matter specifically for clustered or panel data in A/B testing?

**Coding:**
9. Implement the nested ANOVA decomposition (SSA, SSB(A), SSE) from scratch in NumPy/pandas for an arbitrary nested design.
10. Fit both a naive flat ANOVA and a correctly-specified mixed-effects model on the same nested dataset, and compare the resulting significance conclusions.

**Traps:**
11. "We have 12 measurements, so our degrees of freedom for comparing suppliers should be based on n=12." — what's the correct number of truly independent units, and why?
12. "Since the mixed-effects model output looks more complicated, the simpler flat ANOVA is safer to use." — what's actually the more dangerous choice here, and why?
13. A colleague argues that since F=18 (using MSE) is "more significant" and therefore "more conservative to publish," it must be the more careful choice. What's wrong with this reasoning?

---

*This file covers Kutner Ch. 22 — the distinction between nested and crossed factors, the nested-design sum-of-squares decomposition, and the pseudo-replication trap worked in full and dramatically (F=18.0 using the wrong error term vs. F=4.5 using the correct one, reversing the statistical conclusion entirely) — arguably one of the single most practically valuable lessons in applied statistics, and a direct bridge to why hierarchical/mixed-effects models matter for any naturally clustered real-world dataset. Chapter 23 (Repeated Measures Designs) is next if you'd like to continue — formally handling the case where the same experimental unit is measured multiple times across conditions, requiring explicit modeling of within-unit correlation.*
